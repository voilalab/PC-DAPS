import tqdm
import torch
import numpy as np
import torch.nn as nn
from cores.trajectory import Trajectory
from cores.scheduler import get_diffusion_scheduler, DiffusionPFODE
from cores.mcmc import MCMCSampler
from cores.pcdaps import (GaussianMixturePrior, coarse_features, cluster_bag, prior_consistency, prune_bag,
                          within_cluster_score_gaps)


def get_sampler(**kwargs):
    name = kwargs.pop('name', 'daps')
    if name == 'pcdaps':
        return PCDAPS(**kwargs)
    return DAPS(**kwargs)


class DAPS(nn.Module):
    """
    Decoupled Annealing Posterior Sampling (DAPS) implementation.

    Combines diffusion models and MCMC updates for posterior sampling from noisy measurements.
    """

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config):
        """
        Initializes the DAPS sampler with the provided scheduler and sampler configurations.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
            mcmc_sampler_config (dict): Configuration for MCMC sampler.
        """
        super().__init__()
        annealing_scheduler_config, diffusion_scheduler_config = self._check(annealing_scheduler_config,
                                                                             diffusion_scheduler_config)
        self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.mcmc_sampler = MCMCSampler(**mcmc_sampler_config)

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs sampling using the DAPS method.

        Args:
            model (nn.Module): Diffusion model.
            x_start (torch.Tensor): Initial tensor/state.
            operator (nn.Module): Measurement operator.
            measurement (torch.Tensor): Observed measurement tensor.
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation.

        Returns:
            torch.Tensor: Final sampled tensor/state.
        """
        if record:
            self.trajectory = Trajectory()
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1) if verbose else range(self.annealing_scheduler.num_steps - 1)
        xt = x_start
        for step in pbar:
            sigma = self.annealing_scheduler.sigma_steps[step]
            # 1. reverse diffusion
            with torch.no_grad():
                diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma)
                sampler = DiffusionPFODE(model, diffusion_scheduler, solver='euler')
                x0hat = sampler.sample(xt)

            # 2. MCMC update
            x0y = self.mcmc_sampler.sample(xt, model, x0hat, operator, measurement, sigma, step / self.annealing_scheduler.num_steps)

            # 3. forward diffusion
            if step != self.annealing_scheduler.num_steps - 1:
                xt = x0y + torch.randn_like(x0y) * self.annealing_scheduler.sigma_steps[step + 1]
            else:
                xt = x0y

            # 4. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and 'gt' in kwargs:
                with torch.no_grad():
                    gt = kwargs['gt']
                    x0hat_results = evaluator(gt, measurement, x0hat)
                    x0y_results = evaluator(gt, measurement, x0y)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                    })
            if record:
                self._record(xt, x0y, x0hat, sigma, x0hat_results, x0y_results)
        return xt

    def _record(self, xt, x0y, x0hat, sigma, x0hat_results, x0y_results):
        """Records the intermediate states during sampling."""

        self.trajectory.add_tensor(f'xt', xt)
        self.trajectory.add_tensor(f'x0y', x0y)
        self.trajectory.add_tensor(f'x0hat', x0hat)
        self.trajectory.add_value(f'sigma', sigma)
        for name in x0hat_results.keys():
            self.trajectory.add_value(f'x0hat_{name}', x0hat_results[name])
        for name in x0y_results.keys():
            self.trajectory.add_value(f'x0y_{name}', x0y_results[name])

    def _check(self, annealing_scheduler_config, diffusion_scheduler_config):
        """Checks and updates the configurations for the schedulers."""

        # sigma_max of diffusion scheduler change each step
        if 'sigma_max' in diffusion_scheduler_config:
            diffusion_scheduler_config.pop('sigma_max')

        return annealing_scheduler_config, diffusion_scheduler_config

    def get_start(self, batch_size, model):
        """
        Generates initial random state tensors from the Gaussian prior.

        Args:
            batch_size (int): Number of initial states to generate.
            model (nn.Module): Diffusion model.

        Returns:
            torch.Tensor: Random initial tensor.
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        x_start = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
        return x_start


class PCDAPS(DAPS):
    """
    Prior-Consistent DAPS (PC-DAPS), Algorithm 1.

    Anneals a bag of K particles per measurement with the DAPS denoise / Langevin / re-noise loop, and
        * replaces the Gaussian approximation of p(x0 | xt) with a Gaussian mixture estimated from the bag,
        * prunes prior-inconsistent modes once, when the bag occupies several likelihood wells,
        * integrates the PF-ODE with fewer Euler steps at high noise and skips the final Langevin refinement.
    """

    DEFAULTS = dict(
        bag_size=4,                    
        ode_steps_high_noise=1,       
        ode_switch_ratio=0.65,     
        skip_final_langevin=True,     
        prune=True,                    
        gamma=0.065,                   
        residual_test='noisy',         
        stop_after_first_check=True,   
        pruning_stages=(              
            dict(residual_test='tweedie', gamma=0.15, stop_after_first_check=False), 
            dict(residual_test='noisy', gamma=0.065, stop_after_first_check=True, sigma_max=1.0),  
        ),                            
        score_sigmas=(0.5, 1.0, 2.0),  
        score_num_samples=1,           
        cluster_resolution=32,        
        cluster_pca_variance=0.9,      
        cluster_eps=0.15,               
        max_clusters=None,             
        mixture_center='tweedie',      
        var_floor=1e-4,                
        langevin_scheme='semi-implicit',  
        cluster_before_pruning=False,   
        calibrate=False,
        seed=0,                       
    )

    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config, pcdaps_config=None):
        """
        Initializes the PC-DAPS sampler.

        Args:
            annealing_scheduler_config (dict): Configuration for annealing scheduler.
            diffusion_scheduler_config (dict): Configuration for diffusion scheduler; `num_steps` is the number of
                PF-ODE Euler steps at low noise.
            mcmc_sampler_config (dict): Configuration for the Langevin sampler.
            pcdaps_config (dict, optional): PC-DAPS options overriding PCDAPS.DEFAULTS.
        """
        super().__init__(annealing_scheduler_config, diffusion_scheduler_config, mcmc_sampler_config)
        pcdaps_config = dict(pcdaps_config or {})
        unknown = set(pcdaps_config) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown PC-DAPS options: {sorted(unknown)}')
        for key, value in {**self.DEFAULTS, **pcdaps_config}.items():
            setattr(self, key, value)
        if self.mixture_center not in ('tweedie', 'ode'):
            raise ValueError(f'Unknown mixture_center {self.mixture_center}.')
        self.score_sigmas = [float(s) for s in self.score_sigmas]
        self.orient = False   
        self.rng = np.random.default_rng(self.seed)
        stage = dict(residual_test=self.residual_test, gamma=self.gamma, stop_after_first_check=self.stop_after_first_check,
                     sigma_max=None)
        self.stages = [{**stage, **dict(config)} for config in (self.pruning_stages or [{}])]
        for config in self.stages:
            if set(config) != set(stage):
                raise ValueError(f'Unknown pruning stage options: {sorted(set(config) - set(stage))}')

    def sample(self, model, x_start, operator, measurement, evaluator=None, record=False, verbose=False, **kwargs):
        """
        Performs posterior sampling with PC-DAPS.

        Args:
            model (nn.Module): Diffusion model.
            x_start (torch.Tensor): Initial bags [B, K, ...].
            operator (nn.Module): Measurement operator.
            measurement (torch.Tensor): Observed measurements [B, ...].
            evaluator (Evaluator, optional): Evaluator for performance metrics.
            record (bool, optional): If True, records the sampling trajectory.
            verbose (bool, optional): Enables progress bar and logs.
            **kwargs:
                gt (torch.Tensor, optional): Ground truth data for evaluation [B, ...].

        Returns:
            torch.Tensor: Final bags [B, K, ...].
        """
        if record:
            self.trajectory = Trajectory()
        B, K = x_start.shape[:2]
        shape = x_start.shape[2:]
        num_steps = self.annealing_scheduler.num_steps - 1
        y = measurement.repeat_interleave(K, dim=0)
        gt = kwargs.get('gt')
        gt = gt.repeat_interleave(K, dim=0) if gt is not None else None
        self.pruned = np.zeros(B, dtype=bool)   # bags whose mixture is built from clusters (Eqs. 11-12)
        self.checked = np.zeros((len(self.stages), B), dtype=bool)  # bags that consumed each one-shot check
        self.prune_log = []
        oriented = False

        xt = x_start.flatten(0, 1)
        pbar = tqdm.trange(num_steps) if verbose else range(num_steps)
        for step in pbar:
            sigma = float(self.annealing_scheduler.sigma_steps[step])
            sigma_next = float(self.annealing_scheduler.sigma_steps[step + 1])

            # 1. clean estimates: PF-ODE estimate (Langevin initialization) and Tweedie estimate (mixture by default)
            x0hat, x0tilde = self.ode_estimate(model, xt, sigma, self.get_ode_steps(step, num_steps))

            if step == num_steps - 1 and self.skip_final_langevin:
                x0y = x0hat
            else:
                # 2. Langevin refinement with the Gaussian mixture approximation of p(x0 | xt)
                centers = x0tilde if self.mixture_center == 'tweedie' else x0hat
                prior = self.build_prior(centers.view(B, K, *shape))
                x0y = self.mcmc_sampler.sample_mixture(x0hat, prior, operator, y, step / self.annealing_scheduler.num_steps,
                                                       self.langevin_scheme)

                if self.orient and not oriented and sigma <= 0.3:
                    with torch.random.fork_rng(devices=[x0hat.device] if x0hat.is_cuda else []):
                        torch.manual_seed(self.seed + 1)
                        scores, twin_scores = prior_consistency(model, x0hat.view(B, K, *shape), self.score_sigmas, 4,
                                                                twin=True)
                    x0y = torch.where((twin_scores > scores).view(-1, *[1] * len(shape)), x0y.flip(-2, -1), x0y)
                    oriented = True

                # 3. one-shot consensus pruning
                if self.prune and not self.checked.all():
                    x0y = self.consensus_pruning(model, operator, measurement, xt.view(B, K, *shape),
                                                 x0tilde.view(B, K, *shape), x0y.view(B, K, *shape), step, sigma)
                    x0y = x0y.flatten(0, 1)

            # 4. re-noising (sigma_next = 0 at the last step)
            xt = x0y + torch.randn_like(x0y) * sigma_next

            # 5. evaluation
            x0hat_results = x0y_results = {}
            if evaluator and gt is not None and (verbose or record):
                with torch.no_grad():
                    x0hat_results = self._evaluate(evaluator, gt, y, x0hat, full=record)
                    x0y_results = self._evaluate(evaluator, gt, y, x0y, full=record)

                # record
                if verbose:
                    main_eval_fn_name = evaluator.main_eval_fn_name
                    pbar.set_postfix({
                        'x0hat' + '_' + main_eval_fn_name: f"{x0hat_results[main_eval_fn_name].item():.2f}",
                        'x0y' + '_' + main_eval_fn_name: f"{x0y_results[main_eval_fn_name].item():.2f}",
                        'pruned': f"{int(self.pruned.sum())}/{B}",
                    })
            if record:
                self._record(xt.view(B, K, *shape), x0y.view(B, K, *shape), x0hat.view(B, K, *shape), sigma,
                             x0hat_results, x0y_results)
                self.trajectory.add_value('num_pruned', int(self.pruned.sum()))
        return xt.view(B, K, *shape)

    @torch.no_grad()
    def ode_estimate(self, model, xt, sigma, num_steps):
        """
        Integrates the PF-ODE from sigma with `num_steps` Euler steps (as DiffusionPFODE with the euler solver).

        Returns:
            Tuple of (x0hat, x0tilde): the ODE endpoint and the Tweedie estimate D(xt; sigma), which is the denoiser
            output of the first Euler step and thus costs no additional network evaluation.
        """
        config = dict(self.diffusion_scheduler_config)
        config.update(num_steps=num_steps, sigma_max=sigma)
        scheduler = get_diffusion_scheduler(**config)
        time_steps = scheduler.get_discrete_time_steps(scheduler.num_steps).to(xt.device)
        x, x0tilde = xt, None
        for t, t_next in zip(time_steps[:-1], time_steps[1:]):
            st, dst = scheduler.get_scaling(t), scheduler.get_scaling_derivative(t)
            sigma_t, dsigma_t = scheduler.get_sigma(t), scheduler.get_sigma_derivative(t)
            denoised = model.tweedie(x / st, sigma_t)
            if x0tilde is None:
                x0tilde = denoised
            score = (denoised - x / st) / sigma_t ** 2
            x = x + (t_next - t) * (dst / st * x - st * dsigma_t * sigma_t * score)
        return x, x0tilde

    def get_ode_steps(self, step, num_steps):
        """Number of PF-ODE Euler steps at annealing step `step` (adaptive ODE stepping, App. A.4)."""
        if self.ode_steps_high_noise is not None and step < self.ode_switch_ratio * num_steps:
            return self.ode_steps_high_noise
        return self.diffusion_scheduler_config['num_steps']

    def build_prior(self, x0):
        """
        Gaussian mixture approximation of p(x0 | xt) of every bag from its clean estimates [B, K, ...] (Tweedie or
        PF-ODE estimates, see mixture_center): one component per particle before pruning (Eq. 10), one per cluster
        afterwards (Eqs. 11-12).
        """
        B = x0.shape[0]
        labels = [None] * B
        clustered = np.flatnonzero(self.pruned | self.cluster_before_pruning)
        if len(clustered) > 0:
            features = self._features(x0[clustered])
            for i, b in enumerate(clustered):
                labels[b] = self._cluster(features[i])
        return GaussianMixturePrior.from_bag(x0.flatten(2), labels, self.var_floor)

    @torch.no_grad()
    def consensus_pruning(self, model, operator, measurement, xt, x0tilde, x0y, step, sigma):
        """
        Consensus pruning (Alg. 1, lines 16-21) of the post-Langevin particles x0y [B, K, ...].
        """
        for index, stage in enumerate(self.stages):
            if stage['sigma_max'] is None or sigma <= stage['sigma_max']:
                x0y = self._pruning_check(index, stage, model, operator, measurement, xt, x0tilde, x0y, step, sigma)
        return x0y

    def _pruning_check(self, index, stage, model, operator, measurement, xt, x0tilde, x0y, step, sigma):
        """One pruning stage (see consensus_pruning) for the bags that have not consumed it."""
        B, K = x0y.shape[:2]
        active = np.flatnonzero(~self.checked[index])
        if len(active) == 0:
            return x0y
        particles = {'noisy': xt, 'tweedie': x0tilde, 'post_langevin': x0y}[stage['residual_test']][active]
        residuals = operator.loss(particles.flatten(0, 1), measurement[active].repeat_interleave(K, dim=0))
        r_bar = residuals.view(len(active), K).mean(1).cpu().numpy()
        r_star = operator.loss(particles.mean(1), measurement[active]).cpu().numpy()
        fired = active[r_bar < r_star]
        if len(fired) == 0:
            return x0y

        features = self._features(x0y[fired])
        labels = [self._cluster(f) for f in features]
        if not self.calibrate:
            multimodal = [i for i, lab in enumerate(labels) if lab.max() > 0]
            fired, labels = fired[multimodal], [labels[i] for i in multimodal]
            if len(fired) == 0:
                return x0y

        scores = prior_consistency(model, x0y[fired], self.score_sigmas, self.score_num_samples).cpu().numpy()
        # measurement residual of every scored particle (logged only)
        particle_residuals = operator.loss(x0y[fired].flatten(0, 1), measurement[fired].repeat_interleave(K, dim=0))
        particle_residuals = particle_residuals.view(len(fired), K).cpu().numpy()
        position = {b: i for i, b in enumerate(active)}
        x0y = x0y.clone()
        for j, b in enumerate(fired):
            entry = dict(bag=int(b), stage=index, step=int(step), sigma=float(sigma), r_bar=float(r_bar[position[b]]),
                         r_star=float(r_star[position[b]]), labels=labels[j].tolist(), scores=scores[j].tolist(),
                         residuals=particle_residuals[j].tolist())
            if self.calibrate:
                entry['gaps'] = within_cluster_score_gaps(scores[j], labels[j])
                self.prune_log.append(entry)
                continue
            sources, cluster_scores, keep = prune_bag(scores[j], labels[j], stage['gamma'], self.rng)
            if keep.all() and not stage['stop_after_first_check']:
                continue
            x0y[b] = x0y[b, sources]
            self.checked[index, b] = self.pruned[b] = True
            entry.update(cluster_scores=cluster_scores.tolist(), kept_clusters=keep.tolist(), sources=sources.tolist())
            self.prune_log.append(entry)
        return x0y

    def _features(self, x):
        """Coarse clustering features [n, K, F] (numpy) of particles x [n, K, ...]."""
        n, K = x.shape[:2]
        features = coarse_features(x.flatten(0, 1).float(), self.cluster_resolution)
        return features.view(n, K, -1).double().cpu().numpy()

    def _cluster(self, features):
        return cluster_bag(features, self.cluster_pca_variance, self.cluster_eps, self.max_clusters, self.seed)

    def _evaluate(self, evaluator, gt, measurement, x, full):
        """All metrics (recording) or only the main one (progress bar)."""
        if full:
            return evaluator(gt, measurement, x)
        return {evaluator.main_eval_fn_name: evaluator.get_main_eval_fn()(gt, measurement, x, reduction='mean')}

    def get_start(self, batch_size, model):
        """
        Generates initial bags of K particles from the Gaussian prior.

        Args:
            batch_size (int): Number of measurements.
            model (nn.Module): Diffusion model.

        Returns:
            torch.Tensor: Random initial bags [batch_size, K, ...].
        """
        device = next(model.parameters()).device
        in_shape = model.get_in_shape()
        return torch.randn(batch_size, self.bag_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
