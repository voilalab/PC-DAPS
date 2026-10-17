"""
Building blocks of Prior-Consistent DAPS (PC-DAPS).

    * GaussianMixturePrior       -- Gaussian-mixture approximation of the denoising distribution
                                    p(x0 | xt) built from a bag of particles (Sec. 4.2, Eqs. 10-13).
    * coarse_features / cluster_bag
                                 -- partition of a bag into modes (App. A.4): coarse downsampling,
                                    centering, PCA to 90% of the variance, spherical Gaussian mixture
                                    with the number of components selected by BIC, hard assignment.
    * prior_consistency          -- Tweedie prior-consistency score of every particle (Eq. 9).
    * prune_bag                  -- relative-threshold pruning of low-scoring clusters (Alg. 1, App. A.4).
    * within_cluster_score_gaps  -- statistics used to calibrate the pruning threshold gamma (App. A.4).
"""
from itertools import combinations
from math import comb

import numpy as np
import torch
import torch.nn.functional as F


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gaussian mixture approximation of the denoising distribution
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GaussianMixturePrior:
    """
    Diagonal Gaussian mixture q(x0 | xt) = sum_m pi_m N(x0; mu_m, Sigma_m), one mixture per bag.

    Attributes:
        means (torch.Tensor): Component means [B, M, D].
        inv_var (torch.Tensor): Diagonal component precisions [B, M, D].
        log_norm (torch.Tensor): log pi_m - 0.5 log|Sigma_m| [B, M] (float64); -inf marks unused components.
    """

    def __init__(self, means, variances, log_weights):
        self.means = means
        self.inv_var = variances.reciprocal()
        self.mean_prec = means * self.inv_var
        self.log_norm = log_weights.double() - 0.5 * variances.log().sum(-1, dtype=torch.float64)

    @classmethod
    def from_bag(cls, x0, labels=None, var_floor=1e-4):
        """
        Builds the mixture of every bag from the clean estimates of its particles.

        Args:
            x0 (torch.Tensor): Clean estimates [B, K, D].
            labels (list, optional): Per bag, either None -- one uniformly weighted component per particle
                with the bag's diagonal covariance (Eq. 10) -- or an int array [K] of cluster labels -- one
                component per cluster with its empirical weight, mean and diagonal covariance (Eqs. 11-12).
                Singleton clusters use the bag's diagonal covariance.
            var_floor (float): Lower bound on every diagonal variance (numerical stability).

        Returns:
            GaussianMixturePrior: The mixture approximation.
        """
        B, K, D = x0.shape
        bag_var = x0.var(1) if K > 1 else torch.zeros_like(x0[:, 0])
        means = x0.clone()
        variances = bag_var[:, None].expand(B, K, D).clone()
        log_weights = torch.full((B, K), -np.log(K), device=x0.device, dtype=x0.dtype)
        for b in range(B):
            if labels is None or labels[b] is None:
                continue
            lab = torch.as_tensor(labels[b], device=x0.device)
            log_weights[b] = -float('inf')
            for m in range(int(lab.max()) + 1):
                members = x0[b, lab == m]
                means[b, m] = members.mean(0)
                variances[b, m] = members.var(0) if len(members) > 1 else bag_var[b]
                log_weights[b, m] = np.log(len(members) / K)
        return cls(means, variances.clamp_min(var_floor), log_weights)

    def responsibilities(self, x):
        """
        Posterior component probabilities q_m(x) of Eq. (13).

        Args:
            x (torch.Tensor): Points [B, K, D].

        Returns:
            torch.Tensor: Responsibilities [B, K, M].
        """
        B, K, D = x.shape
        M = self.means.shape[1]
        if B * K * M * D <= 2 ** 25:
            # few dimensions or components: all components at once
            diff = x[:, :, None] - self.means[:, None]
            logits = self.log_norm[:, None] - 0.5 * (diff * diff * self.inv_var[:, None]).sum(-1, dtype=torch.float64)
        else:
            # images: loop over the (few) components to avoid a [B, K, M, D] tensor
            logits = self.log_norm[:, None, :].repeat(1, K, 1)
            for m in range(M):
                diff = x - self.means[:, m:m + 1]
                logits[..., m] -= 0.5 * (diff * diff * self.inv_var[:, m:m + 1]).sum(-1, dtype=torch.float64)
        return torch.softmax(logits, dim=-1).to(x.dtype)

    def score_terms(self, x):
        """
        Decomposes the mixture score of Eq. (13) as grad log q(x) = linear - precision * x.

        Args:
            x (torch.Tensor): Points [B, K, D].

        Returns:
            Tuple of (precision, linear), each [B, K, D].
        """
        resp = self.responsibilities(x)
        return torch.bmm(resp, self.inv_var), torch.bmm(resp, self.mean_prec)

    def score(self, x):
        """Mixture score grad_x log q(x) of Eq. (13) for x [B, K, D]."""
        precision, linear = self.score_terms(x)
        return linear - precision * x


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Clustering of a bag into modes
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def coarse_features(x, resolution=32):
    """
    Coarse features used for clustering: images [n, C, H, W] are bilinearly downsampled to
    resolution x resolution; other inputs are only flattened.

    Returns:
        torch.Tensor: Features [n, F].
    """
    if x.dim() == 4 and x.shape[-1] > resolution:
        x = F.interpolate(x, size=(resolution, resolution), mode='bilinear', align_corners=False)
    return x.flatten(1)


def _logsumexp(a, axis):
    a_max = a.max(axis=axis, keepdims=True)
    return (a_max + np.log(np.exp(a - a_max).sum(axis=axis, keepdims=True))).squeeze(axis)


def fit_spherical_gmm(X, num_components, reg, num_iters=100, max_inits=20, tol=1e-6, seed=0):
    """
    Fits a spherical Gaussian mixture to X [n, d] with EM.

    `reg` is added to every component variance: it stops a component from collapsing onto a single point,
    which would otherwise make BIC favour one cluster per particle for the few particles of a bag. Initial
    means are subsets of the points (all subsets when there are at most `max_inits` of them); the fit with
    the highest log-likelihood is kept. One component, and one component per point, are fitted in closed form.

    Returns:
        dict: 'resp' responsibilities [n, M], 'll' log-likelihood, 'bic' Bayesian information criterion.
    """
    n, d = X.shape
    M = num_components
    if M == 1 or M == n:
        # closed-form fits: one component at the mean, or one component per point
        mu = X.mean(0, keepdims=True)[None] if M == 1 else X[None]
        var = np.full((1, M), (X.var(0).mean() if M == 1 else 0.0) + reg)
        num_iters = 1
    else:
        if comb(n, M) <= max_inits:
            inits = np.array(list(combinations(range(n), M)))
        else:
            rng = np.random.default_rng(seed)
            inits = np.stack([rng.choice(n, M, replace=False) for _ in range(max_inits)])
        mu = X[inits]                                               # [R, M, d]
        var = np.full(mu.shape[:2], X.var(0).mean() + reg)         # [R, M]
    w = np.full(mu.shape[:2], 1.0 / M)
    ll_old = None
    for it in range(num_iters):
        # E-step
        sq = ((X[None, :, None, :] - mu[:, None]) ** 2).sum(-1)    # [R, n, M]
        logp = np.log(w)[:, None] - 0.5 * (d * np.log(2 * np.pi * var)[:, None] + sq / var[:, None])
        lse = _logsumexp(logp, axis=-1)                              # [R, n]
        resp = np.exp(logp - lse[..., None])
        ll = lse.sum(-1)                                             # [R]
        converged = ll_old is not None and np.all(np.abs(ll - ll_old) <= tol * (np.abs(ll_old) + 1e-12))
        if converged or it == num_iters - 1:
            break
        ll_old = ll
        # M-step
        nk = resp.sum(1) + 1e-12                                     # [R, M]
        w = nk / n
        mu = np.einsum('rnm,nd->rmd', resp, X) / nk[..., None]
        sq = ((X[None, :, None, :] - mu[:, None]) ** 2).sum(-1)
        var = (resp * sq).sum(1) / (nk * d) + reg
    best = int(np.argmax(ll))
    num_params = (M - 1) + M * d + M                                 # weights, means, variances
    return {'resp': resp[best], 'll': float(ll[best]), 'bic': float(-2 * ll[best] + num_params * np.log(n))}


def cluster_bag(features, pca_variance=0.9, eps=0.15, max_clusters=None, seed=0):
    """
    Partitions the particles of one bag into modes (App. A.4): centering, projection onto the leading
    principal components explaining `pca_variance` of the variance, spherical Gaussian mixture with the
    number of components selected by BIC, hard assignment.

    Args:
        features (np.ndarray): Coarse features of the bag [K, F] (see coarse_features).
        pca_variance (float): Fraction of the variance kept by the PCA projection.
        eps (float): Per-feature RMS distance below which two particles are considered the same mode. It
            sets the variance floor of the mixture: two particles at distance eps * sqrt(F) sit at
            +-(eps * sqrt(F) / 2) around their mean, i.e. a per-dimension variance of eps^2 F / (4 d).
        max_clusters (int, optional): Largest number of components tried (default: K).
        seed (int): Seed of the EM initializations (used only when there are many particles).

    Returns:
        np.ndarray: Cluster labels [K] in 0, ..., M - 1.
    """
    n, num_features = features.shape
    if n < 2:
        return np.zeros(n, dtype=int)
    X = features - features.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    energy = S ** 2
    if energy.sum() <= 1e-12:
        return np.zeros(n, dtype=int)
    d = int(np.searchsorted(np.cumsum(energy) / energy.sum(), pca_variance) + 1)
    d = min(max(d, 1), n - 1)
    X = U[:, :d] * S[:d]
    reg = eps ** 2 * num_features / (4.0 * d)
    max_clusters = n if max_clusters is None else min(int(max_clusters), n)
    fits = [fit_spherical_gmm(X, m, reg, seed=seed) for m in range(1, max_clusters + 1)]
    best = min(fits, key=lambda fit: fit['bic'])
    _, labels = np.unique(best['resp'].argmax(1), return_inverse=True)
    return labels


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prior-consistency scoring and pruning
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
@torch.no_grad()
def prior_consistency(model, x, sigmas=(0.5, 1.0, 2.0), num_samples=1, twin=False):
    """
    Tweedie prior-consistency score of every particle (Eq. 9 before averaging over a cluster):
        s(x) = -1/|S| sum_{sigma in S} ||x - D(x + sigma * eps; sigma)||^2.
    The squared norm is averaged over dimensions instead of summed, a constant factor that leaves the
    relative pruning rule unchanged. The noise eps is shared by the particles of a bag (common random
    numbers), so that score differences within a bag do not depend on the noise draw.

    Args:
        model: Diffusion model with a `tweedie(x, sigma)` denoiser.
        x (torch.Tensor): Particles [num_bags, K, ...].
        sigmas (Sequence[float]): Noise scales S.
        num_samples (int): Noise draws per noise scale.

    Returns:
        torch.Tensor: Scores [num_bags, K] (non-positive, higher is more prior-consistent).
    """
    num_bags, K = x.shape[:2]
    flat = x.flatten(0, 1)
    scores = torch.zeros(num_bags * K, device=x.device)
    twin_scores = torch.zeros_like(scores)
    for sigma in sigmas:
        sigma_t = torch.tensor(float(sigma), device=x.device)
        for _ in range(num_samples):
            eps = torch.randn(num_bags, 1, *x.shape[2:], device=x.device).expand_as(x).flatten(0, 1)
            denoised = model.tweedie(flat + sigma_t * eps, sigma_t)
            scores -= (flat - denoised).pow(2).flatten(1).mean(1)
            if twin:
                denoised = model.tweedie((flat + sigma_t * eps).flip(-2, -1), sigma_t)
                twin_scores -= (flat.flip(-2, -1) - denoised).pow(2).flatten(1).mean(1)
    scores = (scores / (len(sigmas) * num_samples)).view(num_bags, K)
    return (scores, (twin_scores / (len(sigmas) * num_samples)).view(num_bags, K)) if twin else scores


def prune_bag(scores, labels, gamma, rng):
    """
    One-shot pruning of one bag (Alg. 1, lines 17-19): clusters scoring below (1 + gamma) s* are pruned,
    where s* is the best cluster score, and each pruned particle is replaced by a copy of a surviving
    particle, cycling through the survivors in random order.

    Args:
        scores (np.ndarray): Particle scores [K] (non-positive).
        labels (np.ndarray): Cluster labels [K].
        gamma (float): Relative pruning threshold.
        rng (np.random.Generator): Random generator for the order of the survivors.

    Returns:
        Tuple of (sources [K], index of the particle copied into each slot; cluster_scores [M]; keep [M]).
    """
    num_clusters = int(labels.max()) + 1
    cluster_scores = np.array([scores[labels == m].mean() for m in range(num_clusters)])
    keep = cluster_scores >= (1 + gamma) * cluster_scores.max()
    survivors = rng.permutation(np.flatnonzero(keep[labels]))
    victims = np.flatnonzero(~keep[labels])
    sources = np.arange(len(labels))
    sources[victims] = survivors[np.arange(len(victims)) % len(survivors)]
    return sources, cluster_scores, keep


def within_cluster_score_gaps(scores, labels):
    """
    Relative score differences |s_a - s_b| / min(|s_a|, |s_b|) of all particle pairs sharing a cluster.
    A pair with gap g is pruned by the rule of `prune_bag` exactly when gamma < g, so the 95th percentile
    of these gaps over calibration images keeps 95% of same-mode pairs intact (App. A.4).
    """
    gaps = []
    for m in range(int(labels.max()) + 1):
        for a, b in combinations(np.flatnonzero(labels == m), 2):
            gaps.append(float(abs(scores[a] - scores[b]) / max(min(abs(scores[a]), abs(scores[b])), 1e-12)))
    return gaps
