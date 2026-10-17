"""
Toy experiment of the PC-DAPS paper (Fig. 2): premature mode selection on a two-atom prior.

    prior        p(x0) = 1/2 delta_{-a u} + 1/2 delta_{a u} in R^2, u = e1, with exact denoiser
                 D(x, sigma) = a tanh(a <x, u> / sigma^2) u
    measurement  A(x) = f(<x, u>),  f(z) = c / (2ab) (z - a)(z + a - b),  y = c = kappa * beta_y

-a u is measurement-consistent (majority posterior mode x1), a u is not (minority mode x2), and (a + b) u is
measurement-consistent but has no prior mass (x3). DAPS drifts from x2 to x3 and stays trapped there; PC-DAPS
prunes the x3 particles once the annealing noise no longer lets them switch modes. Both samplers are the
repository implementations with the exact denoiser in place of a network.

The figure follows the paper's Fig. 2: the annealed marginal of the noisy iterate along u, <x_sigma, u> / sqrt(1 +
sigma^2), for the exact posterior, DAPS and PC-DAPS, and a zoom on the DAPS Langevin outputs around a and a + b.
Densities are histograms smoothed over (position, noise level) and normalised to a maximum of 1 per noise level.

The measurement is one-dimensional, so the test of Eq. 8 needs a large bag to be reliable; the default is one 
bag of 1000 particles, as in the paper's figure. 

    python toy/two_atom.py --output toy/two_atom.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forward_operator import Operator 
from sampler import DAPS, PCDAPS 

INK, INK2, BLUE, ORANGE = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834'
TEXT_WIDTH = 5.5                       
S_EDGES = np.linspace(-1.9, 1.9, 191)   
SIGMA_VIEW, SIGMA_LO = 10.0, 0.1       


class TwoAtomPrior(torch.nn.Module):
    """Exact denoiser of the two-atom prior (Eq. 15)."""

    def __init__(self, a=1.0):
        super().__init__()
        self.a = a
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)  # device of the model

    def tweedie(self, x, sigma):
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        denoised = torch.zeros_like(x)
        denoised[..., :1] = self.a * torch.tanh(self.a * x[..., :1] / sigma ** 2)
        return denoised

    def score(self, x, sigma):
        return (self.tweedie(x, sigma) - x) / torch.as_tensor(sigma, device=x.device) ** 2

    def get_in_shape(self):
        return (2,)


class QuadraticOperator(Operator):
    """A(x) = f(<x, u>) of Eq. (16)."""

    def __init__(self, a=1.0, b=0.1, c=0.1, sigma=0.05):
        super().__init__(sigma)
        self.a, self.b, self.c = a, b, c

    def __call__(self, x):
        z = x[..., :1]
        return self.c / (2 * self.a * self.b) * (z - self.a) * (z + self.a - self.b)


def exact_posterior(a, b, c, beta):
    """Weight pi of the atom a u in p(x0 | y) = (1 - pi) delta_{-a u} + pi delta_{a u} (Eq. 17)."""
    log_lik = lambda z: -0.5 * ((c / (2 * a * b) * (z - a) * (z + a - b) - c) / beta) ** 2
    return 1 / (1 + np.exp(log_lik(-a) - log_lik(a)))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure in the style of the paper's Fig. 2
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def column_density(values, edges, smooth):
    """Per-level histogram, smoothed over (position, level) and normalised to max 1 per level."""
    from scipy.ndimage import gaussian_filter
    hist = np.stack([np.histogram(v, bins=edges)[0] for v in values], 1).astype(float)   # [bins, levels]
    hist = gaussian_filter(hist, sigma=smooth, mode='nearest')
    return hist / np.maximum(hist.max(0, keepdims=True), 1e-12)


def span_edges(levels):
    """Log-midpoint edges of a decreasing sequence of noise levels, the outer ones at the axis ends."""
    lg = np.log(levels)
    edges = np.exp(np.concatenate([[lg[0]], (lg[1:] + lg[:-1]) / 2, [lg[-1]]]))
    edges[0], edges[-1] = SIGMA_VIEW, SIGMA_LO
    return edges


def exact_density(levels, scale, atoms, masses):
    """Exact p(x_sigma | y) of an atomic posterior along u, VP-scaled and max-normalised per level."""
    s_mid = (S_EDGES[1:] + S_EDGES[:-1]) / 2
    dens = np.stack([sum(p * np.exp(-0.5 * ((s_mid - t * k) / (lv * k)) ** 2) for t, p in zip(atoms, masses))
                     for lv, k in zip(levels, scale)], 1)
    return dens / dens.max(0, keepdims=True)


def plot_figure(output, sigma_steps, daps_xt, daps_x0y, pcdaps_xt, a, b, pi, gate):
    """
    Args:
        sigma_steps: annealing noise levels (the last one 0); step k runs Langevin at sigma_steps[k] and records the
            re-noised iterate at sigma_steps[k + 1].
        daps_xt, pcdaps_xt: <x_t, u> of every particle after every step [T, N]; daps_x0y: DAPS Langevin outputs [T, N].
        pi: exact posterior mass of a u; gate: noise level at which PC-DAPS pruned (None: never).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Nimbus Roman', 'STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.linewidth': 0.6, 'xtick.labelsize': 6.8,
        'xtick.major.width': 0.5, 'xtick.major.size': 2, 'xtick.major.pad': 1.5, 'xtick.color': INK2,
        'xtick.labelcolor': INK, 'legend.frameon': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    levels = sigma_steps[1:-1]                       # noise levels of the recorded x_t (the last one is clean)
    keep = levels <= SIGMA_VIEW
    levels, scale = levels[keep], 1 / np.sqrt(1 + levels[keep] ** 2)
    lv_langevin = sigma_steps[:-1]
    keep_z = lv_langevin <= SIGMA_VIEW
    smooth = (3.0, 2.0) if daps_xt.shape[1] < 5000 else (2.0, 1.5)
    split = b / 2                                    # + side: nearer +a than -a
    pad = max(0.15, 0.75 * b)
    t_edges = np.linspace(a - pad, a + b + pad, 161)
    dens = {
        'exact': exact_density(levels, scale, [-a, a], [1 - pi, pi]),
        'daps': column_density(daps_xt[:-1][keep] * scale[:, None], S_EDGES, smooth),
        'pcdaps': column_density(pcdaps_xt[:-1][keep] * scale[:, None], S_EDGES, smooth),
        'zoom': column_density([v[v > split] for v in daps_x0y[keep_z]], t_edges, smooth),
    }

    W, H, side, y0 = TEXT_WIDTH, 1.95, 0.9, 0.3
    x_in = {'exact': 0.62, 'daps': 1.64, 'pcdaps': 2.66, 'zoom': 3.8}     # panel left edges in inches
    fig = plt.figure(figsize=(W, H))
    axes = {name: fig.add_axes([x / W, y0, side / W, side / H]) for name, x in x_in.items()}
    fig.text(0.5, 0.965, r'$\mathcal{A}(x)=\frac{c}{2ab}\,(z-a)(z+a-b),\;\; z=\langle x,u\rangle,'
                         r'\qquad p(x_0)=\frac{1}{2}(\delta_{-a}+\delta_{a})$', ha='center', va='top', fontsize=9)
    titles = {'exact': 'Ground truth', 'daps': 'DAPS', 'pcdaps': 'PC-DAPS', 'zoom': 'DAPS, zoom'}
    for name, ax in axes.items():
        if name == 'zoom':
            mesh = ax.pcolormesh(span_edges(lv_langevin[keep_z]), t_edges, dens['zoom'], cmap='magma', vmin=0,
                                 vmax=1, rasterized=True, shading='flat')
            ax.set_ylim(t_edges[0], t_edges[-1])
            for y, alpha in ((a, 0.9), (a + b, 0.55)):
                ax.axhline(y, color='white', lw=0.5, alpha=alpha)
        else:
            mesh = ax.pcolormesh(span_edges(levels), S_EDGES, dens[name], cmap='magma', vmin=0, vmax=1,
                                 rasterized=True, shading='flat')
            ax.set_ylim(S_EDGES[0], S_EDGES[-1])
        ax.set_xscale('log')
        ax.set_xlim(SIGMA_VIEW, SIGMA_LO)            # noise on the left, clean on the right
        ax.minorticks_off()
        ax.set_xticks([10, 1, 0.1])
        ax.set_xticklabels(['10', '1', '0.1'])
        labels = ax.get_xticklabels()                # keep the end labels inside their panel
        labels[0].set_horizontalalignment('left')
        labels[-1].set_horizontalalignment('right')
        ax.tick_params(left=False, labelleft=False)
        for s in ax.spines.values():
            s.set_color(INK)
        ax.set_title(titles[name], fontsize=8.5, pad=3)

    trans = axes['zoom'].get_yaxis_transform()       # glyphs just outside the zoom panel
    open_dot = dict(marker='o', mfc='white', mec=INK, mew=0.8, ms=4.2)
    diamond = dict(marker='D', color=ORANGE, ms=3.8, mec='white', mew=0.4)
    blue_dot = dict(marker='o', color=BLUE, ms=4.2, mec='white', mew=0.4)
    for y, kw in ((a, open_dot), (a + b, diamond)):
        axes['zoom'].plot([1.06], [y], transform=trans, clip_on=False, ls='', zorder=6, **kw)
    if gate is not None:
        axes['pcdaps'].axvline(gate, color='white', lw=0.6, alpha=0.85)
        axes['pcdaps'].text(gate, -0.03, f'{gate:.2g}', ha='center', va='top', fontsize=4.5, color=INK,
                            transform=axes['pcdaps'].get_xaxis_transform(), clip_on=False)

    pos = axes['zoom'].get_position()
    cax = fig.add_axes([pos.x1 + 0.2 / W, pos.y0, 0.012, pos.height])
    cb = fig.colorbar(mesh, cax=cax, ticks=[0, 1])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6.3, length=1.5, pad=1)
    cb.set_label('density', fontsize=6.6, labelpad=2)

    left, right = axes['exact'].get_position().x0, axes['zoom'].get_position().x1
    fig.text((left + right) / 2, y0 - 0.1, r'noise level $\sigma$ (sampling $\rightarrow$)', ha='center', va='top',
             fontsize=7.5)
    handles = [Line2D([], [], ls='', label=label, **kw) for label, kw in (
        (r'$-a$: majority posterior mode', blue_dot), (r'$a$: minority posterior mode', open_dot),
        (r'$a{+}b$: DAPS-hallucinated mode', diamond))]
    fig.legend(handles=handles, loc='lower center', ncols=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.6,
               handletextpad=0.25, columnspacing=1.6, borderaxespad=0.2)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num_particles', type=int, default=1000, help='DAPS chains = PC-DAPS bags x bag size')
    parser.add_argument('--bag_size', type=int, default=1000)
    parser.add_argument('--annealing_steps', type=int, default=200)
    parser.add_argument('--langevin_steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--residual_test', default='noisy', choices=['noisy', 'tweedie', 'post_langevin'])
    parser.add_argument('--repeat_checks', action='store_true', help='repeat the pruning check until it prunes')
    parser.add_argument('--check_sigma_max', type=float, default=0.3,
                        help='run the pruning check only at noise levels <= this (10: from the start)')
    parser.add_argument('--max_clusters', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default=str(Path(__file__).with_suffix('.png')))
    args = parser.parse_args()

    a, b, kappa, beta = 1.0, 0.1, 2.0, 0.05
    c = kappa * beta
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    model = TwoAtomPrior(a).to(device)
    operator = QuadraticOperator(a, b, c, beta)

    annealing = dict(name='edm', num_steps=args.annealing_steps, sigma_max=10, sigma_min=0.1, timestep='poly-7')
    diffusion = dict(name='edm', num_steps=5, sigma_min=0.01, timestep='poly-7')
    # tau = sqrt(2) beta makes the DAPS data term -grad ||A(x) - y||^2 / tau^2 the exact log-likelihood gradient
    mcmc = dict(num_steps=args.langevin_steps, lr=args.lr, tau=np.sqrt(2) * beta, lr_min_ratio=1.0,
                mc_algo='langevin', prior_solver='gaussian')
    check = dict(residual_test=args.residual_test, gamma=args.gamma, stop_after_first_check=not args.repeat_checks,
                 sigma_max=args.check_sigma_max)
    pcdaps_config = dict(bag_size=args.bag_size, ode_steps_high_noise=None, skip_final_langevin=False, pruning_stages=[check],
                         score_sigmas=[0.05, 0.1, 0.2], cluster_eps=0.2, max_clusters=args.max_clusters, var_floor=1e-3,
                         seed=args.seed)

    daps = DAPS(dict(annealing), dict(diffusion), dict(mcmc))
    x_start = daps.get_start(args.num_particles, model)
    daps.sample(model, x_start, operator, torch.full((args.num_particles, 1), c, device=device), record=True)
    daps_data = daps.trajectory.compile().tensor_data
    daps_xt, daps_x0y = daps_data['xt'][..., 0].numpy(), daps_data['x0y'][..., 0].numpy()          # [T, N]

    pcdaps = PCDAPS(dict(annealing), dict(diffusion), dict(mcmc), pcdaps_config)
    num_bags = args.num_particles // args.bag_size
    x_start = pcdaps.get_start(num_bags, model)
    pcdaps.sample(model, x_start, operator, torch.full((num_bags, 1), c, device=device), record=True)
    pcdaps_data = pcdaps.trajectory.compile().tensor_data
    pcdaps_xt, pcdaps_x0y = (pcdaps_data[key][..., 0].flatten(1).numpy() for key in ('xt', 'x0y'))  # [T, N]
    sigma_steps = daps.annealing_scheduler.sigma_steps.numpy()

    pi = exact_posterior(a, b, c, beta)
    print(f'exact posterior: P(x1 = -a u) = {1 - pi:.3f}, P(x2 = a u) = {pi:.3f}, P(x3 = (a+b) u) = 0')
    pruned = [e for e in pcdaps.prune_log if not all(e['kept_clusters'])]
    gate = float(np.median([e['sigma'] for e in pruned])) if pruned else None
    print(f'PC-DAPS pruned {len(pruned)} / {num_bags} bags '
          f'(median sigma at pruning {gate if gate is not None else float("nan"):.3f})')
    print(f'{"final samples":14s} {"near x1 (z<0)":>14s} {"near x2 (z~a)":>14s} {"near x3 (z~a+b)":>16s}')
    for name, traj in [('DAPS', daps_x0y), ('PC-DAPS', pcdaps_x0y)]:
        z = traj[-1]
        near_x2 = (z > 0) & (np.abs(z - a) < np.abs(z - a - b))
        print(f'{name:14s} {np.mean(z < 0):14.3f} {np.mean(near_x2):14.3f} {np.mean((z > 0) & ~near_x2):16.3f}')

    plot_figure(args.output, sigma_steps, daps_xt, daps_x0y, pcdaps_xt, a, b, pi, gate)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
