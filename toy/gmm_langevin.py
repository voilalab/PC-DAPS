"""
Toy illustration of the PC-DAPS paper (App. A.3, Fig. 4): the Gaussian-mixture approximation of p(x0 | x_t) keeps the
Langevin refinement on prior-supported modes.

One Langevin run at a single annealing level (sigma = 0.4) for an L-shaped two-component GMM prior in R^2 (a horizontal
bar at (-1, 0) and a vertical bar at (0, 1)), with the measurement y = x2 + n. The same particle, start (its PF-ODE
estimate x0hat), Langevin noise and step size are used under four approximations of p(x0 | x_t):

    DAPS      Gaussian approximation N(x0hat, sigma^2 I)
    PC-DAPS   with the Gaussian approximation: each bag particle keeps its own N(x0hat_k, sigma^2 I), so the run is
              the DAPS one
    PC-DAPS   Gaussian mixture from a bag of K = 4 particles: clusters of their x0hat, cluster mean and diagonal
              variance of Tweedie draws D(x0hat_k + sigma eps; sigma), weights |C| / K (Eqs. 11-12)
    Exact     p(x0 | x_t), itself a two-component GMM for a GMM prior

The Gaussian approximations end on the measurement line but off the prior (a hallucination); the mixture and the exact
p(x0 | x_t) end on the vertical bar. Drawn like the paper's figure: prior (gray levels), likelihood (blue band),
1-sigma ellipses of the approximation (teal) and the run from x0hat, smoothed along time.

    python toy/gmm_langevin.py --output toy/gmm_langevin.png
"""
import argparse
import math
from pathlib import Path

import numpy as np

# prior: horizontal bar at (-1, 0) and vertical bar at (0, 1), axis-aligned so a diagonal covariance can fit them
W = np.array([0.5, 0.5])
MU = np.array([[-1.0, 0.0], [0.0, 1.0]])
LONG, THIN = 0.6, 0.08
VAR = np.array([[LONG ** 2, THIN ** 2], [THIN ** 2, LONG ** 2]])

SIGMA = 0.4                     
Y, SIGMA_Y = 1.0, 0.1           
TAU = math.sqrt(2) * SIGMA_Y     
N_STEPS = 200
N_DRAWS = 8                     
X_T = np.array([-0.45, 0.03])    
BAG_OTHERS = np.array([[-1.15, -0.05], [0.04, 0.85], [-0.03, 1.25]])  
N_SEEDS = 200                    
EXTENT = (-2.1, 0.9, -0.9, 2.1)  
SMOOTH, EVERY = 5, 9            

INK, GRAY, BLUE, ORANGE, TEAL = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#007c7a'
PRIOR_LEVELS = [0.02, 0.1, 0.3, 0.6, 1.01]                   # prior density relative to its maximum
PRIOR_COLORS = ['#eef0f2', '#dce0e4', '#c3c9cf', '#a0a8b1']  # light -> dark, one colour per level band
LIK_ALPHA = 0.2
PANELS = [('daps', 'DAPS', 'Gaussian\nApproximation'), ('pc_gauss', 'PC-DAPS', 'Gaussian\nApproximation'),
          ('bag', 'PC-DAPS', 'Gaussian Mixture\nApproximation'), ('exact', 'Exact', r'$p(x_0\,|\,x_t)$')]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def log_comp(x, w, mu, var):
    """log w_k N(x; mu_k, diag var_k) for x [..., 2] -> [..., K]."""
    d = x[..., None, :] - mu
    return np.log(w) - 0.5 * (d ** 2 / var).sum(-1) - 0.5 * np.log(2 * np.pi * var).sum(-1)


def gmm_score(x, w, mu, var):
    r = softmax(log_comp(x, w, mu, var))
    return (r[..., None] * (mu - x[..., None, :]) / var).sum(-2)


def denoise(x, s):
    """Exact Tweedie denoiser of the prior at noise level s."""
    return x + s ** 2 * gmm_score(x, W, MU, VAR + s ** 2)


def pf_ode(x, s, s_min=0.01, n=100):
    """x0hat: probability-flow ODE from level s to s_min (Heun, geometric sigma grid)."""
    sig = np.geomspace(s, s_min, n + 1)
    for a, b in zip(sig[:-1], sig[1:]):
        d = (x - denoise(x, a)) / a
        xe = x + (b - a) * d
        x = x + (b - a) * 0.5 * (d + (xe - denoise(xe, b)) / b)
    return x


def exact_posterior(x_t, s):
    """p(x0 | x_t) for the GMM prior: components (w', mu', var')."""
    v = 1 / (1 / VAR + 1 / s ** 2)
    mu = v * (MU / VAR + x_t / s ** 2)
    return softmax(log_comp(x_t, W, MU, VAR + s ** 2)), mu, v


def bag_mixture(bag_xt, s, rng):
    """Mixture from the bag: Tweedie draws around each x0hat, clusters of x0hat (the nearest prior branch, unambiguous
    here), cluster mean and floored diagonal variance of its draws, weights |C| / K."""
    x0hat = np.stack([pf_ode(x, s) for x in bag_xt])
    draws = np.stack([denoise(x0hat + s * rng.standard_normal(x0hat.shape), s) for _ in range(N_DRAWS)], 1)
    labels = log_comp(x0hat, W, MU, VAR).argmax(-1)
    w, mu, var = [], [], []
    for m in np.unique(labels):
        pts = draws[labels == m].reshape(-1, 2)
        v = np.maximum(pts.var(0), max(1e-2 * pts.var(0).mean(), 1e-4))
        w.append((labels == m).sum() / len(bag_xt))
        mu.append(pts.mean(0))
        var.append(v)
    return np.array(w), np.array(mu), np.array(var)


def lik_grad(x):
    return np.stack([np.zeros_like(x[..., 1]), -2 * (x[..., 1] - Y) / TAU ** 2], -1)


def langevin(x0, approx, eta, noise):
    x, path = x0.copy(), [x0.copy()]
    for e in noise:
        x = x + eta * (gmm_score(x, *approx) + lik_grad(x)) + math.sqrt(2 * eta) * e
        path.append(x.copy())
    return np.array(path)


def prior_density(x):
    return np.exp(log_comp(x, W, MU, VAR)).sum(-1)


PRIOR_MAX = prior_density(np.stack(np.meshgrid(np.linspace(-2.5, 1.5, 401), np.linspace(-1, 2.5, 351)), -1)).max()


def on_prior(x, thresh=1e-2):
    """Prior density at x above thresh times its maximum."""
    return prior_density(x) / PRIOR_MAX > thresh


def smooth_path(p):
    """The run smoothed along time and thinned to every EVERY-th step (plus the last); the start is kept exactly."""
    from scipy.ndimage import gaussian_filter1d
    q = gaussian_filter1d(p, SMOOTH, axis=0, mode='nearest')
    ramp = np.clip(1 - np.arange(len(p)) / (3 * SMOOTH), 0, 1)[:, None]
    q = q + ramp * (p[0] - q[0])
    return np.vstack([q[::EVERY], q[-1:]]) if (len(q) - 1) % EVERY else q[::EVERY]


def compute():
    rng = np.random.default_rng(0)
    x0hat = pf_ode(X_T, SIGMA)
    approx = {'daps': (np.array([1.0]), x0hat[None], np.full((1, 2), SIGMA ** 2)),
              'bag': bag_mixture(np.vstack([X_T, BAG_OTHERS]), SIGMA, rng),
              'exact': exact_posterior(X_T, SIGMA)}
    approx['pc_gauss'] = approx['daps']                  # each particle keeps its own DAPS Gaussian
    min_var = min(v.min() for _, _, v in approx.values())
    eta = 0.1 / (2 / TAU ** 2 + 1 / min_var)            # step size from the stiffest curvature
    noise = rng.standard_normal((N_STEPS, 2))
    paths = {k: langevin(x0hat, a, eta, noise) for k, a in approx.items()}
    ends = {k: np.array([langevin(x0hat, a, eta, np.random.default_rng(1000 + seed).standard_normal((N_STEPS, 2)))[-1]
                         for seed in range(N_SEEDS)]) for k, a in approx.items()}
    return dict(x0hat=x0hat, approx=approx, eta=eta, paths=paths, ends=ends)


def plot_figure(d, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.legend_handler import HandlerPatch, HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse, Rectangle

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['TeX Gyre Termes', 'Nimbus Roman', 'STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 8, 'legend.frameon': False,
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    x0, x1, y0, y1 = EXTENT
    side, gap, left, bottom = 1.26, 0.08, 0.2, 0.62             # inches
    fig_w, fig_h = 5.5, bottom + side + 0.62
    fig = plt.figure(figsize=(fig_w, fig_h))
    gx, gy = np.linspace(x0, x1, 681), np.linspace(y0, y1, 601)
    dens = prior_density(np.stack(np.meshgrid(gx, gy), -1))
    lik = np.zeros((len(gy), 1, 4))
    lik[..., :3] = to_rgb(BLUE)
    lik[:, 0, 3] = LIK_ALPHA * np.exp(-(gy - Y) ** 2 / TAU ** 2)
    start = dict(marker='o', color=INK, mec='white', mew=0.4, ms=4.4, ls='')
    good = dict(marker='o', color=BLUE, mec='white', mew=0.4, ms=5.6, ls='')
    bad = dict(marker='D', color=ORANGE, mec='white', mew=0.4, ms=4.8, ls='')

    axes = []
    for i, (key, method, title) in enumerate(PANELS):
        ax = fig.add_axes([(left + i * (side + gap)) / fig_w, bottom / fig_h, side / fig_w, side / fig_h])
        axes.append(ax)
        ax.contourf(gx, gy, dens / dens.max(), levels=PRIOR_LEVELS, colors=PRIOR_COLORS, antialiased=True)
        ax.imshow(lik, extent=EXTENT, origin='lower', aspect='auto', interpolation='bilinear')
        ax.axhline(Y, color=BLUE, lw=0.5, alpha=0.8)
        for wk, m, v in zip(*d['approx'][key]):
            ax.add_patch(Ellipse(m, 2 * math.sqrt(v[0]), 2 * math.sqrt(v[1]), fill=False, ec=TEAL, lw=0.9,
                                 alpha=0.9 * max(float(wk), 0.25) ** 0.5))
        p = smooth_path(d['paths'][key])
        ax.plot(p[:, 0], p[:, 1], color=INK, lw=0.8, solid_joinstyle='round')
        ax.plot(*p[0], **start)
        ax.plot(*p[-1], **(good if on_prior(p[-1]) else bad))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
            s.set_color(INK)
        ax.text(0.5, 1.04, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=8, linespacing=1.25)
    axes[0].text(x0 + 0.06, Y + 0.03, r'$x_2 = y$', color=BLUE, fontsize=7, ha='left', va='bottom')

    groups = {}                                                  # method names over their panels, each with a rule
    for i, (_, method, _) in enumerate(PANELS):
        groups.setdefault(method, []).append(i)
    top = (bottom + side + 0.37) / fig_h
    for method, idx in groups.items():
        l_in, r_in = left + idx[0] * (side + gap) + 0.06, left + idx[-1] * (side + gap) + side - 0.06
        fig.text((l_in + r_in) / 2 / fig_w, top + 0.02, method, ha='center', va='bottom', fontsize=8)
        fig.add_artist(Line2D([l_in / fig_w, r_in / fig_w], [top, top], color=GRAY, lw=0.4))

    total = 4 * side + 3 * gap
    fig.text((left + total / 2) / fig_w, (bottom - 0.05) / fig_h, r'$x_1$', ha='center', va='top', fontsize=8)
    fig.text((left - 0.08) / fig_w, (bottom + side / 2) / fig_h, r'$x_2$', ha='right', va='center', rotation=90,
             fontsize=8)

    def ellipse_glyph(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
        return Ellipse((width / 2 - xdescent, height / 2 - ydescent), width * 0.9, height * 0.9, fill=False, ec=TEAL,
                       lw=0.9)

    swatch = tuple(Rectangle((0, 0), 1, 1, fc=c, ec='none') for c in PRIOR_COLORS)
    band = (Rectangle((0, 0), 1, 1, fc=BLUE, alpha=LIK_ALPHA, ec='none'), Line2D([], [], color=BLUE, lw=0.5))
    run = (Line2D([], [], **start), Line2D([], [], color=INK, lw=0.8))
    rows = [([swatch, band, Ellipse((0, 0), 1, 1, fill=False, ec=TEAL, lw=0.9)],
             [r'prior $p(x_0)$', r'likelihood $p(y\,|\,x_0)$', r'approximation of $p(x_0\,|\,x_t)$']),
            ([run, Line2D([], [], **good), Line2D([], [], **bad)],
             [r'Langevin run from $\hat{x}_0$', 'ends on the prior', 'ends off the prior'])]
    handler_map = {swatch: HandlerTuple(ndivide=None, pad=0), band: HandlerTuple(ndivide=1),   # band: line over it
                   run: HandlerTuple(ndivide=None, pad=0), Ellipse: HandlerPatch(patch_func=ellipse_glyph)}
    for r, (handles, labels) in enumerate(rows):
        fig.legend(handles, labels, loc='lower center', ncols=3, bbox_to_anchor=(0.5 * (2 * left + total) / fig_w,
                   (0.17 - 0.17 * r) / fig_h), fontsize=7.5, handler_map=handler_map, handlelength=1.8,
                   handletextpad=0.4, columnspacing=1.4, borderaxespad=0)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default=str(Path(__file__).with_suffix('.png')))
    args = parser.parse_args()

    d = compute()
    print(f"x_t {X_T}  x0hat {np.round(d['x0hat'], 3)}  step size {d['eta']:.3g}  steps {N_STEPS}")
    for key, method, title in PANELS:
        end = smooth_path(d['paths'][key])[-1]
        print(f"{method:8s} {title.replace(chr(10), ' '):31s} end {np.round(end, 3)} "
              f"{'on' if on_prior(end) else 'off'} the prior | {N_SEEDS} noise seeds: "
              f"{on_prior(d['ends'][key]).mean():.0%} end on the prior")
    plot_figure(d, args.output)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
