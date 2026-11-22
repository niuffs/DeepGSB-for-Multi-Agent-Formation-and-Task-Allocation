import math
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from matplotlib.patches import Circle

from . import util
from .constraint import build_constraint


def to_numpy(t):
    return t.detach().cpu().numpy()


def get_lims(opt):
    limits = {
        "Gauss2Gauss_Obs3": [-6, 6],
        "Gauss2Rings_Obs1": [-6, 6],
        "Gauss2Mix_Obs1": [-6, 6],
        "Gauss2Ring_Obs1": [-6, 6],
        "MixGauss2SphereShell_Obs3": [-8, 10],
    }
    return limits.get(opt.problem_name, [-6, 6])


def get_ylims(opt):
    limits = {
        "Gauss2Gauss_Obs3": [-6, 6],
        "Gauss2Rings_Obs1": [-6, 6],
        "Gauss2Mix_Obs1": [-6, 6],
        "Gauss2Ring_Obs1": [-6, 6],
        "MixGauss2SphereShell_Obs3": [-6, 6],
    }
    return limits.get(opt.problem_name, [-6, 6])


def get_colors(n_snapshot):
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"])
    return cm1(np.linspace(0.0, 1.0, n_snapshot))


def create_mesh(opt, n_grid, lims, x_dim=2, convert_to_numpy=True):
    import warnings
    _x = torch.linspace(*(lims + [n_grid]))
    with warnings.catch_warnings():
        X, Y = torch.meshgrid(_x, _x, indexing='xy')
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(opt.device)
    # Pad extra dims with zeros for high-dimensional problems
    if x_dim > 2:
        extra = torch.zeros(xy.shape[0], x_dim - 2, device=opt.device)
        xs = torch.cat([xy, extra], dim=-1)
    else:
        xs = xy
    return [to_numpy(X), to_numpy(Y), xs] if convert_to_numpy else [X, Y, xs]


def get_func_mesher(opt, ts, grid_n, func, out_dim=1):
    if func is None:
        return None
    lims = get_lims(opt)
    X1, X2, XS = create_mesh(opt, grid_n, lims, x_dim=opt.x_dim)
    out_shape = [grid_n, grid_n] if out_dim == 1 else [grid_n, grid_n, out_dim]

    def mesher(idx):
        arg_xs = XS.detach()
        arg_ts = ts[idx].repeat(grid_n ** 2).detach()
        fn_out = func(arg_xs, arg_ts)
        return X1, X2, to_numpy(fn_out.reshape(*out_shape))

    return mesher


def plot_obs(opt, ax, scale=1., zorder=0):
    """Plot obstacles based on problem name."""
    obs_configs = {
        "Gauss2Gauss_Obs3": [((2., 2.), 1.0), ((-3., 1.), 0.5), ((0., -2.), 1.5)],
        "Gauss2Rings_Obs1": [((0., 0.), 1.5)],
        "Gauss2Mix_Obs1": [((0., 0.), 1.0)],
        "Gauss2Ring_Obs1": [((2., 0.), 1.5)],
        "MixGauss2SphereShell_Obs3": [],
    }
    obstacles = obs_configs.get(opt.problem_name, [])
    for (cx, cy), radius in obstacles:
        circle = Circle(xy=(cx, cy), radius=radius, zorder=zorder)
        ax.add_artist(circle)
        circle.set_clip_box(ax.bbox)
        circle.set_facecolor("darkgray")
        circle.set_edgecolor(None)


def setup_ax(ax, title, xlims, ylims, title_fontsize=22):
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)


def plot_traj_snapshot(opt, xs, axes, sample_steps, titles, y_mesher=None):
    n_snapshot = len(axes)
    assert len(sample_steps) == len(titles) == n_snapshot

    if sample_steps is None:
        sample_steps = np.linspace(0, xs.shape[1] - 1, n_snapshot).astype(int)

    xlims = get_lims(opt)
    ylims = get_ylims(opt)
    colors = get_colors(n_snapshot)

    for ax, step, title, color in zip(axes, sample_steps, titles, colors):
        plot_obs(opt, ax, zorder=0)
        ax.scatter(xs[:, step, 0], xs[:, step, 1], s=1.5, color=color, alpha=0.5, zorder=1)
        if y_mesher is not None:
            cp = ax.contour(*y_mesher(step), levels=10, cmap="copper", linewidths=1, zorder=2)
            ax.clabel(cp, inline=True, fontsize=10)
        setup_ax(ax, title, xlims, ylims)


def get_fig_axes_steps(interval, n_snapshot=5, ax_length_in: float = 4):
    n_row, n_col = 1, n_snapshot
    figsize = (n_col * ax_length_in, n_row * ax_length_in)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    axes = fig.subplots(n_row, n_col)
    steps = np.linspace(0, interval - 1, n_snapshot).astype(int)
    return fig, axes, steps


@torch.no_grad()
def sample_traj(opt, mfg, ts, policy_f, policy_b, plot_dim):
    xs_f, _, _, _ = mfg.sde.sample_traj(ts, policy_f)
    xs_b, _, _, _ = mfg.sde.sample_traj(ts, policy_b)
    util.assert_zero_grads(policy_f)
    util.assert_zero_grads(policy_b)

    xs_f_np, xs_b_np = to_numpy(xs_f[..., plot_dim]), to_numpy(xs_b[..., plot_dim])
    return xs_f, xs_b, xs_f_np, xs_b_np


@torch.no_grad()
def snapshot(opt, policy_f, policy_b, mfg, stage, plot_pf_ode=False, plot_dim=[0, 1]):
    """Single-task snapshot: F/B trajectories + PF-ODE log-density."""
    # Dispatch to env-specific rendering when available
    from envs import get_env_class
    env_cls = get_env_class(opt.problem_name)
    if hasattr(env_cls, 'render_trajectories'):
        env = env_cls(device=opt.device, d=getattr(opt, 'x_dim', 2))
        env.render_trajectories(opt, mfg, policy_f, policy_b, stage)
        return

    ts = mfg.ts
    xs_f, xs_b, xs_f_np, xs_b_np = sample_traj(opt, mfg, ts, policy_f, policy_b, plot_dim)
    interval = len(ts)

    # PF-ODE forward trajectory
    g2 = mfg.sde.std ** 2
    dt_pf = (ts[-1] - ts[0]).item() / max(interval - 1, 1)
    n_part = xs_f.shape[0]
    plot_d = min(xs_f.shape[2], 2)
    x_pf = mfg.p0.sample(batch=n_part).to(opt.device)
    xs_pf = torch.zeros(n_part, interval, plot_d, device=opt.device)
    for k in range(interval):
        xs_pf[:, k] = x_pf[..., :plot_d]
        if k < interval - 1:
            t_k = ts[k].expand(n_part, 1).to(opt.device)
            z_f = policy_f(x_pf, t_k)
            z_b = policy_b(x_pf, t_k)
            drift = -0.5 * (z_b - z_f) * g2
            x_pf = x_pf + drift * dt_pf
    xs_pf_np = to_numpy(xs_pf)

    # Plot forward and backward snapshots
    for xs_np, policy in zip([xs_f_np, xs_b_np], [policy_f, policy_b]):
        fig, axes, sample_steps = get_fig_axes_steps(interval, n_snapshot=5)
        titles = [r'$t=0$', r'$t=0.25T$', r'$t=0.50T$', r'$t=0.75T$', r'$t=T$']
        assert len(titles) == len(axes) == len(sample_steps)
        y_mesher = get_func_mesher(opt, ts, 200, policy.compute_value)
        plot_traj_snapshot(opt, xs_np, axes, sample_steps, titles, y_mesher=y_mesher)
        axes[0].set_ylabel(rf'$d={opt.x_dim}$', fontsize=18)

        save_dir = os.path.join(opt.eval_path, policy.direction)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'stage{stage}.pdf'), bbox_inches='tight')
        plt.close(fig)

    # Plot pf-ode (log-density)
    if plot_pf_ode:
        def logp_fn(x, t):
            y_f = policy_f.compute_value(x, t)
            y_b = policy_b.compute_value(x, t)
            return y_f + y_b

        titles = [r'$t=0$', r'$t=0.25T$', r'$t=0.50T$', r'$t=0.75T$', r'$t=T$']
        fig, axes, sample_steps = get_fig_axes_steps(interval, n_snapshot=5)
        logp_mesher = get_func_mesher(opt, ts, 200, logp_fn)
        plot_traj_snapshot(opt, xs_pf_np, axes, sample_steps, titles, y_mesher=logp_mesher)
        axes[0].set_ylabel(rf'$d={opt.x_dim}$', fontsize=18)

        pf_ode_dir = os.path.join(opt.eval_path, "pf-ode")
        os.makedirs(pf_ode_dir, exist_ok=True)
        plt.savefig(os.path.join(pf_ode_dir, f'stage{stage}.pdf'), bbox_inches='tight')
        plt.close(fig)

    return xs_f, xs_b


@torch.no_grad()
def plot_scenario_setup(opt, save_path: str = None):
    """Plot scenario setup: obstacles, initial (blue), terminal (red)."""
    # Dispatch to env-specific rendering when available
    from envs import get_env_class
    env_cls = get_env_class(opt.problem_name)
    if hasattr(env_cls, 'render_setup'):
        env = env_cls(device=opt.device, d=getattr(opt, 'x_dim', 2))
        env.render_setup(opt, save_path)
        return

    import os
    import numpy as np

    constraints = build_constraint(opt.problem_name, 2000, opt.device, x_dim=getattr(opt, 'x_dim', 2))
    p0_samp = constraints.p0.sample(2000).detach().cpu().numpy()
    pT_samp = constraints.pT.sample(2000).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 8))
    setup_ax(ax, opt.problem_name, [-6, 6], [-6, 6])
    ax.set_xticks(range(-6, 7, 2))
    ax.set_yticks(range(-6, 7, 2))
    ax.set_xlabel('$x_1$', fontsize=18)
    ax.set_ylabel('$x_2$', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    plot_obs(opt, ax, zorder=0)

    ax.scatter(p0_samp[:, 0], p0_samp[:, 1], s=3, color='blue', alpha=0.4, zorder=2,
               label=r'$\rho_0$')
    ax.scatter(pT_samp[:, 0], pT_samp[:, 1], s=3, color='red', alpha=0.4, zorder=3,
               label=r'$\rho_T$')
    ax.legend(fontsize=14, loc='upper right')

    out_path = save_path or getattr(opt, 'eval_path', '.')
    os.makedirs(out_path, exist_ok=True)
    fname = f"{opt.problem_name}_d{opt.x_dim}_setup.pdf"
    plt.savefig(os.path.join(out_path, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
