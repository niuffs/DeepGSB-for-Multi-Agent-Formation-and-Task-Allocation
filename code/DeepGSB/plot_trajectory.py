"""
Standalone trajectory plotting -- 3 modes, 2 figures + comparison mode.

Modes:
  pf-ode   (default)  Probability Flow ODE:  drift = 0.5*(z_b - z_f)*g
  sde      Standard Schrodinger Bridge SDE (z_b / z_f with noise)
  fwd-ode  Optimal-control ODE (z_b backward -> z_f forward)

Comparison mode (--compare):
  Samples 10 particles from the initial distribution p0 and propagates them
  forward with SDE, optimal-control ODE, and PF-ODE from the SAME starting
  points.  Produces a single 3-panel PDF with rainbow snapshot format (no
  formation targets).

Formations (per env):
  Gauss2Gauss_Obs3:
    star     (default) 10-pt pentagram terminal positions + dashed connecting lines.
    circles  Two concentric circles (3 inner + 5 outer pts) + dashed circle patches.

  Gauss2Ring_Obs1:
    circle   10 uniformly-spaced pts on the ring (R=2.5) as terminal positions.

Figures:
  trajectory_Step1.png -- backward pass (terminal -> initial), with formation connections.
  trajectory_Step2.png -- forward  pass (initial -> terminal),   with formation connections.
  trajectory_comparison.pdf -- 3-panel comparison of SDE / fwd-ode / pf-ode from p0.

Usage:
  python plot_trajectory.py --model-dir <dir>
  python plot_trajectory.py --model-dir <dir> --formation circles
  python plot_trajectory.py --model-dir <dir> --formation circle   # Gauss2Ring_Obs1
  python plot_trajectory.py --model-dir <dir> --compare            # 3-method comparison
"""

import argparse, math, os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from options import Options
from mfg import MFG
from mfg.constraint import build_constraint
from deepgsb import DeepGSB


# ================================================================
#  Model loading
# ================================================================

def load_deepgsb(model_dir: str, device="cpu"):
    config_path = os.path.join(model_dir, "config.txt")

    # Read saved config
    saved = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    saved[k.strip()] = v.strip()

    problem_name = saved.get("problem_name", "Gauss2Gauss_Obs3")
    x_dim = int(saved.get("x_dim", 2))

    # Start from env defaults, then overlay saved config, then runtime overrides
    from envs import get_env_class

    def _parse(val):
        """Parse a config.txt string value to its Python type."""
        if val in ("None", "none"):
            return None
        if val in ("True", "False"):
            return val == "True"
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    env_class = get_env_class(problem_name)
    defaults = env_class.get_default_config(saved.get("sb_param", "actor-critic"))
    kwargs = dict(defaults)
    for k, v in saved.items():
        kwargs[k] = _parse(v)

    # Runtime overrides
    kwargs.update(
        device=device, cpu=(device == "cpu"), gpu=0, load=None,
        x_dim=x_dim,
        run_dir=model_dir,
        eval_path=os.path.join(model_dir, "plots"),
        model_path=os.path.join(model_dir, "models"),
        log_dir=model_dir,
        use_env_config=True,
    )
    opt = Options(**kwargs)

    mfg = MFG(opt)
    deepgsb = DeepGSB(opt, mfg)

    fwd = os.path.join(model_dir, "models", "forward_policy.pt")
    bwd = os.path.join(model_dir, "models", "backward_policy.pt")
    if not os.path.exists(fwd) or not os.path.exists(bwd):
        print("ERROR: model files not found in", model_dir)
        sys.exit(1)
    deepgsb.z_f.load_state_dict(torch.load(fwd, map_location=device))
    deepgsb.z_b.load_state_dict(torch.load(bwd, map_location=device))
    deepgsb.z_f.freeze()
    deepgsb.z_b.freeze()

    print(f"Model loaded: {problem_name}, dim={x_dim}, device={device}")
    return deepgsb, opt


# ================================================================
#  Trajectory computation
# ================================================================

def _drift_pf(deepgsb, x, t_k):
    """Probability flow drift: 0.5*(z_f - z_b)*g (forward direction)."""
    u_f = deepgsb.z_f(x, t_k)
    u_b = deepgsb.z_b(x, t_k)
    return 0.5 * (u_f - u_b) * deepgsb.z_f.g


@torch.no_grad()
def compute_traj(deepgsb, opt, n_particles=10, mode="pf-ode", formation=None):
    """Run backward + forward, return dict with traj_bwd, traj_fwd, ts, term_pos.

    Args:
        deepgsb: Trained DeepGSB model.
        opt: Options.
        n_particles: Number of particles (overridden by formation if applicable).
        mode: 'pf-ode', 'sde', or 'fwd-ode'.
        formation: None (random pT samples), 'star' (10-pt pentagram),
                   or 'circles' (inner 3 + outer 5 concentric).
    Returns:
        dict with keys: bwd, fwd, ts, term_pos, mode, n_particles, formation.
    """
    ts = deepgsb.mfg.ts
    dt = deepgsb.mfg.dt
    n_steps = len(ts)
    device = opt.device
    dim = opt.x_dim

    g = deepgsb.z_f.g
    noise_scale = math.sqrt(dt) if mode == "sde" else 0.0

    # Determine terminal positions
    # (from formation pattern if given, else random pT samples)
    form_pts = None
    if formation:
        form_pts = _formation_terminal_positions(formation, n_particles, opt)

    if form_pts is not None:
        pT_samples = form_pts.to(device)
        n_particles = pT_samples.shape[0]
        print(f"Using formation '{formation}' with {n_particles} particles")
    else:
        constraints = build_constraint(opt.problem_name, 5000, device, x_dim=dim)
        pT_samples = constraints.pT.sample(n_particles).to(device)
        n_particles = pT_samples.shape[0]

    # Step 1: Backward
    x = pT_samples.clone()
    traj_bwd = torch.zeros(n_steps, n_particles, dim, device=device)
    for k in range(n_steps - 1, -1, -1):
        t_k = ts[k].expand(n_particles, 1).to(device)
        if mode == "pf-ode":
            u = -_drift_pf(deepgsb, x, t_k)
        else:
            u = deepgsb.z_b.g * deepgsb.z_b(x, t_k)
        traj_bwd[k] = x
        if k > 0:
            dw = g * torch.randn_like(x) * noise_scale if noise_scale > 0 else 0
            x = x + u * dt + dw
    init_pos = x.clone()
    print(f"Using {mode}, {n_particles} particles")

    # Step 2: Forward
    x = init_pos.clone()
    traj_fwd = torch.zeros(n_steps, n_particles, dim, device=device)
    for k in range(n_steps):
        t_k = ts[k].expand(n_particles, 1).to(device)
        traj_fwd[k] = x
        if k < n_steps - 1:
            if mode == "pf-ode":
                u = _drift_pf(deepgsb, x, t_k)
            else:
                u = deepgsb.z_f.g * deepgsb.z_f(x, t_k)
            dw = g * torch.randn_like(x) * noise_scale if noise_scale > 0 else 0
            x = x + u * dt + dw

    return {"bwd": traj_bwd, "fwd": traj_fwd, "ts": ts, "term_pos": pT_samples,
            "mode": mode, "n_particles": n_particles, "formation": formation}


@torch.no_grad()
def compute_traj_forward_only(deepgsb, opt, x_init, mode="pf-ode"):
    """Forward-only trajectory from given initial positions (no backward pass).

    Used by plot_trajectory_comparison() to run all 3 modes from the SAME
    p0 samples -- unlike compute_traj() which always starts from pT.

    Args:
        deepgsb: Trained DeepGSB model.
        opt: Options.
        x_init: (n_particles, dim) tensor of initial positions.
        mode: 'pf-ode', 'sde', or 'fwd-ode'.
    Returns:
        traj_fwd: (n_steps, n_particles, dim) tensor.
    """
    ts = deepgsb.mfg.ts
    dt = deepgsb.mfg.dt
    n_steps = len(ts)
    device = opt.device
    n_particles = x_init.shape[0]

    noise_scale = math.sqrt(dt) if mode == "sde" else 0.0

    x = x_init.clone()
    traj_fwd = torch.zeros(n_steps, n_particles, opt.x_dim, device=device)
    for k in range(n_steps):
        t_k = ts[k].expand(n_particles, 1).to(device)
        traj_fwd[k] = x
        if k < n_steps - 1:
            if mode == "pf-ode":
                u = _drift_pf(deepgsb, x, t_k)
            else:
                u = deepgsb.z_f.g * deepgsb.z_f(x, t_k)
            dw = deepgsb.z_f.g * torch.randn_like(x) * noise_scale if noise_scale > 0 else 0
            x = x + u * dt + dw

    return traj_fwd


def optimize_formation_control(deepgsb, opt, x_target, lr=0.05, n_iters=500):
    """Optimise initial positions so forward PF-ODE lands exactly at formation targets.

    Algorithm:
      1. Backward PF-ODE from targets -> initial guess ``x0``.
      2. Gradient descent on ``x0`` to minimise
         ``|| forward_PFODE(x0) - x_target ||^2``.

    Each trajectory is optimised independently; the learned model is frozen.

    Args:
        deepgsb: Trained DeepGSB model.
        opt: Options.
        x_target: (n_particles, dim) target terminal positions.
        lr: Learning rate for Adam (default 0.05).
        n_iters: Number of gradient descent iterations (default 500).
    Returns:
        x0_opt:   (n_particles, dim) optimized initial positions.
        traj_opt: (n_steps, n_particles, dim) forward trajectory from ``x0_opt``.
        loss_hist: List of loss values over iterations.
    """
    ts = deepgsb.mfg.ts
    dt = deepgsb.mfg.dt
    n_steps = len(ts)
    device = opt.device
    n_particles = x_target.shape[0]

    # Step 1: Initial guess via backward PF-ODE
    with torch.no_grad():
        x = x_target.clone()
        for k in range(n_steps - 1, -1, -1):
            t_k = ts[k].expand(n_particles, 1).to(device)
            if k > 0:
                x = x + (-_drift_pf(deepgsb, x, t_k)) * dt
        x0 = x.clone()
        print(f"  Initial guess from backward PF-ODE:  "
              f"x0 in [{x0.min().item():.3f}, {x0.max().item():.3f}]")

    # Step 2: Gradient descent on x0
    x0 = x0.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x0], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    loss_hist = []
    best_x0 = x0.clone().detach()
    best_loss = float("inf")

    for i in range(n_iters):
        optimizer.zero_grad()

        # Forward PF-ODE with gradients enabled
        with torch.enable_grad():
            x = x0
            for k in range(n_steps):
                t_k = ts[k].expand(n_particles, 1).to(device)
                if k < n_steps - 1:
                    x = x + _drift_pf(deepgsb, x, t_k) * dt

            loss = torch.mean((x - x_target) ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([x0], max_norm=10.0)
        optimizer.step()
        scheduler.step()

        val = loss.item()
        loss_hist.append(val)
        if val < best_loss:
            best_loss = val
            best_x0 = x0.clone().detach()

        if (i + 1) % 100 == 0 or i == 0:
            print(f"    iter {i+1:4d}/{n_iters},  loss = {val:.6e},  "
                  f"lr = {scheduler.get_last_lr()[0]:.6f}")

    print(f"  Best loss: {best_loss:.6e}")
    x0_opt = best_x0

    # Final forward trajectory
    with torch.no_grad():
        x = x0_opt.clone()
        traj = torch.zeros(n_steps, n_particles, opt.x_dim, device=device)
        for k in range(n_steps):
            t_k = ts[k].expand(n_particles, 1).to(device)
            traj[k] = x
            if k < n_steps - 1:
                x = x + _drift_pf(deepgsb, x, t_k) * dt

    return x0_opt, traj, loss_hist

# ================================================================

def _obstacles(ax, opt):
    """Draw obstacles (only first 2 dims; HD experiments project to XY plane)."""
    from envs import get_env_class
    try:
        env = get_env_class(opt.problem_name)(device='cpu')
        for center, radius in getattr(env, 'obstacles', []):
            c = center.cpu().numpy() if isinstance(center, torch.Tensor) else np.array(center)
            ax.add_patch(Circle(c[:2], radius, facecolor='darkgray', edgecolor='black', alpha=0.5, zorder=0))
    except Exception:
        pass


def _setup(ax, title, opt):
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_xlabel('$x_1$', fontsize=16); ax.set_ylabel('$x_2$', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(title, fontsize=18)
    ax.grid(True, alpha=0.3)


def _snap(traj, ts, n_snap=5, cmap=plt.cm.rainbow):
    n_steps = traj.shape[0]
    idx = np.linspace(0, n_steps - 1, n_snap).astype(int)
    snap_times = [r'$t=0$', r'$t=0.25T$', r'$t=0.50T$', r'$t=0.75T$', r'$t=T$']
    return idx, snap_times, cmap(np.linspace(0, 1, n_snap))


def _formation_circles(ax, opt):
    """Draw dashed circles around target formations + red triangles at centers."""
    from envs import get_env_class
    try:
        env = get_env_class(opt.problem_name)(device='cpu')
        circles = getattr(env, 'formation_circles', None)
        if circles is None:
            return
        for i, (center, radius) in enumerate(circles):
            c = center.cpu().numpy() if isinstance(center, torch.Tensor) else np.array(center)
            ax.add_patch(Circle(c[:2], radius, facecolor='none', edgecolor="#FDE508",
                                 linestyle='--', linewidth=1.8, alpha=0.6, zorder=4))
            # Red triangle at ring center (encirclement target)
            label = 'encirclement target' if i == 0 else ''
            ax.scatter(c[0], c[1], marker='^', color='red', s=100, zorder=5, label=label)
    except Exception:
        pass


# ================================================================
#  Target formations -- per-environment
# ================================================================

def _star_formation_10pts(center=(4.0, 4.0), R=0.6, r=0.25, rotation=0):
    """10 points tracing a five-pointed star centred at ``center``.

    Alternates outer-vertex -> inner-vertex -> outer-vertex ... so that
    connecting them in order draws the pentagram.
    ``rotation`` is an angular shift (radians) applied to all points.
    """
    cx, cy = center
    pts = []
    for i in range(5):
        outer_angle = -math.pi / 2 + rotation + 2 * math.pi * i / 5
        inner_angle = outer_angle + math.pi / 5
        pts.append([cx + R * math.cos(outer_angle), cy + R * math.sin(outer_angle)])
        pts.append([cx + r * math.cos(inner_angle), cy + r * math.sin(inner_angle)])
    return np.array(pts)


def _concentric_circles_pts(center=(4.0, 4.0), inner_r=0.35, outer_r=0.75):
    """3 points on an inner circle + 5 points on an outer circle."""
    cx, cy = center
    inner = np.array([[cx + inner_r * math.cos(2 * math.pi * i / 3),
                       cy + inner_r * math.sin(2 * math.pi * i / 3)]
                      for i in range(3)])
    outer = np.array([[cx + outer_r * math.cos(2 * math.pi * i / 5),
                       cy + outer_r * math.sin(2 * math.pi * i / 5)]
                      for i in range(5)])
    return inner, outer


def _ring_n_pts(center, R, n=10, rotation=0):
    """n uniformly-spaced points on a circle (optionally rotated)."""
    cx, cy = center
    pts = []
    for i in range(n):
        angle = rotation + 2 * math.pi * i / n
        pts.append([cx + R * math.cos(angle), cy + R * math.sin(angle)])
    return np.array(pts)


def _sphere_shell_terminal_pts():
    """15 specific 3D terminal points: 4+5+6 from the 3 sphere groups.

    Sphere 1 centre (5, 2.5, 0):  4 equatorial pts
    Sphere 2 centre (5, -2.5, 0):  N/S poles + 3 equatorial
    Sphere 3 centre (5.5, 0, 2.5): 3 at 30N + 3 at 30S
    """
    R = 1.5
    c1 = np.array([5.0, 2.5, 0.0])
    c2 = np.array([5.0, -2.5, 0.0])
    c3 = np.array([5.5, 0.0, 2.5])
    pts = []
    # Group 1: 4 equatorial
    for theta in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
        pts.append(c1 + R * np.array([np.cos(theta), np.sin(theta), 0.0]))
    # Group 2: N pole, S pole, 3 equatorial
    pts.append(c2 + np.array([0.0, 0.0, R]))
    pts.append(c2 + np.array([0.0, 0.0, -R]))
    for theta in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
        pts.append(c2 + R * np.array([np.cos(theta), np.sin(theta), 0.0]))
    # Group 3: 3 at 30N + 3 at 30S
    for lat in [np.pi / 6, -np.pi / 6]:
        z_off = R * np.sin(lat)
        r_lat = R * np.cos(lat)
        for theta in [0, 2 * np.pi / 3, 4 * np.pi / 3]:
            pts.append(c3 + r_lat * np.array([np.cos(theta), np.sin(theta), 0.0])
                       + np.array([0.0, 0.0, z_off]))
    return np.array(pts)


# Formation geometry lookup

_FORMATION_GEOM = {
    ("Gauss2Gauss_Obs3", "star"): {
        "n_pts": 10,
        "pts_fn": lambda: _star_formation_10pts((3.8, 4.2), R=0.6, r=0.3, rotation=math.pi/2),
    },
    ("Gauss2Gauss_Obs3", "circles"): {
        "n_pts": 8,
        "pts_fn": lambda: np.vstack(_concentric_circles_pts((4.0, 4.0))),
    },
    ("Gauss2Ring_Obs1", "circle"): {
        "n_pts": 10,
        "pts_fn": lambda: _ring_n_pts((2.0, 0.0), R=2.5, n=10, rotation=math.pi/10),
    },
    ("Gauss2Rings_Obs1", "rings"): {
        "n_pts": 10,
        "pts_fn": lambda: np.vstack([
            _ring_n_pts((3.0, 2.5), R=0.8, n=5),
            _ring_n_pts((3.0, -2.5), R=0.8, n=5),
        ]),
    },
    ("MixGauss2SphereShell_Obs3", "spheres"): {
        "n_pts": 15,
        "pts_fn": _sphere_shell_terminal_pts,
    },
}

# Formation drawing helpers

def _draw_star_connection(ax, term_xy, **kwargs):
    """Connect terminal points with a dashed pentagram."""
    star_x = np.concatenate([term_xy[:, 0], term_xy[0:1, 0]])
    star_y = np.concatenate([term_xy[:, 1], term_xy[0:1, 1]])
    ax.plot(star_x, star_y, '--', color='gray', lw=1.5, alpha=0.7, **kwargs)


def _draw_circles_patches(ax, center=(4.0, 4.0), inner_r=0.35, outer_r=0.75, **kwargs):
    """Draw two actual dashed circle patches (not polygon lines)."""
    for r in (inner_r, outer_r):
        circle = Circle(center[:2], r, facecolor='none', edgecolor='#FFD700',
                        linestyle='--', linewidth=1.8, alpha=0.6, **kwargs)
        ax.add_patch(circle)


def _draw_circle_connection(ax, term_xy, **kwargs):
    """Connect terminal points on a circle (closed polygon)."""
    cx = np.concatenate([term_xy[:, 0], term_xy[0:1, 0]])
    cy = np.concatenate([term_xy[:, 1], term_xy[0:1, 1]])
    ax.plot(cx, cy, '--', color='#FFD700', lw=1.8, alpha=0.6, **kwargs)


def _draw_rings_patches(ax):
    """Dashed circle patches for Gauss2Rings_Obs1 two-ring target."""
    for cx, cy in [(3.0, 2.5), (3.0, -2.5)]:
        ax.add_patch(Circle((cx, cy), 0.8, facecolor='none', edgecolor='#FDE508',
                             linestyle='--', linewidth=1.8, alpha=0.6))


def _draw_rings_connections(ax, term_xy):
    """Connect 5+5 terminal points for two rings separately."""
    n_half = term_xy.shape[0] // 2
    for start in (0, n_half):
        pts = term_xy[start:start + n_half]
        cx = np.concatenate([pts[:, 0], pts[0:1, 0]])
        cy = np.concatenate([pts[:, 1], pts[0:1, 1]])
        ax.plot(cx, cy, '--', color='#FDE508', lw=1.8, alpha=0.6)
        ax.scatter(pts[:, 0], pts[:, 1], s=28, color='#FDE508', alpha=0.6, zorder=6)


def _formation_terminal_positions(formation, n_particles, opt):
    """Generate terminal particle positions in a formation pattern.

    Returns (n_form, dim) tensor, or **None** if problem/formation unknown.
    """
    key = (opt.problem_name, formation)
    if key not in _FORMATION_GEOM:
        return None
    info = _FORMATION_GEOM[key]
    pts_2d = info["pts_fn"]()
    device = opt.device
    dim = opt.x_dim
    pts = torch.zeros(info["n_pts"], dim, device=device)
    pts[:, :2] = torch.tensor(pts_2d, dtype=torch.float, device=device)
    return pts


def plot_step1(result, opt, save_path):
    """Backward pass -- terminal positions -> initial positions."""
    traj, ts = result["bwd"], result["ts"]
    N = result["n_particles"]
    idx, labels, colors = _snap(traj, ts, cmap=plt.cm.viridis)
    formation = result.get("formation", None)

    fig, ax = plt.subplots(figsize=(9, 8))
    _obstacles(ax, opt)

    for i in range(N):
        ax.plot(traj[:, i, 0].cpu(), traj[:, i, 1].cpu(), 'k-', alpha=0.15, lw=0.5, zorder=1)

    for si, c in zip(idx, colors):
        ax.scatter(traj[si, :, 0].cpu(), traj[si, :, 1].cpu(), color=c, s=18, alpha=0.85, zorder=2)

    # Formation connections at terminal (Step1 = backward, t=T is traj[-1])
    term_np = traj[-1].cpu().numpy()
    if formation == 'star' and 'Gauss2Gauss_Obs3' in opt.problem_name:
        _draw_star_connection(ax, term_np)
    elif formation == 'circles' and 'Gauss2Gauss_Obs3' in opt.problem_name:
        _draw_circles_patches(ax)
    elif formation == 'circle' and 'Gauss2Ring_Obs1' in opt.problem_name:
        _draw_circle_connection(ax, term_np)
    elif formation == 'rings' and 'Gauss2Rings_Obs1' in opt.problem_name:
        _draw_rings_patches(ax)
        _draw_rings_connections(ax, term_np)

    # Star markers for terminal positions at T=1
    ax.scatter(traj[-1, :, 0].cpu(), traj[-1, :, 1].cpu(),
               color='red', s=50, marker='*', zorder=3, label="terminal")

    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                  markersize=8, label=labels[i]) for i in range(len(idx))]
    leg += [Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label="terminal")]
    ax.legend(handles=leg, fontsize=13, loc='upper right')
    _setup(ax, f"Step 1: Backward {result['mode'].upper()}\n{opt.problem_name} (d={opt.x_dim})", opt)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {save_path}")


def plot_step2(result, opt, save_path, title=None):
    """Forward pass -- controlled trajectory."""
    traj, ts = result["fwd"], result["ts"]
    N = result["n_particles"]
    idx, labels, colors = _snap(traj, ts, cmap=plt.cm.rainbow)
    formation = result.get("formation", None)

    fig, ax = plt.subplots(figsize=(9, 8))
    _obstacles(ax, opt)
    _formation_circles(ax, opt)

    # Formation connections at terminal (Step2 = forward, t=T is traj[-1])
    term_np = traj[-1].cpu().numpy()
    if formation == 'star' and 'Gauss2Gauss_Obs3' in opt.problem_name:
        _draw_star_connection(ax, term_np)
    elif formation == 'circles' and 'Gauss2Gauss_Obs3' in opt.problem_name:
        _draw_circles_patches(ax)
    elif formation == 'circle' and 'Gauss2Ring_Obs1' in opt.problem_name:
        _draw_circle_connection(ax, term_np)
    elif formation == 'rings' and 'Gauss2Rings_Obs1' in opt.problem_name:
        _draw_rings_patches(ax)
        _draw_rings_connections(ax, term_np)

    for i in range(N):
        ax.plot(traj[:, i, 0].cpu(), traj[:, i, 1].cpu(), 'k-', alpha=0.15, lw=0.5, zorder=1)

    for si, c in zip(idx, colors):
        ax.scatter(traj[si, :, 0].cpu(), traj[si, :, 1].cpu(), color=c, s=18, alpha=0.85, zorder=2)

    leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i],
                  markersize=8, label=labels[i]) for i in range(len(idx))]
    ax.legend(handles=leg, fontsize=13, loc='upper right')
    plot_title = title if title is not None else f"Step 2: Forward {result['mode'].upper()}\n{opt.problem_name} (d={opt.x_dim})"
    _setup(ax, plot_title, opt)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"Saved: {save_path}")


def plot_trajectory_comparison(deepgsb, opt, save_path, n_particles=10):
    """Compare SDE, optimal-control ODE, and PF-ODE from the same p0 samples.

    Samples ``n_particles`` from the initial distribution p0, then propagates
    forward with each of the three methods (SDE, fwd-ode, pf-ode) from the
    SAME starting points.  Produces a 3-panel PDF using the rainbow snapshot
    format (no formation targets or terminal decorations).

    Args:
        deepgsb: Trained DeepGSB model.
        opt: Options.
        save_path: Path to save the PDF.
        n_particles: Number of samples from p0 (default 10).
    """
    device = opt.device

    # Sample from initial distribution p0
    constraints = build_constraint(opt.problem_name, 5000, device, x_dim=opt.x_dim)
    x_init = constraints.p0.sample(n_particles).to(device)
    print(f"Sampled {n_particles} particles from initial distribution p0")

    # Run all 3 modes from the SAME initial positions
    mode_keys = ["sde", "fwd-ode", "pf-ode"]
    mode_labels = ["SDE", "ODE", "PF-ODE"]

    trajs = {}
    for mk in mode_keys:
        trajs[mk] = compute_traj_forward_only(deepgsb, opt, x_init, mode=mk)
        print(f"  {mk}: done")

    # Snapshot indices (same for all 3)
    ts = deepgsb.mfg.ts
    idx, snap_labels, snap_colors = _snap(trajs["pf-ode"], ts)

    # 3-panel figure (horizontal, compact)
    fig, axes = plt.subplots(1, 3, figsize=(28, 8.5))
    fig.suptitle(f"{opt.problem_name}  (d={opt.x_dim})",
                 fontsize=35, y=1.05)

    for col, (mk, ml) in enumerate(zip(mode_keys, mode_labels)):
        ax = axes[col]
        traj = trajs[mk]
        N = traj.shape[1]

        _obstacles(ax, opt)
        # NO _formation_circles -- user explicitly asked to omit terminal targets

        # Trajectory lines (gray, semi-transparent)
        for i in range(N):
            ax.plot(traj[:, i, 0].cpu(), traj[:, i, 1].cpu(),
                    'k-', alpha=0.15, lw=0.5, zorder=1)

        # Snapshot markers (rainbow: blue->red through green/yellow)
        for si, c in zip(idx, snap_colors):
            ax.scatter(traj[si, :, 0].cpu(), traj[si, :, 1].cpu(),
                       color=c, s=28, alpha=0.85, zorder=2)

        # Highlight start (t=0) with slightly larger markers (no black edge)
        ax.scatter(traj[0, :, 0].cpu(), traj[0, :, 1].cpu(),
                   color=snap_colors[0], s=70, marker='o',
                   alpha=0.9, zorder=3)

        # Highlight terminal (t=T) with slightly larger markers (no black edge)
        ax.scatter(traj[-1, :, 0].cpu(), traj[-1, :, 1].cpu(),
                   color=snap_colors[-1], s=70, marker='o',
                   alpha=0.9, zorder=3)

        # Legend for snapshot times
        leg = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=snap_colors[i],
                      markersize=10, label=snap_labels[i])
               for i in range(len(idx))]
        ax.legend(handles=leg, fontsize=14, loc='upper left')

        # Larger-font axis setup
        ax.set_aspect('equal')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('$x_1$', fontsize=24)
        ax.set_ylabel('$x_2$', fontsize=24)
        ax.tick_params(axis='both', labelsize=20)
        ax.set_title(ml, fontsize=30, pad=6)
        ax.grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.04, right=0.98, wspace=0.04, top=0.92, bottom=0.06)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _draw_formation_connections(ax, formation, term_xy, opt):
    """Draw yellow dashed connections at terminal formation positions."""
    if formation == 'star' and 'Gauss2Gauss_Obs3' in opt.problem_name:
        pts = np.concatenate([term_xy[:, 0], term_xy[0:1, 0]])
        pts_y = np.concatenate([term_xy[:, 1], term_xy[0:1, 1]])
        ax.plot(pts, pts_y, '--', color='#FFD700', lw=2.2, alpha=0.8)
    elif formation == 'circles' and 'Gauss2Gauss_Obs3' in opt.problem_name:
        for r in (0.35, 0.75):
            ax.add_patch(Circle((4.0, 4.0), r, facecolor='none',
                         edgecolor='#FFD700', linestyle='--', lw=2.2, alpha=0.8))
    elif formation == 'circle' and 'Gauss2Ring_Obs1' in opt.problem_name:
        pts = np.concatenate([term_xy[:, 0], term_xy[0:1, 0]])
        pts_y = np.concatenate([term_xy[:, 1], term_xy[0:1, 1]])
        ax.plot(pts, pts_y, '--', color='#FFD700', lw=2.2, alpha=0.8)
    elif formation == 'rings' and 'Gauss2Rings_Obs1' in opt.problem_name:
        # Yellow dashed circles only (no point-to-point lines)
        for cx, cy in [(3.0, 2.5), (3.0, -2.5)]:
            ax.add_patch(Circle((cx, cy), 0.8, facecolor='none',
                         edgecolor='#FFD700', linestyle='--', lw=2.2, alpha=0.8))
            # Red triangle at ring centre
            ax.scatter(cx, cy, marker='^', color='red', s=120, zorder=5)


def plot_formation_control(deepgsb, opt, save_path, formation=None,
                           n_particles=10, lr=0.05, n_iters=500):
    """PF-ODE formation control: optimise initial positions for exact terminal fit.

    For the given problem, the appropriate formation is auto-detected:
      Gauss2Gauss_Obs3 -> star     (10 pts centred at (4,4))
      Gauss2Ring_Obs1  -> circle   (10 pts on a ring)
      Gauss2Rings_Obs1 -> rings    (5+5 pts on two rings)

    Trajectories are shown with rainbow snapshot markers and yellow dashed
    formation connections at the terminal positions.

    Args:
        deepgsb: Trained DeepGSB model.
        opt: Options.
        save_path: Path to save the PDF.
        formation: Formation name or ``None`` for auto-detect.
        n_particles: Number of particles (default 10).
        lr: Learning rate for gradient descent (default 0.05).
        n_iters: Number of optimisation iterations (default 500).
    """
    device = opt.device
    if formation is None:
        formation = _default_formation(opt.problem_name)
    if formation is None:
        print(f"No formation defined for {opt.problem_name}, skipping")
        return

    # Target positions from formation geometry
    targets = _formation_terminal_positions(formation, n_particles, opt)
    if targets is None:
        print(f"Unknown formation '{formation}' for {opt.problem_name}, skipping")
        return

    print(f"Running PF-ODE formation control: '{formation}' "
          f"({targets.shape[0]} particles), lr={lr}, iters={n_iters}")

    # Optimize
    x0_opt, traj_opt, loss_hist = optimize_formation_control(
        deepgsb, opt, targets, lr=lr, n_iters=n_iters)

    # Also run un-optimised PF-ODE from p0 for reference
    constraints = build_constraint(opt.problem_name, 5000, device, x_dim=opt.x_dim)
    x_init_p0 = constraints.p0.sample(targets.shape[0]).to(device)
    traj_p0 = compute_traj_forward_only(deepgsb, opt, x_init_p0, mode="pf-ode")

    # Determine if 3D -> show only first 2 dims
    ts = deepgsb.mfg.ts
    idx, snap_labels, snap_colors = _snap(traj_opt, ts)
    N = traj_opt.shape[1]

    fig, (ax_main, ax_ref) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"{opt.problem_name}  (d={opt.x_dim})",
                 fontsize=24, y=1.01)

    # Per-problem settings
    is_gauss2gauss = 'Gauss2Gauss_Obs3' in opt.problem_name
    leg_loc = 'upper left' if is_gauss2gauss else 'upper right'

    # Left panel: optimised formation control
    ax = ax_main
    _obstacles(ax, opt)

    for i in range(N):
        ax.plot(traj_opt[:, i, 0].cpu(), traj_opt[:, i, 1].cpu(),
                'k-', alpha=0.15, lw=0.5, zorder=1)

    for si, c in zip(idx, snap_colors):
        ax.scatter(traj_opt[si, :, 0].cpu(), traj_opt[si, :, 1].cpu(),
                   color=c, s=22, alpha=0.85, zorder=2)

    # Start & terminal markers (slightly smaller)
    ax.scatter(traj_opt[0, :, 0].cpu(), traj_opt[0, :, 1].cpu(),
               color=snap_colors[0], s=45, marker='o', alpha=0.9, zorder=3)
    ax.scatter(traj_opt[-1, :, 0].cpu(), traj_opt[-1, :, 1].cpu(),
               color=snap_colors[-1], s=45, marker='o', alpha=0.9, zorder=3)

    # Yellow dashed formation connections
    term_np = traj_opt[-1].cpu().numpy()
    _draw_formation_connections(ax, formation, term_np, opt)

    leg = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=snap_colors[i],
                  markersize=10, label=snap_labels[i])
           for i in range(len(idx))]
    ax.legend(handles=leg, fontsize=13, loc=leg_loc)
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_xlabel('$x_1$', fontsize=22); ax.set_ylabel('$x_2$', fontsize=22)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title('PF-ODE for DeepGSB', fontsize=22, pad=6)
    ax.grid(True, alpha=0.3)

    # Loss inset -- bottom-left for rings/circle, bottom-right for star
    if is_gauss2gauss:
        ax_inset = fig.add_axes([0.3, 0.13, 0.13, 0.12])
    else:
        ax_inset = fig.add_axes([0.13, 0.15, 0.13, 0.12])
    ax_inset.semilogy(loss_hist, 'b-', lw=1.2)
    ax_inset.set_xlabel('iteration', fontsize=9)
    ax_inset.set_ylabel('MSE loss', fontsize=9)
    ax_inset.tick_params(labelsize=8)
    ax_inset.set_title('Optimization loss', fontsize=10)
    ax_inset.grid(True, alpha=0.3)

    # Right panel: reference (backward PF-ODE initial guess)
    ax = ax_ref
    _obstacles(ax, opt)

    for i in range(N):
        ax.plot(traj_p0[:, i, 0].cpu(), traj_p0[:, i, 1].cpu(),
                'k-', alpha=0.15, lw=0.5, zorder=1)

    for si, c in zip(idx, snap_colors):
        ax.scatter(traj_p0[si, :, 0].cpu(), traj_p0[si, :, 1].cpu(),
                   color=c, s=22, alpha=0.85, zorder=2)
    ax.scatter(traj_p0[0, :, 0].cpu(), traj_p0[0, :, 1].cpu(),
               color=snap_colors[0], s=45, marker='o', alpha=0.9, zorder=3)
    ax.scatter(traj_p0[-1, :, 0].cpu(), traj_p0[-1, :, 1].cpu(),
               color=snap_colors[-1], s=45, marker='o', alpha=0.9, zorder=3)

    leg = [Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=snap_colors[i],
                  markersize=10, label=snap_labels[i])
           for i in range(len(idx))]
    ax.legend(handles=leg, fontsize=13, loc=leg_loc)
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
    ax.set_xlabel('$x_1$', fontsize=22); ax.set_ylabel('$x_2$', fontsize=22)
    ax.tick_params(axis='both', labelsize=18)
    ax.set_title(r'$x_0$ from Backward PF-ODE', fontsize=22, pad=6)
    ax.grid(True, alpha=0.3)

    plt.subplots_adjust(left=0.05, right=0.97, wspace=0.20, top=0.90, bottom=0.07)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_pf_ode_trajectory(deepgsb, opt, save_path, title=None, formation=None):
    """Convenience: run PF ODE mode and save only the forward trajectory.

    Args:
        deepgsb: Trained DeepGSB model.
        opt: Options (must hold problem_name, device, x_dim, ...).
        save_path: Where to save the PNG figure.
        title:  Plot title.  Defaults to ``opt.problem_name``.
        formation: ``None`` (random pT samples), ``'star'``, ``'circles'``, or ``'circle'``.
    """
    result = compute_traj(deepgsb, opt, n_particles=10, mode="pf-ode", formation=formation)
    plot_step2(result, opt, save_path, title=title or opt.problem_name)
    # Also save 3D trajectory when d > 2
    if opt.x_dim > 2:
        save_path_3d = save_path.replace(".pdf", "_3d.pdf")
        plot_trajectory_3d(result, opt, save_path_3d)


def plot_trajectory_3d(result, opt, save_path):
    """3D trajectory -- start/end points + trajectories only, no intermediate markers.

    Intended for d>2 experiments (shows first 3 dims).
    """
    traj = result["fwd"]  # (n_steps, N, dim)
    N = result["n_particles"]
    device = opt.device

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw obstacle as a 3D cylinder
    _obstacles_3d(ax, opt)

    # Trajectories
    for i in range(N):
        ax.plot(traj[:, i, 0].cpu(), traj[:, i, 1].cpu(), traj[:, i, 2].cpu(),
                'k-', alpha=0.2, lw=0.6, zorder=1)

    # Start points (t=0) -- blue
    ax.scatter(traj[0, :, 0].cpu(), traj[0, :, 1].cpu(), traj[0, :, 2].cpu(),
               color='blue', s=40, alpha=0.8, zorder=3, label=r'$t=0$')

    # End points (t=T) -- red
    ax.scatter(traj[-1, :, 0].cpu(), traj[-1, :, 1].cpu(), traj[-1, :, 2].cpu(),
               color='red', s=40, alpha=0.8, zorder=3, label=r'$t=T$')

    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    ax.set_zlabel('$x_3$', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_title(f"{opt.problem_name} (d={opt.x_dim})", fontsize=18)
    ax.legend(fontsize=14, loc='upper right')
    # Fix axis limits: use (-6,6) for x1,x2, auto-scale x3 but match data range
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.view_init(elev=25, azim=-45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def _obstacles_3d(ax, opt):
    """Draw obstacles as 3D cylinders (only first 2 dims for center/radius)."""
    from envs import get_env_class
    try:
        env = get_env_class(opt.problem_name)(device='cpu')
        for center, radius in getattr(env, 'obstacles', []):
            c = center.cpu().numpy()[:2] if isinstance(center, torch.Tensor) else np.array(center)[:2]
            z_vals = np.linspace(-4, 4, 20)
            theta = np.linspace(0, 2 * np.pi, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_vals)
            x_grid = c[0] + radius * np.cos(theta_grid)
            y_grid = c[1] + radius * np.sin(theta_grid)
            ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.15, edgecolor='none')
    except Exception:
        pass


# ================================================================
#  CLI
# ================================================================


def _default_formation(problem_name):
    """Default formation per environment (or None)."""
    if 'Gauss2Gauss_Obs3' in problem_name:
        return 'star'
    if 'Gauss2Ring_Obs1' in problem_name:
        return 'circle'
    if 'Gauss2Rings_Obs1' in problem_name:
        return 'rings'
    return None


def main():
    modes = {"pf-ode": "Probability Flow ODE", "sde": "SDE", "fwd-ode": "Forward ODE"}
    formations = {"star": "10-pt pentagram (Gauss2Gauss)",
                  "circles": "Two concentric circles (Gauss2Gauss)",
                  "circle": "10 pts on ring (Gauss2Ring)"}
    parser = argparse.ArgumentParser(description="Trajectory plotting (3 modes + chain + comparison + formation-control).")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mode", choices=list(modes), default="pf-ode",
                        help=f"default: pf-ode ({modes['pf-ode']})")
    parser.add_argument("--n-particles", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--formation", choices=list(formations) + [""], default="",
                        help="Formation mode (env-dependent default if omitted)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare SDE / fwd-ode / pf-ode from p0 in one 3-panel figure")
    parser.add_argument("--formation-control", action="store_true",
                        help="PF-ODE formation control: optimise x0 so trajectories match target formation")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate for formation-control gradient descent (default 0.05)")
    parser.add_argument("--n-iters", type=int, default=500,
                        help="Iterations for formation-control optimisation (default 500)")
    args = parser.parse_args()

    deepgsb, opt = load_deepgsb(args.model_dir, device=args.device)

    # Formation control mode
    if args.formation_control:
        sd = args.save_dir or os.path.join(args.model_dir, "plots")
        os.makedirs(sd, exist_ok=True)
        formation = args.formation if args.formation else None
        save_path = os.path.join(sd,
                                 f"{opt.problem_name}_d{opt.x_dim}_formation_control.pdf")
        plot_formation_control(deepgsb, opt, save_path, formation=formation,
                               n_particles=args.n_particles, lr=args.lr, n_iters=args.n_iters)
        return

    # Comparison mode
    if args.compare:
        sd = args.save_dir or os.path.join(args.model_dir, "plots")
        os.makedirs(sd, exist_ok=True)
        save_path = os.path.join(sd,
                                 f"{opt.problem_name}_d{opt.x_dim}_trajectory_comparison.pdf")
        plot_trajectory_comparison(deepgsb, opt, save_path, n_particles=args.n_particles)
        return

    formation = args.formation if args.formation else _default_formation(opt.problem_name)
    result = compute_traj(deepgsb, opt, n_particles=args.n_particles,
                          mode=args.mode, formation=formation)

    sd = args.save_dir or os.path.join(args.model_dir, "plots")
    os.makedirs(sd, exist_ok=True)
    plot_step1(result, opt, os.path.join(sd, f"{opt.problem_name}_d{opt.x_dim}_trajectory_Step1.pdf"))
    plot_step2(result, opt, os.path.join(sd, f"{opt.problem_name}_d{opt.x_dim}_trajectory_Step2.pdf"))

    # 3D trajectory when dim > 2
    if opt.x_dim > 2:
        plot_trajectory_3d(result, opt, os.path.join(sd, f"{opt.problem_name}_d{opt.x_dim}_trajectory_Step2_3d.pdf"))


if __name__ == "__main__":
    main()
