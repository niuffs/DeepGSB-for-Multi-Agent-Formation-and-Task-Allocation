"""
Scenario: Mixed Gaussians -> 3 cylindrical obstacles -> 3 spherical shells.
  - Start: Two 3D Gaussian mixtures on ground at z=-2 (left side, x approx -4)
  - Obstacles: 3 cylindrical obstacles
      Left:  2 at x=-2, y=pm 1.5 (symmetric about x-axis)
      Right: 1 at x=2, y=0
  - Target (T=1): 3 spherical shells in the region x > 4
      Sphere 1: center (5,  2.5, 0),   R=1.5
      Sphere 2: center (5, -2.5, 0),   R=1.5
      Sphere 3: center (5.5, 0,  2.5), R=1.5
  - Each spherical shell uses a smooth Gaussian radial profile:
      p(x) ~ exp(-alpha (||x-c||^2 - R^2)^2)
"""
import logging
import math
import torch
import torch.distributions as td
from easydict import EasyDict as edict


log = logging.getLogger(__file__)


class SphericalShellDistribution(td.Distribution):
    """3D spherical shell: p(x) ~ exp(-alpha (||x-c||^2 - R^2)^2)."""

    arg_constraints = {}
    event_shape = torch.Size([3])

    def __init__(self, R: float, alpha: float,
                 center=(0.0, 0.0, 0.0), device='cpu',
                 validate_args=None):
        super().__init__(validate_args=validate_args)
        self.R = float(R)
        self.alpha = float(alpha)
        self.cx = float(center[0])
        self.cy = float(center[1])
        self.cz = float(center[2])
        self.device = device
        self.logZ = (1.5 * math.log(math.pi)
                     + math.log(2 * self.R)
                     - 0.5 * math.log(alpha))

    def log_prob(self, value):
        dx = value[..., 0] - self.cx
        dy = value[..., 1] - self.cy
        dz = value[..., 2] - self.cz
        r2 = dx ** 2 + dy ** 2 + dz ** 2
        return -self.alpha * (r2 - self.R ** 2) ** 2 - self.logZ

    def sample(self, sample_shape=torch.Size()):
        n = sample_shape[0] if len(sample_shape) > 0 else 1
        theta = torch.rand(n, device=self.device) * (2 * math.pi)
        phi = torch.acos(2 * torch.rand(n, device=self.device) - 1)
        sigma_r = 1.0 / (2.0 * self.R * math.sqrt(2.0 * self.alpha))
        r = self.R + torch.randn(n, device=self.device) * sigma_r
        r = r.clamp(min=0)
        x = self.cx + r * torch.sin(phi) * torch.cos(theta)
        y = self.cy + r * torch.sin(phi) * torch.sin(theta)
        z = self.cz + r * torch.cos(phi)
        return torch.stack([x, y, z], dim=1)

    @property
    def support(self):
        return td.constraints.real


class MixtureSphereShells(td.Distribution):
    """Mixture of multiple SphericalShellDistribution objects."""

    arg_constraints = {}
    event_shape = torch.Size([3])

    def __init__(self, spheres, weights=None, device='cpu', validate_args=None):
        super().__init__(validate_args=validate_args)
        self.spheres = spheres
        n = len(spheres)
        if weights is None:
            log_w = torch.full((n,), -math.log(n), device=device)
        else:
            w = torch.tensor(weights, device=device)
            log_w = w.log()
        self._log_weights = log_w - torch.logsumexp(log_w, dim=0)
        self.device = device

    def sample(self, sample_shape=torch.Size()):
        n = sample_shape[0] if len(sample_shape) > 0 else 1
        idx = torch.multinomial(self._log_weights.exp(), n, replacement=True)
        samples = torch.zeros(n, 3, device=self.device)
        for i, sphere in enumerate(self.spheres):
            mask = idx == i
            if mask.any():
                samples[mask] = sphere.sample([mask.sum().item()])
        return samples

    def log_prob(self, value):
        log_probs = torch.stack([s.log_prob(value) for s in self.spheres], dim=-1)
        return torch.logsumexp(log_probs + self._log_weights, dim=-1)


class MixGauss2SphereShell_Obs3:
    """3D: mixed Gaussian start at z=-2 -> 3 cylindrical obstacles -> 3 spherical shells at x > 4."""

    def __init__(self, device='cpu', d=3, init_cov=0.1, diffusion_std=1.0):
        self.dim = d
        self.TT = 1.0
        self.lambda_L = 0.05
        self.lambda_E = 0.05
        self.lambda_P = 10.0
        self.device = device
        self.name = "MixGauss2SphereShell_Obs3"
        self.diffusion_std = diffusion_std

        # 3 cylindrical obstacles
        self.obstacles = [
            (torch.tensor([-2.0, 4.0, 0.0], device=device), 1.0),
            (torch.tensor([-2.0, -4.0, 0.0], device=device), 1.0),
            (torch.tensor([2.0, 0.0, 0.0], device=device), 1.2),
        ]

        # Two Gaussian centres at z=-2 (ground plane), left side
        self.g1_center = torch.tensor([-6.0, 1.5, -2.0], device=device)
        self.g2_center = torch.tensor([-6.0, -1.5, -2.0], device=device)

        # 3 spherical shell targets
        self.sphere_centers = [
            torch.tensor([6.0, 2.5, 0.0], device=device),
            torch.tensor([6.0, -2.5, 0.0], device=device),
            torch.tensor([7.0, 0.0, 2.5], device=device),
        ]
        self.sphere_radius = 1.5
        self.sphere_alpha = 8.0

        self._build_target_dist()

    @property
    def formation_circles(self):
        """Spheres for plot markers."""
        return [(c, self.sphere_radius) for c in self.sphere_centers]

    def _build_target_dist(self):
        spheres = [
            SphericalShellDistribution(R=self.sphere_radius, alpha=self.sphere_alpha,
                                        center=c.tolist(), device=self.device)
            for c in self.sphere_centers
        ]
        mix = MixtureSphereShells(spheres, device=self.device)
        if self.dim == 3:
            self.target_dist = mix
        else:
            from .Gauss2Ring_Obs1 import PaddedDistribution
            self.target_dist = PaddedDistribution(
                mix, full_dim=self.dim, extra_std=0.1, device=self.device
            )

    def sample_rho0(self, num_samples, var_scale=0.1):
        """Sample from two Gaussian clouds at z=-2, left side."""
        n1 = num_samples // 2
        n2 = num_samples - n1
        g1 = self.g1_center.unsqueeze(0) + math.sqrt(0.1) * torch.randn(n1, self.dim, device=self.device)
        g2 = self.g2_center.unsqueeze(0) + math.sqrt(0.1) * torch.randn(n2, self.dim, device=self.device)
        samples = torch.cat([g1, g2], dim=0)
        return samples + var_scale * torch.randn_like(samples) * 0.05

    def terminal_cost(self, x):
        """KL-based terminal cost: -log p_T(x)."""
        return -self.target_dist.log_prob(x)

    @property
    def info_dict(self):
        return {
            'env_name': self.name, 'dim': self.dim,
            'TT': self.TT, 'lambda_L': self.lambda_L,
            'lambda_E': self.lambda_E, 'lambda_P': self.lambda_P,
        }

    # 3D scenario setup

    @torch.no_grad()
    def render_setup(self, opt, save_path=None):
        """3D scenario setup: cylinder obstacles, p0 (blue), pT (red spheres)."""
        import os, numpy as np, matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        from mfg.constraint import build_constraint

        constraints = build_constraint(self.name, 2000, opt.device, x_dim=opt.x_dim)
        p0_samp = constraints.p0.sample(2000).detach().cpu().numpy()
        pT_samp = constraints.pT.sample(2000).detach().cpu().numpy()

        out_path = save_path or getattr(opt, 'eval_path', '.')
        os.makedirs(out_path, exist_ok=True)

        # 2D xy-plane view
        fig, ax = plt.subplots(figsize=(9, 8))
        ax.set_aspect('equal')
        ax.set_xlim(-8, 10); ax.set_ylim(-6, 6)
        ax.set_xticks(range(-8, 11, 2)); ax.set_yticks(range(-6, 7, 2))
        ax.set_xlabel('$x_1$', fontsize=18); ax.set_ylabel('$x_2$', fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f"DeepGSB for {self.name}", fontsize=18)

        for center, radius in self.obstacles:
            circle = Circle((center[0].item(), center[1].item()), radius,
                            facecolor='darkgray', edgecolor='black', alpha=0.4)
            ax.add_patch(circle)

        ax.scatter(p0_samp[:, 0], p0_samp[:, 1], s=3, color='blue', alpha=0.4, label=r'$\rho_0$')
        ax.scatter(pT_samp[:, 0], pT_samp[:, 1], s=3, color='red', alpha=0.4, label=r'$\rho_T$')
        ax.legend(fontsize=10, loc='upper right')
        plt.savefig(os.path.join(out_path, f"{self.name}_d{self.dim}_setup_xy.pdf"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 3D view
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for center, radius in self.obstacles:
            cx, cy = center[0].item(), center[1].item()
            z_vals = np.linspace(-6, 6, 30)
            theta = np.linspace(0, 2 * np.pi, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_vals)
            x_grid = cx + radius * np.cos(theta_grid)
            y_grid = cy + radius * np.sin(theta_grid)
            ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.20, edgecolor='none')

        for c in self.sphere_centers:
            cx, cy, cz = c.tolist()
            R = self.sphere_radius
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            UX, VX = np.meshgrid(u, v)
            sx = cx + R * np.cos(UX) * np.sin(VX)
            sy = cy + R * np.sin(UX) * np.sin(VX)
            sz = cz + R * np.cos(VX)
            ax.plot_wireframe(sx, sy, sz, color='red', alpha=0.15, linewidth=0.4)
            ax.scatter(cx, cy, cz, marker='^', color='red', s=80)

        zdim = min(self.dim, 3)
        ax.scatter(p0_samp[:, 0], p0_samp[:, 1], p0_samp[:, 2] if zdim > 2 else 0,
                   s=3, color='blue', alpha=0.4, label=r'$\rho_0$')
        ax.scatter(pT_samp[:, 0], pT_samp[:, 1], pT_samp[:, 2] if zdim > 2 else 0,
                   s=3, color='red', alpha=0.4, label=r'$\rho_T$')

        ax.set_xlim(-8, 10); ax.set_ylim(-6, 6); ax.set_zlim(-6, 6)
        ax.set_xlabel('$x_1$', fontsize=16); ax.set_ylabel('$x_2$', fontsize=16); ax.set_zlabel('$x_3$', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f"DeepGSB for {self.name}", fontsize=18)
        ax.legend(fontsize=14, loc='upper right')
        ax.view_init(elev=25, azim=-45)
        plt.savefig(os.path.join(out_path, f"{self.name}_d{self.dim}_setup.pdf"), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Training-snapshot trajectories (2D top-down)

    @torch.no_grad()
    def render_trajectories(self, opt, mfg, policy_f, policy_b, stage):
        """2D top-down (xy-plane) forward / backward / logp snapshots."""
        import os, numpy as np, matplotlib.pyplot as plt
        import matplotlib.colors as mcol
        from matplotlib.patches import Circle
        from mfg.plotting import get_func_mesher

        ts = mfg.ts
        n_steps = len(ts)
        snap_indices = np.linspace(0, n_steps - 1, 5).astype(int)
        titles = [r'$t=0$', r'$t=0.25T$', r'$t=0.50T$', r'$t=0.75T$', r'$t=T$']
        xlims, ylims = [-8, 10], [-6, 6]

        xs_f, _, _, _ = mfg.sde.sample_traj(ts, policy_f)
        xs_b, _, _, _ = mfg.sde.sample_traj(ts, policy_b)

        _cmap = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["b", "r"])
        colors = _cmap(np.linspace(0.0, 1.0, 5))

        def get_obstacle_patches():
            patches = []
            for center, radius in self.obstacles:
                patches.append(Circle(
                    (center[0].item(), center[1].item()), radius,
                    facecolor='darkgray', edgecolor='black', alpha=0.4))
            return patches

        def _setup_ax(ax, title):
            ax.set_aspect('equal')
            ax.set_xlim(xlims); ax.set_ylim(ylims)
            ax.set_title(title, fontsize=18)
            ax.set_xticks([]); ax.set_yticks([])

        vf_mesher = get_func_mesher(opt, ts, 200, policy_f.compute_value)
        vb_mesher = get_func_mesher(opt, ts, 200, policy_b.compute_value)

        # Forward (xy-plane)
        fig = plt.figure(figsize=(20, 4))
        for i, idx in enumerate(snap_indices):
            ax = fig.add_subplot(1, 5, i + 1)
            for p in get_obstacle_patches():
                ax.add_patch(p)
            pts = xs_f[:, idx].detach().cpu().numpy()
            ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color=colors[i], alpha=0.5)
            if vf_mesher is not None:
                cp = ax.contour(*vf_mesher(idx), levels=10, cmap="copper", linewidths=1, zorder=2)
                ax.clabel(cp, inline=True, fontsize=10)
            _setup_ax(ax, titles[i])
            if i == 0:
                ax.set_ylabel(rf'$d={opt.x_dim}$', fontsize=18)
        plt.tight_layout()
        os.makedirs(os.path.join(opt.eval_path, "forward"), exist_ok=True)
        plt.savefig(os.path.join(opt.eval_path, "forward", f"stage{stage}.pdf"), bbox_inches='tight')
        plt.close(fig)

        # Backward (xy-plane)
        fig = plt.figure(figsize=(20, 4))
        for i, idx in enumerate(snap_indices):
            ax = fig.add_subplot(1, 5, i + 1)
            for p in get_obstacle_patches():
                ax.add_patch(p)
            pts = xs_b[:, idx].detach().cpu().numpy()
            ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color=colors[i], alpha=0.5)
            if vb_mesher is not None:
                cp = ax.contour(*vb_mesher(idx), levels=10, cmap="copper", linewidths=1, zorder=2)
                ax.clabel(cp, inline=True, fontsize=10)
            _setup_ax(ax, titles[i])
            if i == 0:
                ax.set_ylabel(rf'$d={opt.x_dim}$', fontsize=18)
        plt.tight_layout()
        os.makedirs(os.path.join(opt.eval_path, "backward"), exist_ok=True)
        plt.savefig(os.path.join(opt.eval_path, "backward", f"stage{stage}.pdf"), bbox_inches='tight')
        plt.close(fig)

        # log-density (v_f + v_b = log rho)
        def logp_fn(x, t):
            return policy_f.compute_value(x, t) + policy_b.compute_value(x, t)
        logp_mesher = get_func_mesher(opt, ts, 200, logp_fn)

        fig = plt.figure(figsize=(20, 4))
        for i, idx in enumerate(snap_indices):
            ax = fig.add_subplot(1, 5, i + 1)
            for p in get_obstacle_patches():
                ax.add_patch(p)
            pts = xs_f[:, idx].detach().cpu().numpy()
            ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color=colors[i], alpha=0.5)
            if logp_mesher is not None:
                cp = ax.contour(*logp_mesher(idx), levels=10, cmap="copper", linewidths=1, zorder=2)
                ax.clabel(cp, inline=True, fontsize=10)
            _setup_ax(ax, titles[i])
            if i == 0:
                ax.set_ylabel(rf'$d={opt.x_dim}$', fontsize=18)
        plt.tight_layout()
        os.makedirs(os.path.join(opt.eval_path, "pf-ode"), exist_ok=True)
        plt.savefig(os.path.join(opt.eval_path, "pf-ode", f"stage{stage}.pdf"), bbox_inches='tight')
        plt.close(fig)

    # Custom post-training trajectory plot (3D with sphere wireframes)

    @torch.no_grad()
    def plot_trajectory_custom(self, deepgsb, opt):
        """3D PF ODE trajectory with 2x2 subplot layout.

        Backward (Step 1) + Forward (Step 2) with:
          Main 3D view, xy-plane, xz-plane, yz-plane projections.
        """
        import os, numpy as np, matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d import Axes3D

        ts = deepgsb.mfg.ts
        n_steps = len(ts)
        dt = deepgsb.mfg.dt
        device = opt.device
        dim = opt.x_dim
        g = deepgsb.z_f.g
        n_particles = 15

        term_pos = self.get_terminal_positions(n_particles, device)
        n_particles = term_pos.shape[0]
        if term_pos.shape[1] < dim:
            pad = torch.zeros(n_particles, dim - term_pos.shape[1], device=device)
            term_pos = torch.cat([term_pos, pad], dim=1)

        def _drift_pf(x, t_k):
            u_f = deepgsb.z_f(x, t_k)
            u_b = deepgsb.z_b(x, t_k)
            return 0.5 * (u_f - u_b) * g

        # Step 1: Backward
        x = term_pos.clone()
        traj_bwd = torch.zeros(n_steps, n_particles, dim, device=device)
        for k in range(n_steps - 1, -1, -1):
            t_k = ts[k].expand(n_particles, 1).to(device)
            u = -_drift_pf(x, t_k)
            traj_bwd[k] = x
            if k > 0:
                x = x + u * dt
        init_pos = x.clone()

        # Step 2: Forward
        x = init_pos.clone()
        traj_fwd = torch.zeros(n_steps, n_particles, dim, device=device)
        for k in range(n_steps):
            t_k = ts[k].expand(n_particles, 1).to(device)
            traj_fwd[k] = x
            if k < n_steps - 1:
                u = _drift_pf(x, t_k)
                x = x + u * dt

        save_dir = os.path.join(opt.eval_path, "3D_trajectory")
        os.makedirs(save_dir, exist_ok=True)

        def _draw_spheres(ax, alpha=0.35, linewidth=1.0):
            for sc in self.sphere_centers:
                cx, cy, cz = sc.tolist()
                R = self.sphere_radius
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 20)
                UX, VX = np.meshgrid(u, v)
                sx = cx + R * np.cos(UX) * np.sin(VX)
                sy = cy + R * np.sin(UX) * np.sin(VX)
                sz = cz + R * np.cos(VX)
                ax.plot_wireframe(sx, sy, sz, color='lightgray', alpha=alpha, linewidth=linewidth)

        def _draw_cylinders(ax, alpha=0.20):
            for center, radius in self.obstacles:
                cx, cy = center[0].item(), center[1].item()
                z_vals = np.linspace(-6, 6, 30)
                theta = np.linspace(0, 2 * np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z_vals)
                x_grid = cx + radius * np.cos(theta_grid)
                y_grid = cy + radius * np.sin(theta_grid)
                ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=alpha, edgecolor='none')

        def _draw_trajectories(ax, traj_np):
            for i in range(n_particles):
                ax.plot(traj_np[:, i, 0], traj_np[:, i, 1], traj_np[:, i, 2],
                        'k-', alpha=0.15, lw=0.5)

        def _draw_red_triangles(ax):
            for sc in self.sphere_centers:
                cx, cy, cz = sc.tolist()
                ax.scatter(cx, cy, cz, marker='^', color='red', s=80, zorder=5)

        def _mark_start_end(ax, traj_np, step_label, col_x=0, col_y=1):
            for i in range(n_particles):
                ax.scatter(traj_np[0, i, col_x], traj_np[0, i, col_y],
                           color='green', s=30, zorder=6, edgecolors='white',
                           linewidth=0.3, label='start' if i == 0 else '')
            ax.scatter(traj_np[-1, :, col_x], traj_np[-1, :, col_y],
                       color='#CE93D8', s=60, marker='*', zorder=3, label='terminal')

        def _build_2x2_figure(traj_np, step_label):
            fig = plt.figure(figsize=(18, 12))

            # Main 3D view (top-left)
            ax1 = fig.add_subplot(2, 2, 1, projection='3d')
            _draw_cylinders(ax1, alpha=0.20)
            _draw_spheres(ax1, alpha=0.35, linewidth=1.0)
            _draw_red_triangles(ax1)
            _draw_trajectories(ax1, traj_np)
            ax1.scatter(traj_np[0, :, 0], traj_np[0, :, 1], traj_np[0, :, 2],
                       color='green', s=30, zorder=6, marker='o',
                       edgecolors='white', linewidth=0.5, label='start')
            ax1.scatter(traj_np[-1, :, 0], traj_np[-1, :, 1], traj_np[-1, :, 2],
                       color='#CE93D8', s=60, marker='*', zorder=3, label='terminal')
            ax1.set_xlim(-8, 10); ax1.set_ylim(-6, 6); ax1.set_zlim(-6, 6)
            ax1.set_xlabel('$x_1$', fontsize=18); ax1.set_ylabel('$x_2$', fontsize=18); ax1.set_zlabel('$x_3$', fontsize=18)
            ax1.set_title(f"d={opt.x_dim}", fontsize=20)
            ax1.tick_params(axis='both', labelsize=14)
            ax1.view_init(elev=25, azim=-45)
            ax1.legend(fontsize=14, loc='upper right')

            # xy-plane projection (bottom-left), cols (0,1)
            ax2 = fig.add_subplot(2, 2, 3)
            for i in range(n_particles):
                ax2.plot(traj_np[:, i, 0], traj_np[:, i, 1], 'k-', alpha=0.2, lw=0.5)
            for center, radius in self.obstacles:
                cx, cy = center[0].item(), center[1].item()
                circle = plt.Circle((cx, cy), radius, facecolor='darkgray', edgecolor='black', alpha=0.3)
                ax2.add_patch(circle)
            for sc in self.sphere_centers:
                cx, cy = sc[0].item(), sc[1].item()
                ax2.add_patch(plt.Circle((cx, cy), self.sphere_radius, fill=False, color='gray', lw=1.0, alpha=0.5))
                ax2.scatter(cx, cy, marker='^', color='red', s=60, zorder=5)
            _mark_start_end(ax2, traj_np, step_label, col_x=0, col_y=1)
            ax2.set_xlim(-8, 10); ax2.set_ylim(-6, 6)
            ax2.set_aspect('equal')
            ax2.set_xlabel('$x_1$', fontsize=18); ax2.set_ylabel('$x_2$', fontsize=18)
            ax2.tick_params(axis='both', labelsize=14)
            ax2.set_title("$x_1$-$x_2$", fontsize=20)
            ax2.grid(True, alpha=0.3)

            # yz-plane projection (top-right), cols (1,2)
            ax3 = fig.add_subplot(2, 2, 2)
            for i in range(n_particles):
                ax3.plot(traj_np[:, i, 1], traj_np[:, i, 2], 'k-', alpha=0.2, lw=0.5)
            for sc in self.sphere_centers:
                cy, cz = sc[1].item(), sc[2].item()
                ax3.add_patch(plt.Circle((cy, cz), self.sphere_radius, fill=False, color='gray', lw=1.0, alpha=0.5))
                ax3.scatter(cy, cz, marker='^', color='red', s=60, zorder=5)
            _mark_start_end(ax3, traj_np, step_label, col_x=1, col_y=2)
            ax3.set_xlim(-6, 6); ax3.set_ylim(-6, 6)
            ax3.set_aspect('equal')
            ax3.set_xlabel('$x_2$', fontsize=18); ax3.set_ylabel('$x_3$', fontsize=18)
            ax3.tick_params(axis='both', labelsize=14)
            ax3.set_title("$x_2$-$x_3$", fontsize=20)
            ax3.grid(True, alpha=0.3)

            # xz-plane projection (bottom-right), cols (0,2)
            ax4 = fig.add_subplot(2, 2, 4)
            for i in range(n_particles):
                ax4.plot(traj_np[:, i, 0], traj_np[:, i, 2], 'k-', alpha=0.2, lw=0.5)
            for sc in self.sphere_centers:
                cx, cz = sc[0].item(), sc[2].item()
                ax4.add_patch(plt.Circle((cx, cz), self.sphere_radius, fill=False, color='gray', lw=1.0, alpha=0.5))
                ax4.scatter(cx, cz, marker='^', color='red', s=60, zorder=5)
            _mark_start_end(ax4, traj_np, step_label, col_x=0, col_y=2)
            ax4.set_xlim(-8, 10); ax4.set_ylim(-6, 6)
            ax4.set_aspect('equal')
            ax4.set_xlabel('$x_1$', fontsize=18); ax4.set_ylabel('$x_3$', fontsize=18)
            ax4.tick_params(axis='both', labelsize=14)
            ax4.set_title("$x_1$-$x_3$", fontsize=20)
            ax4.grid(True, alpha=0.3)

            plt.suptitle(f"{step_label}\n{self.name}", fontsize=18, y=0.98)
            plt.subplots_adjust(top=0.92, wspace=0.25, hspace=0.25)
            return fig

        # Step 1: Backward
        traj_bwd_np = traj_bwd.cpu().numpy()
        fig = _build_2x2_figure(traj_bwd_np, "Step 1: Backward PF-ODE")
        plt.savefig(os.path.join(save_dir, f"{self.name}_d{self.dim}_trajectory_Step1.pdf"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Step 2: Forward
        traj_fwd_np = traj_fwd.cpu().numpy()
        fig = _build_2x2_figure(traj_fwd_np, "Step 2: Forward PF-ODE")
        plt.savefig(os.path.join(save_dir, f"{self.name}_d{self.dim}_trajectory_Step2.pdf"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        log.info("3D trajectory (2x2 layout) saved to %s", save_dir)

    # Terminal positions for trajectory plotting (15 specific points)

    @staticmethod
    def get_terminal_positions(n_particles: int, device: str):
        """15 specific terminal points: 4 + 5 + 6 from the 3 sphere groups.

        Returns a tensor of shape (15, 3).
        """
        R = 1.5
        c1 = torch.tensor([6.0, 2.5, 0.0])
        c2 = torch.tensor([6.0, -2.5, 0.0])
        c3 = torch.tensor([7.0, 0.0, 2.5])

        pts = []

        # Group 1 (Sphere 1): 4 equatorial points (z=0 plane)
        for theta in [0, math.pi / 2, math.pi, 3 * math.pi / 2]:
            p = c1 + R * torch.tensor([math.cos(theta), math.sin(theta), 0.0])
            pts.append(p)

        # Group 2 (Sphere 2): N pole, S pole, 3 equatorial
        pts.append(c2 + torch.tensor([0.0, 0.0, R]))
        pts.append(c2 + torch.tensor([0.0, 0.0, -R]))
        for theta in [0, 2 * math.pi / 3, 4 * math.pi / 3]:
            p = c2 + R * torch.tensor([math.cos(theta), math.sin(theta), 0.0])
            pts.append(p)

        # Group 3 (Sphere 3): 3 at 30N + 3 at 30S
        for lat in [math.pi / 6, -math.pi / 6]:
            z_off = R * math.sin(lat)
            r_lat = R * math.cos(lat)
            for theta in [0, 2 * math.pi / 3, 4 * math.pi / 3]:
                p = c3 + r_lat * torch.tensor([math.cos(theta), math.sin(theta), 0.0]) + torch.tensor([0.0, 0.0, z_off])
                pts.append(p)

        result = torch.stack(pts).to(device)
        return result[:n_particles] if n_particles < len(pts) else result

    # Constraints (p0: mixture of Gaussians, pT: mixture of spheres)

    @staticmethod
    def build_constraint(batch_size, device, x_dim=None):
        import torch.distributions as td
        from mfg.constraint import Sampler, ProblemDists

        d = x_dim or 3

        # p0: two Gaussians at z=-2 (x1=-6)
        mu1 = torch.zeros(d, device=device)
        mu1[0] = -6.0; mu1[1] = 1.5; mu1[2] = -2.0
        mu2 = torch.zeros(d, device=device)
        mu2[0] = -6.0; mu2[1] = -1.5; mu2[2] = -2.0
        cov = 0.1 * torch.eye(d, device=device)

        mix = td.Categorical(torch.ones(2, device=device))
        comp = td.Independent(
            td.Normal(
                torch.stack([mu1, mu2]),
                torch.stack([cov.diag().sqrt(), cov.diag().sqrt()]),
            ), 1)
        p0_dist = td.MixtureSameFamily(mix, comp)
        p0 = Sampler(p0_dist, batch_size, device)

        # pT: 3 spherical shells at x > 4
        sphere_params = [
            (6.0, 2.5, 0.0),
            (6.0, -2.5, 0.0),
            (7.0, 0.0, 2.5),
        ]
        spheres = [
            SphericalShellDistribution(R=1.5, alpha=8.0, center=c, device=device)
            for c in sphere_params
        ]
        mixT = MixtureSphereShells(spheres, device=device)
        if d == 3:
            pT_dist = mixT
        else:
            from .Gauss2Ring_Obs1 import PaddedDistribution
            pT_dist = PaddedDistribution(mixT, full_dim=d, extra_std=0.1, device=device)

        pT = Sampler(pT_dist, batch_size, device)

        return ProblemDists(p0, pT)

    @staticmethod
    def get_default_config(sb_param='actor-critic'):
        config = edict()
        config.problem_name = 'MixGauss2SphereShell_Obs3'
        config.x_dim = 3
        config.seed = 42
        config.t0 = 0.0
        config.T = 1.0
        config.interval = 100
        config.diffusion_std = 1.0
        config.num_itr = 250
        config.train_bs_x = 128
        config.rb_bs_x = 128
        config.samp_bs = 5000
        config.snapshot_freq = 1
        config.optimizer = 'AdamW'
        config.l2_norm = 0.0
        config.lr_step = 1000
        config.noise_type = 'gaussian'
        config.ema = 0.99
        config.ckpt_freq = 0
        config.buffer_size = 20000
        config.use_rb_loss = False
        config.weighted_loss = False
        config.lambda_L = 0.05
        config.lambda_E = 0.05
        config.lambda_P = 10.0

        if sb_param == 'actor-critic':
            config.sb_param = 'actor-critic'
            config.policy_net = 'toy'
            config.lr = 5e-4
            config.lr_y = 1e-3
            config.lr_gamma = 0.999
            config.num_stage = 20
            config.multistep_td = True
            config.samp_method = 'gauss'
        elif sb_param == 'critic':
            config.sb_param = 'critic'
            config.policy_net = 'toy'
            config.lr = 5e-4
            config.lr_gamma = 0.999
            config.num_stage = 20
            config.multistep_td = True
            config.samp_method = 'gauss'
        else:
            raise ValueError(f"Unknown sb_param: {sb_param}")
        return config
