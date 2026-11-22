"""
PINN solver: solves the Hamilton-Jacobi-Bellman equation via
physics-informed neural networks for single-agent optimal control.

The value function phi(x,t) is learned by minimizing the HJB PDE residual:

    -phi_t - (sigma^2/2) * div(phi_x) + 1/(2*lambda_L) * ||phi_x||^2 - f_obs = 0

The optimal control is then u(x,t) = -grad_x(phi) / lambda_L.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def compute_divergence(X, Y):
    """Compute divergence of vector field Y w.r.t. input X.

    Args:
        X: Input tensor (batch, d), requires gradients
        Y: Output vector field (batch, d)

    Returns:
        div: Scalar divergence per batch element (batch,)
    """
    div = 0.0
    for i in range(Y.shape[1]):
        div += torch.autograd.grad(
            Y[:, i], X,
            grad_outputs=torch.ones_like(Y[:, i]),
            retain_graph=True, create_graph=True,
        )[0][:, i]
    return div


class PINNSolver:
    """PINN solver for optimal control with obstacle avoidance.

    Learns the value function phi(x,t) by minimizing the HJB PDE residual
    on randomly sampled collocation points.

    Args:
        lambda_L: Running cost weight
        obstacles: List of dicts with 'center' and 'radius'
        endpoint_B: Target endpoint tensor (d,)
        batch_size: Number of collocation points per iteration
        model: MLP network for phi(x,t)
        optimizer: Optimizer class
        sigma: Noise intensity (sigma^2/2 appears in PDE)
        T: Time horizon
        lr: Learning rate
        device: Device
        save_path: Path to save plots
    """

    def __init__(
        self,
        lambda_L,
        obstacles,
        endpoint_B,
        batch_size,
        model,
        optimizer=optim.Adam,
        sigma=1.095,
        T=1.0,
        lr=1e-3,
        device="cpu",
        save_path=None,
        lambda_P=10.0,
    ):
        self.lambda_L = lambda_L
        self.lambda_P = lambda_P
        self.obstacles = obstacles
        self.endpoint_B = endpoint_B
        self.N = batch_size
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr)
        self.sigma = sigma
        self.T = T
        self.device = device
        self.save_path = save_path
        self.dim = endpoint_B.shape[0] if endpoint_B is not None else 2
        self.name = ""

        # Logging
        self.step_log = []
        self.loss_log = []

    # ------------------------------------------------------------------
    #  Obstacle potential
    # ------------------------------------------------------------------

    def f_obs(self, X):
        """Obstacle penalty potential.

        Args:
            X: Spatial coordinates (batch, d)

        Returns:
            Obstacle penalty (batch, 1)
        """
        f = torch.zeros(X.shape[0], 1, device=X.device)
        # Obstacles only constrain the first 2 dims; extra dims are zero-padded
        X_2d = X[:, :2] if X.shape[1] > 2 else X
        for obs in self.obstacles:
            center = obs["center"]
            radius = obs["radius"]
            # Use only first 2 dims of center for distance computation
            center_2d = center[:2] if center.shape[0] > 2 else center
            distance = torch.norm(X_2d - center_2d, dim=1, keepdim=True)
            f += self.lambda_P * torch.relu(radius - distance)
        return f

    # ------------------------------------------------------------------
    #  PDE loss (HJB residual)
    # ------------------------------------------------------------------

    def loss(self):
        """HJB PDE residual loss.

        Returns:
            Scalar PDE loss
        """
        self.X_sam.requires_grad_(True)
        self.t_sam.requires_grad_(True)

        phi = self.model(torch.cat([self.X_sam, self.t_sam], dim=1))

        phi_t = torch.autograd.grad(
            phi, self.t_sam,
            grad_outputs=torch.ones_like(phi), create_graph=True,
        )[0]

        phi_x = torch.autograd.grad(
            phi, self.X_sam,
            grad_outputs=torch.ones_like(phi), create_graph=True,
        )[0]

        # Hamiltonian: H = 1/(2*lambda_L) * ||phi_x||^2
        H = 0.5 * torch.sum(phi_x ** 2, dim=1)[:, None] / self.lambda_L

        # PDE residual: -phi_t - (sigma^2/2) * div(phi_x) + H - f_obs = 0
        residual = (
            -phi_t
            - 0.5 * self.sigma ** 2 * compute_divergence(self.X_sam, phi_x)[:, None]
            + H
            - self.f_obs(self.X_sam)
        )
        return torch.mean(residual ** 2)

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def train(self, iterations, log_freq=100):
        """Train the PINN by minimizing the HJB PDE residual.

        Args:
            iterations: Number of Adam iterations
            log_freq: Logging frequency
        """
        train_start = time.time()
        for epoch in range(iterations):
            # Resample collocation points every 10 iterations
            if epoch % 10 == 0:
                # First 2 dims in [-4, 6]; extra dims uniform in [-3, 3] to prevent z-drift
                d = getattr(self, 'dim', 2)
                left = torch.tensor((-4, -4), device=self.device)
                right = torch.tensor((6, 6), device=self.device)
                X_2d = left + (right - left) * torch.rand(self.N, 2, device=self.device)
                if d > 2:
                    extra = 4 * torch.rand(self.N, d - 2, device=self.device) - 2
                    self.X_sam = torch.cat([X_2d, extra], dim=1)
                else:
                    self.X_sam = X_2d
                self.t_sam = torch.rand(self.N, 1, device=self.device)

            self.model.train()
            self.optimizer.zero_grad()
            loss = self.loss()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % log_freq == 0:
                print(f"  Iteration {epoch + 1}, PDE Loss: {loss.item():.2e},"
                      f" elapsed={(time.time() - train_start) / 60.0:.1f} min")
                self.step_log.append(epoch)
                self.loss_log.append(loss.item())

        total_h = (time.time() - train_start) / 3600.0
        print(f"Training complete --- total time: {total_h:.2f} hours")

    # ------------------------------------------------------------------
    #  Control extraction
    # ------------------------------------------------------------------

    def get_control(self, x, t):
        """Compute the optimal control u = -grad_x(phi) / lambda_L.

        Args:
            x: Spatial coordinates (batch, d) — full d-dim
            t: Time values (batch, 1)

        Returns:
            u: Control vector (batch, d)
        """
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        phi = self.model(xt)
        grad_phi = torch.autograd.grad(
            phi, xt,
            grad_outputs=torch.ones_like(phi), create_graph=False,
        )[0]
        return -grad_phi[:, :-1] / self.lambda_L

    # ------------------------------------------------------------------
    #  Plotting
    # ------------------------------------------------------------------

    def plot_loss(self, save_path=None):
        """Plot PDE loss curve."""
        path = save_path or self.save_path
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.step_log, self.loss_log, label='PDE Loss', color='#2196F3', linewidth=1.5)
        ax.set_xlabel('Iteration', fontsize=16)
        ax.set_ylabel('Loss', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_yscale('symlog')
        ax.set_title('PINN Training Loss', fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(path, "training_loss.pdf"), dpi=150)
        plt.close()

    def plot_phi(self, save_path=None):
        """Plot heatmaps of the learned value function phi at 10 time snapshots."""
        path = save_path or self.save_path

        x = np.linspace(-4, 6, 50)
        y = np.linspace(-4, 6, 50)
        X, Y = np.meshgrid(x, y)
        grid_2d = np.stack([X.ravel(), Y.ravel()], axis=-1)

        # For high-dim: pad remaining dims with zeros
        d = 2
        if hasattr(self, 'dim') and self.dim > 2:
            d = self.dim
        grid = np.zeros((grid_2d.shape[0], d))
        grid[:, :2] = grid_2d

        times = np.linspace(0, 1, 10)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()

        for i, t in enumerate(times):
            inputs = torch.tensor(
                np.hstack([grid, np.full((grid.shape[0], 1), t)]),
                dtype=torch.float32, device=self.device,
            )
            with torch.no_grad():
                outputs = self.model(inputs).cpu().numpy()

            Z = outputs.reshape(50, 50)
            ax = axes[i]
            c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='rainbow')
            fig.colorbar(c, ax=ax, shrink=0.7, orientation='vertical')
            ax.set_title(f't = {t:.2f}', fontsize=16)
            ax.set_xlabel('$x_1$', fontsize=14)
            ax.set_ylabel('$x_2$', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.set_aspect('equal')

            for obs in self.obstacles:
                center = obs['center'].cpu().numpy()
                circle = plt.Circle(
                    center[:2], obs['radius'],
                    color='red', fill=False, linewidth=2,
                )
                ax.add_patch(circle)
            ax.scatter(
                self.endpoint_B[0].cpu(), self.endpoint_B[1].cpu(),
                marker='*', color='red', s=80,
            )

        fig.suptitle("Value function $\\phi(x,t)$", fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(path, "Value_function.pdf"), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_samples(self, startpoint_A, save_path=None):
        """Swarm trajectory from a given start point.

        Args:
            startpoint_A: Starting position tensor (2,) — only first 2 dims used
            save_path: Directory to save plot
        """
        path = save_path or self.save_path

        N = 100
        n_snap = 20
        n_steps = 100
        dt = self.T / n_steps
        d = getattr(self, 'dim', 2)

        # Initialize in d dimensions: first 2 dims from startpoint, rest ~ N(0, 0.01)
        particles_2d = np.random.multivariate_normal(
            startpoint_A.cpu().numpy(), 0.1 * np.eye(2), N,
        ).astype(np.float32)
        if d > 2:
            extra = 0.01 * np.random.randn(N, d - 2).astype(np.float32)
            particles = np.concatenate([particles_2d, extra], axis=1)
        else:
            particles = particles_2d

        # Store full trajectory (only first 2 dims for plotting)
        traj = np.zeros((n_steps + 1, N, 2))
        traj[0] = particles[:, :2]

        for k in range(n_steps):
            xt = torch.tensor(particles, dtype=torch.float32, device=self.device)
            t_k = torch.full((N, 1), k * dt, device=self.device)
            u = self.get_control(xt, t_k)
            particles = particles + dt * u.cpu().detach().numpy()
            traj[k + 1] = particles[:, :2]

        # Rainbow scatter snapshots style
        sw_idx = np.linspace(0, n_steps, n_snap).astype(int)
        sw_colors = plt.cm.rainbow(np.linspace(0, 1, n_snap))

        fig, ax = plt.subplots(figsize=(7, 6))
        # Obstacles: darkgray fill + black edge
        for obs in self.obstacles:
            cx, cy = obs['center'].cpu().numpy()
            radius = obs['radius']
            ax.add_patch(plt.Circle((cx, cy), radius,
                                    facecolor='darkgray', edgecolor='black', alpha=0.4, zorder=1))
        # Rainbow scatter snapshots, no lines, no legend
        for sidx, c in zip(sw_idx, sw_colors):
            ax.scatter(traj[sidx, :, 0], traj[sidx, :, 1],
                       s=6, color=c, alpha=0.6, zorder=2)
        ax.set_aspect('equal')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('$x_1$', fontsize=16)
        ax.set_ylabel('$x_2$', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        Xstr = f"({startpoint_A[0].item():.1f},{startpoint_A[1].item():.1f})"
        ax.set_title(f"$x_A$={Xstr}", fontsize=18)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"x_A={Xstr}.pdf"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved x_A={Xstr}.pdf")

    def plot_trajectory_3d(self, save_path=None):
        """3D trajectory plot for d>2 scenarios.

        Shows particle trajectories in (x1, x2, x3) space with obstacles as cylinders.
        """
        path = save_path or self.save_path
        d = getattr(self, 'dim', 2)
        if d <= 2:
            return

        N = 50
        n_steps = 100
        dt = self.T / n_steps

        particles_2d = np.random.multivariate_normal(
            np.array([-3.0, -2.0]), 0.1 * np.eye(2), N,
        ).astype(np.float32)
        extra = 0.01 * np.random.randn(N, d - 2).astype(np.float32)
        particles = np.concatenate([particles_2d, extra], axis=1)

        traj = np.zeros((n_steps + 1, N, 3))
        traj[0] = particles[:, :3]

        for k in range(n_steps):
            xt = torch.tensor(particles, dtype=torch.float32, device=self.device)
            t_k = torch.full((N, 1), k * dt, device=self.device)
            u = self.get_control(xt, t_k)
            particles = particles + dt * u.cpu().detach().numpy()
            traj[k + 1] = particles[:, :3]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Obstacles as 3D cylinders
        for obs in self.obstacles:
            c = obs['center'].cpu().numpy()
            z_vals = np.linspace(-4, 4, 20)
            theta = np.linspace(0, 2 * np.pi, 30)
            theta_grid, z_grid = np.meshgrid(theta, z_vals)
            x_grid = c[0] + obs['radius'] * np.cos(theta_grid)
            y_grid = c[1] + obs['radius'] * np.sin(theta_grid)
            ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.20, edgecolor='none')

        # Trajectories
        for i in range(N):
            ax.plot(traj[:, i, 0], traj[:, i, 1], traj[:, i, 2],
                    'k-', alpha=0.15, lw=0.5, zorder=1)

        # Start points (blue)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], traj[0, :, 2],
                   color='blue', s=30, alpha=0.8, zorder=3, label=r'$t=0$')
        # End points (red)
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1], traj[-1, :, 2],
                   color='red', s=30, alpha=0.8, zorder=3, label=r'$t=T$')

        ax.set_xlabel('$x_1$', fontsize=16)
        ax.set_ylabel('$x_2$', fontsize=16)
        ax.set_zlabel('$x_3$', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)
        ax.set_title(f'PINN for {self.name} (d={d})', fontsize=18)
        ax.legend(fontsize=14, loc='upper right')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.view_init(elev=25, azim=-45)
        plt.tight_layout()
        prefix = self.name if self.name else "trajectory"
        plt.savefig(os.path.join(path, f"{prefix}_d{d}_trajectory_3d.pdf"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved trajectory_3d_d{d}.pdf")
