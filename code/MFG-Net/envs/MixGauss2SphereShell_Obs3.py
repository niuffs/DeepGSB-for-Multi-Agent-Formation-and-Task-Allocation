"""
MFG-Net environment: MixGauss2SphereShell_Obs3.

Terminal cost uses KL divergence (negative log-probability of 3-spherical-shell target).

Scenario: Mixed Gaussian start at z=-2 -> 3 cylindrical obstacles -> 3 spherical shells at x>4.
For d>3, extra dimensions are zero-padded with independent Gaussian noise.
"""
import math
import numpy as np
import torch
import torch.distributions as td


# ------------------------------------------------------------
#  PaddedDistribution  --  extend a k-dim distribution to d-dim
# ------------------------------------------------------------

class PaddedDistribution(td.Distribution):
    """Extend a ``k``-dim distribution to ``full_dim`` by appending
    independent ``Normal(0, extra_std^2)`` dimensions."""
    arg_constraints = {}

    def __init__(self, base_dist, full_dim, extra_std=0.1,
                 device='cpu', validate_args=None):
        super().__init__(
            batch_shape=base_dist.batch_shape,
            event_shape=torch.Size([full_dim]),
            validate_args=validate_args,
        )
        self.base = base_dist
        self.full_dim = full_dim
        k = base_dist.event_shape[0] if base_dist.event_shape else 1
        self.base_dim = k
        self.extra_std = extra_std
        self.device = device
        self._extra_logZ = 0.5 * (full_dim - k) * math.log(2 * math.pi * extra_std ** 2)

    def sample(self, sample_shape=torch.Size()):
        base = self.base.sample(sample_shape)
        n_extra = self.full_dim - self.base_dim
        extra = torch.randn(*sample_shape, n_extra, device=self.device) * self.extra_std
        return torch.cat([base, extra], dim=-1)

    def log_prob(self, value):
        base_lp = self.base.log_prob(value[..., :self.base_dim])
        extra = value[..., self.base_dim:]
        extra_lp = -0.5 * (extra / self.extra_std).pow(2).sum(-1) - self._extra_logZ
        return base_lp + extra_lp


# ------------------------------------------------------------
#  SphericalShellDistribution -- smooth 3D shell
# ------------------------------------------------------------

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
        # 3D spherical shell: Z = 2*pi*R*sqrt(pi/alpha) = 2*pi^(3/2)*R/sqrt(alpha) (large-alpha Laplace approx)
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


# ------------------------------------------------------------
#  MixtureSphereShells -- mixture of several 3D shells
# ------------------------------------------------------------

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


# ------------------------------------------------------------
#  Environment
# ------------------------------------------------------------

class MixGauss2SphereShell_Obs3:
    """3D: mixed Gaussian start at z=-2 -> 3 cylindrical obstacles -> 3 spherical shells.
    For d>3, extra dims are zero-padded with independent Gaussian noise."""

    def __init__(self, device='cpu', d=3):
        self.dim = d
        self.nu = 0.1
        self.TT = 1.0
        self.lambda_L = 0.05
        self.lambda_E = 0.05
        self.lambda_P = 500.0
        self.device = device
        self.name = "MixGauss2SphereShell_Obs3"

        # 3 cylindrical obstacles (xy-plane)
        self.obstacles = [
            (torch.tensor((-2.0, 4.0), device=device), 1.0),
            (torch.tensor((-2.0, -4.0), device=device), 1.0),
            (torch.tensor((2.0, 0.0), device=device), 1.2),
        ]

        # Two Gaussian centres at z=-2
        self.g1_center = torch.zeros(d, device=device)
        self.g1_center[0] = -6.0; self.g1_center[1] = 1.5; self.g1_center[2] = -2.0

        self.g2_center = torch.zeros(d, device=device)
        self.g2_center[0] = -6.0; self.g2_center[1] = -1.5; self.g2_center[2] = -2.0

        # Start (approx center -- kept for compatibility)
        self.start_point = torch.zeros(d, device=device)
        self.start_point[0] = -6.0; self.start_point[2] = -2.0
        self.init_cov = 0.1 * torch.eye(d, device=device)

        # End point (reference)
        self.end_point = torch.zeros(d, device=device)
        self.end_point[0] = 6.0

        # 3 spherical shell targets (two at x=6, one at x=7)
        self.sphere_centers_3d = [
            (6.0, 2.5, 0.0),
            (6.0, -2.5, 0.0),
            (7.0, 0.0, 2.5),
        ]
        self._build_target_dist()

    def _build_target_dist(self):
        spheres = [
            SphericalShellDistribution(R=1.5, alpha=8.0, center=c, device=self.device)
            for c in self.sphere_centers_3d
        ]
        mix = MixtureSphereShells(spheres, device=self.device)
        if self.dim == 3:
            self.target_dist = mix
        else:
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

    def rho0_log_pdf(self, x):
        """Log-pdf of two-Gaussian mixture (matches sample_rho0)."""
        def _log_gaussian(center):
            diff = x - center
            return -0.5 * torch.sum(diff ** 2 / self.init_cov.diag(), dim=1) \
                   - 0.5 * self.dim * math.log(2 * math.pi) \
                   - 0.5 * torch.log(self.init_cov.diag()).sum()
        log_p1 = _log_gaussian(self.g1_center)
        log_p2 = _log_gaussian(self.g2_center)
        return torch.logaddexp(log_p1 - math.log(2), log_p2 - math.log(2))

    def rho0_pdf(self, x):
        return torch.exp(self.rho0_log_pdf(x))

    def obstacle_potential(self, x):
        """Linear barrier on first 2 dims: max(0, r - ||x-c||)."""
        out = torch.zeros(x.shape[0], device=self.device)
        x_2d = x[:, :2] if x.shape[1] > 2 else x
        for center, radius in self.obstacles:
            dist = torch.norm(x_2d - center, dim=1)
            out = out + torch.clamp(radius - dist, min=0)
        return out

    @property
    def plot_xlim(self):
        return (-8, 10)

    @property
    def plot_ylim(self):
        return (-6, 6)

    def terminal_cost(self, x):
        """KL-based terminal cost: -log p_T(x)."""
        return -self.target_dist.log_prob(x)

    @property
    def info_dict(self):
        return {
            'env_name': self.name, 'dim': self.dim, 'nu': self.nu,
            'TT': self.TT, 'lambda_L': self.lambda_L,
            'lambda_E': self.lambda_E, 'lambda_P': self.lambda_P,
        }
