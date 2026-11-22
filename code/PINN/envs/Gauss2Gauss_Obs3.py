"""Gauss2Gauss_Obs3 scenario for PINN (single-agent, no logp / congestion).

3 obstacles, Gaussian start to Gaussian target at (4,4).
Obstacle functions defined only in solver/evaluator, not here.
"""
import math
import torch


class Gauss2Gauss_Obs3:
    """3 obstacles, Gaussian start to Gaussian target at (4,4)."""

    def __init__(self, device='cpu', d=2):
        self.dim = d
        self.nu = 1.0
        self.TT = 1.0
        self.lambda_L = 0.05
        self.lambda_E = 0.0                     # No congestion term
        self.lambda_P = 10.0
        self.device = device
        self.name = "Gauss2Gauss_Obs3"

        # Obstacles definition (used by evaluator, not training)
        self.obstacles = [
            (torch.tensor((2., 2.), device=device),   1.0),
            (torch.tensor((-3., 1.), device=device),  0.5),
            (torch.tensor((0., -2.), device=device),  1.5),
        ]
        self.obstacle_dicts = [
            {"center": c, "radius": r} for c, r in self.obstacles
        ]

        # Start / Target
        self.start_point = torch.zeros(d, device=device)
        self.start_point[0] = -3.0
        self.start_point[1] = -2.0
        self.init_cov = 0.1 * torch.eye(d, device=device)

        self.end_point = torch.zeros(d, device=device)
        self.end_point[0] = 4.0
        self.end_point[1] = 4.0

    def sample_rho0(self, num_samples, var_scale=0.1):
        mu = self.start_point.unsqueeze(0)
        return math.sqrt(var_scale) * torch.randn(num_samples, self.dim, device=self.device) + mu

    def terminal_cost(self, x, delta=1.0):
        """Huber terminal cost to end_point=(4,4,0,...)."""
        r = x - self.end_point
        dist = torch.norm(r, dim=1)
        huber = torch.where(
            dist <= delta,
            dist ** 2 / (2.0 * delta) + delta / 2.0,
            dist,
        )
        return huber

    def rho0_pdf(self, x):
        diff = x - self.start_point
        cov_diag = self.init_cov.diag()
        return -0.5 * torch.sum(diff ** 2 / cov_diag, dim=1) \
               - 0.5 * self.dim * math.log(2 * math.pi) \
               - 0.5 * torch.log(cov_diag).sum()

    @property
    def info_dict(self):
        return {
            'env_name': self.name, 'dim': self.dim, 'nu': self.nu,
            'TT': self.TT, 'lambda_L': self.lambda_L,
            'lambda_E': self.lambda_E, 'lambda_P': self.lambda_P,
        }
