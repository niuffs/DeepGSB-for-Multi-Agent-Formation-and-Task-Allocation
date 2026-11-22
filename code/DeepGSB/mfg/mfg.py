import logging
from typing import Callable, Optional, Tuple

import torch

from options import Options
from envs import get_env_class

from .constraint import Sampler, build_constraint
from .plotting import snapshot
from .sde import SampledTraj, build_sde
from .state_cost import build_state_cost_fn

log = logging.getLogger(__file__)

Bound = Tuple[torch.Tensor, torch.Tensor, Sampler]
InitTermBound = Tuple[torch.Tensor, torch.Tensor, Sampler, torch.Tensor, torch.Tensor, Sampler]

# Per-problem feature toggle lists
logp_list = []
xs_all_list = []
mf_drift_list = []
state_dependent_drift_list = []


def get_bound_index(direction: str, bound: str) -> int:
    assert direction in ["forward", "backward"]
    assert bound in ["init", "term"]
    if direction == "forward" and bound == "init":
        return 0
    elif direction == "forward" and bound == "term":
        return -1
    elif direction == "backward" and bound == "init":
        return -1
    elif direction == "backward" and bound == "term":
        return 0


class MFGPolicy(torch.nn.Module):
    def __init__(self, direction, dyn):
        super(MFGPolicy, self).__init__()
        self.direction = direction
        self.dyn = dyn


class MFG:
    def __init__(self, opt: Options):
        self.opt = opt
        self.problem_name = opt.problem_name

        self.ts = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device)

        constraints = build_constraint(opt.problem_name, opt.samp_bs, opt.device, x_dim=opt.x_dim)
        self.p0, self.pT = constraints.p0, constraints.pT
        self.constraints = sorted(constraints.constraints, key=lambda x: x[0])

        self.state_cost_fn, self.obstacle_cost_fn = build_state_cost_fn(
            opt.problem_name, lambda_E=opt.lambda_E, lambda_P=opt.lambda_P,
            dim=opt.x_dim,
        )
        self.pbound = [self.p0, self.pT]
        self.sde = build_sde(opt, self.p0, self.pT)

    @property
    def dt(self) -> float:
        return self.sde.dt

    def f(self, x: torch.Tensor, t: torch.Tensor, direction: str) -> torch.Tensor:
        return self.sde.f(x, t, direction)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        return self.sde.g(t)

    def uses_logp(self) -> bool:
        return self.problem_name in logp_list

    def uses_xs_all(self) -> bool:
        return self.problem_name in xs_all_list

    def uses_mf_drift(self) -> bool:
        return self.problem_name in mf_drift_list

    def uses_state_dependent_drift(self) -> bool:
        return self.problem_name in state_dependent_drift_list

    @property
    def has_midpoint_constraint(self) -> bool:
        """Whether this problem has intermediate distribution constraints."""
        return len(self.constraints) > 0

    def get_constraint_bounds(self, xs: torch.Tensor):
        """Get trajectory states and target distributions at all constraint times.

        Returns a list of (t, x_at_t, Sampler) tuples, one per constraint,
        in increasing time order.
        """
        b, T, nx = xs.shape
        assert len(self.ts) == T and nx == self.opt.x_dim
        results = []
        for t, p_target in self.constraints:
            t_idx = (self.ts - t).abs().argmin().item()
            results.append((self.ts[t_idx], xs[:, t_idx], p_target))
        return results

    def initialize_mf_drift(self, policy: MFGPolicy) -> None:
        self.sde.initialize_mf_drift(self.ts, policy)

    def sample_traj(self, policy: MFGPolicy, **kwargs) -> SampledTraj:
        return self.sde.sample_traj(self.ts, policy, **kwargs)

    def compute_state_cost(self, xs, ts, xs_all, log_p):
        """Compute F = (lambda_P * Q + lambda_E * log(rho) / d) / (lambda_L * sigma^2)."""
        s_cost, mf_cost = self.state_cost_fn(xs, ts, log_p, xs_all)
        scale = self.opt.lambda_L * self.sde.std ** 2
        return (s_cost + mf_cost) / scale

    def get_bound(self, xs: torch.Tensor, direction: str, bound: str) -> Bound:
        t_index = get_bound_index(direction, bound)
        b, T, nx = xs.shape
        assert len(self.ts) == T and nx == self.opt.x_dim
        return self.ts[t_index], xs[:, t_index, ...], self.pbound[t_index]

    def get_init_term_bound(self, xs: torch.Tensor, direction: str) -> InitTermBound:
        return (*self.get_bound(xs, direction, "init"), *self.get_bound(xs, direction, "term"))

    def save_snapshot(self, policy_f, policy_b, stage: int) -> None:
        plot_pf_ode = True
        snapshot(self.opt, policy_f, policy_b, self, stage, plot_pf_ode=plot_pf_ode)
