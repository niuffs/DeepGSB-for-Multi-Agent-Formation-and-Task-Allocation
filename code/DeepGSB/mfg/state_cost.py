from typing import Callable
import torch

StateCostFn = Callable[[torch.Tensor], torch.Tensor]
MFCostFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def build_state_cost_fn(problem_name: str, lambda_E: float = 1.0, lambda_P: float = 500.0,
                         return_obs_cost_fn: bool = False, dim: int = 2):
    obstacle_cost_fn = get_obstacle_cost_fn(problem_name, lambda_P)
    mf_cost_fn = get_mf_cost_fn(lambda_E, dim)

    def state_cost_fn(xs, ts, logp, xs_all):
        obstacle_cost = obstacle_cost_fn(xs)
        mf_cost = mf_cost_fn(xs, logp, xs_all)
        return obstacle_cost, mf_cost

    return state_cost_fn, obstacle_cost_fn if return_obs_cost_fn else state_cost_fn


def get_obstacle_cost_fn(problem_name: str, weight: float = 500.0) -> StateCostFn:
    """Map problem name to obstacle cost function.

    All returned functions compute indicator(dist < radius) * weight,
    checking only the first 2 spatial dimensions.
    """
    registry = {
        'Gauss2Gauss_Obs3': lambda xs: obstacle_cost_fn_obs3(xs, weight),
        'Gauss2Rings_Obs1': lambda xs: obstacle_cost_fn_single(xs, weight, radius=1.5),
        'Gauss2Mix_Obs1':   lambda xs: obstacle_cost_fn_single(xs, weight, radius=1.0),
        'Gauss2Ring_Obs1': lambda xs: obstacle_cost_fn_ring_obs1(xs, weight),
        'MixGauss2SphereShell_Obs3': lambda xs: obstacle_cost_fn_mixgauss_shell(xs, weight),
    }
    for key, fn in registry.items():
        if key.lower() == problem_name.lower():
            return fn
    return _zero_obstacle_cost


def get_mf_cost_fn(lambda_E: float = 1.0, dim: int = 2) -> MFCostFn:
    """Mean-field cost: lambda_E * log(rho) / d.

    log(rho) scales linearly with dimension d. Normalising by d keeps the
    congestion cost consistent regardless of dimension.
    """
    d = max(dim, 1)
    def mf_cost_fn(x, logp, xs_all):
        return lambda_E * logp / d
    return mf_cost_fn


# Obstacle cost functions -- indicator * lambda_P, 2D only

def _zero_obstacle_cost(xs: torch.Tensor) -> torch.Tensor:
    """Fallback: no obstacle."""
    return torch.zeros(xs.shape[:-1], device=xs.device)


def _xy(xs: torch.Tensor) -> torch.Tensor:
    """Extract first 2 spatial dims, flattening leading dims."""
    return xs.reshape(-1, xs.shape[-1])[:, :2]


def obstacle_cost_fn_obs3(xs: torch.Tensor, weight: float = 500.0) -> torch.Tensor:
    """Gauss2Gauss_Obs3: 3 obstacles at (2,2)/r=1, (-3,1)/r=0.5, (0,-2)/r=1.5.

    Returns (sum_i indicator(||xy - c_i|| < r_i)) * weight.
    """
    xy = _xy(xs)
    device = xs.device
    centers = [
        torch.tensor([2., 2.], device=device),
        torch.tensor([-3., 1.], device=device),
        torch.tensor([0., -2.], device=device),
    ]
    radii = [1.0, 0.5, 1.5]
    cost = torch.zeros(xy.shape[0], device=device)
    for c, r in zip(centers, radii):
        dist = torch.norm(xy - c, dim=-1)
        cost = cost + (dist < r).float()
    return (cost * weight).reshape(xs.shape[:-1])


def obstacle_cost_fn_single(xs: torch.Tensor, weight: float,
                            radius: float) -> torch.Tensor:
    """Single circular obstacle at origin, indicator * weight, first 2 dims only."""
    xy = _xy(xs)
    dist = torch.norm(xy, dim=-1)
    return ((dist < radius).float() * weight).reshape(xs.shape[:-1])


def obstacle_cost_fn_ring_obs1(xs: torch.Tensor, weight: float = 500.0) -> torch.Tensor:
    """Gauss2Ring_Obs1: 1 obstacle at (2, 0)/r=1.5."""
    xy = _xy(xs)
    dist = torch.norm(xy - torch.tensor([2.0, 0.0], device=xs.device), dim=-1)
    return ((dist < 1.5).float() * weight).reshape(xs.shape[:-1])


def obstacle_cost_fn_mixgauss_shell(xs: torch.Tensor, weight: float = 500.0) -> torch.Tensor:
    """MixGauss2SphereShell_Obs3: 3 obstacles at (-2,1.5)/r0.8, (-2,-1.5)/r0.8, (2,0)/r0.8."""
    xy = _xy(xs)
    device = xs.device
    centers = [
        torch.tensor([-2.0, 1.5], device=device),
        torch.tensor([-2.0, -1.5], device=device),
        torch.tensor([2.0, 0.0], device=device),
    ]
    cost = torch.zeros(xy.shape[0], device=device)
    for c in centers:
        dist = torch.norm(xy - c, dim=-1)
        cost = cost + (dist < 0.8).float()
    return (cost * weight).reshape(xs.shape[:-1])
