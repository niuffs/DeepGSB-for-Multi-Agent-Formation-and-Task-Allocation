import torch


def msigma(x):
    """Activation: log(2*cosh(x)) = |x| + log(1 + exp(-2|x|))."""
    ax = torch.abs(x)
    return ax + torch.log1p(torch.exp(-2 * ax))


def mdsigma(x):
    """First derivative: tanh(x)."""
    return torch.tanh(x)


def md2sigma(x):
    """Second derivative: 1 - tanh^2(x)."""
    return 1 - torch.tanh(x).pow(2)


def linInter1D(tk, T, theta):
    """Linear interpolation of time-dependent parameters.

    Args:
        tk: current time in [0, T]
        T: final time
        theta: (nTime, *shape) tensor

    Returns:
        interpolated value with shape (*shape)
    """
    nTime = theta.shape[0]
    if nTime == 1:
        return theta[0]
    H = T / (nTime - 1)
    tk_c = max(0.0, min(float(tk), T))
    idx = max(0, min(int(tk_c / H), nTime - 2))
    w = max(0.0, min((H * (idx + 1) - tk_c) / H, 1.0))
    return w * theta[idx] + (1 - w) * theta[idx + 1]
