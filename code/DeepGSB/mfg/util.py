import torch


def assert_zero_grads(network: torch.nn.Module) -> None:
    for param in network.parameters():
        if not (param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))):
            raise RuntimeError("Non-zero gradient detected!")
        assert param.grad is None or torch.allclose(param.grad, torch.zeros_like(param.grad))
