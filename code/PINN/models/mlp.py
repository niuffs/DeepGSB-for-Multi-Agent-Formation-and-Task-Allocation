"""
Neural network modules for PINN.

Extends the base MLP with an optional output transform
for enforcing boundary / initial conditions.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with configurable activation and initialization.

    Supports an optional output transform (via add_output_transform) for
    enforcing boundary / initial conditions on the network output.

    Args:
        layer_sizes: List of [input_dim, hidden_1, ..., output_dim]
        activation: Activation function (single or list per layer)
        kernel_initializer: Weight initializer
        dtype: Tensor dtype
        device: Device to place parameters
    """

    def __init__(
        self,
        layer_sizes,
        activation=torch.tanh,
        kernel_initializer=nn.init.xavier_normal_,
        dtype=torch.float32,
        device="cpu",
    ):
        super().__init__()
        if isinstance(activation, list) and len(activation) != len(layer_sizes) - 1:
            raise ValueError(
                "Number of activation functions must match number of layers."
            )

        self.activation = activation
        self.linears = nn.ModuleList()
        self._output_transform = None

        for i in range(1, len(layer_sizes)):
            linear = nn.Linear(layer_sizes[i - 1], layer_sizes[i], dtype=dtype).to(device)
            kernel_initializer(linear.weight)
            nn.init.zeros_(linear.bias)
            self.linears.append(linear)

    def add_output_transform(self, transform):
        """Register an output transform function.

        The transform receives (inputs, outputs) and returns modified outputs,
        typically used to enforce terminal / boundary conditions.
        """
        self._output_transform = transform

    def forward(self, inputs):
        outputs = inputs
        for j, linear in enumerate(self.linears[:-1]):
            act = self.activation[j] if isinstance(self.activation, list) else self.activation
            outputs = act(linear(outputs))
        outputs = self.linears[-1](outputs)

        if self._output_transform is not None:
            outputs = self._output_transform(inputs, outputs)
        return outputs
