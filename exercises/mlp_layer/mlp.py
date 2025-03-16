import torch
import torch.nn as nn
from torch import Tensor


class MLP(nn.Module):
    """
    A standard MLP has two linear layers separated by an activation,
    with an optional dropout layer at the end.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ):
        """
        Initialize a standard transformer MLP layer.

        Args:
            hidden_dim: Dimension of the input and output features
            intermediate_dim: Dimension of the intermediate features after the first linear layer
                              Often set to 4 * hidden_dim as in the original transformer
            activation: Activation function to use, defaults to GELU
            dropout: Output dropout probability (0.0 means no dropout)
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for standard MLP layer")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]
        """
        raise NotImplementedError("Implement forward pass for standard MLP layer")


class GLU(nn.Module):
    """
    The Gated Linear Unit has two parallel linear transforms: one for the gate and one for the value.
    Apply the activation only to the gate, then multiply elementwise with the value, followed by a
    final linear projection and optional dropout.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ):
        """
        Initialize a GLU.

        Args:
            hidden_dim: Dimension of the input and output features
            intermediate_dim: Dimension of each intermediate branch
                              Often set to 2/3 * 4 * hidden_dim to maintain similar parameter
                              count to a standard MLP with 4x expansion
            activation: Activation function to use, defaults to GELU
            dropout: Output dropout probability (0.0 means no dropout)
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for GLU")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]
        """
        raise NotImplementedError("Implement forward pass for GLU")
