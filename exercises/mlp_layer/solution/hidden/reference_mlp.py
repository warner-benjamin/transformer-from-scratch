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

        # First linear layer (expansion)
        self.Wi = nn.Linear(hidden_dim, intermediate_dim)

        # Store activation function
        self.activation = activation()

        # Second linear layer (contraction)
        self.Wo = nn.Linear(intermediate_dim, hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]
        """
        # First linear layer
        x = self.Wi(x)

        # Apply activation function
        x = self.activation(x)

        # Second linear layer
        x = self.Wo(x)

        # Apply dropout
        x = self.dropout(x)

        return x


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

        # Gate branch linear layer
        self.Wg = nn.Linear(hidden_dim, intermediate_dim)

        # Value branch linear layer
        self.Wv = nn.Linear(hidden_dim, intermediate_dim)

        # Store activation function
        self.activation = activation()

        # Output projection
        self.Wo = nn.Linear(intermediate_dim, hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] or [total_seq_len, hidden_dim]
        """
        # Compute gate and value projections
        gate = self.Wg(x)
        value = self.Wv(x)

        # Apply activation to the gate
        gate = self.activation(gate)

        # Element-wise multiplication
        intermediate = gate * value

        # Output projection
        output = self.Wo(intermediate)

        # Apply dropout
        output = self.dropout(output)

        return output
