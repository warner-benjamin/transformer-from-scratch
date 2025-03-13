import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, BoolTensor

# Try to import Flash Attention
FLASH_ATTN_AVAILABLE = False
try:
    from flash_attn import flash_attn_varlen_func

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    # Flash Attention is not available
    pass


class EagerBidirectionalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head bidirectional (full) attention using
    the eager approach.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Projection layers for query, key, and value
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

        # Output projection layer
        self.Wo = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask: Optional boolean mask of shape [batch_size, seq_len] where False indicates a masked position.

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
        """
        batch_size, seq_len, hidden_dim = x.size()

        # Compute Q, K, V projections
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        # Reshape to separate the heads
        # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to get [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
        attn = q @ k.transpose(-2, -1)

        # Scale by square root of head dimension
        attn = attn / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            # Reshape mask to [batch_size, 1, 1, seq_len]
            # And invert since we want to mask out the positions where it is False
            mask = ~mask.view(batch_size, 1, 1, seq_len)
            attn = attn.masked_fill(mask, float("-inf"))

        # Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)

        # Apply attention weights to values
        # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
        # -> [batch_size, num_heads, seq_len, head_dim]
        output = attn @ v

        # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

        # Apply dropout
        output = self.dropout(output)

        # Final projection
        output = self.Wo(output)
        return output


class EagerCausalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head causal (masked) attention using
    the eager approach.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        max_seq_len: int = 128,
    ):
        """
        Initialize the causal attention block with eager implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)
            max_seq_len: Maximum sequence length (for masking)
        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - You'll need to create projection layers for query, key, and value
            - Don't forget the output projection layer
            - Create a dropout layer if dropout > 0.0, otherwise use nn.Identity
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for causal attention block using PyTorch operations")

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask: Optional boolean mask of shape [batch_size, seq_len] where True indicates a masked position.

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
        """
        raise NotImplementedError("Implement causal attention block using PyTorch operations")


class SDPABidirectionalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head bidirectional (full) attention using
    PyTorch's scaled_dot_product_attention (SDPA).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the bidirectional attention block with SDPA implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - You'll need to create projection layers for query, key, and value
            - Don't forget the output projection layer
            - Create a dropout layer if dropout > 0.0, otherwise use nn.Identity
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for bidirectional attention block using PyTorch's SDPA")

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask: Optional boolean mask where True indicates a masked position.

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
        """
        raise NotImplementedError("Implement bidirectional attention block using PyTorch's SDPA")


class SDPACausalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head causal (masked) attention using
    PyTorch's scaled_dot_product_attention (SDPA).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the causal attention block with SDPA implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - You'll need to create projection layers for query, key, and value
            - Don't forget the output projection layer
            - Create a dropout layer if dropout > 0.0, otherwise use nn.Identity
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for causal attention block using PyTorch's SDPA")

    def forward(self, x: Tensor, mask: BoolTensor | None = None) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            mask: Optional boolean mask where True indicates a masked position.

        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim] after attention.
        """
        raise NotImplementedError("Implement causal attention block using PyTorch's SDPA")


class FlashBidirectionalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head bidirectional (full) attention using
    Flash Attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the bidirectional attention block with Flash Attention implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - Check if Flash Attention is available (FLASH_ATTN_AVAILABLE)
            - You'll need to create projection layers for query, key, and value
            - Don't forget the output projection layer
            - Create a dropout layer if dropout > 0.0, otherwise use nn.Identity
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for bidirectional attention block using Flash Attention")

    def forward(self, x: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [total_seq_len, hidden_dim].
            cu_seqlens: Cumulative sequence lengths tensor of shape [batch_size + 1]
                    Used instead of an attention mask for both masking and
                    variable-length sequences. Example:
                        cu_seqlens = torch.tensor([0, 10, 30, 60])
                    This means there are three sequences in the batch:
                        - First sequence has 10 tokens
                        - Second sequence has 20 tokens
                        - Third sequence has 30 tokens
            max_seqlen: Maximum sequence length in the batch. In the example above,
                        the maximum sequence length is 30.

        Returns:
            Tensor of shape [total_seq_len, hidden_dim] after attention.
        """
        raise NotImplementedError("Implement bidirectional attention block using Flash Attention")


class FlashCausalAttentionBlock(nn.Module):
    """
    Attention block implementing multi-head causal (masked) attention using
    Flash Attention.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Initialize the causal attention block with Flash Attention implementation.

        Args:
            hidden_dim: Dimension of the input and output features
            num_heads: Number of attention heads
            dropout: Output dropout probability (0.0 means no dropout)

        Note:
            - Make sure to check that hidden_dim is divisible by num_heads
            - Check if Flash Attention is available (FLASH_ATTN_AVAILABLE)
            - You'll need to create projection layers for query, key, and value
            - Don't forget the output projection layer
            - Create a dropout layer if dropout > 0.0, otherwise use nn.Identity
        """
        super().__init__()
        raise NotImplementedError("Implement initialization for causal attention block using Flash Attention")

    def forward(self, x: Tensor, cu_seqlens: Tensor, max_seqlen: int) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [total_seq_len, hidden_dim].
            cu_seqlens: Cumulative sequence lengths tensor of shape [batch_size + 1]
                    Used instead of an attention mask for both masking and
                    variable-length sequences. Example:
                        cu_seqlens = torch.tensor([0, 10, 30, 60])
                    This means there are three sequences in the batch:
                        - First sequence has 10 tokens
                        - Second sequence has 20 tokens
                        - Third sequence has 30 tokens
            max_seqlen: Maximum sequence length in the batch. In the example above,
                        the maximum sequence length is 30.

        Returns:
            Tensor of shape [total_seq_len, hidden_dim] after attention.
        """
        raise NotImplementedError("Implement causal attention block using Flash Attention")
