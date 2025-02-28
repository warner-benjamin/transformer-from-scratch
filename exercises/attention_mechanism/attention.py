from __future__ import annotations

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


def eager_bidirectional_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates masked positions
) -> Tensor:
    """
    Implement bidirectional (full) attention using only PyTorch operations.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask of shape [batch_size, sequence_length]
              where True indicates masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        You need to reshape the inputs to separate the heads, perform the
        attention computation, and then merge the heads back.
    """
    raise NotImplementedError("Implement bidirectional attention using PyTorch operations")


def eager_causal_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates masked positions
) -> Tensor:
    """
    Implement causal (masked) attention using only PyTorch operations.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask of shape [batch_size, sequence_length]
              where True indicates masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        A causal mask ensures that a position i can only attend to positions j ≤ i.
    """
    raise NotImplementedError("Implement causal attention using PyTorch operations")


def sdp_bidirectional_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates masked positions
) -> Tensor:
    """
    Implement bidirectional (full) attention using PyTorch's scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask of shape [batch_size, sequence_length]
              where True indicates masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        Note that there's a difference in mask interpretation between our interface and
        PyTorch's SDPA function. In our interface, True means "masked out", while in
        PyTorch's SDPA, True means "participate in attention".
    """
    raise NotImplementedError("Implement bidirectional attention using PyTorch's SDPA")


def sdp_causal_attention(
    q: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    k: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    v: Tensor,  # shape: [batch_size, sequence_length, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    mask: BoolTensor | None = None,  # shape: [batch_size, sequence_length] where True indicates masked positions
) -> Tensor:
    """
    Implement causal (masked) attention using PyTorch's scaled_dot_product_attention.

    Args:
        q: Query tensor of shape [batch_size, sequence_length, hidden_dim]
        k: Key tensor of shape [batch_size, sequence_length, hidden_dim]
        v: Value tensor of shape [batch_size, sequence_length, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        mask: Optional boolean mask of shape [batch_size, sequence_length]
              where True indicates masked positions

    Returns:
        Output tensor of shape [batch_size, sequence_length, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        Note that there's a difference in mask interpretation between our interface and
        PyTorch's SDPA function. In our interface, True means "masked out", while in
        PyTorch's SDPA, True means "participate in attention".

        You can use the `is_causal` argument to enable causal masking instead of
        creating a causal mask.
    """
    raise NotImplementedError("Implement causal attention using PyTorch's SDPA")


def flash_bidirectional_attention(
    q: Tensor,  # shape: [total_seq_len, hidden_dim]
    k: Tensor,  # shape: [total_seq_len, hidden_dim]
    v: Tensor,  # shape: [total_seq_len, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    cu_seqlens: Tensor,  # shape: [batch_size + 1], cumulative sequence lengths
    max_seqlen: int,  # maximum sequence length
) -> Tensor:
    """
    Implement bidirectional (full) attention using Flash Attention.

    Args:
        q: Query tensor of shape [total_seq_len, hidden_dim]
        k: Key tensor of shape [total_seq_len, hidden_dim]
        v: Value tensor of shape [total_seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
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
        Output tensor of shape [total_seq_len, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        Flash Attention uses a different interface than scaled_dot_product_attention.
        Instead of using an attention mask, it uses cumulative sequence lengths (cu_seqlens)
        and the maximum sequence length (max_seqlen) to .
    """
    raise NotImplementedError("Implement bidirectional attention using Flash Attention")


def flash_causal_attention(
    q: Tensor,  # shape: [total_seq_len, hidden_dim]
    k: Tensor,  # shape: [total_seq_len, hidden_dim]
    v: Tensor,  # shape: [total_seq_len, hidden_dim]
    num_heads: int,  # number of attention heads
    head_dim: int,  # dimension of each attention head
    cu_seqlens: Tensor,  # shape: [batch_size + 1], cumulative sequence lengths
    max_seqlen: int,  # maximum sequence length
) -> Tensor:
    """
    Implement causal (masked) attention using Flash Attention.

    Args:
        q: Query tensor of shape [total_seq_len, hidden_dim]
        k: Key tensor of shape [total_seq_len, hidden_dim]
        v: Value tensor of shape [total_seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
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
        Output tensor of shape [total_seq_len, hidden_dim]
        This is the result after the attention computation but before
        the final linear projection.

    Note:
        Flash Attention uses a different interface than scaled_dot_product_attention.
        Instead of using masks, it uses cumulative sequence lengths (cu_seqlens)
        and the maximum sequence length (max_seqlen) to handle variable-length sequences.

        For causal attention, you'll need to set the causal flag to True when using
        the Flash Attention function.
    """
    raise NotImplementedError("Implement causal attention using Flash Attention")
