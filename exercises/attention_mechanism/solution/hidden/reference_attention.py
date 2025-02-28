from __future__ import annotations

import math
import torch
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
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: int,
    head_dim: int,
    mask: BoolTensor | None = None,
) -> Tensor:
    """
    Reference implementation of bidirectional (full) attention using the eager approach.

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
    """
    # Get batch size and sequence length
    batch_size, seq_len, hidden_dim = q.shape

    # Reshape to separate the heads
    # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
    q = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose to get [batch_size, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
    attn = q @ k.transpose(-2, -1)

    # Scale by square root of head dimension
    attn = attn / math.sqrt(head_dim)

    # Apply mask if provided
    if mask is not None:
        # Reshape mask to [batch_size, 1, 1, seq_len]
        mask = mask.view(batch_size, 1, 1, seq_len)
        attn = attn.masked_fill(mask, float("-inf"))

    # Apply softmax to get attention weights
    attn = F.softmax(attn, dim=-1)

    # Apply attention weights to values
    # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
    # -> [batch_size, num_heads, seq_len, head_dim]
    output = attn @ v

    # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
    output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

    return output


def eager_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: int,
    head_dim: int,
    mask: BoolTensor | None = None,
) -> Tensor:
    """
    Reference implementation of causal (masked) attention using the eager approach.

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
    """
    # Get batch size and sequence length
    batch_size, seq_len, hidden_dim = q.shape

    # Reshape to separate the heads
    # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
    q = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose to get [batch_size, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Compute attention scores: [batch_size, num_heads, seq_len, seq_len]
    attn = q @ k.transpose(-2, -1)

    # Scale by square root of head dimension
    attn = attn / math.sqrt(head_dim)

    # Create causal mask (upper triangular)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1).view(1, 1, seq_len, seq_len)

    # Apply causal mask
    attn = attn.masked_fill(causal_mask, float("-inf"))

    # Apply additional mask if provided
    if mask is not None:
        # Reshape mask to [batch_size, 1, 1, seq_len]
        mask = mask.view(batch_size, 1, 1, seq_len)
        attn = attn.masked_fill(mask, float("-inf"))

    # Apply softmax to get attention weights
    attn = F.softmax(attn, dim=-1)

    # Apply attention weights to values
    # [batch_size, num_heads, seq_len, seq_len] @ [batch_size, num_heads, seq_len, head_dim]
    # -> [batch_size, num_heads, seq_len, head_dim]
    output = attn @ v

    # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
    output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

    return output


def sdp_bidirectional_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: int,
    head_dim: int,
    mask: BoolTensor | None = None,
) -> Tensor:
    """
    Reference implementation of bidirectional (full) attention using PyTorch's SDPA.

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
    """
    # Get batch size and sequence length
    batch_size, seq_len, hidden_dim = q.shape

    # Reshape to separate the heads
    # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
    q = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose to get [batch_size, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Prepare attention mask if provided
    # For SDPA, we need to invert the mask since True means "participate in attention"
    # in SDPA, but in our interface True means "masked out"
    attn_mask = None
    if mask is not None:
        # Reshape to [batch_size, 1, 1, seq_len] for broadcasting
        attn_mask = mask.view(batch_size, 1, 1, seq_len)
        # Invert the mask since SDPA expects True to mean "participate in attention"
        attn_mask = ~attn_mask

    # Use PyTorch's scaled_dot_product_attention
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
    output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

    return output


def sdp_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: int,
    head_dim: int,
    mask: BoolTensor | None = None,
) -> Tensor:
    """
    Reference implementation of causal (masked) attention using PyTorch's SDPA.

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
    """
    # Get batch size and sequence length
    batch_size, seq_len, hidden_dim = q.shape

    # Reshape to separate the heads
    # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, num_heads, head_dim]
    q = q.reshape(batch_size, seq_len, num_heads, head_dim)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim)

    # Transpose to get [batch_size, num_heads, seq_len, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Prepare attention mask if provided
    # For SDPA, we need to invert the mask since True means "participate in attention"
    # in SDPA, but in our interface True means "masked out"
    attn_mask = None
    if mask is not None:
        # Reshape to [batch_size, 1, 1, seq_len] for broadcasting
        attn_mask = mask.view(batch_size, 1, 1, seq_len)
        # Invert the mask since SDPA expects True to mean "participate in attention"
        attn_mask = ~attn_mask

    # Use PyTorch's scaled_dot_product_attention with is_causal=True
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=True)

    # Transpose and reshape back to [batch_size, seq_len, hidden_dim]
    output = output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)

    return output


def flash_bidirectional_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """
    Reference implementation of bidirectional (full) attention using Flash Attention.

    Args:
        q: Query tensor of shape [total_seq_len, hidden_dim]
        k: Key tensor of shape [total_seq_len, hidden_dim]
        v: Value tensor of shape [total_seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        cu_seqlens: Cumulative sequence lengths tensor of shape [batch_size + 1]
                   Used for variable sequence length batching in Flash Attention
        max_seqlen: Maximum sequence length in the batch

    Returns:
        Output tensor of shape [total_seq_len, hidden_dim]
    """
    if not FLASH_ATTN_AVAILABLE:
        raise ImportError("Flash Attention is not available. Please install flash-attn package or skip Flash Attention tests.")

    # Get total sequence length and hidden dimension
    total_seq_len, hidden_dim = q.shape

    # Reshape to separate the heads: [total_seq_len, hidden_dim] -> [total_seq_len, num_heads, head_dim]
    q = q.view(total_seq_len, num_heads, head_dim)
    k = k.view(total_seq_len, num_heads, head_dim)
    v = v.view(total_seq_len, num_heads, head_dim)

    # Use Flash Attention with variable sequence lengths
    attn_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=False,
    )

    # Reshape back to original format: [total_seq_len, num_heads, head_dim] -> [total_seq_len, hidden_dim]
    attn_output = attn_output.view(total_seq_len, hidden_dim)

    return attn_output


def flash_causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: Tensor,
    max_seqlen: int,
) -> Tensor:
    """
    Reference implementation of causal (masked) attention using Flash Attention.

    Args:
        q: Query tensor of shape [total_seq_len, hidden_dim]
        k: Key tensor of shape [total_seq_len, hidden_dim]
        v: Value tensor of shape [total_seq_len, hidden_dim]
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        cu_seqlens: Cumulative sequence lengths tensor of shape [batch_size + 1]
                   Used for variable sequence length batching in Flash Attention
        max_seqlen: Maximum sequence length in the batch

    Returns:
        Output tensor of shape [total_seq_len, hidden_dim]
    """
    if not FLASH_ATTN_AVAILABLE:
        raise ImportError("Flash Attention is not available. Please install flash-attn package or skip Flash Attention tests.")

    # Get total sequence length and hidden dimension
    total_seq_len, hidden_dim = q.shape

    # Reshape to separate the heads: [total_seq_len, hidden_dim] -> [total_seq_len, num_heads, head_dim]
    q = q.view(total_seq_len, num_heads, head_dim)
    k = k.view(total_seq_len, num_heads, head_dim)
    v = v.view(total_seq_len, num_heads, head_dim)

    # Use Flash Attention with variable sequence lengths and causal masking
    attn_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )

    # Reshape back to original format: [total_seq_len, num_heads, head_dim] -> [total_seq_len, hidden_dim]
    attn_output = attn_output.view(total_seq_len, hidden_dim)

    return attn_output
