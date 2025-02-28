# Attention Mechanism Exercise

This exercise is designed to help you understand and implement the attention mechanism used in transformer models. You will implement different variants of attention mechanisms and test them against reference implementations.

> **Note:** The reference implementation is available in `solution/hidden/reference_attention.py`, but we strongly encourage you to implement the solutions yourself before looking at it. The learning happens in the struggle!

## Overview

The attention mechanism is a key component of transformer models. It allows the model to focus on different parts of the input sequence when making predictions. In this exercise, you will implement:

1. **Eager Bidirectional Attention**: The standard attention mechanism that allows each position to attend to all positions in the sequence.
2. **Eager Causal Attention**: A masked attention mechanism that ensures each position can only attend to itself and previous positions (used in decoder-only models like GPT).
3. **SDPA Bidirectional Attention**: Use PyTorch's `scaled_dot_product_attention` function for bidirectional attention.
4. **SDPA Causal Attention**: Use PyTorch's `scaled_dot_product_attention` function with causal masking.
5. **Flash Bidirectional Attention**: Use Flash Attention for bidirectional attention.
6. **Flash Causal Attention**: Usie Flash Attention with causal masking.

## Getting Started

1. `attention.py` contains six function stubs that you need to implement. (The Flash Attention stubs require an Nvidia GPU.)
2. Each function receives query, key, and value tensors of shape `[batch_size, sequence_length, hidden_dim]`, along with the number of heads and head dimension.
3. Your task is to perform the attention computation and return an output tensor of shape `[batch_size, sequence_length, hidden_dim]`.

## Implementation Details

Do not apply any attention dropout.

### Eager Implementations

For the eager implementations, you need to manually compute the entire attention mechanism.

### SDPA and Flash Implementations

For these implementations, you'll use PyTorch's [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) function, which provides an optimized implementation of the attention mechanism.

### Flash Attention Implementations

For these implementations, you'll use Flash Attention, a highly optimized implementation of the attention mechanism for Nvidia GPUs. Flash Attention uses a significantly different interface:

- The input tensors for Flash Attention have a different shape: `[total_seq_len, hidden_dim]` where `total_seq_len` is the total number of tokens across all sequences in the batch (i.e., `batch_size * seq_len`) excluding padding tokens.
- Instead of masks, it uses cumulative sequence lengths (`cu_seqlens`) and maximum sequence length (`max_seqlen`) to handle variable-length sequences.
- The `cu_seqlens` tensor should contain the cumulative sum of sequence lengths, starting with 0. For example, if you have 3 sequences of lengths [2, 3, 4], `cu_seqlens` would be [0, 2, 5, 9].
- The `max_seqlen` parameter specifies the maximum sequence length in the batch.

Your implementation should:
1. Skip the transpose required for Eager and SDPA implementations.
2. Use the Flash Attention function with the provided `cu_seqlens` and `max_seqlen` to pass to `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, and `max_seqlen_k`.

#### Mask Interpretation

Be aware that there's a difference in how masks are interpreted:
- In our interface, `True` in the mask means "masked out" (don't attend to this position).
- In PyTorch's SDPA, `True` means "participate in attention" (do attend to this position).
- In Flash Attention, we use `cu_seqlens` and `max_seqlen` instead of an attention mask.

## Testing Your Implementation

Run the tests to check your implementation:

```bash
pytest tests/
```

The tests will compare your implementation against a reference implementation. If your implementation is correct, all tests should pass.

## Notes

- The mask is a boolean tensor of shape `[batch_size, sequence_length]` where `True` indicates masked positions.
- For causal attention, don't forget to create the causal mask.
- For Flash Attention implementations, you'll need to use the `cu_seqlens` and `max_seqlen` parameters instead of masks.
- Flash Attention is an optimized implementation that may not be available on all hardware.

## If You Get Stuck

If you're stuck or helping others with this exercise, you cam prompt your favorite coding assistant (Copilot, Cursor, etc.) with the [Socratic Prompt](solution/hidden/socratic_prompt.md). This prompt should instruct the LLM to guide you through the problem, rather than give you the solution outright.

Good luck!