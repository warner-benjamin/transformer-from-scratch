# Attention Mechanism Exercise

This exercise is designed to help you understand and implement the attention mechanism used in transformer models. You will implement different variants of attention mechanisms and test them against reference implementations.

> **Note:** The reference implementation is available in `solution/hidden/reference_attention.py`, but you are strongly encouraged to implement the solutions yourself before looking at it. The learning happens in the struggle!

If you haven't already, you should first read an Attention explainer, whether that's [*Creating a Transformer From Scratch - Part One: The Attention Mechanism*](https://benjaminwarner.dev/2023/07/01/attention-mechanism.html), Chapter 3 of [*Build a Large Language Model (From Scratch)*](https://sebastianraschka.com/books) by Sebastian Raschka, the original Transformers paper [*Attention is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al, or your favorite online resource. Then wait a bit and come back to this exercise.

Make sure to turn off code completions in your IDE.

## Overview

The attention mechanism is a key component of transformer models. It allows the model to focus on different parts of the input sequence when making predictions. In this exercise, you will implement:

1. **Eager Bidirectional Attention**: The standard attention mechanism that allows each position to attend to all positions in the sequence.
2. **Eager Causal Attention**: A masked attention mechanism that ensures each position can only attend to itself and previous positions (used in decoder-only models like GPT).
3. **SDPA Bidirectional Attention**: Use PyTorch's `scaled_dot_product_attention` function for bidirectional attention.
4. **SDPA Causal Attention**: Use PyTorch's `scaled_dot_product_attention` function with causal masking.
5. **Flash Bidirectional Attention**: Use Flash Attention for bidirectional attention.
6. **Flash Causal Attention**: Use Flash Attention with causal masking.

Remember, the attention mechanism is defined by the following equation:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

## Getting Started

1. `attention.py` contains six function stubs that you need to implement. (The Flash Attention stubs require an Nvidia GPU.)
2. Each function receives query, key, and value tensors of shape `[batch_size, sequence_length, hidden_dim]`, along with the number of heads and head dimension.
3. Your task is to perform the attention computation and return an output tensor of shape `[batch_size, sequence_length, hidden_dim]`.

## If You Get Stuck

If you are stuck, you can prompt your [ChatGPT](https://chatgpt.com) or [Claude](https://claude.ai) with the [Socratic Prompt](socratic_prompt.md) to help you complete this exercise (don't read it yourself, though). This prompt should instruct ChatGPT or Claude to guide you through the problem, rather than give you the solution outright.

Don't use Sonnet 3.7, it has a tendency to ignore the Socratic prompt and give you the entire solution. Use Sonnet 3.5, GPT-4o, or o3-mini instead.

Don't use Cursor, Copilot, or other AI powered IDEs, they will not follow the Socratic prompt as well as the ChatGPT or Claude chatbots and might ignore it altogether and give you the entire solution.

## Implementation Details

Do not apply any attention dropout.

### Eager Implementations

For the eager implementations, you need to manually compute the entire attention mechanism.

Remember that the attention mask is True for tokens that should be attended to and False for tokens which should be masked out, so depending on your implementation you will probably need to invert the mask.

### SDPA Implementations

For these implementations, you'll use PyTorch's [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) function, which provides an optimized implementation of the attention mechanism.

If passed a boolean attention mask, SDPA expects True means "participate in attention" and False means "masked out", which matches our attention mask.

### Flash Attention Implementations

For these implementations, you'll use [Flash Attention](https://github.com/Dao-AILab/flash-attention), a highly optimized implementation of the attention mechanism for Nvidia GPUs. Flash Attention uses a significantly different interface:

- The input tensors for Flash Attention have a different shape: `[total_seq_len, hidden_dim]` where `total_seq_len` is the total number of tokens across all sequences in the batch (i.e., `batch_size * seq_len`) excluding padding tokens.
- Instead of masks, it uses cumulative sequence lengths (`cu_seqlens`) and maximum sequence length (`max_seqlen`) to handle variable-length sequences.
- The `cu_seqlens` tensor should contain the cumulative sum of sequence lengths, starting with 0. For example, if you have 3 sequences of lengths [2, 3, 4], `cu_seqlens` would be [0, 2, 5, 9].
- The `max_seqlen` parameter specifies the maximum sequence length in the batch.

Your implementation should:
1. Skip the transpose required for Eager and SDPA implementations.
2. Use the Flash Attention function with the provided `cu_seqlens` and `max_seqlen` to pass to `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, and `max_seqlen_k`.

### Mask Interpretation

Be aware that there's a difference in how masks are interpreted:
- In our interface, `False` in the mask means "masked out" (don't attend to this position).
    - This makes intuitive sense (attend to True, don't attend to False), but adds potential additional step of inverting the mask.
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
- Flash Attention only works on Ampere or newer Nvidia GPUs.

Good luck!