# Attention Block Exercise

This exercise is designed to help you understand and implement complete attention blocks used in transformer models. You will implement different variants of attention blocks and test them against reference implementations.

> **Note:** The reference implementation is available in `solution/hidden/reference_attention_block.py`, but you are strongly encouraged to implement the solutions yourself before looking at it. The learning happens in the struggle!

If you haven't already, you should first read about attention mechanisms and transformer architecture, whether that's [*Creating a Transformer From Scratch*](https://benjaminwarner.dev/2023/07/01/attention-mechanism.html), Chapter 3 of [*Build a Large Language Model (From Scratch)*](https://sebastianraschka.com/books) by Sebastian Raschka, the original Transformers paper [*Attention is All You Need*](https://arxiv.org/abs/1706.03762) by Vaswani et al, or your favorite online resource. Then wait a bit and come back to this exercise.

Make sure to turn off code completions in your IDE.

## Overview

The attention block is a key component of transformer models. It includes the full pipeline of transforming inputs through linear projections, applying multi-head attention, and projecting the output back to the original dimension. In this exercise, you will implement:

1. **Eager Bidirectional Attention Block**: Standard attention block that allows each position to attend to all positions in the sequence.
2. **Eager Causal Attention Block**: A masked attention block that ensures each position can only attend to itself and previous positions (used in decoder-only models like GPT).
3. **SDPA Bidirectional Attention Block**: PyTorch's `scaled_dot_product_attention` function for bidirectional attention within a full block.
4. **SDPA Causal Attention Block**: PyTorch's `scaled_dot_product_attention` function with causal masking within a full block.
5. **Flash Bidirectional Attention Block**: Flash Attention for bidirectional attention within a full block.
6. **Flash Causal Attention Block**: Flash Attention with causal masking within a full block.

A complete attention block performs the following operations:
1. Linear projections of the input to create queries (Q), keys (K), and values (V)
2. Multi-head attention computation
3. Output projection back to the original dimension
4. Optional projection dropout

Or, in equation form (ignoring final dropout):

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V, \\[0.5em]
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\Bigl(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\Bigr)\mathbf{V}, \\[0.5em]
\mathbf{X} = \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})\,\mathbf{W}^O,
$$

where $\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V})$ is a simplification of multi-head attention:

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathrm{concat}(\text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h), \text{ for all } h), \\[0.5em]
$$

where $h$ is the number of attention heads, $\mathbf{X}$ is the input, and $\mathbf{X}\mathbf{W}$ is shorthand for $\mathbf{X} \cdot \mathbf{A} + \mathbf{b}$ (where $\mathbf{A}$ and $\mathbf{b}$ are learnable parameters).

## Getting Started

1. `attention_block.py` contains six class stubs that you need to implement.
2. Each class should be a complete `nn.Module` that processes input tensors of shape `[batch_size, sequence_length, hidden_dim]`.
3. Your task is to implement the full attention block and return an output tensor of the same shape.
4. If you already completed the attention mechanism exercise, you can copy your code from `attention_mechanism.py` to help complete the attention block.

## Implementation Details

Do not apply any attention dropout as this is rarely used in practice. The projection dropout should be applied after the output linear layer.

The test code assumes that your linear layers are named `self.Wq`, `self.Wk`, `self.Wv`, and `self.Wo`.

### Common Structure

All attention block implementations should:
1. Have projection layers for query, key, and value
2. Support multi-head attention
3. Include an output projection
4. Support optional projection dropout

### Eager Implementations

For the eager implementations, you need to manually compute the entire attention mechanism:
1. Project input to Q, K, V
2. Reshape to separate the heads
3. Compute attention scores
4. Apply appropriate masking
5. Scale, apply softmax, and compute the weighted sum
6. Reshape and project back to the original dimension

Remember that the attention mask is True for tokens that should be attended to and False for tokens which should be masked out, so depending on your implementation you will probably need to invert the mask.

For the causal implementation, you'll need to create an upper triangular mask to ensure each position can only attend to itself and previous positions.

### SDPA Implementations

For these implementations, you'll use PyTorch's [`scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) function, which provides an optimized implementation of the attention mechanism.

If passed a boolean attention mask, SDPA expects True means "participate in attention" and False means "masked out", which matches our attention mask.

### Flash Attention Implementations

For these implementations, you'll use [Flash Attention](https://github.com/Dao-AILab/flash-attention), a highly optimized implementation of the attention mechanism for Nvidia GPUs. Flash Attention uses a significantly different interface:

- The input tensors for Flash Attention have a different shape: `[total_seq_len, hidden_dim]` where `total_seq_len` is the total number of tokens across all sequences in the batch (i.e., `batch_size * seq_len`) excluding padding tokens.
- Instead of masks, it uses cumulative sequence lengths (`cu_seqlens`) and maximum sequence length (`max_seqlen`) to handle variable-length sequences.
- The `cu_seqlens` tensor should contain the cumulative sum of sequence lengths, starting with 0. For example, if you have 3 sequences of lengths [2, 3, 4], `cu_seqlens` would be [0, 2, 5, 9].
- The `max_seqlen` parameter specifies the maximum sequence length in the batch.

The Flash Attention implementations require an Nvidia GPU (Ampere or newer).

## If You Get Stuck

If you are stuck, you can prompt your [ChatGPT](https://chatgpt.com) or [Claude](https://claude.ai) with the Socratic Prompt (see `socratic_prompt.md`) to help you complete this exercise (don't read it yourself, though). This prompt should instruct the AI to guide you through the problem, rather than give you the solution outright.

Don't use Sonnet 3.7, it has a tendency to ignore the Socratic prompt and give you the entire solution. Use Sonnet 3.5, GPT-4o, or o3-mini instead.

Don't use Cursor, Copilot, or other AI powered IDEs, they will not follow the Socratic prompt as well as the ChatGPT or Claude chatbots and might ignore it altogether and give you the entire solution.

## Testing Your Implementation

Run the tests to check your implementation:

```bash
pytest tests/
```

The tests will compare your implementation against a reference implementation. If your implementation is correct, all tests should pass.

## Notes

- For bidirectional attention, each position can attend to all positions in the sequence.
- For causal attention, each position can only attend to itself and previous positions.
- The mask is a boolean tensor of shape `[batch_size, sequence_length]` where `False` indicates masked positions.
- Flash Attention only works on Ampere or newer Nvidia GPUs.

Good luck!