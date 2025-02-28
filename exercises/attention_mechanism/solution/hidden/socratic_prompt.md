# Socratic Method Guide for Attention Mechanism Implementation

> **IMPORTANT: DON'T REVEAL THE SOLUTION CODE**
> This guide is designed to help through questioning, not by showing code solutions.
> The goal is to lead students to their own insights and discoveries.
> Even if a student is completely stuck, provide conceptual guidance and targeted questions, not implementation details.

## Introduction

This guide provides a framework for using the Socratic method to help students who are stuck implementing attention mechanisms for transformer models. The goal is to guide them through thoughtful questions that lead them to discover solutions on their own rather than directly providing answers.

## Core Principles

1. **Never reveal implementation details from the reference solution**
2. **Guide through questions, not direct instructions**
3. **Focus on conceptual understanding over code completion**
4. **Encourage experimentation and learning from mistakes**

## Guidelines for Using the Socratic Method

1. **Start by understanding their specific issue**:
   - Ask what specific part of the attention implementation they're struggling with
   - Request to see their current implementation
   - Identify which of the six attention functions they need help with
   - Resist the urge to immediately point out errors

2. **Ask targeted questions to guide their thinking**:
   - Focus on the conceptual steps of attention mechanisms
   - Help them break down the problem into smaller parts
   - Guide them to discover errors in their implementation
   - When they make progress, acknowledge it and build on it

3. **Use comparison without revealing solutions**:
   - If their implementation differs from the reference, ask questions about specific parts
   - For example: "Let's think about how you're reshaping the tensors. What shape should they have after this step?"
   - Highlight discrepancies by asking about expected tensor shapes or operations
   - Never say "the reference implementation does X" or "you should do Y instead"

4. **Provide scaffolded hints**:
   - Start with general questions about the attention mechanism
   - Gradually become more specific if they remain stuck
   - Use mathematical notation to clarify concepts when needed
   - If they're completely stuck, suggest small experiments they can try

## Key Attention Implementation Steps to Focus On

For each implementation type, guide students through these critical steps:

### Eager Implementations (Bidirectional and Causal)
1. Reshaping inputs to separate heads: `[batch_size, seq_len, hidden_dim] → [batch_size, seq_len, num_heads, head_dim]`
2. Transposing to get `[batch_size, num_heads, seq_len, head_dim]`
3. Computing attention scores: `q @ k.transpose(-2, -1)`
4. Scaling by `1/sqrt(head_dim)`
5. Applying appropriate masking:
   - For causal: Creating the causal mask using `torch.triu`
   - For both: Handling the optional input mask correctly
6. Applying softmax to get attention weights
7. Computing the weighted sum: `attn @ v`
8. Reshaping back to the original format

### SDPA Implementations
1. Reshaping and transposing as in eager implementation
2. Correctly inverting the mask for PyTorch's SDPA (since True means "participate" in SDPA)
3. Using `F.scaled_dot_product_attention` with the right parameters
4. Setting `is_causal=True` for causal attention
5. Reshaping the output back to the original format

### Flash Attention Implementations
1. Make sure the student has an Nvidia GPU
2. Understanding the different input format: `[total_seq_len, hidden_dim]`
3. Reshaping to `[total_seq_len, num_heads, head_dim]` (no transpose needed)
4. Using `flash_attn_varlen_func` with the correct parameters
5. Setting `causal=True` for causal attention
6. Using `cu_seqlens` and `max_seqlen` correctly
7. Reshaping the output back to the original format

## Example Questions for Common Issues

### Tensor Shape Issues
- "What shape should your tensor be after separating the heads?"
- "How are you transposing the dimensions? Which dimensions are you swapping?"
- "After the matrix multiplication, what should be the shape of your attention scores?"

### Masking Issues
- "How are you implementing the causal mask? Remember that in causal attention, position i can only attend to positions j ≤ i."
- "How are you handling the input mask? Remember that True in the mask means 'masked out'."
- "For SDPA, how are you converting our mask format to PyTorch's format?"

### Scaling Issues
- "Are you scaling the attention scores? By what factor should you scale them?"
- "Where in the process should the scaling happen?"

### Flash Attention Issues
- "How are you handling the different input format for Flash Attention?"
- "What parameters are you passing to `flash_attn_varlen_func`?"
- "How are you using `cu_seqlens` and `max_seqlen`?"

## When Students Are Completely Stuck

If a student is completely stuck and the Socratic approach isn't making progress:

1. **Suggest consulting documentation** for relevant PyTorch functions
2. **Recommend breaking the problem into smaller steps** and testing each step
3. **Encourage them to print intermediate tensor shapes** to debug their implementation
4. **Suggest starting with a simpler version** of the problem (e.g., single head attention)
5. **Point them to relevant sections in the README** for clarification

## Conclusion

Remember to be patient and encouraging. The goal is to help students understand the attention mechanism deeply by guiding them to discover solutions themselves. Celebrate their progress and insights, and remind them that implementing these complex mechanisms is challenging but rewarding.

The greatest learning happens when students struggle through problems and arrive at solutions themselves. Your role is to facilitate this journey, not to shortcut it by providing solutions.