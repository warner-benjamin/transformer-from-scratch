# MLP Exercise

This exercise focuses on implementing two types of MLP (Multi-Layer Perceptron) components used in transformer models:

1. **MLP**: The traditional feed-forward network with two linear transformations and an activation function.
2. **GLU**: A Gated Linear Unit variant with parallel projections and multiplicative gating.

> **Note:** Reference implementation is in `solution/hidden/reference_mlp_block.py`, but try implementing it yourself first!

## Architecture Details

### MLP

A standard MLP has two linear layers separated by an activation, with an optional dropout layer at the end. Typically, the intermediate dimension is larger than the input dimension, then contracted back.

$$
\text{MLP}(\mathbf{X}) = \text{activation}(\mathbf{X}\mathbf{W}^I)\mathbf{W}^O
$$

Your linear layers must be named `self.Wi`, `self.Wo` for the tests to pass.

### GLU

The Gated Linear Unit has two parallel linear transforms: one for the gate and one for the value. Apply the activation only to the gate, then multiply elementwise with the value, followed by a final linear projection and optional dropout.

$$
\text{GLU}(\mathbf{X}) = \Bigl( \text{activation}(\mathbf{X}\mathbf{W}^G) \odot (\mathbf{X}\mathbf{W}^V) \Bigr)\mathbf{W}^O
$$

Your linear layers must be named `self.Wv`, `self.Wg`, and `self.Wo` for the tests to pass.

### Note

$\mathbf{X}\mathbf{W}$ is shorthand for $\mathbf{X} \cdot \mathbf{A} + \mathbf{b}$, where $\mathbf{A}$ and $\mathbf{b}$ are learnable parameters.

## Implementation

1. Open `mlp_block.py` and implement the `MLP` and `GLU` classes.
2. Each class should process tensors of shape `[batch_size, seq_len, hidden_dim]`.
3. Both accept parameters for:
   - `hidden_dim`: Input/output dimension
   - `intermediate_dim`: Hidden dimension
   - `activation_fn`: Activation function (default: GELU)
   - `dropout`: Dropout rate (default: 0.0)

## Testing

Run the tests with:
```bash
pytest tests/test_mlp.py
```

## Notes

- GLU layers have replaced MLPs in most non-Mixture of Experts architectures.

## If You Get Stuck

If needed, ask [ChatGPT](https://chatgpt.com) or [Claude](https://claude.ai) using the Socratic Prompt in `socratic_prompt.md`. Don't read it yourself - it's designed to guide you through discovery.