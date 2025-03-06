import pytest
import torch

# Import student implementations
from attention import (
    eager_bidirectional_attention as student_eager_bidirectional,
    eager_causal_attention as student_eager_causal,
    sdp_bidirectional_attention as student_sdp_bidirectional,
    sdp_causal_attention as student_sdp_causal,
    flash_bidirectional_attention as student_flash_bidirectional,
    flash_causal_attention as student_flash_causal,
)

# Import reference implementations
from solution.hidden.reference_attention import (
    eager_bidirectional_attention as ref_eager_bidirectional,
    eager_causal_attention as ref_eager_causal,
    sdp_bidirectional_attention as ref_sdp_bidirectional,
    sdp_causal_attention as ref_sdp_causal,
    flash_bidirectional_attention as ref_flash_bidirectional,
    flash_causal_attention as ref_flash_causal,
)

# Set random seed for reproducibility
torch.manual_seed(42)

# Test parameters
BATCH_SIZE = 4
SEQ_LEN = 16
NUM_HEADS = 8
HEAD_DIM = 32
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
RTOL = 1e-3
ATOL = 1e-3

# Determine the device to use
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def generate_test_data(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM, with_mask=False):
    """Generate random test data for attention functions."""
    q = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    k = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    v = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    mask = None
    if with_mask:
        # Create a padding mask where the last few positions are masked
        # Simulate sequences of varying lengths by masking the last few positions
        mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
        for i in range(batch_size):
            # Randomly choose a length for each sequence
            seq_length = torch.randint(low=1, high=seq_len, size=(1,)).item()
            mask[i, seq_length:] = True  # Mask the positions beyond the chosen length

    return q, k, v, mask


def generate_flash_attention_data(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM):
    """Generate test data specifically for Flash Attention."""
    # Create standard inputs
    q, k, v, _ = generate_test_data(batch_size, seq_len, hidden_dim)

    # Reshape for Flash Attention: [batch_size, seq_len, hidden_dim] -> [batch_size * seq_len, hidden_dim]
    q_flash = q.reshape(-1, hidden_dim)
    k_flash = k.reshape(-1, hidden_dim)
    v_flash = v.reshape(-1, hidden_dim)

    # Create cumulative sequence lengths tensor
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)

    # Maximum sequence length
    max_seqlen = seq_len

    return q, k, v, q_flash, k_flash, v_flash, cu_seqlens, max_seqlen


def test_eager_bidirectional_attention():
    """Test student's eager bidirectional attention implementation."""
    q, k, v, mask = generate_test_data(with_mask=True)

    try:
        student_output = student_eager_bidirectional(q, k, v, NUM_HEADS, HEAD_DIM, mask)
        ref_output = ref_eager_bidirectional(q, k, v, NUM_HEADS, HEAD_DIM, mask)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's eager bidirectional attention output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's eager bidirectional attention not implemented yet")


def test_eager_causal_attention():
    """Test student's eager causal attention implementation."""
    q, k, v, mask = generate_test_data(with_mask=True)

    try:
        student_output = student_eager_causal(q, k, v, NUM_HEADS, HEAD_DIM, mask)
        ref_output = ref_eager_causal(q, k, v, NUM_HEADS, HEAD_DIM, mask)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's eager causal attention output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's eager causal attention not implemented yet")


def test_sdp_bidirectional_attention():
    """Test student's SDPA bidirectional attention implementation."""
    q, k, v, mask = generate_test_data(with_mask=True)

    try:
        student_output = student_sdp_bidirectional(q, k, v, NUM_HEADS, HEAD_DIM, mask)
        ref_output = ref_sdp_bidirectional(q, k, v, NUM_HEADS, HEAD_DIM, mask)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's SDPA bidirectional attention output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's SDPA bidirectional attention not implemented yet")


def test_sdp_causal_attention():
    """Test student's SDPA causal attention implementation."""
    q, k, v, mask = generate_test_data(with_mask=True)

    try:
        student_output = student_sdp_causal(q, k, v, NUM_HEADS, HEAD_DIM, mask)
        ref_output = ref_sdp_causal(q, k, v, NUM_HEADS, HEAD_DIM, mask)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's SDPA causal attention output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's SDPA causal attention not implemented yet")


def test_flash_bidirectional_attention():
    """Test student's Flash bidirectional attention implementation."""
    # Generate data for both standard and Flash Attention formats
    q, k, v, q_flash, k_flash, v_flash, cu_seqlens, max_seqlen = generate_flash_attention_data()

    try:
        # Call student implementation with Flash Attention format
        student_output = student_flash_bidirectional(q_flash, k_flash, v_flash, NUM_HEADS, HEAD_DIM, cu_seqlens, max_seqlen)

        # Call reference implementation with Flash Attention format
        ref_output = ref_flash_bidirectional(q_flash, k_flash, v_flash, NUM_HEADS, HEAD_DIM, cu_seqlens, max_seqlen)

        # Reshape outputs back to standard format for comparison if needed
        if student_output.shape != ref_output.shape:
            student_output = student_output.reshape(BATCH_SIZE, SEQ_LEN, -1)
            ref_output = ref_output.reshape(BATCH_SIZE, SEQ_LEN, -1)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's Flash bidirectional attention output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's Flash bidirectional attention not implemented yet")
    except ImportError:
        pytest.skip("Flash Attention is not available")


def test_flash_causal_attention():
    """Test student's Flash causal attention implementation."""
    # Generate data for both standard and Flash Attention formats
    q, k, v, q_flash, k_flash, v_flash, cu_seqlens, max_seqlen = generate_flash_attention_data()

    try:
        # Call student implementation with Flash Attention format
        student_output = student_flash_causal(q_flash, k_flash, v_flash, NUM_HEADS, HEAD_DIM, cu_seqlens, max_seqlen)

        # Call reference implementation with Flash Attention format
        ref_output = ref_flash_causal(q_flash, k_flash, v_flash, NUM_HEADS, HEAD_DIM, cu_seqlens, max_seqlen)

        # Reshape outputs back to standard format for comparison if needed
        if student_output.shape != ref_output.shape:
            student_output = student_output.reshape(BATCH_SIZE, SEQ_LEN, -1)
            ref_output = ref_output.reshape(BATCH_SIZE, SEQ_LEN, -1)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's Flash causal attention output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's Flash causal attention not implemented yet")
    except ImportError:
        pytest.skip("Flash Attention is not available")


# Additional tests with no mask
def test_eager_bidirectional_attention_no_mask():
    """Test student's eager bidirectional attention implementation without a mask."""
    q, k, v, _ = generate_test_data(with_mask=False)

    try:
        student_output = student_eager_bidirectional(q, k, v, NUM_HEADS, HEAD_DIM)
        ref_output = ref_eager_bidirectional(q, k, v, NUM_HEADS, HEAD_DIM)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's eager bidirectional attention output (no mask) doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's eager bidirectional attention not implemented yet")


def test_eager_causal_attention_no_mask():
    """Test student's eager causal attention implementation without a mask."""
    q, k, v, _ = generate_test_data(with_mask=False)

    try:
        student_output = student_eager_causal(q, k, v, NUM_HEADS, HEAD_DIM)
        ref_output = ref_eager_causal(q, k, v, NUM_HEADS, HEAD_DIM)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's eager causal attention output (no mask) doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's eager causal attention not implemented yet")
