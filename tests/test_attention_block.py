import pytest
import torch

# Import student implementations
from exercises.attention_block.attention_block import (
    EagerBidirectionalAttentionBlock as StudentEagerBidirectional,
    EagerCausalAttentionBlock as StudentEagerCausal,
    SDPABidirectionalAttentionBlock as StudentSDPABidirectional,
    SDPACausalAttentionBlock as StudentSDPACausal,
    FlashBidirectionalAttentionBlock as StudentFlashBidirectional,
    FlashCausalAttentionBlock as StudentFlashCausal,
)

# Import reference implementations
from exercises.attention_block.solution.hidden.reference_attention_block import (
    EagerBidirectionalAttentionBlock as RefEagerBidirectional,
    EagerCausalAttentionBlock as RefEagerCausal,
    SDPABidirectionalAttentionBlock as RefSDPABidirectional,
    SDPACausalAttentionBlock as RefSDPACausal,
    FlashBidirectionalAttentionBlock as RefFlashBidirectional,
    FlashCausalAttentionBlock as RefFlashCausal,
)

# Test parameters
BATCH_SIZE = 4
SEQ_LEN = 16
NUM_HEADS = 8
HEAD_DIM = 32
HIDDEN_DIM = NUM_HEADS * HEAD_DIM
MAX_SEQ_LEN = 128  # For causal attention with a static mask
DROPOUT = 0.1
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
    """Generate random test data for attention blocks."""
    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    mask = None
    if with_mask:
        # Create a padding mask where the last few positions are masked
        # Simulate sequences of varying lengths by masking the last few positions
        mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        for i in range(batch_size):
            # Randomly choose a length for each sequence
            seq_length = torch.randint(low=1, high=seq_len, size=(1,)).item()
            mask[i, seq_length:] = False  # Mask the positions beyond the chosen length

    return x, mask


def generate_flash_attention_data(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM):
    """Generate test data specifically for Flash Attention blocks."""
    # Create standard input
    x, _ = generate_test_data(batch_size, seq_len, hidden_dim)

    # Reshape for Flash Attention: [batch_size, seq_len, hidden_dim] -> [batch_size * seq_len, hidden_dim]
    x_flash = x.reshape(-1, hidden_dim)

    # Create cumulative sequence lengths tensor
    cu_seqlens = torch.arange(0, (batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32, device=device)

    # Maximum sequence length
    max_seqlen = seq_len

    return x, x_flash, cu_seqlens, max_seqlen


@pytest.mark.parametrize("with_mask", [True, False], ids=lambda x: "with_mask" if x else "no_mask")
def test_eager_bidirectional_attention_block(with_mask):
    """Test student's EagerBidirectionalAttentionBlock implementation."""
    x, mask = generate_test_data(with_mask=with_mask)

    random_seed = torch.randint(0, 100000, size=(1,)).item()

    try:
        # Initialize student and reference models
        student_model = StudentEagerBidirectional(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)
        ref_model = RefEagerBidirectional(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)

        # Copy weights from reference to student model to ensure identical initialization
        student_model.load_state_dict(ref_model.state_dict())

        # Forward pass
        with torch.no_grad():
            torch.manual_seed(random_seed)
            student_output = student_model(x, mask)
            torch.manual_seed(random_seed)
            ref_output = ref_model(x, mask)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's EagerBidirectionalAttentionBlock output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's EagerBidirectionalAttentionBlock not implemented yet")
    except AttributeError as e:
        pytest.fail(f"Attribute error in EagerBidirectionalAttentionBlock: {e}")


@pytest.mark.parametrize("with_mask", [True, False], ids=lambda x: "with_mask" if x else "no_mask")
def test_eager_causal_attention_block(with_mask):
    """Test student's EagerCausalAttentionBlock implementation."""
    x, mask = generate_test_data(with_mask=with_mask)

    random_seed = torch.randint(0, 100000, size=(1,)).item()

    try:
        # Initialize student and reference models
        student_model = StudentEagerCausal(HIDDEN_DIM, NUM_HEADS, DROPOUT, MAX_SEQ_LEN).to(device)
        ref_model = RefEagerCausal(HIDDEN_DIM, NUM_HEADS, DROPOUT, MAX_SEQ_LEN).to(device)

        # Copy weights from reference to student model to ensure identical initialization
        student_model.load_state_dict(ref_model.state_dict())

        # Forward pass
        with torch.no_grad():
            torch.manual_seed(random_seed)
            student_output = student_model(x, mask)
            torch.manual_seed(random_seed)
            ref_output = ref_model(x, mask)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's EagerCausalAttentionBlock output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's EagerCausalAttentionBlock not implemented yet")
    except AttributeError as e:
        pytest.fail(f"Attribute error in EagerCausalAttentionBlock: {e}")


@pytest.mark.parametrize("with_mask", [True, False], ids=lambda x: "with_mask" if x else "no_mask")
def test_sdpa_bidirectional_attention_block(with_mask):
    """Test student's SDPABidirectionalAttentionBlock implementation."""
    x, mask = generate_test_data(with_mask=with_mask)

    random_seed = torch.randint(0, 100000, size=(1,)).item()

    try:
        # Initialize student and reference models
        student_model = StudentSDPABidirectional(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)
        ref_model = RefSDPABidirectional(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)

        # Copy weights from reference to student model to ensure identical initialization
        student_model.load_state_dict(ref_model.state_dict())

        # Forward pass
        with torch.no_grad():
            torch.manual_seed(random_seed)
            student_output = student_model(x, mask)
            torch.manual_seed(random_seed)
            ref_output = ref_model(x, mask)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's SDPABidirectionalAttentionBlock output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's SDPABidirectionalAttentionBlock not implemented yet")
    except AttributeError as e:
        pytest.fail(f"Attribute error in SDPABidirectionalAttentionBlock: {e}")


@pytest.mark.parametrize("with_mask", [True, False], ids=lambda x: "with_mask" if x else "no_mask")
def test_sdpa_causal_attention_block(with_mask):
    """Test student's SDPACausalAttentionBlock implementation."""
    x, mask = generate_test_data(with_mask=with_mask)

    random_seed = torch.randint(0, 100000, size=(1,)).item()

    try:
        # Initialize student and reference models
        student_model = StudentSDPACausal(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)
        ref_model = RefSDPACausal(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)

        # Copy weights from reference to student model to ensure identical initialization
        student_model.load_state_dict(ref_model.state_dict())

        # Forward pass
        with torch.no_grad():
            torch.manual_seed(random_seed)
            student_output = student_model(x, mask)
            torch.manual_seed(random_seed)
            ref_output = ref_model(x, mask)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's SDPACausalAttentionBlock output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's SDPACausalAttentionBlock not implemented yet")
    except AttributeError as e:
        pytest.fail(f"Attribute error in SDPACausalAttentionBlock: {e}")


def test_flash_bidirectional_attention_block():
    """Test student's FlashBidirectionalAttentionBlock implementation."""
    # Generate data for both standard and Flash Attention formats
    x, x_flash, cu_seqlens, max_seqlen = generate_flash_attention_data()

    random_seed = torch.randint(0, 100000, size=(1,)).item()

    try:
        # Initialize student and reference models
        student_model = StudentFlashBidirectional(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)
        ref_model = RefFlashBidirectional(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)

        # Copy weights from reference to student model to ensure identical initialization
        student_model.load_state_dict(ref_model.state_dict())

        # Forward pass
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            torch.manual_seed(random_seed)
            student_output = student_model(x_flash, cu_seqlens, max_seqlen)
            torch.manual_seed(random_seed)
            ref_output = ref_model(x_flash, cu_seqlens, max_seqlen)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's FlashBidirectionalAttentionBlock output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's FlashBidirectionalAttentionBlock not implemented yet")
    except ImportError:
        pytest.skip("Flash Attention is not available")
    except AttributeError as e:
        pytest.fail(f"Attribute error in FlashBidirectionalAttentionBlock: {e}")


def test_flash_causal_attention_block():
    """Test student's FlashCausalAttentionBlock implementation."""
    # Generate data for both standard and Flash Attention formats
    x, x_flash, cu_seqlens, max_seqlen = generate_flash_attention_data()

    random_seed = torch.randint(0, 100000, size=(1,)).item()

    try:
        # Initialize student and reference models
        student_model = StudentFlashCausal(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)
        ref_model = RefFlashCausal(HIDDEN_DIM, NUM_HEADS, DROPOUT).to(device)

        # Copy weights from reference to student model to ensure identical initialization
        student_model.load_state_dict(ref_model.state_dict())

        # Forward pass
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            torch.manual_seed(random_seed)
            student_output = student_model(x_flash, cu_seqlens, max_seqlen)
            torch.manual_seed(random_seed)
            ref_output = ref_model(x_flash, cu_seqlens, max_seqlen)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's FlashCausalAttentionBlock output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's FlashCausalAttentionBlock not implemented yet")
    except ImportError:
        pytest.skip("Flash Attention is not available")
    except AttributeError as e:
        pytest.fail(f"Attribute error in FlashCausalAttentionBlock: {e}")
