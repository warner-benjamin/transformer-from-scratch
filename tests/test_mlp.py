import pytest
import torch
import torch.nn.functional as F
import torch.nn as nn

# Import student implementations
from exercises.mlp_layer.mlp import (
    MLP as StudentMLP,
    GLU as StudentGLU,
)

# Import reference implementations
from exercises.mlp_layer.solution.hidden.reference_mlp import (
    MLP as RefMLP,
    GLU as RefGLU,
)

# Test parameters
BATCH_SIZE = 4
SEQ_LEN = 16
HIDDEN_DIM = 128
INTERMEDIATE_DIM = 384  # 3x expansion for testing
RTOL = 1e-3
ATOL = 1e-3

# Determine the device to use
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def generate_test_data(batch_size=BATCH_SIZE, seq_len=SEQ_LEN, hidden_dim=HIDDEN_DIM):
    """Generate random test data for MLP blocks."""
    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    return x


@pytest.mark.parametrize("activation", [nn.GELU, nn.ReLU, nn.SiLU], ids=["gelu", "relu", "silu"])
@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.2])
def test_mlp_block(activation, dropout):
    """Test student's MLP implementation."""
    x = generate_test_data()

    random_seed = torch.randint(0, 100000, size=(1,)).item()

    try:
        # Initialize student and reference
        student_model = StudentMLP(HIDDEN_DIM, INTERMEDIATE_DIM, activation, dropout).to(device)
        ref_model = RefMLP(HIDDEN_DIM, INTERMEDIATE_DIM, activation, dropout).to(device)

        # Copy weights from reference to student model to ensure identical initialization
        student_model.load_state_dict(ref_model.state_dict())

        # Forward pass
        with torch.no_grad():
            torch.manual_seed(random_seed)
            student_output = student_model(x)
            torch.manual_seed(random_seed)
            ref_output = ref_model(x)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's MLP output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's MLP not implemented yet")
    except AttributeError as e:
        pytest.fail(f"Attribute error in MLP: {e}")


@pytest.mark.parametrize("activation", [nn.GELU, nn.ReLU, nn.SiLU], ids=["gelu", "relu", "silu"])
@pytest.mark.parametrize("dropout", [0.0, 0.1, 0.2])
def test_glu_block(activation, dropout):
    """Test student's GLU implementation."""
    x = generate_test_data()

    random_seed = torch.randint(0, 100000, size=(1,)).item()

    try:
        # Initialize student and reference models
        student_model = StudentGLU(HIDDEN_DIM, INTERMEDIATE_DIM, activation, dropout).to(device)
        ref_model = RefGLU(HIDDEN_DIM, INTERMEDIATE_DIM, activation, dropout).to(device)

        # Copy weights from reference to student model to ensure identical initialization
        student_model.load_state_dict(ref_model.state_dict())

        # Forward pass
        with torch.no_grad():
            torch.manual_seed(random_seed)
            student_output = student_model(x)
            torch.manual_seed(random_seed)
            ref_output = ref_model(x)

        assert torch.allclose(student_output, ref_output, rtol=RTOL, atol=ATOL), (
            "Student's GLU output doesn't match reference implementation"
        )
    except NotImplementedError:
        pytest.skip("Student's GLU not implemented yet")
    except AttributeError as e:
        pytest.fail(f"Attribute error in GLU: {e}")
