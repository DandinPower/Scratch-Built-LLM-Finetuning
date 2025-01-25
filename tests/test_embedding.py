import torch
import pytest
from src.ops.embedding import ScratchEmbedding

# Set up common parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define parameter combinations for testing
VOCAB_SIZES = [1024, 4094, 16384]
HIDDEN_SIZES = [4096, 8192]
BATCH_SIZES = [1, 8]
SEQUENCE_LENGTHS = [16, 512, 8192]

@pytest.fixture
def setup_embedding_layers(request):
    vocab_size, hidden_size = request.param

    # Create a PyTorch embedding layer
    torch_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size).to(DEVICE)
    randn_initialized_weight = torch_embedding.weight.detach()

    # Create custom embedding layer
    scratch_embedding = ScratchEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=hidden_size,
        weight=randn_initialized_weight,
        device=DEVICE
    )

    return torch_embedding, scratch_embedding

@pytest.mark.parametrize("setup_embedding_layers", [(vocab_size, hidden_size) for vocab_size in VOCAB_SIZES for hidden_size in HIDDEN_SIZES], indirect=True)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("sequence_length", SEQUENCE_LENGTHS)
def test_forward_pass(setup_embedding_layers, batch_size, sequence_length):
    torch_embedding, scratch_embedding = setup_embedding_layers

    # Create input tensor
    input_tensor = torch.randint(0, torch_embedding.num_embeddings, (batch_size, sequence_length), dtype=torch.long).to(DEVICE)

    # Forward pass
    torch_output = torch_embedding(input_tensor)
    scratch_output = scratch_embedding.forward(input_tensor)

    # Check if outputs are the same
    assert torch.allclose(torch_output, scratch_output), "Forward pass outputs do not match!"

@pytest.mark.parametrize("setup_embedding_layers", [(vocab_size, hidden_size) for vocab_size in VOCAB_SIZES for hidden_size in HIDDEN_SIZES], indirect=True)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("sequence_length", SEQUENCE_LENGTHS)
def test_backward_pass(setup_embedding_layers, batch_size, sequence_length):

    torch_embedding, scratch_embedding = setup_embedding_layers

    # Create input tensor
    input_tensor = torch.randint(0, torch_embedding.num_embeddings, (batch_size, sequence_length), dtype=torch.long).to(DEVICE)

    # Forward pass
    torch_output = torch_embedding(input_tensor)
    scratch_output = scratch_embedding.forward(input_tensor.detach().clone())

    # Create a target tensor
    target = torch.randn(torch_output.shape, device=DEVICE)

    # Compute loss and backward pass for PyTorch embedding
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(torch_output, target)
    loss.backward()

    # Compute loss and backward pass for ScratchEmbedding
    difference = scratch_output - target
    N = scratch_output.numel()  # Total number of elements in the result tensor
    scratch_grad = difference * (2 / N)  # Gradient of loss w.r.t. result
    scratch_embedding.backward(scratch_grad)

    # Check if gradients are the same
    assert torch.allclose(torch_embedding.weight.grad, scratch_embedding.grad), "Backward pass gradients do not match!"

if __name__ == "__main__":
    pytest.main()