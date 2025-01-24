import torch

class ScratchEmbedding:
    """
    Custom embedding layer with manual backward pass support.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, weight: torch.Tensor, device: str) -> None:
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = weight.detach().to(device)
        self.weight.requires_grad = False
        self.grad = torch.zeros_like(self.weight)
        self.activation = None
        self.accumulate_flops = False  # If True, the FLOPs accumulate counter won't reset
        self.fwd_flops = 0
        self.bwd_flops = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.
        """
        if not self.accumulate_flops:
            self.fwd_flops = 0
            
        # Lookup operation
        self.activation = inputs.detach()
        output = self.weight[inputs]
        return output

    def backward(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Backward pass for the custom embedding layer.
        """
        if not self.accumulate_flops:
            self.bwd_flops = 0
        self.bwd_flops += self.activation.view(-1).numel() * self.embedding_dim
        
        self.grad.index_add_(dim=0, index=self.activation.view(-1), source=grads.view(-1, self.embedding_dim))
        self.activation = None  # clear the activation after backward propagation
        
    def get_fwd_flops(self) -> int:
        """
        Returns the number of FLOPs for the forward pass.
        """
        return self.fwd_flops
    
    def get_bwd_flops(self) -> int:
        """
        Returns the number of FLOPs for the backward pass.
        """
        return self.bwd_flops


if __name__ == "__main__":
    from ..utils.seed import set_seed
    set_seed(42)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    VOCAB_SIZE = 1024
    HIDDEN_SIZE = 4096

    # Create a PyTorch embedding layer and move it to the GPU
    torch_embedding = torch.nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=HIDDEN_SIZE).to(device)
    randn_initialized_weight = torch_embedding.weight.detach()

    # Create custom embedding layer and move it to the GPU
    scratch_embedding = ScratchEmbedding(
        num_embeddings=VOCAB_SIZE,
        embedding_dim=HIDDEN_SIZE,
        weight=randn_initialized_weight,
        device=device
    )
    
    input_tensor = torch.tensor([[0, 1, 2],], dtype=torch.long).to(device)
    other_input_tensor = input_tensor.detach()

    result = torch_embedding(input_tensor)
    other_result = scratch_embedding.forward(other_input_tensor)

    target = torch.randn(result.shape, device=device)
    other_target = target.detach()

    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(result, target)
    loss.backward()
    
    # Check the gradients of the embedding layer
    print("[Torch] Gradients of the embedding layer:")
    print(torch_embedding.weight.grad[0])

    difference = other_result.detach() - other_target.detach()
    N = other_result[0].numel()  # Total number of elements in the result tensor of one batch
    other_result_grad = difference * (2 / N)  # Gradient of loss w.r.t. result
    scratch_embedding.backward(other_result_grad)
    
    # Check the gradients of the embedding layer
    print("[Scratch] Gradients of the embedding layer:")
    print(scratch_embedding.grad[0])
    
    print(scratch_embedding.get_fwd_flops())
    print(scratch_embedding.get_bwd_flops())