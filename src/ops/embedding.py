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