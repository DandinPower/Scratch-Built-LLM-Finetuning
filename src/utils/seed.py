import torch
import numpy as np
import random

def set_seed(seed: int):
    """
    Set the seed for all components involved in PyTorch-related computations.
    
    Args:
        seed (int): The seed value to set.
    """
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    
    # Set the seed for CUDA (GPU operations)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmarking for reproducibility
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for Python's random module
    random.seed(seed)