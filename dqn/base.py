import torch

_GLOBAL_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
