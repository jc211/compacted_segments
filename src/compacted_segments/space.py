
from typing import Optional 
from dataclasses import dataclass

import torch

class Normalizer:

    def __init__(self, min: torch.Tensor, max: torch.Tensor):
        self.min = min
        self.max = max
    
    def normalize(self, x: torch.Tensor):
        ndata = (x - self.min) / (self.max - self.min)
        ndata = ndata * 2 - 1
        return ndata
    
    def unnormalize(self, x: torch.Tensor):
        ndata = (x + 1) / 2
        ndata = ndata * (self.max - self.min) + self.min
        return ndata
    
    def __eq__(self, other: "Normalizer"):
        return torch.allclose(self.min, other.min) and torch.allclose(self.max, other.max)

@dataclass
class Space:
    dim: int
    normalizer : Optional[Normalizer] = None

    def __eq__(self, other: "Space"):
        return self.dim == other.dim # and self.normalizer == other.normalizer # Removed because it can't be in a cuda graph
