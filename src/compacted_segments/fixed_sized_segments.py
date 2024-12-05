import torch

from .space import Normalizer, Space

class FixedSizedSegments:
    def __init__(self, data: torch.Tensor, normalized: bool, normalizer: Normalizer):
        assert data.dim() == 3, f"Expected 3 (B, T, I), got {data.dim()}"
        self.data = data
        self._normalized = normalized
        self.normalizer = normalizer
        self._batch_size = data.shape[0]
        self._segment_size = data.shape[1]  
        self._element_dim = data.shape[2]
        assert self.normalizer is not None, "No normalizer found"

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def normalized(self):
        return self._normalized
    
    @property
    def segment_size(self):
        return self._segment_size
    
    @property
    def element_dim(self):
        return self._element_dim
    
    @property
    def device(self):
        return str(self.data.device)
    
    @property
    def shape(self):
        return self.data.shape
    
    def copy(self) -> "FixedSizedSegments":
        return FixedSizedSegments(
            data=self.data.clone(),
            normalized=self._normalized,
            normalizer=self.normalizer
        )
    
    def normalize(self) -> "FixedSizedSegments":
        if self._normalized:
            return self
        c = self.copy()
        c.data = c.normalizer.normalize(c.data) 
        c._normalized = True
        return c
    
    def unnormalize(self) -> "FixedSizedSegments":
        if not self._normalized:
            return self
        c = self.copy()
        c.data = c.normalizer.unnormalize(c.data)
        c._normalized = False
        return c
    
    def space(self) -> Space:
        return Space(self.element_dim, self.normalizer)
    
    def __repr__(self) -> str:
        return f"FixedSizedSegments(B={self.batch_size}, T={self.segment_size}, I={self.element_dim}, data={self.data})"