import numpy as np
import torch
import warp as wp

from .space import Normalizer, Space
from .fixed_sized_segments import FixedSizedSegments

class CompactedSegments:
    """
    Stores data of variable length in a compacted form


    |  s0  |    s1     | ... |    s S-1    |
    |eeeeee|eeeeeeeeeee| ... |eeeeeeeeeeeee|

    | start_0, length_0 | start_1, length_1 | ... | start_S-1, length_S-1 |
    """
    def __init__(self):
        self.device = None
        self._num_segments = 0 # S
        self._num_elements = 0 # N
        self._compacted_segments = torch.tensor([], dtype=torch.float32) # (E, D)
        self._segment_start_and_length = torch.tensor([], dtype=torch.int32) # (S, 2)
        self._segment_ids = torch.tensor([], dtype=torch.int32) # (E,)
        self._min_element = None
        self._max_element = None
        self._normalized = False

    @property
    def compacted_segments(self):
        return self._compacted_segments

    @property
    def start_and_length(self):
        return self._segment_start_and_length
    
    @property
    def segment_ids(self):
        return self._segment_ids
    
    @property
    def compacted_segments_warp(self):
        return wp.from_torch(self._compacted_segments, requires_grad=False)
    
    @property
    def segment_start_and_length_warp(self):
        return wp.from_torch(self._segment_start_and_length, requires_grad=False)
    
    @property
    def segment_ids_warp(self):
        return wp.from_torch(self._segment_ids, requires_grad=False)
    
    @property
    def num_segments(self):
        return self._num_segments

    @property
    def num_elements(self):
        return self._num_elements
    
    @property
    def dim_element(self):
        return self._compacted_segments.shape[-1]
    
    @property
    def min_element(self):
        return self._min_element
    
    @property
    def max_element(self):
        return self._max_element
    
    @property
    def normalized(self):
        return self._normalized
    
    def get_segment(self, segment_id: int):
        assert segment_id < self._num_segments, f"Segment {segment_id} does not exist - only {self._num_segments} segments"
        start, length = self._segment_start_and_length[segment_id]
        res = FixedSizedSegments(
            data=self._compacted_segments[start:start+length].view(1, length, self.dim_element),
            normalized=self._normalized,
            normalizer=self.get_normalizer()
        )
        return res
    
    def get_normalizer(self):
        return Normalizer(self._min_element, self._max_element)
    
    def copy(self):
        res = CompactedSegments()
        res.device = self.device
        res._num_segments = self._num_segments
        res._num_elements = self._num_elements
        res._compacted_segments = self._compacted_segments.clone()
        res._segment_start_and_length = self._segment_start_and_length.clone()
        res._segment_ids = self._segment_ids.clone()
        res._min_element = self._min_element.clone()
        res._max_element = self._max_element.clone()
        res._normalized = self._normalized
        return res
    
    def normalize(self):
        if self._normalized:
            return self
        normalizer = Normalizer(self._min_element, self._max_element)
        res = self.copy()
        res._compacted_segments = normalizer.normalize(res._compacted_segments)
        res._normalized = True
        return res
    
    def unnormalize(self):
        if not self._normalized:
            return self
        normalizer = Normalizer(self._min_element, self._max_element)
        res = self.copy()
        res._compacted_segments = normalizer.unnormalize(res._compacted_segments)
        res._normalized = False
        return res
    
    def space(self):
        return Space(self.dim_element, self.get_normalizer())
    
    def __repr__(self) -> str:
        return f"CompactedSegments(num_segments={self._num_segments}, num_elements={self._num_elements}, dim_element={self.dim_element}, normalized={self._normalized})"
    

class CompactedSegmentsBuilder:
    """
    Used to build CompactedSegments
    """

    def __init__(self, device:str, normalize: bool = False):
        self.device = device
        self._temp_segments = []
        self._temp_last_start = 0
        self._temp_start_and_length = []
        self._temp_segment_id = []
        self._num_segments = 0 # S
        self._num_elements = 0 # N
        self._normalize = normalize
        self._built = False

    def append(self, segment: np.ndarray):
        assert not self._built, "Already built"
        assert segment.ndim == 2, "Expected 2D array, got {segment.ndim}D - Must be (T, D)"
        self._temp_start_and_length.append((self._temp_last_start, len(segment)))
        self._temp_segments.append(segment)
        self._temp_last_start += len(segment)
        self._temp_segment_id.extend([self._num_segments] * len(segment))
        self._num_segments += 1
        self._num_elements += len(segment)
    
    def build(self) -> CompactedSegments:
        assert not self._built, "Already built"
        assert self._num_segments > 0, "No segments added"
        self._temp_segments = np.concatenate(self._temp_segments, axis=0)
        res = CompactedSegments()
        res.device = self.device
        res._compacted_segments = torch.from_numpy(self._temp_segments).to(self.device)
        res._segment_start_and_length = torch.tensor(self._temp_start_and_length, dtype=torch.int32, device=self.device)
        res._segment_ids = torch.tensor(self._temp_segment_id, dtype=torch.int32, device=self.device)
        res._min_element = torch.min(res._compacted_segments, dim=0).values
        res._max_element = torch.max(res._compacted_segments, dim=0).values
        res._num_segments = self._num_segments
        res._num_elements = self._num_elements
        if self._normalize:
            normalizer = Normalizer(res._min_element, res._max_element)
            res._compacted_segments = normalizer.normalize(res._compacted_segments)
            res._normalized = True
        self._built = True
        return res
    
