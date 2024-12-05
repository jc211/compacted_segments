from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
import warp as wp
import torch

from .compacted_segments import CompactedSegments
from .fixed_sized_segments import FixedSizedSegments

@dataclass
class SamplerResult:
    subsegments: FixedSizedSegments
    ids: torch.Tensor

class CompactedSegmentSampler:
    def __init__(self, segments: CompactedSegments):
        self._compacted_segments = segments
        self._random_indeces = None
        self._fixed_size_subsegments : Optional[FixedSizedSegments] = None
        self._gather_subsegments_kernel = create_gather_subsegments_kernel(segments.dim_element)
    
    @property
    def device(self):
        return self._compacted_segments.device
    
    @property
    def normalized(self):
        return self._compacted_segments._normalized
    
    @property
    def batch_size(self):
        return self._fixed_size_subsegments.batch_size if self._fixed_size_subsegments is not None else 0
    
    @property
    def subsegment_size(self):
        return self._fixed_size_subsegments.segment_size if self._fixed_size_subsegments is not None else 0
    
    def sample(self, batch_size: int = 32, subsegment_size: int = 16) -> SamplerResult:
        self.reserve_memory(batch_size, subsegment_size)
        inds = self.sample_indeces(batch_size)
        return self.gather_subsegments(inds)

    def sample_indeces(self, batch_size: int = 32):
        assert self._fixed_size_subsegments is not None, "Memory not reserved. Call reserve_memory() first"
        assert self.batch_size == batch_size, f"Memory reserved for {self.batch_size}, got {batch_size}. Call reserve_memory() first"

        self._random_indeces.random_(0, self._compacted_segments.num_elements)
        self._fix_indices()
        return self._random_indeces
    
    def get_normalizer(self):
        return self._compacted_segments.get_normalizer()
    
    def gather_subsegments(self, indeces: torch.Tensor) -> SamplerResult:
        assert self._fixed_size_subsegments is not None, "Memory not reserved. Call reserve_memory() first"
        assert indeces.shape[0] == self.batch_size, f"Memory reserved for {self.batch_size}, got {indeces.shape[0]}. Call reserve_memory() first"

        random_indeces_warp = wp.from_torch(indeces, requires_grad=False)
        _sampled_subsegments_warp = wp.from_torch(self._fixed_size_subsegments.data, requires_grad=False)
        wp.launch(
            kernel=self._gather_subsegments_kernel,
            dim=(self.batch_size, self.subsegment_size),
            inputs=[
                random_indeces_warp,
                self._compacted_segments.compacted_segments_warp,
                _sampled_subsegments_warp
            ],
            device=self.device
        )
        return SamplerResult(
            subsegments=self._fixed_size_subsegments,
            ids=self._subsegment_ids
        )

    def reserve_memory(self, batch_size: int, subsegment_size: int):
        assert batch_size > 0, "Batch size must be positive"
        assert subsegment_size > 0, "Segment size must be positive"
        if self.batch_size == batch_size and self.subsegment_size == subsegment_size:
            return

        _sampled_subsegments = torch.zeros(
            (batch_size, subsegment_size, self._compacted_segments.dim_element),
            dtype=torch.float32,
            device=self.device
        )
        self._subsegment_ids = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=self.device
        )

        self._random_indeces = torch.zeros(
            (batch_size,),
            dtype=torch.int32,
            device=self.device
        )

        self._fixed_size_subsegments = FixedSizedSegments(
            data=_sampled_subsegments,
            normalized=self._compacted_segments.normalized,
            normalizer=self._compacted_segments.get_normalizer()
        )

    def _fix_indices(self):
        subsegment_ids_warp = wp.from_torch(self._subsegment_ids, requires_grad=False)
        random_indeces_warp = wp.from_torch(self._random_indeces, requires_grad=False)
        wp.launch(
            kernel=fix_indices_kernel,
            dim=(self.batch_size, ),
            inputs=[
                self.subsegment_size,
                self._compacted_segments.segment_ids_warp,
                self._compacted_segments.segment_start_and_length_warp,
                subsegment_ids_warp,
                random_indeces_warp,
            ],
            device=self.device
        )


class CompositeSegmentSampler:
    def __init__(self, segments: dict[str, CompactedSegments]):
        assert len(segments) > 0, "No segments provided"
        first_segment = list(segments.values())[0]  
        for _, s in segments.items():
            assert s.device == first_segment.device, "All segments must be on the same device"
            assert s.num_segments == first_segment.num_segments, "All segments must have the same number of segments"
        self.segments = segments
        self.samplers = {k: CompactedSegmentSampler(s) for k, s in segments.items()}
        self.names = list(segments.keys())
    
    def sample(self, batch_size: int = 32, subsegment_sizes: list[int] | int = 16) -> dict[str, SamplerResult]:
        if isinstance(subsegment_sizes, int):
            subsegment_sizes = [subsegment_sizes] * len(self.names)
        
        for i, name in enumerate(self.names):
            self.samplers[name].reserve_memory(batch_size, subsegment_sizes[i])
        
        first_sampler = self.samplers[self.names[0]]
        inds = first_sampler.sample_indeces(batch_size)
        res = {}
        for name, s in self.samplers.items():
            res[name] = s.gather_subsegments(inds)
        return res

def create_gather_subsegments_kernel(action_dim: int):
    A = wp.constant(action_dim)

    @wp.kernel
    def gather_subsegments( # one thread per action sampled
        random_indeces: wp.array(dtype=wp.int32), # (B,)
        segments: wp.array2d(dtype=wp.float32), # (E, A)
        subsegments: wp.array3d(dtype=wp.float32), # (E, S, A)
    ):
        bi, si = wp.tid() # batch index, subsegment index
        ri = random_indeces[bi] # random index
        current_random_action_ind = ri + si
        for i in range(A):
            subsegments[bi, si, i] = segments[current_random_action_ind, i]

    return gather_subsegments


@wp.kernel
def fix_indices_kernel(
    segment_size: wp.int32,
    segment_ids: wp.array(dtype=wp.int32), # (E,)
    segment_start_and_length: wp.array2d(dtype=wp.int32), # (S, 2)
    subsegment_ids: wp.array(dtype=wp.int32), # (B, )
    random_inds: wp.array(dtype=wp.int32), # (B,)
):
    """
    Runs one thread per sample in the batch and corrects the random inds generated so they do not lie on the edge of the segment
    """
    bi = wp.tid() # batch index
    ri = random_inds[bi] # random index
    si = segment_ids[ri] # segment index

    start_index = segment_start_and_length[si][0]
    num_elements = segment_start_and_length[si][1]

    subsegment_ids[bi] = si

    if ri >= start_index + num_elements - segment_size:
        res = start_index + num_elements - segment_size - 1
        random_inds[bi] = res

