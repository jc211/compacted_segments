import numpy as np 
import warp as wp
from compacted_segments import CompactedSegments, CompactedSegmentsBuilder, CompactedSegmentSampler, CompositeSegmentSampler
import pytest

wp.init()

def test_builder_bad_input():
    with pytest.raises(AssertionError):
        device = "cuda:0"
        builder = CompactedSegmentsBuilder(device=device, normalize=False)
        builder.append(np.arange(10, dtype=np.float32))

def test_builder_good_input():
    device = "cuda:0"
    builder = CompactedSegmentsBuilder(device=device, normalize=False)
    builder.append(np.arange(10, dtype=np.float32).reshape(-1,1))

def test_sample():
    device = "cuda:0"
    builder = CompactedSegmentsBuilder(device=device, normalize=False)
    builder.append(np.arange(10, dtype=np.float32).reshape(-1,1))
    builder.append(np.arange(10, 18, dtype=np.float32).reshape(-1,1))
    builder.append(np.arange(30, 35, dtype=np.float32).reshape(-1,1))
    segments = builder.build()
    sampler = CompactedSegmentSampler(segments)
    res = sampler.sample(batch_size=4, subsegment_size=8)

def test_composite_sample():
    device = "cuda:0"
    builder_1 = CompactedSegmentsBuilder(device=device, normalize=False)
    builder_2 = CompactedSegmentsBuilder(device=device, normalize=False)

    builder_1.append(np.arange(10, dtype=np.float32).reshape(-1,1))
    builder_2.append(np.arange(10, dtype=np.float32).reshape(-1,1)*10)

    builder_1.append(np.arange(10, 18, dtype=np.float32).reshape(-1,1))
    builder_2.append(np.arange(10, 18, dtype=np.float32).reshape(-1,1)*10)

    builder_1.append(np.arange(30, 35, dtype=np.float32).reshape(-1,1))
    builder_2.append(np.arange(30, 35, dtype=np.float32).reshape(-1,1)*10)

    segments_1 = builder_1.build()
    segments_2 = builder_2.build()

    sampler = CompositeSegmentSampler(
        {
            "seg1": segments_1,
            "seg2": segments_2
        }
    )
    res = sampler.sample(batch_size=4, subsegment_sizes=[8, 5])

def test_normalize():
    device = "cuda:0"
    builder = CompactedSegmentsBuilder(device=device, normalize=True)
    builder.append(np.arange(10, dtype=np.float32).reshape(-1,1))
    builder.append(np.arange(10, 18, dtype=np.float32).reshape(-1,1))
    builder.append(np.arange(30, 35, dtype=np.float32).reshape(-1,1))
    segments = builder.build()
    assert segments.min_element == 0.0, f"Expected 10.0, got {segments.min_element}"
    assert segments.max_element == 34.0, f"Expected 34.0, got {segments.max_element}"
    assert segments.compacted_segments.min() == -1.0, f"Expected -1.0, got {segments.compacted_segments.min()}"
    assert segments.compacted_segments.max() == 1.0, f"Expected 1.0, got {segments.compacted_segments.max()}"

    segments = segments.unnormalize()
    assert segments.compacted_segments.min() == 0.0, f"Expected 0.0, got {segments.compacted_segments.min()}"
    assert segments.compacted_segments.max() == 34.0, f"Expected 34.0, got {segments.compacted_segments.max()}"
    
if __name__ == "__main__":
    pass

    
    