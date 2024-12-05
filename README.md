# Compacted Segments
Provides a means of storing variable length data such as (T1, D), (T2, D), (T3, D). It also provides a sampler that can output (F, D) where F is a fixed dimension.

<div style="display:flex">
<img src="assets/compacted_data.png" style="margin-right: 16px; width:65%"/>
<img src="assets/fixed_sized_data.png" style="width:30%"/>
</div>



# Usage

```python
from compacted_segments import CompactedSegmentsBuilder, CompactedSegmentSampler, CompositeSegmentSampler

device = "cuda:0"
builder = CompactedSegmentsBuilder(device=device, normalize=False)
builder.append(np.arange(10, dtype=np.float32).reshape(-1,1))
builder.append(np.arange(10, 18, dtype=np.float32).reshape(-1,1))
builder.append(np.arange(30, 35, dtype=np.float32).reshape(-1,1))
segments = builder.build()
sampler = CompactedSegmentSampler(segments)
res = sampler.sample(batch_size=4, subsegment_size=8)
# res.subsegments is fixed sized data and res.ids is the segment id that sample came from
```