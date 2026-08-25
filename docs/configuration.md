# Configuration reference

Placeholder -- see SPEC.md §6 (per-stage Hydra configs) and params.yaml
(repo root, every default parameter) until this is written for real
post-implementation, epic by epic. This page currently only has real
content for the piece Epic 1 resolved: `BUILD_DATASET`'s WebDataset shard
sizing.

## `BUILD_DATASET` shard sizing (SPEC.md §10, item 2)

`shard_maxcount` (default `2000`, `BuildDatasetConfig.shard_maxcount`)
controls how many cells `write_dataset_shards()` packs into each
`dataset-*.tar` shard. No real experiment was available to measure this
against directly in this environment (only the pipeline source repos are
mounted, not phenotyping data) -- the number below is a computed estimate
from real defaults elsewhere in the stack, not a real measurement, and
should be re-checked against one real experiment's actual
`*_crops_{window}.tif` byte size once available.

**Inputs to the estimate:**

- Channel count: **4**, from `starcall-workflow`'s (`origin/devel`)
  `default-config.yaml`: `phenotyping_channels: ['DAPI', 'GFP', 'Ph+WGA', 'Mito']`.
- Crop window: **224** (`window`/`crop_size`), from Cell-DINO's
  channel-adaptive eval config (`global_crops_size: 224`, SPEC.md §6.3).
- Crop dtype: **uint16**, the standard bit depth for fluorescence
  microscopy TIFFs -- assumed, not confirmed against a real
  `*_crops_224.tif` (`starcall-workflow`'s `make_cell_images` rule writes
  crops in the source phenotype image's own dtype, whatever that turns out
  to be for a real acquisition).
- Mask dtype: **uint8** label mask (fixed -- `make_cell_images` always
  writes `uint8`, not memory-mappable `bool`).

**Per-sample size:**

| Component | Formula | Size |
| --- | --- | --- |
| `crop.npy` | 4 × 224 × 224 × 2 bytes | ≈ 392 KB |
| `mask.npy` | 224 × 224 × 1 byte | ≈ 49 KB |
| `meta.json` + tar per-file headers (3 files/sample) | -- | a few KB |
| **Total** | | **≈ 440 KB/sample** |

**Per-shard size** at the default `shard_maxcount=2000`:
440 KB × 2000 ≈ **~880 MB/shard** -- within the "hundreds of MB to ~1GB"
band `SPEC.md`/`IMPLEMENTATION_CHECKLIST.md` call reasonable for a
webdataset shard, so **the `2000` default is kept as-is**.

**Still open:** re-run this estimate (or better, measure directly) once a
real experiment's `phenotyping_dir` is available -- channel count, crop
dtype, and window are all assumption-flagged above and could shift the
number meaningfully (e.g. a channel count above ~9 or a `uint32`/float
crop dtype would push a shard past 1GB at the current default).
