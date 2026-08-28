# Configuration reference

## `params.yaml`, not `nextflow.config`

`fisseq-data-pipeline`'s `nextflow.config` declares every default
parameter directly in a `params { ... }` block. This repo splits that in
two:

- **`params.yaml`** (repo root) -- every default value, nothing else.
  Loaded explicitly via `-params-file params.yaml`; Nextflow merges a
  `-params-file` YAML/JSON document into `params` natively. A per-run
  override still works the normal way, either as a bare CLI flag
  (`--ovwt_min_cells 500`, which wins over `params.yaml`) or a whole
  separate copy of `params.yaml` passed to `-params-file` instead --
  Nextflow accepts only one `-params-file` per run (`Can only specify
  option -params-file once`), so there's no way to layer a second, partial
  file on top of it.
- **`nextflow.config`** -- executor/profile/process directives only
  (`process.container`, per-`label` resource requests, `-profile local`,
  `docker.enabled`). No default parameter values live here at all.

`pipeline_dir`, `cell_dino_checkpoint` are required with no default --
`EmbeddingsPipeline` fails fast with a specific message if either is unset
(see [Nextflow Workflow](nextflow.md)), rather than letting Nextflow's own
less-specific "no such property" error surface first.

`BUILD_DATASET`'s own fields (`phenotyping_dir`, `wells`, `grid_size`,
`window`, ...) are per-experiment, not pipeline-wide, so they don't have a
single default value in the rest of `params.yaml` -- instead each
experiment supplies its own map of these fields as one entry of
`params.yaml`'s `experiments:` list (see
[Nextflow Workflow](nextflow.md#per-experiment-configs)).
`BUILD_CP_FEATURES`' fields (`phenotyping_dir`, `wells`, `grid_size`,
`cellprofiler_cycle`, `cellprofiler_pipeline`, ...) work the same way, via
the separate, optional `cp_features_experiments:` list -- see
[Nextflow Workflow](nextflow.md#cellprofiler-feature-track).

### Fields

| Key | Default | Stage(s) |
| --- | --- | --- |
| `pipeline_dir` | *(required)* | all |
| `container_image` | `"fisseq-embeddings-pipeline:latest"` | all |
| `cell_dino_checkpoint` | *(required)* | `EMBED_CELLS` |
| `experiments` | `[]` (required non-empty) | `BUILD_DATASET` (list of per-experiment maps, each requiring `batch_stem`; see above) |
| `cp_features_experiments` | `[]` (optional) | `BUILD_CP_FEATURES` (list of per-experiment maps, each requiring `batch_stem`; empty means no CellProfiler-feature track for this run -- see below) |
| `random_seed` | `0` | every stochastic stage |
| `barcode_count_threshold` | `10` | `QC_FILTER` |
| `variant_barcode_count_threshold` | `4` | `QC_FILTER` |
| `edit_distance_threshold` | `1` | `QC_FILTER` |
| `cell_dino_arch` | `"vit_large"` | `EMBED_CELLS` |
| `cell_dino_patch_size` | `16` | `EMBED_CELLS` |
| `cell_dino_crop_size` | `224` | `EMBED_CELLS` (must match `BUILD_DATASET`'s per-experiment `window`) |
| `cell_dino_channels` | `[0, 1, 2, 3]` | `EMBED_CELLS` |
| `cell_dino_channel_apply_mask` | `[true, true, true, true]` | `EMBED_CELLS` |
| `cell_dino_channel_pool` | `"mean"` | `EMBED_CELLS` |
| `cell_dino_device` | `"cuda"` | `EMBED_CELLS` |
| `cell_dino_batch_size` | `256` | `EMBED_CELLS` |
| `cell_dino_num_workers` | `4` | `EMBED_CELLS` |
| `filter_label_column` | `"meta_aa_changes"` | `QC_FILTER`, `FILTER_EMBEDDINGS`, `AGGREGATE_EMBEDDINGS`, `OVWT_BATCHWISE`, both global stages, and their CellProfiler-track counterparts |
| `aggregate_methods` | `["median", "KS", "AUROC"]` | `AGGREGATE_EMBEDDINGS` |
| `aggregate_methods_cp_features` | `["median"]` | `AGGREGATE_CP_FEATURES` |
| `ovwt_wt_label` | `"WT"` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_n_folds` | `5` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_calibrate` | `true` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_min_cells` | `250` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `ovwt_downsample_wt` | `true` | `OVWT_BATCHWISE`, `OVWT_BATCHWISE_CP_FEATURES` |
| `global_variant_embeddings_cumulative_variance_explained` | `0.9` | `GLOBAL_VARIANT_EMBEDDINGS` |
| `global_variant_cp_features_cumulative_variance_explained` | `0.9` | `GLOBAL_VARIANT_CP_FEATURES` |

`filter_label_column` is shared pipeline-wide so overriding it changes the
variant label column everywhere at once, rather than each stage needing
its own override. `aggregate_methods` defaults to `["median", "KS",
"AUROC"]` -- since that's not the literal single-element `["median"]`,
`AGGREGATE_EMBEDDINGS`' default output columns are suffixed by method
(`emb_0000_median`, `emb_0000_KS`, `emb_0000_AUROC`, ...); the
CellProfiler-feature track's own `aggregate_methods_cp_features` stays
`["median"]`, so `AGGREGATE_CP_FEATURES`' default output columns remain
bare. The two `*_cumulative_variance_explained` params each have their own
CellProfiler-track counterpart above; `ovwt_*`, by contrast, is genuinely
shared between both tracks' OVWT stages (scoring methodology, not tied to
feature type) -- see [Nextflow Workflow](nextflow.md#cellprofiler-feature-track).
See each [Stage Reference](cli/dataset.md) page for the full field list a
given stage's Hydra config accepts beyond what `params.yaml` exposes (e.g.
`QC_FILTER`'s optional `n_variants` downsampling cap, off by default).

## Docker image versioning & publishing

- **Registry:** GitHub Container Registry, `ghcr.io/<owner>/<repo>`
  (derived from the repo's own `${{ github.repository }}` at build time).
- **Tags, on every push to `main`:** `:latest` (moving -- convenience/dev
  use) and `:<short-sha>` (exact, 7-character commit SHA -- what
  `params.yaml`'s `container_image` should point at for anything that
  needs to pin a specific build instead of floating on `:latest`, e.g. a
  reproducibility-sensitive run).
- **Tags, on a pushed `v*` git tag** (a real release): additionally
  `:<version>` (the tag with its `v` prefix stripped, e.g. `v0.1.0` ->
  `0.1.0`) -- not tied to `pyproject.toml`'s own `version` field
  automatically; bump that field and push a matching `vX.Y.Z` tag together
  when cutting a release.
- **Every PR:** build-only, no push, no registry credentials needed -- a
  smoke test against Dockerfile regressions.

One CUDA-capable base image serves every stage, including the CPU-only
ones (`QC_FILTER`, `FILTER_EMBEDDINGS`, etc.) -- simpler to build/publish/
version as a single artifact than a GPU image plus a slimmer CPU image, at
the cost of a larger pull for CPU-only processes. Worth splitting into two
images later if that pull cost matters in practice; not required for v1.

## `BUILD_DATASET` shard sizing

`shard_maxcount` (default `2000`, `BuildDatasetConfig.shard_maxcount`)
controls how many cells `write_dataset_shards()` packs into each
`dataset-*.tar` shard.

**Inputs to the estimate:**

- Channel count: **4**, from `starcall-workflow`'s default single
  phenotype cycle (`phenotype_cycles: ['PT']`, `phenotyping_channels:
  ['DAPI', 'GFP', 'Ph+WGA', 'Mito']`). `crop.npy`'s actual channel
  dimension is `num_phenotyping_cycles × num_channels` (cycle-major
  flattened), so a deployment configuring more than one phenotyping cycle
  scales this estimate proportionally.
- Crop window: **224** (`window`/`crop_size`), matching Cell-DINO's
  channel-adaptive eval config (`global_crops_size: 224`).
- Crop dtype: **uint16**, the standard bit depth for fluorescence
  microscopy TIFFs.
- Mask dtype: **uint8** label mask.

**Per-sample size:**

| Component | Formula | Size |
| --- | --- | --- |
| `crop.npy` | 4 × 224 × 224 × 2 bytes | ≈ 392 KB |
| `mask.npy` | 224 × 224 × 1 byte | ≈ 49 KB |
| `meta.json` + tar per-file headers (3 files/sample) | -- | a few KB |
| **Total** | | **≈ 440 KB/sample** |

**Per-shard size** at the default `shard_maxcount=2000`:
440 KB × 2000 ≈ **~880 MB/shard** -- within the "hundreds of MB to ~1GB"
band generally considered reasonable for a WebDataset shard, so the
`2000` default is kept as-is. Re-check this estimate (or better, measure
directly) against a real experiment's actual byte sizes if channel count,
crop dtype, or window change meaningfully -- e.g. a channel count above
~9 or a `uint32`/float crop dtype would push a shard past 1GB at the
current default.
