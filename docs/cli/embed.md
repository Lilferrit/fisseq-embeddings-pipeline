# Cell Embeddings (`EMBED_CELLS`)

`python -m fisseq_embeddings_pipeline.embed` (Nextflow process
`EMBED_CELLS`, the pipeline's only GPU-bound stage) streams every cell in
an experiment's per-tile WebDataset shards (cut by `BUILD_CELL_IMAGES`'
nested `make_cell_shard` rule -- see [Cell Shards](tile_shard.md)) through
a pretrained Cell-DINO checkpoint (Meta's `dinov2`) and writes one row per
cell to `embeddings.parquet`. Not gated by `QC_FILTER` -- this GPU pass
runs once per experiment regardless of how many times QC thresholds get
retuned afterward.

The shards are read in place, under `phenotyping_dir`, from the paths
`BUILD_CELL_IMAGES`' `tiles.parquet` lists (nextflow.config binds that
directory into this task's container). A shard's `meta.json` carries only
the cell's location, so every other `meta_*` column is joined on from
`BUILD_CELL_METADATA`'s `metadata.parquet` (`attach_metadata`): left join
on `(meta_well, meta_tile, meta_cell_index)`, raising if any embedded cell
has no row there -- the shards and `cell_table.parquet` are built from the
same segmentation tables, so a miss means they came from different runs.

See [Architecture](../architecture.md#embed_cells-cell-dino-inference-internals)
for how the checkpoint's architecture is inferred from its own state dict
rather than assumed, and what real checkpoint shapes have been verified.

## Config fields

Extends the [common config fields](#common-config-fields) below.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `tiles_path` | `null` | `BUILD_CELL_IMAGES`' `tiles.parquet`: every row's `shard_tar` is one of this experiment's shards; a listed shard that doesn't exist raises. What the pipeline sets. Exactly one of `tiles_path` / `shard_pattern` must be set. |
| `shard_pattern` | `null` | A path/brace pattern for a set of shards instead, e.g. `"dataset-{000000..000042}.tar"`. A bare glob (`"dataset-*.tar"`) also works. |
| `metadata_path` | **required** | `BUILD_CELL_METADATA`'s `metadata.parquet` for the same experiment, joined onto each embedded cell. |
| `checkpoint_path` | **required** | Path to the Cell-DINO checkpoint (`.pth`). |
| `arch` | `"vit_large"` | Backbone architecture: `vit_small` / `vit_base` / `vit_large` / `vit_giant2`. |
| `patch_size` | `16` | ViT patch size. |
| `crop_size` | `224` | Expected crop size -- must match the shards' `window`. Only a fallback for model construction; doesn't constrain what crop size can actually be embedded. |
| `channels` | `[0, 1, 2, 3]` | Which of the crop's channel indices to feed into the model, in order. |
| `apply_mask` | `true` | Whether `mask.npy`-based background zeroing is applied before embedding -- to every selected channel, or to none. A single flag because there is only ever one mask per cell to apply. |
| `channel_pool` | `"mean"` | How per-channel CLS embeddings are pooled (`"mean"` or `"max"`) -- only consulted for a bag-of-channels model. |
| `device` | `"cuda"` | torch device string. |
| `batch_size` | `256` | Cells per dataloader batch. |
| `num_workers` | `4` | `webdataset`/`DataLoader` worker processes. |

## Output file

`embeddings.parquet` -- one row per `(meta_well, meta_tile,
meta_cell_index)` key streamed off the dataloader, in shard order: the
seven `meta_*` columns `QC_FILTER` sees (`CELL_METADATA_SCHEMA`), then
`emb_0000`..`emb_{D-1}` (`D` = the loaded
checkpoint's actual embed dim -- e.g. 1024 for ViT-L/16, 384 for
ViT-S/8). Column-naming convention: zero-padded `emb_%04d`, matched
downstream by `EMBEDDING_SELECTOR = cs.matches(r"^emb_\d+$")`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.embed \
    output_dir=./out \
    tiles_path=/pipeline/cell_images/experiment1/tiles.parquet \
    metadata_path=/pipeline/cell_metadata/experiment1/metadata.parquet \
    checkpoint_path=/path/to/checkpoint.pth \
    device=cpu \
    'channels=[0,1,2,3]' \
    apply_mask=true \
    random_seed=0
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage (unused by this stage -- inference is deterministic given a fixed checkpoint). |

See [API Reference: embed](../api/embed.md) for full function documentation.
