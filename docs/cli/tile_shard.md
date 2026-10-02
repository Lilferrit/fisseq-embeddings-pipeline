# Cell Shards (`make_cell_shard`)

`python -m fisseq_embeddings_pipeline.tile_shard` crops every cell of
**one tile** out of `starcall-workflow`'s whole-tile phenotype image and
writes them into that tile's **WebDataset** shard (a `.tar`, one sample per
cell, unfiltered), which `EMBED_CELLS` streams from.

It is not a Nextflow process. It's the body of `rule make_cell_shard`,
which this repo adds on top of starcall's own Snakefile
(`snakemake/Snakefile`), so it runs inside `BUILD_CELL_IMAGES`' nested
snakemake, once per tile, as phase 2 of that stage. `BUILD_CELL_IMAGES`
requests every tile's shard as a target. As a result, cropping fans out one
job per tile under a `starcall_profile`, and snakemake's own mtime check
caches each shard, so a tile already cut isn't cut again. See
[Architecture](../architecture.md) decisions 17 and 24.

Building the shards (and running `EMBED_CELLS` over them) is deliberately
decoupled from `QC_FILTER`. QC thresholds get tuned and re-run often. Keeping
embedding as a separate, unconditional branch means changing a QC threshold
never re-triggers the expensive GPU embedding pass. You pay for embedding
every cell once, up front.

## The rule

```text
input:   {phenotyping_dir}{well}_grid{N}/tile{x}x{y}y/{raw|corrected}_pt.tif
         {phenotyping_dir}{well}_grid{N}/tile{x}x{y}y/{segmentation_type}_mask.tif
         {phenotyping_dir}{well}_grid{N}/tile{x}x{y}y/{segmentation_type}.csv
output:  {phenotyping_dir}{well}_grid{N}/tile{x}x{y}y/{segmentation_type}_{raw|corrected}_shard_{window}.tar
log:     the same path, .log
```

`raw` vs `corrected` (from the experiment's `use_corrected`) and `window`
are both part of the shard's filename. Changing either requests a
different shard instead of reusing a stale one.

It's a `shell:` rule, not `run:`, so the body runs in this pipeline's own
Python 3.13 venv (`config['fisseq_python']`, which `BUILD_CELL_IMAGES` sets
to its own `$(command -v python)`) rather than in the `ops` env's 3.10
snakemake interpreter. Every child job re-enters the same image, so that
path is valid on a cluster node too. Hydra's run directory and this
module's own log file go to a scratch `mktemp -d` directory, so the job's
working directory (the experiment directory) doesn't collect a Hydra
`outputs/` tree per tile. The console log lands in the rule's `log:` file.

The whole-tile image is a `temp()` output upstream (`rule stitch_tile_pt`)
and is no longer requested as a target itself. Once the shard is cut,
snakemake deletes it. The tile mask is `temp()` only when
`stitch_tile_segmentation` builds it (`relabel_segmentation`/
`stitch_tile_from_well_segmentation` write a plain output), so it may
stay.

## Cropping

`write_tile_shard` reads the tile's segmentation table with
`build_cell_images_table.read_segmentation_table`, the same reader
`BUILD_CELL_IMAGES`' table phase uses, so the two can't disagree on which
row is which cell. Each row has a `tile_cell_index` (the CSV's own index)
and a `crop_index` (its 0-based on-disk row position). The module then reads
the whole-tile image (starcall's `(cycles, channels, H, W)` flattened to
`(C, H, W)`) and the label mask (`(H, W)`) once, and cuts every cell out
with `crop_cell`:

- the crop is `window` x `window`, centred on the bbox midpoint
  (`((bbox_x1 + bbox_x2) // 2, (bbox_y1 + bbox_y2) // 2)`, with `bbox_x*`
  on image axis 0 and `bbox_y*` on axis 1, starcall's own convention);
- where the window runs off the tile it is zero-padded;
- the mask crop is `mask == crop_index + 1` (row *i* of the tile's cell
  table is mask label *i+1*), stored as uint8, so neighbouring cells
  inside the window are masked out.

This is starcall-workflow's own `make_cell_images` crop with the centre
fixed: upstream reads `xpos`/`ypos` columns its cell table doesn't have.
See [Architecture](../architecture.md) decision 17. A tile with no cells
still gets a valid, empty tar (snakemake needs the output to exist), and
its image and mask are never opened. A mask whose shape doesn't match its
image raises.

## Config fields

Extends the [common config fields](#common-config-fields) below.
`output_dir` holds only this run's log; the shard itself goes to
`output_tar`.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `image_tif` | **required** | The tile's whole-tile phenotype image, `(cycles, channels, H, W)`. |
| `mask_tif` | **required** | The tile's `(H, W)` segmentation label mask. |
| `segmentation_csv` | **required** | The tile's `{segmentation_type}.csv`. |
| `well`, `tile` | **required** | This tile's identifiers, written into each sample's key and `meta.json`. |
| `window` | **required** | Side length, in pixels, of each cell's square crop. Must match the loaded Cell-DINO checkpoint's expected input (`cell_dino_crop_size`). Comes from the shard target's filename, which `BUILD_CELL_IMAGES` names from the experiment's `window` (falling back to `params.yaml`'s global default; see [Configuration](../configuration.md)). |
| `output_tar` | **required** | Path of the shard to write. |

## Output

One `.tar` per tile. Each sample is keyed `"{well}_{tile}_{tile_cell_index}"`
and carries:

- `crop.npy`: `(num_phenotyping_cycles × num_channels, window, window)`,
  in the image's dtype;
- `mask.npy`: `(window, window)` uint8 foreground mask;
- `meta.json`: `meta_well`, `meta_tile`, `meta_cell_index` only.

`meta.json` carries just the cell's location on purpose. The genotype
columns live in a different starcall tree (`sequencing_dir`) under
per-experiment column names, and `meta_batch` is a pipeline-level name.
Baking either into a file snakemake caches would leave it stale whenever
they change, and would copy what `BUILD_CELL_METADATA`'s
`metadata.parquet` already holds. `EMBED_CELLS` joins that table back on
(see [Cell Embeddings](embed.md)).

At the defaults (4 uint16 channels, `window` 224) a sample is about
440 KB (`crop.npy` ≈ 392 KB, `mask.npy` ≈ 49 KB, plus `meta.json` and tar
headers), so a tile's shard is roughly 440 KB × its cell count; channel
count (`num_phenotyping_cycles × num_channels`), dtype and `window` scale
it.

`BUILD_CELL_IMAGES`' `tiles.parquet` lists every tile's shard path
(`shard_tar`). The shards are read in place and never copied into
`pipeline_dir`.

## Example

```bash
uv run python -m fisseq_embeddings_pipeline.tile_shard \
    output_dir=/tmp/log \
    image_tif=phenotyping/well1_grid4/tile0x0y/raw_pt.tif \
    mask_tif=phenotyping/well1_grid4/tile0x0y/cells_mask.tif \
    segmentation_csv=phenotyping/well1_grid4/tile0x0y/cells.csv \
    well=well1 tile=tile0x0y window=224 \
    output_tar=phenotyping/well1_grid4/tile0x0y/cells_raw_shard_224.tar
```

## Common config fields

Every CLI tool's config extends `AppConfig`, which supplies:

| Field | Default | Description |
| ----- | ------- | ----------- |
| `output_dir` | **required** | Directory for all output files; created if absent. |
| `output_root` | `null` | If set, output files are prefixed `{output_root}.{name}` instead of being placed directly under `output_dir`. |
| `log_level` | `"info"` | Logging verbosity (`debug`, `info`, `warning`, `error`, `critical`). |
| `random_seed` | `0` | Shared seed for every stochastic pipeline stage. |

See [API Reference: tile_shard](../api/tile_shard.md) for full function documentation.
