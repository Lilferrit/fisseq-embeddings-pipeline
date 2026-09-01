# Quickstart

## 1. Lay out one experiment's inputs

Each experiment needs an entry in `params.yaml`'s `experiments:` list, plus
a `--pipeline_dir` directory holding that experiment's raw data:

```yaml
# params.yaml
window: 224   # must match your Cell-DINO checkpoint's crop size -- global
              # default, used by every experiment below that doesn't set
              # its own `window`
starcall_container_image: "starcall-workflow:latest"  # required once any
                                                        # experiment is present

experiments:
  - batch_stem: experiment1
    starcall_workflow_dir: /data/experiment1/starcall-workflow  # a checkout Snakemake can run against
    phenotyping_dir: /data/experiment1/phenotyping   # starcall-workflow output root
    segmentation_dir: /data/experiment1/segmentation
    sequencing_dir: /data/experiment1/sequencing
    wells: [well1, well2]
    # grid_size omitted -- auto-detected per well from phenotyping_dir's
    # own {well}_grid<N> directory naming; set it explicitly to override.
    # window omitted -- falls back to the global `window` default above.
```

These starcall-workflow-facing fields all belong to `BUILD_CELL_IMAGES`,
the one stage that touches `starcall-workflow`'s tree -- see
[`BUILD_CELL_IMAGES`' section of the Architecture doc](architecture.md#cell-images-buildcellimages-output-from-starcall-workflow)
for every field it accepts, and
[`BUILD_DATASET`'s stage reference](cli/dataset.md) for `BuildDatasetConfig`'s
own remaining fields (`window`, `shard_maxcount`, ...).

## 2. Get a Cell-DINO checkpoint

`--cell_dino_checkpoint` is required, with no default (a checkpoint path is
inherently deployment-specific). Point it at a real `.pth` file; see
[Architecture](architecture.md#embed_cells-cell-dino-inference-internals)
for what checkpoint shapes are supported.

## 3. Run the pipeline

```bash
nextflow run . \
    --pipeline_dir /path/to/experiment1 \
    --cell_dino_checkpoint /path/to/checkpoint.pth \
    -params-file params.yaml
```

`-params-file params.yaml` is mandatory -- there is no
`nextflow.config`-embedded fallback for pipeline defaults (see
[Configuration](configuration.md)). Any field in `params.yaml` can be
overridden with a bare CLI flag, e.g. `--ovwt_min_cells 500`.

By default every process runs inside the Docker image named by
`container_image`. To run directly against your own Python environment
instead (no Docker image needed -- useful for local development or CI),
add `-profile local`:

```bash
nextflow run . \
    --pipeline_dir /path/to/experiment1 \
    --cell_dino_checkpoint /path/to/checkpoint.pth \
    -params-file params.yaml \
    -profile local
```

## 4. Read the outputs

Every stage's output lands under `<pipeline_dir>/`, one subdirectory per
stage (`cell_images/`, `dataset/`, `qc_filter/`, `embeddings/`,
`filter_embeddings/`, `feature_select_batchwise/`, `ovwt_batchwise/`,
`global/`). See
[Nextflow Workflow](nextflow.md#output-directory-layout) for the full tree
and [Architecture](architecture.md#data-contracts) for what each Parquet
file's columns mean.

## Running multiple experiments together

Add another map to `params.yaml`'s `experiments:` list (with its own
unique `batch_stem`) -- every per-experiment stage runs once per entry,
and the two global stages (`GLOBAL_VARIANT_EMBEDDINGS`,
`GLOBAL_VARIANT_DISTINGUISHABILITY`) automatically pool across however
many experiments are present.
