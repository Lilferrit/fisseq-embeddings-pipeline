# Quickstart

## 1. Lay out one experiment's inputs

Each experiment needs an entry in `params.yaml`'s `experiments:` list, plus
a `--pipeline_dir` directory holding that experiment's raw data:

```yaml
# params.yaml
experiments:
  - batch_stem: experiment1
    phenotyping_dir: /data/experiment1/phenotyping   # starcall-workflow output root
    wells: [well1, well2]
    grid_size: 12
    window: 224                                       # must match your Cell-DINO checkpoint's crop size
```

See [`BUILD_DATASET`'s stage reference](cli/dataset.md) for every field
`BuildDatasetConfig` accepts.

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
stage (`dataset/`, `qc_filter/`, `embeddings/`, `filter_embeddings/`,
`feature_select_batchwise/`, `ovwt_batchwise/`, `global/`). See
[Nextflow Workflow](nextflow.md#output-directory-layout) for the full tree
and [Architecture](architecture.md#data-contracts) for what each Parquet
file's columns mean.

## Running multiple experiments together

Add another map to `params.yaml`'s `experiments:` list (with its own
unique `batch_stem`) -- every per-experiment stage runs once per entry,
and the two global stages (`GLOBAL_VARIANT_EMBEDDINGS`,
`GLOBAL_VARIANT_DISTINGUISHABILITY`) automatically pool across however
many experiments are present.
