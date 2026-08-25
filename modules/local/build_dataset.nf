// SPEC.md §6.1 / §7 -- BUILD_DATASET (Epic 1). Reads a per-experiment
// config (phenotyping_dir, wells, grid_size, segmentation_type, window,
// shard_maxcount, batch_stem) from <pipeline_dir>/configs/<batch>.yaml.
// TODO(Epic 9): implement per IMPLEMENTATION_CHECKLIST.md Epic 9, following
// embed_cells.nf's shape (container / errorStrategy / publishDir / CLI args
// incl. random_seed=${params.random_seed}).

process BUILD_DATASET {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/dataset/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(config_yaml)

    output:
    tuple val(batch_stem), path("dataset-*.tar"), path("metadata.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.dataset \\
        output_dir=. \\
        --config-path=. --config-name=${config_yaml} \\
        random_seed=${params.random_seed}
    """
}
