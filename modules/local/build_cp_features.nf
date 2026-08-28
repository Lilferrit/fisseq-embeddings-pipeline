// BUILD_CP_FEATURES. Reads a per-experiment config (phenotyping_dir, wells,
// grid_size, segmentation_type, cellprofiler_cycle, cellprofiler_pipeline,
// barcode_col_name, aa_changes_col_name, edit_distance_col_name) from one
// entry of params.yaml's `cp_features_experiments:` list -- same blind
// per-experiment-map CLI-override forwarding as build_dataset.nf. Its own
// process, not depending on BUILD_DATASET's task output -- it reads the
// same phenotyping_dir tiles directly, plus each tile's already-computed
// CellProfiler CSV alongside them.

process BUILD_CP_FEATURES {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/cp_features/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(batch_config)

    output:
    tuple val(batch_stem), path("cp_features.parquet")

    script:
    def overrides = batch_config.collect { key, value ->
        (value instanceof List) ? "'${key}=[${value.join(",")}]'" : "${key}=${value}"
    }.join(' \\\n        ')
    """
    python -m fisseq_embeddings_pipeline.cp_features \\
        output_dir=. \\
        batch_stem=${batch_stem} \\
        ${overrides} \\
        random_seed=${params.random_seed}
    """
}
