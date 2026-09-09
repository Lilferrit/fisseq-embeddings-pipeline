// BUILD_CP_FEATURES. Reads a per-experiment config (cell_images_dir,
// barcode_col_name, aa_changes_col_name, edit_distance_col_name) from one
// entry of params.yaml's `experiments:` list that sets `cp_features: true`
// -- same blind per-experiment-map CLI-override forwarding as
// build_dataset/main.nf, plus cell_images_dir itself (BUILD_CELL_IMAGES' own
// per-experiment output directory, injected by workflows/embeddings.nf --
// see that workflow's cp_config_ch). Its own process, not depending on
// BUILD_DATASET's task output -- it reads cell_images_dir/cell_table.parquet
// directly (BUILD_CELL_IMAGES already joined in this experiment's
// CellProfiler columns; see that module's own docstring), no starcall-
// workflow-facing field or tile discovery of its own any more.
// cell_images_dir is also bind-mounted into this process' container -- see
// nextflow.config's own withName:'BUILD_DATASET|BUILD_CP_FEATURES'
// containerOptions comment.

process BUILD_CP_FEATURES {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/cp_features/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(batch_config)

    output:
    tuple val(batch_stem), path("cp_features.parquet"), emit: cp_features

    when:
    task.ext.when == null || task.ext.when

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
