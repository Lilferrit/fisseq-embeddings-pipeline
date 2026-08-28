// FILTER_CP_FEATURES. Publishes only the QC-passed join key + fitted
// Normalizer -- no CellProfiler feature columns. `filtered_cells_parquet`
// is QC_FILTER's existing output (the same one FILTER_EMBEDDINGS
// consumes), not a second QC run -- see workflows/embeddings.nf.

process FILTER_CP_FEATURES {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/filter_cp_features/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(cp_features_parquet), path(filtered_cells_parquet)

    output:
    tuple val(batch_stem), path("filtered_keys.parquet"), path("normalizer.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.filter_cp_features \\
        output_dir=. \\
        cp_features_file=${cp_features_parquet} \\
        qc_passed_file=${filtered_cells_parquet} \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
