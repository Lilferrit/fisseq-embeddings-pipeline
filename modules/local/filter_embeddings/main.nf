// FILTER_EMBEDDINGS. Publishes only the QC-passed join key + fitted
// Normalizer -- no emb_* columns.

process FILTER_EMBEDDINGS {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/filter_embeddings/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(embeddings_parquet), path(filtered_cells_parquet)

    output:
    tuple val(batch_stem), path("filtered_keys.parquet"), path("normalizer.parquet"), emit: filtered

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python -m fisseq_embeddings_pipeline.filter \\
        output_dir=. \\
        embeddings_file=${embeddings_parquet} \\
        qc_passed_file=${filtered_cells_parquet} \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
