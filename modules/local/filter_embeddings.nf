// SPEC.md §6.4 -- FILTER_EMBEDDINGS (Epic 4). Publishes only the QC-passed
// join key + fitted Normalizer -- no emb_* columns (§3 decision 10).
// TODO(Epic 9): implement per IMPLEMENTATION_CHECKLIST.md Epic 9.

process FILTER_EMBEDDINGS {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/filter_embeddings/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(embeddings_parquet), path(filtered_cells_parquet)

    output:
    tuple val(batch_stem), path("filtered_keys.parquet"), path("normalizer.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.filter \\
        output_dir=. \\
        embeddings_file=${embeddings_parquet} \\
        qc_passed_file=${filtered_cells_parquet} \\
        random_seed=${params.random_seed}
    """
}
