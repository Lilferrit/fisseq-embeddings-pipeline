// SPEC.md §6.7 -- GLOBAL_VARIANT_EMBEDDINGS (Epic 7). Runs once,
// unconditionally, over every experiment's aggregate.parquet (§3 decision 8).
// TODO(Epic 9): implement per IMPLEMENTATION_CHECKLIST.md Epic 9.

process GLOBAL_VARIANT_EMBEDDINGS {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/embeddings" }, mode: 'copy'

    input:
    path(aggregate_parquets)
    val(batch_stems)

    output:
    path("median_aggregate.parquet"), path("pca_scores.parquet"), path("pca_components.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.global_embeddings \\
        output_dir=. \\
        n_components=${params.global_variant_embeddings_n_components} \\
        random_seed=${params.random_seed}
    """
}
