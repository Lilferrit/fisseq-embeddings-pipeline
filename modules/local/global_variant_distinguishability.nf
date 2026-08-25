// SPEC.md §6.8 -- GLOBAL_VARIANT_DISTINGUISHABILITY (Epic 8). Per-experiment
// synonymous z-score, then cross-experiment median (§3 decision 9) -- not a
// direct median of raw AUROC.
// TODO(Epic 9): implement per IMPLEMENTATION_CHECKLIST.md Epic 9.

process GLOBAL_VARIANT_DISTINGUISHABILITY {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/distinguishability" }, mode: 'copy'

    input:
    path(results_parquets)

    output:
    path("global_scores.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.global_distinguishability \\
        output_dir=. \\
        random_seed=${params.random_seed}
    """
}
