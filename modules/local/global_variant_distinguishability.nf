// SPEC.md §6.8 -- GLOBAL_VARIANT_DISTINGUISHABILITY (Epic 8). Per-experiment
// synonymous z-score, then cross-experiment median (§3 decision 9) -- not a
// direct median of raw AUROC.
//
// `stageAs: "res_input_*.parquet"` avoids every experiment's identically-
// named results.parquet colliding when collected into this one task, same
// pattern/caveat as global_variant_embeddings.nf.

process GLOBAL_VARIANT_DISTINGUISHABILITY {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/distinguishability" }, mode: 'copy'

    input:
    path(results_parquets, stageAs: "res_input_*.parquet")
    val(batch_stems)

    output:
    path("global_scores.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.global_distinguishability \\
        output_dir=. \\
        'batch_stems=[${batch_stems.join(",")}]' \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
