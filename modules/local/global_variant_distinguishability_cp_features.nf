// GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES. Per-experiment
// synonymous z-score, then cross-experiment median -- CellProfiler analog
// of global_variant_distinguishability.nf.
//
// `stageAs: "res_input_*.parquet"` avoids every experiment's identically-
// named results.parquet colliding when collected into this one task, same
// pattern/caveat as global_variant_distinguishability.nf.

process GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/distinguishability_cp_features" }, mode: 'copy'

    input:
    path(results_parquets, stageAs: "res_input_*.parquet")
    val(batch_stems)

    output:
    path("global_scores.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.global_variant_distinguishability_cp_features \\
        output_dir=. \\
        'batch_stems=[${batch_stems.join(",")}]' \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
