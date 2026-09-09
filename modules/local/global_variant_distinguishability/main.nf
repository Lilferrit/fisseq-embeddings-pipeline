// GLOBAL_VARIANT_DISTINGUISHABILITY. Per-experiment synonymous z-score,
// then cross-experiment median -- not a direct median of raw AUROC.
//
// `stageAs: "res_input_*.parquet"` avoids every experiment's identically-
// named results.parquet colliding when collected into this one task, same
// pattern/caveat as global_variant_embeddings/main.nf.

process GLOBAL_VARIANT_DISTINGUISHABILITY {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/distinguishability" }, mode: 'copy'

    input:
    path(results_parquets, stageAs: "res_input_*.parquet")
    val(batch_stems)

    output:
    path("global_scores.parquet"), emit: global_scores

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    python -m fisseq_embeddings_pipeline.global_distinguishability \\
        output_dir=. \\
        'batch_stems=[${batch_stems.join(",")}]' \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
