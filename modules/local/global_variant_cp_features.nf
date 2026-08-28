// GLOBAL_VARIANT_CP_FEATURES. Runs once, unconditionally, over every
// experiment's CellProfiler-track aggregate.parquet -- CellProfiler
// analog of global_variant_embeddings.nf.
//
// `stageAs: "agg_input_*.parquet"` avoids every experiment's identically-
// named aggregate.parquet colliding when collected into this one task --
// see global_variant_embeddings.nf's comment for the full explanation
// (including the n==1 staging gotcha handled by
// utils/nextflow_staging.reconstruct_staged_paths).

process GLOBAL_VARIANT_CP_FEATURES {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/cp_features" }, mode: 'copy'

    input:
    path(aggregate_parquets, stageAs: "agg_input_*.parquet")
    val(batch_stems)

    output:
    tuple path("median_aggregate.parquet"), path("pca_scores.parquet"), path("pca_components.parquet"), path("pca_variance_explained.parquet"), path("pca_reduced.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.global_variant_cp_features \\
        output_dir=. \\
        'batch_stems=[${batch_stems.join(",")}]' \\
        label_column=${params.filter_label_column} \\
        cumulative_variance_explained=${params.global_variant_cp_features_cumulative_variance_explained} \\
        random_seed=${params.random_seed}
    """
}
