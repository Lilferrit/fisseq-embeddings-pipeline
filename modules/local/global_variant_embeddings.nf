// SPEC.md §6.7 -- GLOBAL_VARIANT_EMBEDDINGS (Epic 7). Runs once,
// unconditionally, over every experiment's aggregate.parquet (§3 decision 8).
//
// `stageAs: "agg_input_*.parquet"` avoids every experiment's identically-
// named aggregate.parquet colliding when collected into this one task --
// Nextflow numbers staged files 1-indexed in the same order as the list it
// received, which global_embeddings.py's main() reverses positionally
// against the paired batch_stems list (utils/nextflow_staging.py) rather
// than reading a directory glob (fisseq-data-pipeline's own
// globalfeatureselect.py precedent for this exact collision). Unverified
// until Epic 9's real Nextflow run, same caveat as embed_cells.nf's/
// aggregate_embeddings.nf's own list-interpolation precedents.
//
// No n_components param (revision, per request) -- global_embeddings.py
// always computes the full retained PCA rank itself.

process GLOBAL_VARIANT_EMBEDDINGS {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/embeddings" }, mode: 'copy'

    input:
    path(aggregate_parquets, stageAs: "agg_input_*.parquet")
    val(batch_stems)

    output:
    tuple path("median_aggregate.parquet"), path("pca_scores.parquet"), path("pca_components.parquet"), path("pca_variance_explained.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.global_embeddings \\
        output_dir=. \\
        'batch_stems=[${batch_stems.join(",")}]' \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
