// GLOBAL_VARIANT_EMBEDDINGS. Cross-experiment median pooling, then PCA at
// the full retained rank; cumulative_variance_explained controls only the
// extra pca_reduced.parquet.
//
// Reads each experiment's UNFILTERED aggregate.parquet and applies the
// global blocklist itself, rather than reading the per-experiment
// filtered_aggregate.parquet: median_across_batches intersects feature
// columns across experiments, so consuming the filtered files would
// silently reduce every setting to "reproducible in every experiment" and
// make reproducibility_global_min_batches_ok inert.
//
// Every experiment's file has the same basename, so they're staged under
// numbered names and passed as an explicit input_files list, paired
// positionally with batch_stems -- the workflow hands both over as one
// sorted tuple so the pairing can't drift.

include { threadEnv; hydraList } from '../functions'

process GLOBAL_VARIANT_EMBEDDINGS {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/embeddings" }, mode: 'copy'

    input:
    tuple val(batch_stems), path(aggregate_parquets, stageAs: "agg_input_*.parquet")
    path(blocklist_parquet)

    output:
    tuple path("median_aggregate.parquet"), path("pca_scores.parquet"), path("pca_components.parquet"), path("pca_variance_explained.parquet"), path("pca_reduced.parquet"), emit: global

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.global_embeddings \\
        output_dir=. \\
        ${hydraList('input_files', aggregate_parquets)} \\
        ${hydraList('batch_stems', batch_stems)} \\
        blocklist_file=${blocklist_parquet} \\
        label_column=${params.filter_label_column} \\
        cumulative_variance_explained=${params.global_variant_embeddings_cumulative_variance_explained} \\
        random_seed=${params.random_seed}
    """
}
