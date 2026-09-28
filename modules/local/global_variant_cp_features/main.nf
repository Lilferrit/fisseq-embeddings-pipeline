// GLOBAL_VARIANT_CP_FEATURES. The CellProfiler track's counterpart of
// GLOBAL_VARIANT_EMBEDDINGS (no reproducibility filtering on this track).
//
// Every experiment's file has the same basename, so they're staged under
// numbered names and passed as an explicit input_files list, paired
// positionally with batch_stems -- the workflow hands both over as one
// sorted tuple so the pairing can't drift.

include { threadEnv; hydraList } from '../functions'

process GLOBAL_VARIANT_CP_FEATURES {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/cp_features" }, mode: 'copy'

    input:
    tuple val(batch_stems), path(aggregate_parquets, stageAs: "agg_input_*.parquet")

    output:
    tuple path("median_aggregate.parquet"), path("pca_scores.parquet"), path("pca_components.parquet"), path("pca_variance_explained.parquet"), path("pca_reduced.parquet"), emit: global

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.global_variant_cp_features \\
        output_dir=. \\
        ${hydraList('input_files', aggregate_parquets)} \\
        ${hydraList('batch_stems', batch_stems)} \\
        label_column=${params.filter_label_column} \\
        cumulative_variance_explained=${params.global_variant_cp_features_cumulative_variance_explained} \\
        random_seed=${params.random_seed}
    """
}
