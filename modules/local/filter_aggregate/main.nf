// FILTER_AGGREGATE. Aggregates -> filtered aggregates, plus the terminal
// passthrough view. Two outputs on purpose: filtered_aggregate.parquet
// carries only reproducible columns; aggregate_with_passthrough.parquet
// adds the passthrough methods and is terminal -- nothing in-pipeline reads
// it. The file split is what keeps passthrough columns out of the PCA:
// GLOBAL_VARIANT_EMBEDDINGS picks features with FEATURE_SELECTOR, which
// would happily match emb_0000_KSnegLogP.

include { threadEnv } from '../functions'

process FILTER_AGGREGATE {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(aggregate_parquet), path(blocklist_parquet), path(passthrough_parquets, stageAs: 'passthrough/*')

    output:
    tuple val(batch_stem), path("filtered_aggregate.parquet"), path("aggregate_with_passthrough.parquet"), emit: filtered

    script:
    // An empty pattern means "no passthrough columns"; the default.
    def pattern = passthrough_parquets ? "'passthrough_files=passthrough/*.parquet'" : 'passthrough_files=null'
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.filter_aggregate \\
        output_dir=. \\
        aggregate_file=${aggregate_parquet} \\
        blocklist_file=${blocklist_parquet} \\
        ${pattern} \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
