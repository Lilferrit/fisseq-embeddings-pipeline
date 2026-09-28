// AGGREGATE_HALF. One method's lean aggregate over one pseudo-replicate
// half: [label] + this method's stat columns, no normalizer output and no
// metadata join. One task per (replicate, half, method) -- one method per
// task, together with aggregate_feature_chunk_size, is what keeps the
// reference-based aggregators' peak memory bounded.
//
// bare_columns comes from the workflow, which can see the whole
// aggregate_methods list: column names must match aggregate.parquet's
// exactly or FILTER_AGGREGATE's blocklist finds nothing to drop.

include { threadEnv } from '../functions'

process AGGREGATE_HALF {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}/half_aggregates/rep${rep}/half${half}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(embeddings_parquet), path(filtered_keys_parquet), path(normalizer_parquet), val(rep), val(half), path(split_parquet), val(method)
    val(bare_columns)

    output:
    tuple val(batch_stem), val(rep), val(method), val(half), path("${method}.parquet"), emit: half_aggregate

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.aggregate_half \\
        output_dir=. \\
        embeddings_file=${embeddings_parquet} \\
        filtered_keys_file=${filtered_keys_parquet} \\
        normalizer_file=${normalizer_parquet} \\
        aggregator=${method} \\
        split_file=${split_parquet} \\
        label_column=${params.filter_label_column} \\
        feature_chunk_size=${params.aggregate_feature_chunk_size == null ? 'null' : params.aggregate_feature_chunk_size} \\
        bare_columns=${bare_columns} \\
        output_name=${method} \\
        random_seed=${params.random_seed}
    """
}
