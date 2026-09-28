// AGGREGATE_PASSTHROUGH. One passthrough method's lean aggregate over
// EVERY QC-passed cell: aggregate_half.py with no split_file. Nothing
// downstream of it but FILTER_AGGREGATE's final join -- a passthrough
// method never reaches the bootstrap halves, the correlation or the
// blocklist.

include { threadEnv } from '../functions'

process AGGREGATE_PASSTHROUGH {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}/passthrough_aggregates" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(embeddings_parquet), path(filtered_keys_parquet), path(normalizer_parquet), val(method)
    val(bare_columns)

    output:
    tuple val(batch_stem), path("${method}.parquet"), emit: passthrough

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.aggregate_half \\
        output_dir=. \\
        embeddings_file=${embeddings_parquet} \\
        filtered_keys_file=${filtered_keys_parquet} \\
        normalizer_file=${normalizer_parquet} \\
        aggregator=${method} \\
        label_column=${params.filter_label_column} \\
        feature_chunk_size=${params.aggregate_feature_chunk_size == null ? 'null' : params.aggregate_feature_chunk_size} \\
        bare_columns=${bare_columns} \\
        output_name=${method} \\
        random_seed=${params.random_seed}
    """
}
