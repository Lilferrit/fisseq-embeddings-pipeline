// AGGREGATE_EMBEDDINGS. Takes embeddings.parquet + filtered_keys.parquet +
// normalizer.parquet (three inputs, not a pre-normalized single file) and
// reconstructs filtered_lf itself via load_filtered_embeddings() before
// aggregating.

include { threadEnv } from '../functions'

process AGGREGATE_EMBEDDINGS {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(embeddings_parquet), path(filtered_keys_parquet), path(normalizer_parquet)

    output:
    tuple val(batch_stem), path("aggregate.parquet"), emit: aggregate

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.aggregate \\
        output_dir=. \\
        embeddings_file=${embeddings_parquet} \\
        filtered_keys_file=${filtered_keys_parquet} \\
        normalizer_file=${normalizer_parquet} \\
        label_column=${params.filter_label_column} \\
        'aggregators=[${params.aggregate_methods.join(",")}]' \\
        feature_chunk_size=${params.aggregate_feature_chunk_size == null ? 'null' : params.aggregate_feature_chunk_size} \\
        random_seed=${params.random_seed}
    """
}
