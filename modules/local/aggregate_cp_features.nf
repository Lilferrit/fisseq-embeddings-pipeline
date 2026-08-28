// AGGREGATE_CP_FEATURES. Takes cp_features.parquet + filtered_keys.parquet
// + normalizer.parquet (three inputs, not a pre-normalized single file)
// and reconstructs filtered_lf itself via load_filtered_embeddings()
// before aggregating -- same shape as aggregate_embeddings.nf.
// `aggregate_methods_cp_features` defaults to `["median"]`, unlike
// AGGREGATE_EMBEDDINGS' `aggregate_methods` (`["median", "KS", "AUROC"]`).

process AGGREGATE_CP_FEATURES {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise_cp_features/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(cp_features_parquet), path(filtered_keys_parquet), path(normalizer_parquet)

    output:
    tuple val(batch_stem), path("aggregate.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.aggregate_cp_features \\
        output_dir=. \\
        cp_features_file=${cp_features_parquet} \\
        filtered_keys_file=${filtered_keys_parquet} \\
        normalizer_file=${normalizer_parquet} \\
        label_column=${params.filter_label_column} \\
        'aggregators=[${params.aggregate_methods_cp_features.join(",")}]' \\
        random_seed=${params.random_seed}
    """
}
