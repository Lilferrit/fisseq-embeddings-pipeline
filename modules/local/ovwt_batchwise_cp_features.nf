// OVWT_BATCHWISE_CP_FEATURES. Same three-input shape as
// aggregate_cp_features.nf; reuses the SAME `ovwt_*` params as
// OVWT_BATCHWISE -- OVWT hyperparameters are about scoring methodology,
// not feature type, so there is no parallel `ovwt_*_cp_features` set.

process OVWT_BATCHWISE_CP_FEATURES {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/ovwt_batchwise_cp_features/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(cp_features_parquet), path(filtered_keys_parquet), path(normalizer_parquet)

    output:
    tuple val(batch_stem), path("results.parquet"), path("cell_scores.parquet"), path("models.pkl")

    script:
    """
    python -m fisseq_embeddings_pipeline.ovwt_cp_features \\
        output_dir=. \\
        cp_features_file=${cp_features_parquet} \\
        filtered_keys_file=${filtered_keys_parquet} \\
        normalizer_file=${normalizer_parquet} \\
        label_column=${params.filter_label_column} \\
        wt_label=${params.ovwt_wt_label} \\
        n_folds=${params.ovwt_n_folds} \\
        calibrate=${params.ovwt_calibrate} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        random_seed=${params.random_seed}
    """
}
