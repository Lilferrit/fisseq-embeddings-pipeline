// OVWT_BATCHWISE. Same three-input shape as aggregate_embeddings.nf;
// k-fold CV controlled by ovwt_n_folds/ovwt_calibrate, all randomness from
// random_seed. `label_column` reuses the same `filter_label_column` param
// every other stage (FILTER_EMBEDDINGS/AGGREGATE_EMBEDDINGS/the two global
// stages) keys off, so overriding it changes every stage's label column
// together; `wt_label` gets its own `ovwt_wt_label` (default "WT").

process OVWT_BATCHWISE {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/ovwt_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(embeddings_parquet), path(filtered_keys_parquet), path(normalizer_parquet)

    output:
    tuple val(batch_stem), path("results.parquet"), path("cell_scores.parquet"), path("models.pkl")

    script:
    """
    python -m fisseq_embeddings_pipeline.ovwt \\
        output_dir=. \\
        embeddings_file=${embeddings_parquet} \\
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
