// SPEC.md §6.6 -- OVWT_BATCHWISE (Epic 6). Same three-input shape as
// aggregate_embeddings.nf (§3 decision 10); k-fold CV controlled by
// ovwt_n_folds/ovwt_calibrate, all randomness from random_seed (§3 decision 11).
// TODO(Epic 9): implement per IMPLEMENTATION_CHECKLIST.md Epic 9.

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
        n_folds=${params.ovwt_n_folds} \\
        calibrate=${params.ovwt_calibrate} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        random_seed=${params.random_seed}
    """
}
