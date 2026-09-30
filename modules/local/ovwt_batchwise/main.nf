// OVWT_BATCHWISE. Same three-input shape as aggregate_embeddings/main.nf;
// cross-validation controlled by ovwt_cv_mode/ovwt_n_folds/ovwt_calibrate,
// all randomness from random_seed. ovwt_cv_mode is "kfold" (ovwt_n_folds
// folds, every barcode in every fold's training set) or "barcode_holdout"
// (whole barcodes held out of training, one barcode group per fold, so a
// single-barcode variant is skipped). Under "barcode_holdout" ovwt_n_folds
// caps the fold count, and null -- which Groovy renders as the literal
// `null` that Hydra parses back into None -- means one fold per barcode.
// Both are validated up front by PLAN_EXPERIMENTS (config/experiments.py),
// since errorStrategy 'ignore' would otherwise swallow a bad value here. `label_column` reuses the same `filter_label_column` param
// every other stage (FILTER_EMBEDDINGS/AGGREGATE_EMBEDDINGS/the two global
// stages) keys off, so overriding it changes every stage's label column
// together; `wt_label` gets its own `ovwt_wt_label` (default "WT").

include { threadEnv } from '../functions'

process OVWT_BATCHWISE {
    errorStrategy 'ignore'
    label 'process_high'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/ovwt_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(embeddings_parquet), path(filtered_keys_parquet), path(normalizer_parquet)

    output:
    tuple val(batch_stem), path("results.parquet"), path("cell_scores.parquet"), path("models.pkl"), emit: ovwt

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.ovwt \\
        output_dir=. \\
        embeddings_file=${embeddings_parquet} \\
        filtered_keys_file=${filtered_keys_parquet} \\
        normalizer_file=${normalizer_parquet} \\
        label_column=${params.filter_label_column} \\
        wt_label=${params.ovwt_wt_label} \\
        cv_mode=${params.ovwt_cv_mode} \\
        n_folds=${params.ovwt_n_folds} \\
        calibrate=${params.ovwt_calibrate} \\
        min_cells=${params.ovwt_min_cells} \\
        downsample_wt=${params.ovwt_downsample_wt} \\
        random_seed=${params.random_seed}
    """
}
