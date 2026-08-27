// SPEC.md §6.6 -- OVWT_BATCHWISE (Epic 6). Same three-input shape as
// aggregate_embeddings.nf (§3 decision 10); k-fold CV controlled by
// ovwt_n_folds/ovwt_calibrate, all randomness from random_seed (§3 decision 11).
// Wired into workflows/embeddings.nf (Epic 9); confirmed against a real
// `nextflow -profile local` run (Epic 9 Story 9.3).
//
// Epic 10 fix: `label_column`/`wt_label` were previously left off this
// script block entirely, so OVWT_BATCHWISE silently used OvwtEmbeddingConfig's
// own Hydra defaults ("meta_aa_changes"/"WT") no matter what params.yaml
// said -- harmless while every default happened to agree, but a latent trap
// the moment `filter_label_column` was ever overridden (every other stage
// -- FILTER_EMBEDDINGS/AGGREGATE_EMBEDDINGS/the two global stages -- already
// keyed off it). `label_column` now reuses that same param;
// `wt_label` gets its own `ovwt_wt_label` (default "WT", SPEC.md §6.6).

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
