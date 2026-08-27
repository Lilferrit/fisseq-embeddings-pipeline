// SPEC.md §6.2 -- QC_FILTER (Epic 2, vendored close to verbatim from
// fisseq-data-pipeline's qcfilter.py/modules/local/qc_filter.nf). Reads
// BUILD_DATASET's metadata.parquet only, never the WebDataset shards.
//
// Epic 9 fix: QcFilterConfig's actual field names are `bc_threshold` /
// `variant_bc_threshold` (not `barcode_count_threshold` /
// `variant_barcode_count_threshold`, which are only params.yaml's key
// names) -- the CLI overrides below were passing values under the wrong
// keys, silently leaving Hydra's own defaults (10 / 4) in effect no
// matter what params.yaml set.
//
// Epic 10 addition: `label_column` now reuses `params.filter_label_column`
// (same param FILTER_EMBEDDINGS/AGGREGATE_EMBEDDINGS/OVWT_BATCHWISE/the two
// global stages all key off) so overriding it changes every stage's label
// column together instead of leaving QC_FILTER's own copy silently at its
// Hydra default. `QcFilterConfig`'s `barcode_col_name`/`aa_changes_col_name`/
// `edit_distance_col_name` are deliberately left unwired -- they name
// BUILD_DATASET's fixed `meta_barcode`/`meta_aa_changes`/`meta_edit_distance`
// output columns (Epic 1's meta_* convention), not something a per-run
// override should ever need to change. The `n_variants`/
// `variant_downsample_classes`/`variant_downsample_mode`/
// `variant_allow_list_file` opt-in downsampling cap (off by default,
// `n_variants=null`) is also deliberately left unwired for v1 -- see
// params.yaml's QC_FILTER comment for why.

process QC_FILTER {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/qc_filter/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(metadata_parquet)

    output:
    tuple val(batch_stem), path("filtered_cells.parquet"), path("barcode_counts.parquet"), path("variants_per_barcode.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.qcfilter \\
        output_dir=. \\
        cell_files=${metadata_parquet} \\
        bc_threshold=${params.barcode_count_threshold} \\
        variant_bc_threshold=${params.variant_barcode_count_threshold} \\
        edit_distance_threshold=${params.edit_distance_threshold} \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
