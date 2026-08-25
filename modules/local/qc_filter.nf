// SPEC.md §6.2 -- QC_FILTER (Epic 2, vendored close to verbatim from
// fisseq-data-pipeline's qcfilter.py/modules/local/qc_filter.nf). Reads
// BUILD_DATASET's metadata.parquet only, never the WebDataset shards.
// TODO(Epic 9): implement per IMPLEMENTATION_CHECKLIST.md Epic 9.

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
        barcode_count_threshold=${params.barcode_count_threshold} \\
        variant_barcode_count_threshold=${params.variant_barcode_count_threshold} \\
        edit_distance_threshold=${params.edit_distance_threshold} \\
        random_seed=${params.random_seed}
    """
}
