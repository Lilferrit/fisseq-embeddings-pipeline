// BUILD_DATASET. Reads a per-experiment config (phenotyping_dir, wells,
// grid_size, segmentation_type, use_corrected, window, shard_maxcount,
// barcode_col_name, aa_changes_col_name, edit_distance_col_name) from
// <pipeline_dir>/configs/<batch>.yaml.
//
// dataset.py's `hydra.main` is registered with a fixed
// `config_name="dataset_main"`/`config_path=None` (ConfigStore, not a
// loadable file) -- unlike fisseq-data-pipeline's INPUT process, there is
// no config-file-loading mode to hand it an external YAML directly. So
// workflows/embeddings.nf's config_ch parses each configs/*.yaml itself
// (batch_stem from the filename, everything else from the file's own
// keys) and this process threads every key through as an individual Hydra
// CLI override -- same list-interpolation convention as
// aggregate_embeddings.nf's `aggregators=[...]`/embed_cells.nf's
// `channels=[...]`. A key the YAML omits simply falls back to
// BuildDatasetConfig's own default (or raises Hydra's own "missing
// mandatory value" error if that field has none, e.g. phenotyping_dir).

process BUILD_DATASET {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/dataset/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(batch_config)

    output:
    tuple val(batch_stem), path("dataset-*.tar"), path("metadata.parquet")

    script:
    def overrides = batch_config.collect { key, value ->
        (value instanceof List) ? "'${key}=[${value.join(",")}]'" : "${key}=${value}"
    }.join(' \\\n        ')
    """
    python -m fisseq_embeddings_pipeline.dataset \\
        output_dir=. \\
        batch_stem=${batch_stem} \\
        ${overrides} \\
        random_seed=${params.random_seed}
    """
}
