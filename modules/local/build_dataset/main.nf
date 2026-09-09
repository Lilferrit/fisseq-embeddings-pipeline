// BUILD_DATASET. Reads a per-experiment config (cell_images_dir, window,
// shard_maxcount, barcode_col_name, aa_changes_col_name,
// edit_distance_col_name) from one entry of params.yaml's `experiments:`
// list, plus cell_images_dir itself -- BUILD_CELL_IMAGES' own per-experiment
// output directory, injected by workflows/embeddings.nf (not a
// params.experiments field) -- see that workflow's config_ch. No
// starcall-workflow-facing field (phenotyping_dir/wells/grid_size/
// segmentation_type/use_corrected/...) reaches this process any more:
// BUILD_CELL_IMAGES is the only stage that still needs those (see its own
// module comment).
//
// dataset.py's `hydra.main` is registered with a fixed
// `config_name="dataset_main"`/`config_path=None` (ConfigStore, not a
// loadable file) -- unlike fisseq-data-pipeline's INPUT process, there is
// no config-file-loading mode to hand it an external YAML directly. So
// workflows/embeddings.nf validates params.experiments itself (batch_stem
// required and unique per entry) and this process threads every other key
// through as an individual Hydra CLI override -- same list-interpolation
// convention as aggregate_embeddings/main.nf's `aggregators=[...]`/
// embed_cells/main.nf's `channels=[...]`. A key an entry omits simply falls
// back to BuildDatasetConfig's own default (or raises Hydra's own
// "missing mandatory value" error if that field has none, e.g.
// cell_images_dir). cell_images_dir is also bind-mounted into this
// process' container -- see nextflow.config's own
// withName:'BUILD_DATASET|BUILD_CP_FEATURES' containerOptions comment.

process BUILD_DATASET {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/dataset/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(batch_config)

    output:
    tuple val(batch_stem), path("dataset-*.tar"), path("metadata.parquet"), emit: dataset

    when:
    task.ext.when == null || task.ext.when

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
