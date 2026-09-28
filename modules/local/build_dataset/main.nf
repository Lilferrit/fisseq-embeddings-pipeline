// BUILD_DATASET. One experiment's cells, cropped straight out of
// starcall's whole-tile images into a sharded WebDataset. tiles.parquet
// names those images by their real paths under phenotyping_dir, which
// nextflow.config binds into this task's container (the `phenotyping_dir`
// input). cell_images_dir=. because both parquets are staged here.

include { threadEnv } from '../functions'

process BUILD_DATASET {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/dataset/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(dataset_args), path(cell_table), path(tiles), val(phenotyping_dir)

    output:
    tuple val(batch_stem), path("dataset-*.tar"), path("metadata.parquet"), emit: dataset

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.dataset \\
        output_dir=. \\
        batch_stem=${batch_stem} \\
        cell_images_dir=. \\
        ${dataset_args} \\
        random_seed=${params.random_seed}
    """
}
