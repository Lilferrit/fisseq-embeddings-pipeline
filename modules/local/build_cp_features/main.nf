// BUILD_CP_FEATURES. A flat read + column-select against cell_table.parquet
// -- BUILD_CELL_IMAGES already folded this experiment's CellProfiler
// columns in, so there's no tile discovery or CSV reading here.

include { threadEnv } from '../functions'

process BUILD_CP_FEATURES {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/cp_features/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(cp_features_args), path(cell_table)

    output:
    tuple val(batch_stem), path("cp_features.parquet"), emit: cp_features

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.cp_features \\
        output_dir=. \\
        batch_stem=${batch_stem} \\
        cell_images_dir=. \\
        ${cp_features_args} \\
        random_seed=${params.random_seed}
    """
}
