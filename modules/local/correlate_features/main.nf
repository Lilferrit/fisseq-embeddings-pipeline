// CORRELATE_FEATURES. Per-dimension Pearson r between one replicate's two
// halves, for one method.

include { threadEnv } from '../functions'

process CORRELATE_FEATURES {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}/correlations/rep${rep}" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(rep), val(method), path(half1_parquet, stageAs: 'half1/*'), path(half2_parquet, stageAs: 'half2/*')

    output:
    tuple val(batch_stem), val(method), path("${method}.parquet"), emit: correlations

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.correlatefeatures \\
        output_dir=. \\
        half1_file=${half1_parquet} \\
        half2_file=${half2_parquet} \\
        label_column=${params.filter_label_column} \\
        output_name=${method} \\
        random_seed=${params.random_seed}
    """
}
