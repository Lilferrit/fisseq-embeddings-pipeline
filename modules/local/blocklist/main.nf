// BLOCKLIST. One method's verdict: the median r across every bootstrap
// replicate against reproducibility_min_correlation. The one
// synchronization point across replicates -- every other task in the chain
// fans out per replicate, this one gathers them.

include { threadEnv } from '../functions'

process BLOCKLIST {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}/blocklists" }, mode: 'copy'

    input:
    tuple val(batch_stem), val(method), path(correlation_parquets, stageAs: 'correlations/rep*.parquet')

    output:
    tuple val(batch_stem), path("${method}.parquet"), emit: blocklist

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.blocklist \\
        output_dir=. \\
        'correlation_files=correlations/*.parquet' \\
        minimum_correlation=${params.reproducibility_min_correlation} \\
        output_name=${method} \\
        random_seed=${params.random_seed}
    """
}
