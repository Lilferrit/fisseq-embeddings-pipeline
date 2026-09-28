// GLOBAL_BLOCKLIST. The cross-experiment reproducibility vote: each
// experiment decides from its own cells which dimensions are reproducible;
// this requires agreement (in every experiment that reported on a
// dimension, or in at least reproducibility_global_min_batches_ok of them).

include { threadEnv; hydraList } from '../functions'

process GLOBAL_BLOCKLIST {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/embeddings" }, mode: 'copy'

    input:
    tuple val(batch_stems), path(blocklists, stageAs: 'input_*.parquet')

    output:
    path("blocklist.parquet"), emit: blocklist

    script:
    def min_ok = params.reproducibility_global_min_batches_ok == null ? 'null' : params.reproducibility_global_min_batches_ok
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.global_blocklist \\
        output_dir=. \\
        ${hydraList('input_files', blocklists)} \\
        min_batches_ok=${min_ok} \\
        random_seed=${params.random_seed}
    """
}
