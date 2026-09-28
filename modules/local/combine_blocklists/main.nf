// COMBINE_BLOCKLISTS. One experiment's per-method blocklists,
// concatenated. Stat suffixes (emb_0000_median vs emb_0000_KS) make each
// method's feature names disjoint, so no deduplication is needed.

include { threadEnv } from '../functions'

process COMBINE_BLOCKLISTS {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(method_blocklists, stageAs: 'blocklists/*')

    output:
    tuple val(batch_stem), path("blocklist.parquet"), emit: blocklist

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.combineblocklists \\
        output_dir=. \\
        'blocklist_files=blocklists/*.parquet' \\
        random_seed=${params.random_seed}
    """
}
