// GLOBAL_VARIANT_DISTINGUISHABILITY. Per-experiment OVWT results pooled
// across experiments.
//
// Every experiment's file has the same basename, so they're staged under
// numbered names and passed as an explicit input_files list, paired
// positionally with batch_stems -- the workflow hands both over as one
// sorted tuple so the pairing can't drift.

include { threadEnv; hydraList } from '../functions'

process GLOBAL_VARIANT_DISTINGUISHABILITY {
    errorStrategy 'ignore'
    label 'process_medium'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/global/distinguishability" }, mode: 'copy'

    input:
    tuple val(batch_stems), path(results_parquets, stageAs: "res_input_*.parquet")

    output:
    path("global_scores.parquet"), emit: global_scores

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.global_distinguishability \\
        output_dir=. \\
        ${hydraList('input_files', results_parquets)} \\
        ${hydraList('batch_stems', batch_stems)} \\
        label_column=${params.filter_label_column} \\
        random_seed=${params.random_seed}
    """
}
