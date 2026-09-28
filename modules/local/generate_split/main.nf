// GENERATE_SPLIT. One bootstrap replicate's stratified 50/50
// pseudo-replicate split, seeded at random_seed + rep.
//
// Reads filtered_keys.parquet alone -- it already carries the composite
// cell key, meta_is_control and the label column. The halves are written
// as JOIN_KEYS rows, not row indices: both this stage and AGGREGATE_HALF
// reconstruct the cell table through a join, whose row order Polars does
// not guarantee to be stable across processes.

include { threadEnv } from '../functions'

process GENERATE_SPLIT {
    errorStrategy 'ignore'
    label 'process_low'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}/splits/rep${rep}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(filtered_keys_parquet), val(rep)

    output:
    tuple val(batch_stem), val(rep), path("half1.parquet"), path("half2.parquet"), emit: split

    script:
    """
    ${threadEnv(task.cpus)}
    python -m fisseq_embeddings_pipeline.generatesplit \\
        output_dir=. \\
        filtered_keys_file=${filtered_keys_parquet} \\
        label_column=${params.filter_label_column} \\
        bootstrap_idx=${rep} \\
        random_seed=${params.random_seed}
    """
}
