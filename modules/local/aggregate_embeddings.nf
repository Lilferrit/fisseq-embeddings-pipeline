// SPEC.md §6.5 -- AGGREGATE_EMBEDDINGS (Epic 5). Takes embeddings.parquet +
// filtered_keys.parquet + normalizer.parquet (three inputs, §3 decision 10 --
// NOT a pre-normalized single file) and reconstructs filtered_lf itself via
// load_filtered_embeddings() before aggregating.
// Wired into workflows/embeddings.nf (Epic 9); the `aggregators=[...]`
// list interpolation below is unverified against a real `nextflow run`
// (no nextflow/docker available in this sandbox -- see
// IMPLEMENTATION_CHECKLIST.md Epic 9 Story 9.3's note).

process AGGREGATE_EMBEDDINGS {
    errorStrategy 'ignore'
    container "${params.container_image}"
    publishDir { "${params.pipeline_dir}/feature_select_batchwise/${batch_stem}" }, mode: 'copy'

    input:
    tuple val(batch_stem), path(embeddings_parquet), path(filtered_keys_parquet), path(normalizer_parquet)

    output:
    tuple val(batch_stem), path("aggregate.parquet")

    script:
    """
    python -m fisseq_embeddings_pipeline.aggregate \\
        output_dir=. \\
        embeddings_file=${embeddings_parquet} \\
        filtered_keys_file=${filtered_keys_parquet} \\
        normalizer_file=${normalizer_parquet} \\
        label_column=${params.filter_label_column} \\
        'aggregators=[${params.aggregate_methods.join(",")}]' \\
        random_seed=${params.random_seed}
    """
}
