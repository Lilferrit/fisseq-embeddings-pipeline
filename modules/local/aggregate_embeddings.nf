// SPEC.md §6.5 -- AGGREGATE_EMBEDDINGS (Epic 5). Takes embeddings.parquet +
// filtered_keys.parquet + normalizer.parquet (three inputs, §3 decision 10 --
// NOT a pre-normalized single file) and reconstructs filtered_lf itself via
// load_filtered_embeddings() before aggregating.
// TODO(Epic 9): wire into workflows/embeddings.nf and verify against a real
// (small) Nextflow run per IMPLEMENTATION_CHECKLIST.md Epic 9 / Epic 5
// Story 5.3's last bullet. The `aggregators=[...]` list interpolation below
// is new, unverified-until-Epic-9 territory -- no other module in this repo
// passes a List-typed param through to a Hydra CLI override yet.

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
