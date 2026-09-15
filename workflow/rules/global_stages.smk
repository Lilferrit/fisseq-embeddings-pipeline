# The four cross-experiment stages. Each collects one file per experiment
# into a single job.
#
# Under Nextflow these needed a stageAs numbering workaround, because every
# experiment's file has the same basename (aggregate.parquet /
# results.parquet) and they all landed in one flat task directory. expand()
# hands over real, distinct paths instead, so each stage now takes an
# explicit input_files list paired positionally with batch_stems.
#
# One deliberate difference from Nextflow: its .collect() would run a global
# stage over whatever experiments happened to survive, so a crash in one
# experiment silently changed what the cross-experiment median was computed
# over. Here a missing input simply means the global job doesn't run. That is
# the stricter and more reproducible behavior; see docs/snakemake.md.


rule global_variant_embeddings:
    """Cross-experiment median pooling, then PCA at the full retained rank.
    cumulative_variance_explained controls only the extra pca_reduced.parquet;
    the full-rank scores/components/variance files are unaffected."""
    input:
        expand("feature_select_batchwise/{batch}/aggregate.parquet", batch=BATCHES),
    output:
        median="global/embeddings/median_aggregate.parquet",
        scores="global/embeddings/pca_scores.parquet",
        components="global/embeddings/pca_components.parquet",
        variance="global/embeddings/pca_variance_explained.parquet",
        reduced="global/embeddings/pca_reduced.parquet",
    params:
        input_files=lambda wc, input: hydra_list("input_files", input),
        batch_stems=hydra_list("batch_stems", BATCHES),
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.global_embeddings \
            output_dir=$(dirname {output.median}) \
            {params.input_files} \
            {params.batch_stems} \
            label_column={config[filter_label_column]} \
            cumulative_variance_explained={config[global_variant_embeddings_cumulative_variance_explained]} \
            random_seed={config[random_seed]}
        """


rule global_variant_distinguishability:
    """Global Variant Distinguish-ability Scores, pooled across experiments."""
    input:
        expand("ovwt_batchwise/{batch}/results.parquet", batch=BATCHES),
    output:
        "global/distinguishability/global_scores.parquet",
    params:
        input_files=lambda wc, input: hydra_list("input_files", input),
        batch_stems=hydra_list("batch_stems", BATCHES),
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.global_distinguishability \
            output_dir=$(dirname {output}) \
            {params.input_files} \
            {params.batch_stems} \
            label_column={config[filter_label_column]} \
            random_seed={config[random_seed]}
        """


rule global_variant_cp_features:
    input:
        expand(
            "feature_select_batchwise_cp_features/{batch}/aggregate.parquet",
            batch=CP_BATCHES,
        ),
    output:
        median="global/cp_features/median_aggregate.parquet",
        scores="global/cp_features/pca_scores.parquet",
        components="global/cp_features/pca_components.parquet",
        variance="global/cp_features/pca_variance_explained.parquet",
        reduced="global/cp_features/pca_reduced.parquet",
    params:
        input_files=lambda wc, input: hydra_list("input_files", input),
        batch_stems=hydra_list("batch_stems", CP_BATCHES),
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.global_variant_cp_features \
            output_dir=$(dirname {output.median}) \
            {params.input_files} \
            {params.batch_stems} \
            label_column={config[filter_label_column]} \
            cumulative_variance_explained={config[global_variant_cp_features_cumulative_variance_explained]} \
            random_seed={config[random_seed]}
        """


rule global_variant_distinguishability_cp_features:
    input:
        expand("ovwt_batchwise_cp_features/{batch}/results.parquet", batch=CP_BATCHES),
    output:
        "global/distinguishability_cp_features/global_scores.parquet",
    params:
        input_files=lambda wc, input: hydra_list("input_files", input),
        batch_stems=hydra_list("batch_stems", CP_BATCHES),
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.global_variant_distinguishability_cp_features \
            output_dir=$(dirname {output}) \
            {params.input_files} \
            {params.batch_stems} \
            label_column={config[filter_label_column]} \
            random_seed={config[random_seed]}
        """
