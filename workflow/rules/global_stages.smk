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


rule global_blocklist:
    """The cross-experiment reproducibility vote.

    Each experiment decides independently which dimensions are reproducible
    from its own cells; this gathers those verdicts and requires agreement
    (in every experiment that reported on a dimension, or in at least
    reproducibility_global_min_batches_ok of them).
    """
    input:
        expand("feature_select_batchwise/{batch}/blocklist.parquet", batch=BATCHES),
    output:
        "global/embeddings/blocklist.parquet",
    params:
        input_files=lambda wc, input: hydra_list("input_files", input),
        min_batches_ok=config["reproducibility_global_min_batches_ok"] or "null",
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.global_blocklist \
            output_dir=$(dirname {output}) \
            {params.input_files} \
            min_batches_ok={params.min_batches_ok} \
            random_seed={config[random_seed]}
        """


rule global_variant_embeddings:
    """Cross-experiment median pooling, then PCA at the full retained rank.
    cumulative_variance_explained controls only the extra pca_reduced.parquet;
    the full-rank scores/components/variance files are unaffected.

    Reads each experiment's UNFILTERED aggregate.parquet and applies the
    global blocklist itself, rather than reading the per-experiment
    filtered_aggregate.parquet. That is deliberate: median_across_batches
    intersects feature columns across experiments, so consuming the filtered
    files would silently reduce every setting to "reproducible in every
    experiment" and make reproducibility_global_min_batches_ok inert.
    filtered_aggregate.parquet is the per-experiment deliverable; this is the
    global one.
    """
    input:
        aggregates=expand(
            "feature_select_batchwise/{batch}/aggregate.parquet", batch=BATCHES
        ),
        blocklist="global/embeddings/blocklist.parquet",
    output:
        median="global/embeddings/median_aggregate.parquet",
        scores="global/embeddings/pca_scores.parquet",
        components="global/embeddings/pca_components.parquet",
        variance="global/embeddings/pca_variance_explained.parquet",
        reduced="global/embeddings/pca_reduced.parquet",
    params:
        input_files=lambda wc, input: hydra_list("input_files", input.aggregates),
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
            blocklist_file={input.blocklist} \
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
