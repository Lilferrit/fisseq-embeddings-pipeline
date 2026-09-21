# The cellDINO track: BUILD_DATASET -> EMBED_CELLS -> FILTER_EMBEDDINGS ->
# {AGGREGATE_EMBEDDINGS, OVWT_BATCHWISE}.


rule build_dataset:
    """One experiment's cells gathered into a sharded WebDataset.

    A `directory()` output because the shard count isn't known ahead of time
    (dataset-000000.tar, dataset-000001.tar, ...). Declaring only
    metadata.parquet would leave those tars unmodelled, so a rerun over a
    smaller input would leave stale high-numbered shards behind for
    embed_cells' glob to silently ingest.

    metadata.parquet is published for the record -- it's which cells actually
    made it into the shards, a possible subset of the cell table -- but
    nothing consumes it; QC runs off build_cell_metadata instead.
    """
    input:
        cell_images="cell_images/{batch}",
    output:
        directory("dataset/{batch}"),
    params:
        overrides=lambda wc: dataset_args(wc.batch),
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.dataset \
            output_dir={output} \
            batch_stem={wildcards.batch} \
            cell_images_dir={input.cell_images} \
            {params.overrides} \
            random_seed={config[random_seed]}

        test -s {output}/metadata.parquet
        """


rule embed_cells:
    """The Cell-DINO forward pass -- the pipeline's one heavy GPU stage.

    Streams build_dataset's shards directly and has no QC dependency: the
    whole point of building the WebDataset up front is that this runs once
    per experiment however many times QC thresholds get retuned afterwards.
    """
    input:
        dataset="dataset/{batch}",
    output:
        "embeddings/{batch}/embeddings.parquet",
    params:
        channels=hydra_list("channels", config["cell_dino_channels"]),
        apply_mask=str(config["cell_dino_apply_mask"]).lower(),
    threads: 8
    resources:
        mem_mb=lambda wc, attempt: 64000 * attempt,
        gpu=1,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.embed \
            output_dir=$(dirname {output}) \
            'shard_pattern={input.dataset}/*.tar' \
            checkpoint_path={config[cell_dino_checkpoint]} \
            arch={config[cell_dino_arch]} \
            patch_size={config[cell_dino_patch_size]} \
            crop_size={config[cell_dino_crop_size]} \
            {params.channels} \
            apply_mask={params.apply_mask} \
            channel_pool={config[cell_dino_channel_pool]} \
            device={config[cell_dino_device]} \
            batch_size={config[cell_dino_batch_size]} \
            num_workers={config[cell_dino_num_workers]} \
            random_seed={config[random_seed]}
        """


rule filter_embeddings:
    """QC-passed join key + the fitted synonymous z-score normalizer. Writes
    no emb_* columns -- no stage copies another stage's data wholesale."""
    input:
        embeddings="embeddings/{batch}/embeddings.parquet",
        qc_passed="qc_filter/{batch}/filtered_cells.parquet",
    output:
        filtered_keys="filter_embeddings/{batch}/filtered_keys.parquet",
        normalizer="filter_embeddings/{batch}/normalizer.parquet",
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.filter \
            output_dir=$(dirname {output.filtered_keys}) \
            embeddings_file={input.embeddings} \
            qc_passed_file={input.qc_passed} \
            label_column={config[filter_label_column]} \
            random_seed={config[random_seed]}
        """


rule aggregate_embeddings:
    """Experiment N Aggregates. Takes the raw embeddings plus the join key
    and normalizer, and reconstructs the QC-passed, synonymous-corrected
    table itself -- it does not read a pre-normalized file."""
    input:
        embeddings="embeddings/{batch}/embeddings.parquet",
        filtered_keys="filter_embeddings/{batch}/filtered_keys.parquet",
        normalizer="filter_embeddings/{batch}/normalizer.parquet",
    output:
        "feature_select_batchwise/{batch}/aggregate.parquet",
    params:
        aggregators=hydra_list("aggregators", config["aggregate_methods"]),
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.aggregate \
            output_dir=$(dirname {output}) \
            embeddings_file={input.embeddings} \
            filtered_keys_file={input.filtered_keys} \
            normalizer_file={input.normalizer} \
            label_column={config[filter_label_column]} \
            {params.aggregators} \
            random_seed={config[random_seed]}
        """


rule ovwt_batchwise:
    """Experiment N Distinguish-ability Scores -- one-vs-wildtype XGBoost
    with cross-validated, optionally calibrated per-cell scores."""
    input:
        embeddings="embeddings/{batch}/embeddings.parquet",
        filtered_keys="filter_embeddings/{batch}/filtered_keys.parquet",
        normalizer="filter_embeddings/{batch}/normalizer.parquet",
    output:
        results="ovwt_batchwise/{batch}/results.parquet",
        cell_scores="ovwt_batchwise/{batch}/cell_scores.parquet",
        models="ovwt_batchwise/{batch}/models.pkl",
    params:
        calibrate=str(config["ovwt_calibrate"]).lower(),
        downsample_wt=str(config["ovwt_downsample_wt"]).lower(),
    threads: 8
    resources:
        mem_mb=lambda wc, attempt: 64000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.ovwt \
            output_dir=$(dirname {output.results}) \
            embeddings_file={input.embeddings} \
            filtered_keys_file={input.filtered_keys} \
            normalizer_file={input.normalizer} \
            label_column={config[filter_label_column]} \
            wt_label={config[ovwt_wt_label]} \
            n_folds={config[ovwt_n_folds]} \
            calibrate={params.calibrate} \
            min_cells={config[ovwt_min_cells]} \
            downsample_wt={params.downsample_wt} \
            random_seed={config[random_seed]}
        """


# ── Reproducibility filtering (cellDINO track only) ─────────────────────────
# GENERATE_SPLIT -> AGGREGATE_HALF -> CORRELATE_FEATURES -> BLOCKLIST ->
# COMBINE_BLOCKLISTS -> FILTER_AGGREGATE. A dimension is kept when the
# variant-to-variant pattern it reports from one random half of an
# experiment's cells is the pattern it reports from the other half, at median
# Pearson r >= reproducibility_min_correlation across
# reproducibility_bootstrap_reps independent splits.
#
# The fan-out mirrors fisseq-data-pipeline's: a separate job per (replicate,
# half, method). One method per job is what keeps the reference-based
# aggregators' peak memory bounded -- together with
# aggregate_feature_chunk_size, which is why that knob was ported at the same
# time. See docs/architecture.md and params.yaml.


rule generate_split:
    """One bootstrap replicate's stratified 50/50 pseudo-replicate split.

    Reads filtered_keys.parquet alone -- it already carries the composite cell
    key, meta_is_control and the label column, so there's no reason to touch
    the much larger embeddings.parquet just to decide which cells go where.
    The two halves are written as JOIN_KEYS rows, not row indices: both this
    rule and aggregate_half reconstruct the cell table through a join, whose
    row order Polars does not guarantee to be stable across processes.
    """
    input:
        filtered_keys="filter_embeddings/{batch}/filtered_keys.parquet",
    output:
        half1="feature_select_batchwise/{batch}/splits/rep{rep}/half1.parquet",
        half2="feature_select_batchwise/{batch}/splits/rep{rep}/half2.parquet",
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.generatesplit \
            output_dir=$(dirname {output.half1}) \
            filtered_keys_file={input.filtered_keys} \
            label_column={config[filter_label_column]} \
            bootstrap_idx={wildcards.rep} \
            random_seed={config[random_seed]}
        """


rule aggregate_half:
    """One method's lean aggregate over one pseudo-replicate half.

    A separate job per (replicate, half, method) -- the fan-out the sibling
    repo uses. Lean output ([label] + this method's stat columns): no
    normalizer, no metadata join. A half's meta_num_cells would be actively
    misleading, and correlate_features wants nothing but the stat columns.
    """
    input:
        embeddings="embeddings/{batch}/embeddings.parquet",
        filtered_keys="filter_embeddings/{batch}/filtered_keys.parquet",
        normalizer="filter_embeddings/{batch}/normalizer.parquet",
        split="feature_select_batchwise/{batch}/splits/rep{rep}/half{half}.parquet",
    output:
        "feature_select_batchwise/{batch}/half_aggregates/rep{rep}/half{half}/{method}.parquet",
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.aggregate_half \
            output_dir=$(dirname {output}) \
            embeddings_file={input.embeddings} \
            filtered_keys_file={input.filtered_keys} \
            normalizer_file={input.normalizer} \
            aggregator={wildcards.method} \
            split_file={input.split} \
            label_column={config[filter_label_column]} \
            feature_chunk_size={config[aggregate_feature_chunk_size]} \
            bare_columns={BARE_COLUMNS} \
            output_name={wildcards.method} \
            random_seed={config[random_seed]}
        """


rule aggregate_passthrough:
    """One passthrough method's lean aggregate over EVERY QC-passed cell.

    Same module as aggregate_half with no split_file -- fisseq-data-pipeline
    aliases one process the same way. Deliberately nothing downstream of it
    but filter_aggregate's final join: a passthrough method never reaches the
    bootstrap halves, the correlation or the blocklist.
    """
    input:
        embeddings="embeddings/{batch}/embeddings.parquet",
        filtered_keys="filter_embeddings/{batch}/filtered_keys.parquet",
        normalizer="filter_embeddings/{batch}/normalizer.parquet",
    output:
        "feature_select_batchwise/{batch}/passthrough_aggregates/{method}.parquet",
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.aggregate_half \
            output_dir=$(dirname {output}) \
            embeddings_file={input.embeddings} \
            filtered_keys_file={input.filtered_keys} \
            normalizer_file={input.normalizer} \
            aggregator={wildcards.method} \
            label_column={config[filter_label_column]} \
            feature_chunk_size={config[aggregate_feature_chunk_size]} \
            bare_columns={BARE_COLUMNS} \
            output_name={wildcards.method} \
            random_seed={config[random_seed]}
        """


rule correlate_features:
    """Per-dimension Pearson r between one replicate's two halves."""
    input:
        half1="feature_select_batchwise/{batch}/half_aggregates/rep{rep}/half1/{method}.parquet",
        half2="feature_select_batchwise/{batch}/half_aggregates/rep{rep}/half2/{method}.parquet",
    output:
        "feature_select_batchwise/{batch}/correlations/rep{rep}/{method}.parquet",
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.correlatefeatures \
            output_dir=$(dirname {output}) \
            half1_file={input.half1} \
            half2_file={input.half2} \
            label_column={config[filter_label_column]} \
            output_name={wildcards.method} \
            random_seed={config[random_seed]}
        """


rule blocklist:
    """One method's verdict: median r across every bootstrap replicate.

    The one intentional synchronization point across replicates -- every other
    rule in the chain fans out per replicate, this one gathers them.
    """
    input:
        lambda wc: expand(
            "feature_select_batchwise/{batch}/correlations/rep{rep}/{method}.parquet",
            batch=wc.batch,
            rep=REPS,
            method=wc.method,
        ),
    output:
        "feature_select_batchwise/{batch}/blocklists/{method}.parquet",
    params:
        pattern=lambda wc: (
            f"feature_select_batchwise/{wc.batch}/correlations/rep*/{wc.method}.parquet"
        ),
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.blocklist \
            output_dir=$(dirname {output}) \
            'correlation_files={params.pattern}' \
            minimum_correlation={config[reproducibility_min_correlation]} \
            output_name={wildcards.method} \
            random_seed={config[random_seed]}
        """


rule combine_blocklists:
    """One experiment's per-method blocklists, concatenated.

    A plain concat: stat suffixes (emb_0000_median vs emb_0000_KS) make each
    method's feature names disjoint, so no deduplication is needed.
    """
    input:
        expand(
            "feature_select_batchwise/{{batch}}/blocklists/{method}.parquet",
            method=AGG_METHODS,
        ),
    output:
        "feature_select_batchwise/{batch}/blocklist.parquet",
    params:
        pattern=lambda wc: (
            f"feature_select_batchwise/{wc.batch}/blocklists/*.parquet"
        ),
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.combineblocklists \
            output_dir=$(dirname {output}) \
            'blocklist_files={params.pattern}' \
            random_seed={config[random_seed]}
        """


rule filter_aggregate:
    """Aggregates -> filtered aggregates, and the terminal passthrough view.

    Two outputs on purpose. filtered_aggregate.parquet carries only
    reproducible columns; aggregate_with_passthrough.parquet adds the
    passthrough methods and is terminal -- nothing in-pipeline reads it. The
    split is what keeps passthrough columns out of the PCA across a file
    boundary: global_variant_embeddings re-reads its input from disk and picks
    features with FEATURE_SELECTOR, which would match emb_0000_KSnegLogP.
    """
    input:
        aggregate="feature_select_batchwise/{batch}/aggregate.parquet",
        blocklist="feature_select_batchwise/{batch}/blocklist.parquet",
        passthrough=expand(
            "feature_select_batchwise/{{batch}}/passthrough_aggregates/{method}.parquet",
            method=PASSTHROUGH_METHODS,
        ),
    output:
        filtered="feature_select_batchwise/{batch}/filtered_aggregate.parquet",
        with_passthrough=(
            "feature_select_batchwise/{batch}/aggregate_with_passthrough.parquet"
        ),
    params:
        # Empty when aggregate_methods_passthrough is [] -- the default. The
        # module treats a null/empty pattern as "no passthrough columns", and
        # a non-empty pattern matching nothing as a warning, not an error.
        pattern=lambda wc: (
            f"feature_select_batchwise/{wc.batch}/passthrough_aggregates/*.parquet"
            if PASSTHROUGH_METHODS
            else "null"
        ),
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.filter_aggregate \
            output_dir=$(dirname {output.filtered}) \
            aggregate_file={input.aggregate} \
            blocklist_file={input.blocklist} \
            'passthrough_files={params.pattern}' \
            label_column={config[filter_label_column]} \
            random_seed={config[random_seed]}
        """
