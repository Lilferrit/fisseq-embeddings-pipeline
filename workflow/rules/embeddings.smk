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
