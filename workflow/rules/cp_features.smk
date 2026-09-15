# The optional CellProfiler-feature track: the same cells as the cellDINO
# track, in a different feature space.
#
# An experiments: entry opts itself in with `cp_features: true` -- there's no
# separate list to keep in sync. That does two things: build_cell_images
# (which always runs) additionally forces that experiment's CellProfiler CSV
# to exist and folds its columns into cell_table.parquet, and build_cp_features
# selects them back out.
#
# This track reuses the SAME qc_filter outputs as the cellDINO track -- there
# is no second QC pass. Every rule after build_cp_features is a thin wrapper
# around the cellDINO track's own function with a different feature selector.


rule build_cp_features:
    """A flat read + column-select against cell_table.parquet. No tile
    discovery and no CSV reads of its own -- build_cell_images already
    folded this experiment's CellProfiler columns in."""
    input:
        cell_images="cell_images/{batch}",
    output:
        "cp_features/{batch}/cp_features.parquet",
    params:
        overrides=lambda wc: cp_features_args(wc.batch),
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.cp_features \
            output_dir=$(dirname {output}) \
            batch_stem={wildcards.batch} \
            cell_images_dir={input.cell_images} \
            {params.overrides} \
            random_seed={config[random_seed]}
        """


rule filter_cp_features:
    input:
        cp_features="cp_features/{batch}/cp_features.parquet",
        qc_passed="qc_filter/{batch}/filtered_cells.parquet",
    output:
        filtered_keys="filter_cp_features/{batch}/filtered_keys.parquet",
        normalizer="filter_cp_features/{batch}/normalizer.parquet",
    threads: 2
    resources:
        mem_mb=lambda wc, attempt: 8000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.filter_cp_features \
            output_dir=$(dirname {output.filtered_keys}) \
            cp_features_file={input.cp_features} \
            qc_passed_file={input.qc_passed} \
            label_column={config[filter_label_column]} \
            random_seed={config[random_seed]}
        """


rule aggregate_cp_features:
    """Experiment N CP Aggregates. Median-only by default -- CellProfiler
    features are hand-engineered, interpretable columns where a single median
    is the established baseline; KS/AUROC remain an explicit opt-in."""
    input:
        cp_features="cp_features/{batch}/cp_features.parquet",
        filtered_keys="filter_cp_features/{batch}/filtered_keys.parquet",
        normalizer="filter_cp_features/{batch}/normalizer.parquet",
    output:
        "feature_select_batchwise_cp_features/{batch}/aggregate.parquet",
    params:
        aggregators=hydra_list(
            "aggregators", config["aggregate_methods_cp_features"]
        ),
    threads: 4
    resources:
        mem_mb=lambda wc, attempt: 32000 * attempt,
    container:
        config["container_image"]
    shell:
        THREAD_ENV
        + """
        python -m fisseq_embeddings_pipeline.aggregate_cp_features \
            output_dir=$(dirname {output}) \
            cp_features_file={input.cp_features} \
            filtered_keys_file={input.filtered_keys} \
            normalizer_file={input.normalizer} \
            label_column={config[filter_label_column]} \
            {params.aggregators} \
            random_seed={config[random_seed]}
        """


rule ovwt_batchwise_cp_features:
    """The OVWT scoring hyperparameters are shared verbatim with the cellDINO
    track -- they're scoring methodology, not tied to feature type, so there
    is no parallel *_cp_features set of them in params.yaml."""
    input:
        cp_features="cp_features/{batch}/cp_features.parquet",
        filtered_keys="filter_cp_features/{batch}/filtered_keys.parquet",
        normalizer="filter_cp_features/{batch}/normalizer.parquet",
    output:
        results="ovwt_batchwise_cp_features/{batch}/results.parquet",
        cell_scores="ovwt_batchwise_cp_features/{batch}/cell_scores.parquet",
        models="ovwt_batchwise_cp_features/{batch}/models.pkl",
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
        python -m fisseq_embeddings_pipeline.ovwt_cp_features \
            output_dir=$(dirname {output.results}) \
            cp_features_file={input.cp_features} \
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
