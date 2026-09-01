// EmbeddingsPipeline. Wires BUILD_DATASET -> {QC_FILTER, EMBED_CELLS} ->
// FILTER_EMBEDDINGS -> {AGGREGATE_EMBEDDINGS, OVWT_BATCHWISE} ->
// {GLOBAL_VARIANT_EMBEDDINGS, GLOBAL_VARIANT_DISTINGUISHABILITY}.
//
// A second, parallel CellProfiler-feature track shares QC_FILTER's output
// (the same cells, just a different feature space) rather than running a
// second QC pass: BUILD_CP_FEATURES -> FILTER_CP_FEATURES ->
// {AGGREGATE_CP_FEATURES, OVWT_BATCHWISE_CP_FEATURES} ->
// {GLOBAL_VARIANT_CP_FEATURES, GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES}.
nextflow.enable.dsl = 2

include { BUILD_CELL_IMAGES } from '../modules/local/build_cell_images'
include { BUILD_DATASET } from '../modules/local/build_dataset'
include { QC_FILTER } from '../modules/local/qc_filter'
include { EMBED_CELLS } from '../modules/local/embed_cells'
include { FILTER_EMBEDDINGS } from '../modules/local/filter_embeddings'
include { AGGREGATE_EMBEDDINGS } from '../modules/local/aggregate_embeddings'
include { OVWT_BATCHWISE } from '../modules/local/ovwt_batchwise'
include { GLOBAL_VARIANT_EMBEDDINGS } from '../modules/local/global_variant_embeddings'
include { GLOBAL_VARIANT_DISTINGUISHABILITY } from '../modules/local/global_variant_distinguishability'
include { BUILD_CP_FEATURES } from '../modules/local/build_cp_features'
include { FILTER_CP_FEATURES } from '../modules/local/filter_cp_features'
include { AGGREGATE_CP_FEATURES } from '../modules/local/aggregate_cp_features'
include { OVWT_BATCHWISE_CP_FEATURES } from '../modules/local/ovwt_batchwise_cp_features'
include { GLOBAL_VARIANT_CP_FEATURES } from '../modules/local/global_variant_cp_features'
include { GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES } from '../modules/local/global_variant_distinguishability_cp_features'

workflow EmbeddingsPipeline {
    // -params-file params.yaml is mandatory (there's no
    // nextflow.config-embedded fallback), so fail fast here with a
    // specific message for every required-with-no-default param --
    // params.yaml's own "Required, no default" section -- rather than
    // letting Nextflow's generic "no such property" surface first.
    if (params.pipeline_dir == null) {
        error "ERROR: --pipeline_dir is required."
    }
    if (params.cell_dino_checkpoint == null) {
        error "ERROR: --cell_dino_checkpoint is required (path to a Cell-DINO .pth checkpoint)."
    }
    // Every entry in params.experiments (a YAML list of maps, declared
    // directly in params.yaml under the `experiments:` key -- see that
    // file's own comment) supplies BUILD_CELL_IMAGES' starcall-workflow-
    // facing fields (phenotyping_dir, wells, grid_size, ...) and/or
    // BuildDatasetConfig's own remaining fields (window, ...) directly.
    // batch_stem is a required key *inside* each map now (there's no
    // filename to derive it from any more, unlike the old
    // configs/<batch_stem>.yaml-per-experiment mechanism this replaces).
    // Nextflow's own -params-file YAML loader already parses this into a
    // real List<Map> -- no manual SnakeYAML parsing needed, unlike the
    // old config_ch, since there's no file to read here any more.
    if (!(params.experiments instanceof List) || params.experiments.isEmpty()) {
        error "ERROR: params.experiments must be a non-empty list of experiment maps (see params.yaml)."
    }
    params.experiments.eachWithIndex { entry, i ->
        if (!(entry instanceof Map)) {
            error "ERROR: params.experiments[${i}] must be a map, got ${entry?.getClass()?.simpleName}."
        }
        if (!(entry.batch_stem instanceof String) || entry.batch_stem.trim().isEmpty()) {
            error "ERROR: params.experiments[${i}] is missing a required, non-empty 'batch_stem' field."
        }
        if (entry.containsKey('cp_features') && !(entry.cp_features instanceof Boolean)) {
            error "ERROR: params.experiments[${i}].cp_features must be a boolean (true/false), got ${entry.cp_features?.getClass()?.simpleName}."
        }
    }
    def batch_stems = params.experiments.collect { it.batch_stem }
    def duplicate_stems = batch_stems.findAll { s -> batch_stems.count(s) > 1 }.unique()
    if (duplicate_stems) {
        error "ERROR: params.experiments has duplicate batch_stem value(s): ${duplicate_stems.join(', ')}. Every experiment's batch_stem must be unique."
    }

    // BUILD_CELL_IMAGES is the ONLY thing that touches starcall-workflow's
    // tree (phenotyping_dir/segmentation_dir/sequencing_dir) or invokes
    // Snakemake -- every starcall-workflow-facing key in an experiment's
    // map (starcall_workflow_dir, phenotyping_dir, segmentation_dir,
    // sequencing_dir, wells, grid_size, segmentation_type, use_corrected,
    // sequencing_reads_params, cp_features, cellprofiler_pipeline,
    // cellprofiler_cycle) routes to it, not to BUILD_DATASET/
    // BUILD_CP_FEATURES. Runs unconditionally for every experiment (not
    // gated on cp_features) -- both tracks below depend on its
    // cell_images_dir output. `cellprofiler_pipeline`/`cellprofiler_cycle`
    // fall back to their global params.yaml defaults the same way `window`
    // does for config_ch below -- an entry's own value always wins.
    // (cell_images_hard_copy is NOT per-experiment: it's read directly off
    // params by build_cell_images.nf's own publishDir directive, since
    // Nextflow's publishDir `mode:` must be a static value at process-
    // definition time, unlike `path:` -- confirmed against a real
    // Nextflow 26.04.6 run. See that module's own comment.)
    def cell_images_field_includes = [
        'starcall_workflow_dir', 'phenotyping_dir', 'segmentation_dir', 'sequencing_dir',
        'wells', 'grid_size', 'segmentation_type', 'use_corrected', 'sequencing_reads_params',
        'cp_features', 'cellprofiler_pipeline', 'cellprofiler_cycle',
    ] as Set
    cell_images_config_ch = channel.fromList(params.experiments).map { entry ->
        def overrides = entry.findAll { k, v -> k in cell_images_field_includes }
        if (!overrides.containsKey('cellprofiler_pipeline') && params.cellprofiler_pipeline != null) {
            overrides = overrides + [cellprofiler_pipeline: params.cellprofiler_pipeline]
        }
        if (!overrides.containsKey('cellprofiler_cycle') && params.cellprofiler_cycle != null) {
            overrides = overrides + [cellprofiler_cycle: params.cellprofiler_cycle]
        }
        tuple(entry.batch_stem, overrides)
    }
    cell_images_ch = BUILD_CELL_IMAGES(cell_images_config_ch) // (batch_stem, cell_table.parquet, cell_images_dir)

    // `cp_features` opts an experiment into the CellProfiler-feature track
    // below and isn't a BuildDatasetConfig field itself, so BUILD_DATASET
    // never sees it; every starcall-workflow-facing key above is now
    // BUILD_CELL_IMAGES-only, so BUILD_DATASET never sees those either --
    // it only needs cell_images_dir (injected below) plus whatever
    // BuildDatasetConfig fields remain (window, shard_maxcount,
    // barcode_col_name/aa_changes_col_name/edit_distance_col_name).
    //
    // Parsed here (not passed through as a raw file) so BUILD_DATASET's
    // -resume cache key is the actual scalar values -- matching this
    // repo's prior configs/*.yaml-parsing precedent for the same reason.
    // `window` falls back to the global params.window default (see
    // params.yaml's "Shared per-experiment defaults" section) whenever an
    // entry doesn't set its own -- an entry's own value always wins.
    def dataset_field_excludes = (['batch_stem', 'cp_features', 'cell_images_hard_copy'] + cell_images_field_includes) as Set
    config_ch = channel.fromList(params.experiments)
        .map { entry ->
            def overrides = entry.findAll { k, v -> !(k in dataset_field_excludes) }
            if (!overrides.containsKey('window') && params.window != null) {
                overrides = overrides + [window: params.window]
            }
            tuple(entry.batch_stem, overrides)
        }
        // parquet.getParent(), not the tuple's own third (*_grid* glob)
        // element -- path("*_grid*", type: 'dir') resolves to each matched
        // grid subdirectory itself (e.g. well1_grid1/), not its containing
        // directory; cell_table.parquet's parent is the one unambiguous
        // reference to BUILD_CELL_IMAGES' actual per-experiment output
        // directory, regardless of how many *_grid* subdirectories exist
        // (confirmed against a real Nextflow 26.04.6 run -- see
        // modules/local/build_cell_images.nf).
        .join(cell_images_ch.map { stem, parquet, dir -> tuple(stem, parquet.getParent()) })
        .map { stem, overrides, cell_images_dir -> tuple(stem, overrides + [cell_images_dir: cell_images_dir.toString()]) }

    // Per-batch (per-experiment) chain -- identical shape to
    // fisseq-data-pipeline's per-batch resolution pattern (BatchParams.resolve).
    dataset_ch = BUILD_DATASET(config_ch)                 // (batch_stem, [dataset-*.tar shards], metadata.parquet)
    qc_ch      = QC_FILTER(dataset_ch.map { s, shards, meta -> tuple(s, meta) })     // (batch_stem, filtered_cells, barcode_counts, variants_per_barcode) -- reads metadata.parquet only
    embed_ch   = EMBED_CELLS(dataset_ch.map { s, shards, meta -> tuple(s, shards) }) // (batch_stem, embeddings.parquet) -- streams the shards; no QC dependency, matches diagram
    // FILTER_EMBEDDINGS only wants the join key (filtered_cells.parquet),
    // not qc_ch's other two (informational, QC-report-only) outputs.
    filtered_ch = FILTER_EMBEDDINGS(
        embed_ch.join(qc_ch.map { s, filtered_cells, barcode_counts, variants_per_barcode -> tuple(s, filtered_cells) })
    ) // (batch_stem, filtered_keys.parquet, normalizer.parquet) -- no emb_* columns, §3 decision 10

    // Both downstream consumers need the raw embeddings.parquet *and*
    // filtered_ch's join key + normalizer -- neither reads a pre-normalized
    // file, each reconstructs it itself via load_filtered_embeddings() (§6.4).
    embed_and_filtered_ch = embed_ch.join(filtered_ch)     // (batch_stem, embeddings.parquet, filtered_keys.parquet, normalizer.parquet)
    agg_ch  = AGGREGATE_EMBEDDINGS(embed_and_filtered_ch)  // (batch_stem, aggregate.parquet)  -> "Experiment N Aggregates"
    ovwt_ch = OVWT_BATCHWISE(embed_and_filtered_ch)        // (batch_stem, results.parquet, cell_scores.parquet, models.pkl) -> "Experiment N Distinguish-ability Scores"

    // Global stages -- real path channels via .collect(), not a directory
    // glob (see fisseq-data-pipeline's stage_channel.nf / AGENTS.md gotcha:
    // a val glob string only hashes the glob text, not the resolved file
    // set, and silently breaks -resume cache invalidation).
    GLOBAL_VARIANT_EMBEDDINGS(
        agg_ch.map { stem, path -> path }.collect(),
        agg_ch.map { stem, path -> stem }.collect(),
    )
    GLOBAL_VARIANT_DISTINGUISHABILITY(
        ovwt_ch.map { stem, results, cell_scores, models -> results }.collect(),
        ovwt_ch.map { stem, results, cell_scores, models -> stem }.collect(),
    )

    // ── CellProfiler-feature track (optional second track) ──────────────
    // An `experiments:` entry opts itself into this track by setting
    // `cp_features: true` -- there's no separate list to keep in sync with
    // `experiments:` any more, so batch_stem existence/uniqueness are
    // already guaranteed by the validation above. No entries opting in
    // (the default) skips BUILD_CP_FEATURES onward entirely, so existing
    // cellDINO-only runs work unchanged.
    def cp_experiments = params.experiments.findAll { it.cp_features == true }
    if (cp_experiments) {
        // CpFeaturesConfig no longer needs any starcall-workflow-facing
        // field at all -- BUILD_CELL_IMAGES already folded this
        // experiment's CellProfiler columns into cell_table.parquet (see
        // cell_images_config_ch/cell_images_ch above), so this stage just
        // needs cell_images_dir (injected below) plus barcode_col_name/
        // aa_changes_col_name/edit_distance_col_name/batch_stem.
        def cp_field_excludes = (['batch_stem', 'cp_features', 'cell_images_hard_copy'] + cell_images_field_includes) as Set
        cp_config_ch = channel.fromList(cp_experiments)
            .map { entry -> tuple(entry.batch_stem, entry.findAll { k, v -> !(k in cp_field_excludes) }) }
            // parquet.getParent(), not the tuple's own third (*_grid* glob)
            // element -- path("*_grid*", type: 'dir') resolves to each
            // matched grid subdirectory itself (e.g. well1_grid1/), not
            // its containing directory; cell_table.parquet's parent is the
            // one unambiguous reference to BUILD_CELL_IMAGES' actual
            // per-experiment output directory, regardless of how many
            // *_grid* subdirectories exist (confirmed against a real
            // Nextflow 26.04.6 run -- see modules/local/build_cell_images.nf).
            .join(cell_images_ch.map { stem, parquet, dir -> tuple(stem, parquet.getParent()) })
            .map { stem, overrides, cell_images_dir -> tuple(stem, overrides + [cell_images_dir: cell_images_dir.toString()]) }

        cp_ch = BUILD_CP_FEATURES(cp_config_ch)   // (batch_stem, cp_features.parquet)
        // Reuses the SAME qc_ch computed above for the cellDINO track --
        // no second QC_FILTER run.
        cp_qc_join_ch = qc_ch.map { s, filtered_cells, barcode_counts, variants_per_barcode -> tuple(s, filtered_cells) }
        cp_filtered_ch = FILTER_CP_FEATURES(cp_ch.join(cp_qc_join_ch)) // (batch_stem, filtered_keys.parquet, normalizer.parquet)

        cp_and_filtered_ch = cp_ch.join(cp_filtered_ch) // (batch_stem, cp_features.parquet, filtered_keys.parquet, normalizer.parquet)
        cp_agg_ch  = AGGREGATE_CP_FEATURES(cp_and_filtered_ch)
        cp_ovwt_ch = OVWT_BATCHWISE_CP_FEATURES(cp_and_filtered_ch)

        GLOBAL_VARIANT_CP_FEATURES(
            cp_agg_ch.map { stem, path -> path }.collect(),
            cp_agg_ch.map { stem, path -> stem }.collect(),
        )
        GLOBAL_VARIANT_DISTINGUISHABILITY_CP_FEATURES(
            cp_ovwt_ch.map { stem, results, cell_scores, models -> results }.collect(),
            cp_ovwt_ch.map { stem, results, cell_scores, models -> stem }.collect(),
        )
    }
}
