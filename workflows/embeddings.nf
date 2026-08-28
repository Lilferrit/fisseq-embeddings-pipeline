// EmbeddingsPipeline. Wires BUILD_DATASET -> {QC_FILTER, EMBED_CELLS} ->
// FILTER_EMBEDDINGS -> {AGGREGATE_EMBEDDINGS, OVWT_BATCHWISE} ->
// {GLOBAL_VARIANT_EMBEDDINGS, GLOBAL_VARIANT_DISTINGUISHABILITY}.
nextflow.enable.dsl = 2

include { BUILD_DATASET } from '../modules/local/build_dataset'
include { QC_FILTER } from '../modules/local/qc_filter'
include { EMBED_CELLS } from '../modules/local/embed_cells'
include { FILTER_EMBEDDINGS } from '../modules/local/filter_embeddings'
include { AGGREGATE_EMBEDDINGS } from '../modules/local/aggregate_embeddings'
include { OVWT_BATCHWISE } from '../modules/local/ovwt_batchwise'
include { GLOBAL_VARIANT_EMBEDDINGS } from '../modules/local/global_variant_embeddings'
include { GLOBAL_VARIANT_DISTINGUISHABILITY } from '../modules/local/global_variant_distinguishability'

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
    // file's own comment) supplies BuildDatasetConfig's per-experiment
    // fields (phenotyping_dir, wells, grid_size, window, ...) directly.
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
    }
    def batch_stems = params.experiments.collect { it.batch_stem }
    def duplicate_stems = batch_stems.findAll { s -> batch_stems.count(s) > 1 }.unique()
    if (duplicate_stems) {
        error "ERROR: params.experiments has duplicate batch_stem value(s): ${duplicate_stems.join(', ')}. Every experiment's batch_stem must be unique."
    }

    // Parsed here (not passed through as a raw file) so BUILD_DATASET's
    // -resume cache key is the actual scalar values -- matching this
    // repo's prior configs/*.yaml-parsing precedent for the same reason.
    config_ch = channel.fromList(params.experiments).map { entry ->
        tuple(entry.batch_stem, entry.findAll { k, v -> k != 'batch_stem' })
    }

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
}
