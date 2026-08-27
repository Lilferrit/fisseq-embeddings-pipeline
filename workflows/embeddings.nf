// SPEC.md §7.2 -- EmbeddingsPipeline. Wires BUILD_DATASET -> {QC_FILTER,
// EMBED_CELLS} -> FILTER_EMBEDDINGS -> {AGGREGATE_EMBEDDINGS,
// OVWT_BATCHWISE} -> {GLOBAL_VARIANT_EMBEDDINGS, GLOBAL_VARIANT_
// DISTINGUISHABILITY} per IMPLEMENTATION_CHECKLIST.md Epic 9.
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
    // SPEC.md §9.1's Resolved note: -params-file params.yaml is mandatory
    // (there's no nextflow.config-embedded fallback), so fail fast here
    // with a specific message for every required-with-no-default param --
    // params.yaml's own "Required, no default" section -- rather than
    // letting Nextflow's generic "no such property" surface first.
    if (params.pipeline_dir == null) {
        error "ERROR: --pipeline_dir is required."
    }
    if (params.cell_dino_checkpoint == null) {
        error "ERROR: --cell_dino_checkpoint is required (path to a Cell-DINO .pth checkpoint -- see SPEC.md §6.3)."
    }
    def configsDir = file("${params.pipeline_dir}/configs")
    if (!configsDir.isDirectory()) {
        error "ERROR: ${params.pipeline_dir}/configs does not exist or is not a directory"
    }
    def config_files = configsDir.listFiles()?.findAll { it.name.endsWith('.yaml') } ?: []
    if (config_files.size() == 0) {
        error "ERROR: No .yaml files found in ${params.pipeline_dir}/configs"
    }

    // Every configs/<batch_stem>.yaml supplies BuildDatasetConfig's
    // per-experiment fields (phenotyping_dir, wells, grid_size, window,
    // ...) directly -- batch_stem comes from the filename, not a key
    // inside the file. Parsed here (not passed through as a raw file) so
    // BUILD_DATASET's -resume cache key is the actual scalar values, not
    // the YAML's bytes -- a non-semantic edit (e.g. reordering keys,
    // touching a comment) shouldn't bust the cache, matching
    // fisseq-data-pipeline's own INPUT/BatchParams.resolve precedent for
    // this exact problem (see modules/local/input.nf there). A
    // `batch_stem` key inside the YAML itself, if present, is dropped --
    // the filename is always authoritative -- so it can never disagree
    // with the value BUILD_DATASET is actually invoked with.
    config_ch = channel.fromList(config_files).map { f ->
        def batch_stem = f.baseName
        def batch_config = (new org.yaml.snakeyaml.Yaml().load(f.text) ?: [:]) as Map
        tuple(batch_stem, batch_config.findAll { k, v -> k != 'batch_stem' })
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
