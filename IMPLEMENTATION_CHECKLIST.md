# Implementation Checklist

An Agile-style backlog for implementing `SPEC.md` in `fisseq-embeddings-pipeline`. Work through the epics in order — later epics assume earlier ones exist. Each story lists concrete, verifiable acceptance criteria (a file, a column set, a passing test) rather than vague done-ness; check items off (`- [x]`) as you complete them and commit the updated checklist alongside the code, so it stays a true record of progress.

Every story cites the `SPEC.md` section it implements — read that section (and, where noted, the real vendored file in `fisseq-data-pipeline`) before writing code; the code blocks in `SPEC.md` are illustrative sketches, not guaranteed-correct source to copy verbatim.

**Suggested sequencing** (four increments, each independently demoable):

- **M1 — Foundations**: Epic 0, Epic 1, Epic 2
- **M2 — Embedding & scoring core**: Epic 3, Epic 4, Epic 5, Epic 6
- **M3 — Global stages & orchestration**: Epic 7, Epic 8, Epic 9
- **M4 — Hardening & handoff**: Epic 10, Epic 11, Epic 12

---

## Epic 0 — Repo & shared foundations

*Goal: every later epic has a working `AppConfig`, vendored utilities, and a package that imports cleanly.*

### Story 0.1 — Vendor shared utilities from `fisseq-data-pipeline`
- [x] `utils/constants.py`: vendored unchanged (`CONTROL_COLUMN`/`CONTROL_COLUMN_NAME`, `FEATURE_SELECTOR`, `META_SELECTOR`, `EPS`, `META_BATCH_COL`, `META_BARCODE_COL`, `META_EDIT_DISTANCE_COL`, etc.) **plus** a new `EMBEDDING_SELECTOR = cs.matches(r"^emb_\d+$")` (SPEC.md §6.3's Output note).
- [x] `utils/variant.py`: vendored unchanged (`classify_variant`). **Checklist correction**: `variant_classification` does not live in the real source repo's `utils/variant.py` — it's defined in `aggregate.py` and will be ported alongside that file in Epic 4/5, not here.
- [x] `utils/batches.py`: vendored unchanged (`load_batches`).
- [x] `utils/log.py`: vendored unchanged (`setup_logging`).
- [x] `utils/xgbparams.py`: vendored, **with exactly one line changed** — `train_binary_xgboost`'s `params["seed"] = cfg.random_state` becomes `params["seed"] = cfg.random_seed` (SPEC.md §6.6's Seed note). Everything else (`XGBoostParams`, `XGBoostConfig`, `get_dmatrix`, `get_dmatrix_multiclass`, `resolve_feature_importance`, `split_indices_stratified`, `evaluate_binary`) unchanged.
- [x] `utils/dimreduction.py`: vendored, **with one added parameter** — `compute_pca` gains `random_state: int = 0`, threaded into its internal `PCA(...)` call instead of the hardcoded `0` (SPEC.md §6.7's Seed note).
- [x] `config/input.py`: vendored unchanged (`InputConfig`, `LabeledInputConfig`). `config/__init__.py` populated to export `AppConfig`/`InputConfig`/`LabeledInputConfig`, matching the source repo's, since `utils/log.py`'s `setup_logging` imports `AppConfig` from `..config`.
- [x] Unit test per vendored-with-changes file confirming the deviation behaves as documented (e.g. `xgbparams`'s `train_binary_xgboost` actually reads `cfg.random_seed`, not `cfg.random_state`).

### Story 0.2 — `AppConfig` with the shared seed
- [x] `config/app.py` implements `AppConfig` per SPEC.md §3's code block: `output_dir`, `output_root`, `log_level`, `random_seed: int = 0`.
- [x] Every stage config in later epics extends `AppConfig` — no stage adds its own `random_state`/seed field (SPEC.md §3 decision 11).
- [x] Unit test: a `ConfigStore`-registered stage config's default `random_seed` is `0` and is overridable via Hydra CLI override syntax.

### Story 0.3 — Package/build sanity
- [x] `uv sync --group dev` succeeds from a clean clone.
- [x] `python -c "import fisseq_embeddings_pipeline"` succeeds.
- [x] `uv run ruff check .` and `uv run pre-commit run --all-files` pass on the scaffold as-is.
- [x] CI (or at minimum a documented local command) runs `pytest tests/unit` and `ruff check` on every push — decide and document where (GitHub Actions, matching `fisseq-data-pipeline`'s `.github/`, or elsewhere). **Decided:** `.github/workflows/ci.yml`, mirroring `fisseq-data-pipeline`'s `.github/workflows/pr-checks.yml` shape (`astral-sh/setup-uv`, `uv sync --group dev`) with `lint`/`unit-tests` jobs against `master` (this repo's actual default branch). No `integration-tests` job yet — `tests/integration` has no content until Epic 11; add one then, mirroring that same workflow's Java+Nextflow setup.

---

## Epic 1 — Cell Dataset: `BUILD_DATASET` (SPEC.md §6.1)

*Goal: crop one experiment's cells directly from `starcall-workflow`'s stitched phenotype images into a WebDataset, unconditionally (no QC gating). Stories 1.1–1.3 below describe the original design (reading `make_cell_images`'s pre-cropped output); Story 1.4 supersedes that with `BUILD_DATASET` doing its own cropping, since `make_cell_images` isn't reliably run for every experiment — see Story 1.4 for why.*

### Story 1.1 — Config & tile discovery
- [x] `BuildDatasetConfig(AppConfig)` implemented with every field from SPEC.md §6.1's dataclass (`phenotyping_dir`, `wells`, `grid_size`, `segmentation_type="cells"`, `window`, `shard_maxcount=2000`, `batch_stem`, `barcode_col_name="upBarcode"`, `aa_changes_col_name="aaChanges"`, `edit_distance_col_name="editDistance"`).
- [x] `discover_tiles()` globs `starcall-workflow`'s `{well}_grid{grid_size}/tile{x}x{y}y/` layout (**verified against a real `starcall-workflow` `origin/devel` checkout**, now mounted read-only: `workflow/rules/stitching.smk`'s `rule stitch_tile_pt` writes `'{well}_grid{grid_size}/tile{x}x{y}y/{corrected}_pt.tif'`, confirming the glob pattern). One correction versus SPEC.md's own sketch: rows are sorted by `(well, int(tile_x), int(tile_y))` instead of lexical `sorted(glob.glob(...))` order, since lexical sort misorders double-digit tile indices (`tile10x0y` would sort before `tile2x0y`).
- [x] Unit test: `discover_tiles()` against a small synthetic directory tree returns the expected `(well, tile, cell_table_csv, cell_crops_tif, mask_crops_tif)` rows, sorted deterministically (including a case with >10 tiles to actually exercise the numeric-sort fix above).

### Story 1.2 — Shard writing
- [x] `write_dataset_shards()` writes one WebDataset sample per cell (`crop.npy`, `mask.npy`, `meta.json` with `meta_batch`/`meta_well`/`meta_tile`/`meta_cell_index`/`meta_barcode`/`meta_aa_changes`/`meta_edit_distance`), keyed `"{well}_{tile}_{cell_index}"`, via `webdataset.ShardWriter(maxcount=cfg.shard_maxcount)`.
- [x] Empty-tile CSVs are skipped without erroring (matches `make_cell_images`'s empty-tile behavior). **Correction versus SPEC.md's sketch**: a real empty tile is a genuinely 0-byte file (`os.system('touch ...')` in the real `starcall-workflow` rule), which makes `pandas.read_csv` raise `EmptyDataError` rather than return a 0-row frame — SPEC.md's sketch only guards `len(table.index) == 0`, which alone only catches a header-only CSV. `write_dataset_shards()` catches both cases.
- [x] `metadata.parquet` is written alongside the shards with the same per-cell `meta_*` fields, no image data — confirm `QC_FILTER` (Epic 2) never needs to touch the `.tar` shards.
- [x] Hydra `main()` entry point wired up; `python -m fisseq_embeddings_pipeline.dataset <overrides>` runs end to end against a small synthetic `phenotyping_dir`.
- [x] Unit test: round-trip a tiny synthetic tile (2-3 cells) through `write_dataset_shards()` and confirm the shard's decoded samples and `metadata.parquet` rows match.

### Story 1.3 — Shard sizing sanity (SPEC.md §10, item 2)
- [x] Measure real per-sample byte size (crop + mask, your actual `window`/channel count) and confirm `shard_maxcount=2000` still lands shards in a reasonable size band (~hundreds of MB to ~1GB); adjust the default if not. **No real experiment data was available in this environment** (only the pipeline source repos are mounted) — computed an estimate instead from real defaults elsewhere in the stack (`starcall-workflow`'s 4-channel `phenotyping_channels` default, Cell-DINO's `224` crop size, assumed `uint16` crops): ≈440 KB/sample × `shard_maxcount=2000` ≈ ~880 MB/shard, within the target band, so the `2000` default is kept. Flagged as needing re-verification once real experiment data is available.
- [x] Document the measured number in `docs/configuration.md` (replacing the placeholder).

### Story 1.4 — Revision: crop from `stitch_tile_pt` instead of depending on `make_cell_images` (SPEC.md §6.1 revision)

*Why: per Alyssa La Fleur (starcall-workflow maintainer, Slack), `make_cell_images` (`phenotyping.smk`) is not reliably run for every experiment. Reading its real `origin/devel` source confirmed a likely cause: it reads `cell_table['xpos']`/`['ypos']` columns that don't exist in the real cell-table schema (`tabulate_cells` only ever produces `bbox_x1/y1/x2/y2`) and would raise `KeyError` against real data. `rule stitch_tile_pt` (`stitching.smk`) and `rule stitch_tile_from_well_segmentation` (`segmentation.smk`), by contrast, are always produced regardless of cell count — every experiment needs stitching/segmentation. This story replaces `BUILD_DATASET`'s dependency on `make_cell_images`'s output with `BUILD_DATASET` doing its own cropping, porting `make_cell_images`'s crop-window algorithm directly.*

- [x] `BuildDatasetConfig` gains `use_corrected: bool = False` (selects `corrected_pt.tif`/`raw_pt.tif`, mirroring starcall-workflow's own `config.phenotyping.use_corrected` default). `discover_tiles()` returns `pt_tif`/`mask_tif` columns pointing at `stitch_tile_pt`'s `{raw|corrected}_pt.tif` and `stitch_tile_from_well_segmentation`'s `{segmentation_type}_mask.tif`, replacing the old `cell_crops_tif`/`mask_crops_tif` columns (`make_cell_images`'s `*_crops_{window}.tif` outputs). `cfg.window` no longer affects `discover_tiles()` at all — only crop time.
- [x] `write_dataset_shards()` computes each cell's crop center as the integer midpoint of `bbox_x1/x2` and `bbox_y1/y2` (bbox columns are fixed structural constants, not new config fields — see SPEC.md §5.1's verified `xpos`/`ypos` correction), and crops via a new `_crop_cell()` helper ported from `make_cell_images`'s algorithm, including its positional `mask == i + 1` labeling convention (carried over from upstream as-is — flagged as an open risk, not a validated-correct choice; see below).
- [x] Empty-tile handling simplified: the old `pd.errors.EmptyDataError` branch (modeling `make_cell_images`'s own `touch`-emptied output) is removed — `tabulate_cells` always writes at least a header row, so only `len(table.index) == 0` needs guarding.
- [x] `image.ndim == 3` is normalized to 4D before flattening (`image[None]`), defending against the single-phenotype-cycle TIFF-squeeze ambiguity `make_cell_images` itself never guarded against.
- [x] `tests/unit/test_dataset.py` rewritten: fixtures now write a deterministic stitched tile image + label mask + bbox-columned cell table (position-encoding pixel content, so expected crops are computable via an independent pad-based oracle, not by re-running the code under test) instead of pre-cropped per-cell TIFFs. New tests cover centered/interior crops, edge-clipped crops (low and high sides), the positional-label-vs-cell_index regression case (non-sequential cell ids), multi-cycle/multi-channel cycle-major flattening, and the 3D-squeeze `ndim` guard.
- [x] SPEC.md §5.1 (drops `xpos`/`ypos`, documents the bbox-midpoint centroid and the verified schema gap), §5.2 (`stitch_tile_pt`/`stitch_tile_from_well_segmentation` promoted to what `BUILD_DATASET` reads; `make_cell_images` demoted to "algorithm ported, not output consumed"), §5.3, and §6.1 (config/`discover_tiles`/`write_dataset_shards` sketches, empty-tile note, output table) updated to match. `docs/configuration.md`'s shard-sizing note updated with a multi-cycle caveat and corrected dtype provenance.

**Open risks carried forward, not resolved by this story** (no real experiment data was available to validate against in this environment):
1. The positional `mask == i + 1` labeling assumes contiguous `1..N` mask labels in table row order — an assumption `make_cell_images` itself apparently never validated against real data either. Worth spot-checking against one real tile once available.
2. `stitch_tile_pt` and `stitch_tile_from_well_segmentation` are assumed to write into the same `phenotyping_dir` root (verified consistent with the real rule signatures) — worth confirming against an actual deployment's layout; add a separate `segmentation_dir` config field later if they ever diverge.

---

## Epic 2 — QC Filtering: `QC_FILTER` (SPEC.md §6.2)

*Goal: near-verbatim port of `qcfilter.py`, retargeted to read `BUILD_DATASET`'s `metadata.parquet`.*

### Story 2.1 — Port `qcfilter.py`
- [x] `QcFilterConfig` fields match SPEC.md's list (`bc_threshold=10`, `variant_bc_threshold=4`, `edit_distance_threshold=1`) with `downsample_amounts`/`downsample_classes`/`downsample_seed` **omitted** (SPEC.md §6.2's Resolved note — deliberately dropped, not an oversight).
- [x] `cell_files` input accepts `BUILD_DATASET`'s `metadata.parquet` directly (no separate CSV-reading path from the old `input.py`).
- [x] Outputs `filtered_cells.parquet` (composite join key + `meta_*` for QC-passed cells — a key list, not a copy), `barcode_counts.parquet`, `variants_per_barcode.parquet`, matching `fisseq-data-pipeline`'s existing contract.
- [x] Unit tests ported/adapted from `fisseq-data-pipeline/tests/unit/test_qcfilter.py`, updated for the `metadata.parquet` input path.

**Two deviations from the vendored source, flagged during implementation (beyond the
Resolved-note downsample drop above):**
1. `barcode_col_name`/`aa_changes_col_name`/`edit_distance_col_name` default to
   `"meta_barcode"`/`"meta_aa_changes"`/`"meta_edit_distance"` instead of the upstream raw-CSV
   names (`"upBarcode"`/`"aaChanges"`/`"editDistance"`) — `metadata.parquet` (Epic 1) already
   writes those columns under their canonical `meta_*` names, so this pipeline's only real
   `cell_files` input is never the raw, unrenamed cell table `filter_columns`'s rename step was
   originally written against. `filter_columns` itself is otherwise unchanged (renaming a column
   to its own existing name is a harmless no-op in Polars).
2. `select_variants`'s `mode="random"` seed now comes from `AppConfig.random_seed` (SPEC.md §3
   decision 11) instead of the dropped `downsample_seed` field, since no stage config is meant
   to add its own seed field.

Also fixed in passing: `read_file`'s upstream `if`/`elif` on file suffix has no `else`, so an
unrecognized suffix raised an opaque `UnboundLocalError` deep inside a later `.collect()` call —
the port raises a clear `ValueError` instead.

`modules/local/qc_filter.nf`'s CLI arg names (`barcode_count_threshold`/
`variant_barcode_count_threshold` vs. the real field names `bc_threshold`/`variant_bc_threshold`)
remain unreconciled — left for Epic 9 / Story 9.2, matching `build_dataset.nf`'s own still-open
`TODO(Epic 9)` even after Epic 1 shipped.

---

## Epic 3 — Cell Embeddings: `EMBED_CELLS` (SPEC.md §6.3)

*Goal: stream every cell through Cell-DINO in bag-of-channels mode. This is the epic SPEC.md flags as resting most heavily on assumptions — budget real time to read `dinov2` source before trusting any of its sketch.*

### Story 3.1 — Verify Cell-DINO's real inference API (SPEC.md §10, item 1 — do this first)
- [ ] Read `dinov2/eval/setup.py` and `dinov2/models/vision_transformer.py` (or the channel-adaptive equivalent) against your actual checkpoint.
- [ ] Confirm (or correct) `load_cell_dino`'s construction path — does it really go through `dinov2.eval.setup`/`build_model_from_cfg`, or something else?
- [ ] Confirm the checkpoint's state-dict key (`"teacher"` vs. top-level) against a real `.pth`.
- [ ] Confirm the correct pooling operator for per-channel CLS tokens (SPEC.md assumes mean/max over a channel-adaptive `in_chans=1` backbone — verify this matches how Cell-DINO was actually evaluated).
- [ ] Record what you found in `docs/architecture.md` (replacing SPEC.md's placeholder assumptions) so this verification isn't silently lost.

### Story 3.2 — Config & dataloader
- [ ] `EmbedCellsConfig(AppConfig)` implemented per SPEC.md's dataclass (`shard_pattern`, `checkpoint_path`, `arch="vit_large"`, `patch_size=16`, `crop_size=224`, `channel_pool="mean"`, `mask_mode="none"`, `device="cuda"`, `batch_size=256`, `num_workers=4`).
- [ ] `load_embedding_dataloader()` streams `(key, crop, mask, meta)` from `BUILD_DATASET`'s shards via `webdataset.WebDataset(...).decode().to_tuple(...).batched(...)`.

### Story 3.3 — Model wrapper & masking
- [ ] `load_cell_dino()` implemented against the real API confirmed in Story 3.1 (not the SPEC.md placeholder).
- [ ] `embed_batch()` implements bag-of-channels embedding: split `(B, C, H, W)` into `C` single-channel images, run through the shared backbone, pool per-channel CLS tokens (`channel_pool="mean"|"max"`) into `(B, D)`.
- [ ] `mask_mode="zero_background"` zeroes non-target pixels using `mask.npy` before embedding; `mask_mode="none"` passes crops through untouched. Both paths covered by tests.
- [ ] Unit test: `embed_batch()` against a random-weight (not pretrained) model of the same architecture, confirming output shape `(B, D)` and that `mask_mode="zero_background"` actually zeroes the expected pixels before the forward pass.

### Story 3.4 — Output & Nextflow wiring
- [ ] Output `embeddings.parquet`: one row per cell, `meta.json` fields passed through, plus zero-padded `emb_0000..emb_{D-1}` columns (`EMBEDDING_SELECTOR` from Epic 0 matches these).
- [ ] `modules/local/embed_cells.nf` given `label 'process_gpu'` and wired to a GPU-capable executor/queue in `nextflow.config`.
- [ ] End-to-end smoke test (small synthetic shard, real or stub checkpoint) produces `embeddings.parquet` with the expected shape.

---

## Epic 4 — Filter Embeddings: `FILTER_EMBEDDINGS` (SPEC.md §6.4)

*Goal: the no-copy/foreign-key redesign (SPEC.md §3 decision 10) — publish a join key + fitted stats, never a second copy of the embedding matrix.*

### Story 4.1 — `filter_and_fit_normalizer()`
- [ ] Inner-joins `embeddings_lf` to `qc_passed_lf` on the composite key (`meta_batch`/`meta_well`/`meta_tile`/`meta_cell_index`).
- [ ] Calls `variant_classification()` to mark `meta_is_control` (synonymous/untagged).
- [ ] Fits `Normalizer.from_lazyframe(filtered, fit_only_on_control=True)`.
- [ ] Returns `filtered_keys` (join key + `meta_is_control`/`meta_aa_changes`/other `meta_*` — **verified to contain zero `emb_*` columns**) and the fitted `Normalizer` — **never** a materialized normalized embedding table.
- [ ] Unit test asserting `filtered_keys`'s column set contains no `emb_*` columns (this is the single most important regression test for decision 10 — write it before moving on).

### Story 4.2 — `load_filtered_embeddings()` (shared helper)
- [ ] Joins `embeddings_lf` to `filtered_keys_lf` by composite key, then applies the normalizer.
- [ ] Importable from both `aggregate.py` (Epic 5) and `ovwt.py` (Epic 6) without duplicating the join/apply logic.
- [ ] Unit test: `load_filtered_embeddings()` output matches what the *old* single-step `filter_and_normalize()` (SPEC.md's superseded version) would have produced, on the same synthetic input — i.e. the redesign is output-equivalent, not just differently-shaped.

### Story 4.3 — Output & Nextflow wiring
- [ ] Outputs `filtered_keys.parquet`, `normalizer.parquet` (no `filtered_embeddings.parquet`).
- [ ] `modules/local/filter_embeddings.nf` takes `embeddings.parquet` + `filtered_cells.parquet` as input, matching the stub's `input:` block.
- [ ] `workflows/embeddings.nf`'s `embed_and_filtered_ch = embed_ch.join(filtered_ch)` wiring (feeding both Epic 5 and Epic 6) is in place and tested against a real (small) Nextflow run once Epic 9 starts.

---

## Epic 5 — Aggregation: `AGGREGATE_EMBEDDINGS` (SPEC.md §6.5)

*Goal: per-variant median pooling of the (reconstructed) synonymous-corrected embeddings.*

### Story 5.1 — `aggregate_embeddings()`
- [ ] Takes `filtered_lf` (built by the **caller** via `load_filtered_embeddings()`, Epic 4 — not read from a file directly).
- [ ] Median-pools all `emb_*` columns per variant (`group_by(label_column).agg(median)`) — **no** `per_barcode` option (SPEC.md §6.5's first Resolved note — deliberate, not a gap).
- [ ] Joins in `get_aggregate_meta_data()` (vendored unchanged) for `meta_num_cells`/`meta_barcode_num_unique`/etc.
- [ ] Confirmed **no** WT-null bootstrap/blocklist reproducibility-gate machinery is ported (SPEC.md §6.5's second Resolved note — explicitly out of scope for v1).
- [ ] Output `aggregate.parquet` matches SPEC.md's column description: one row per variant, `emb_0000..emb_{D-1}` plus `meta_*` aggregate columns.
- [ ] Unit test against a small synthetic `filtered_lf` with a known median.

---

## Epic 6 — OVWT Distinguish-ability Scores: `OVWT_BATCHWISE` (SPEC.md §6.6)

*Goal: k-fold CV stratified jointly on `(meta_barcode, is_wt)`, with per-fold calibration, producing an out-of-fold score for every cell and two distinguish-ability numbers per variant. The most algorithmically involved epic — budget real review time for the fold/stratification logic.*

### Story 6.1 — Config
- [ ] `OvwtEmbeddingConfig(AppConfig)` with `label_column`, `wt_label="WT"`, `n_folds: int = 5`, `calibrate: bool = True`, `min_cells=250`, `downsample_wt=True` (thresholds carried over from `ovwt.py` as a starting point, per SPEC.md's Resolved note), plus the vendored `xgboost: XGBoostConfig` sub-config.
- [ ] No `random_state` field — inherits `random_seed` from `AppConfig` only (Epic 0).

### Story 6.2 — `predict_binary()` helper
- [ ] Thin wrapper around vendored `get_dmatrix()` + `model.predict()`, no metric computation.
- [ ] Unit test against a trivially-separable synthetic dataset (predicted scores near 0/1 in the expected direction).

### Story 6.3 — `ovwt_batchwise()` core loop
- [ ] Per non-WT variant: builds the variant-vs-WT `subset`, computes the composite `(meta_barcode, is_wt)` stratification key.
- [ ] `sklearn.model_selection.StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.random_seed)` splits the subset.
- [ ] Per fold: inner fit/calibration split via `split_indices_stratified` (seeded `cfg.random_seed + fold_idx`); `train_binary_xgboost(train_df, calib_df, ..., cfg)` — **passing the full `cfg`**, not `cfg.xgboost` (a real bug in an earlier draft — confirm the fixed call site, SPEC.md §6.6's code comment flags this explicitly).
- [ ] If `cfg.calibrate`: fits a sigmoid/Platt calibrator on the calibration slice's raw scores; stores `(model, calibrator)` per fold.
- [ ] Every cell in `subset` receives exactly one out-of-fold score (`oof_scores[test_idx] = ...`) — **write a unit test asserting no `NaN`s remain in `oof_scores` after all folds run**, since a stratification bug could silently leave cells unscored.
- [ ] `auroc_pooled` computed over the full subset's OOF scores.
- [ ] `auroc_median_barcode`: per non-WT barcode, AUROC of that barcode's cells vs. all WT cells (same OOF scores), then median across barcodes — **WT barcodes excluded** from this median (verify with a unit test using >1 barcode per variant with deliberately different separability).
- [ ] Output columns exactly `meta_aa_changes`, `auroc_pooled`, `auroc_median_barcode`, `meta_n_barcodes`, `meta_n_cells` (note the `meta_` prefix on the count columns — required for Epic 8's z-score step to exclude them automatically via `FEATURE_SELECTOR`).
- [ ] `cell_scores.parquet` populated directly from the OOF loop: `meta_*` + `score` + `meta_variant_scored_against`, one row per cell per variant it was scored against.
- [ ] `models.pkl`: `dict[str, list[tuple[Booster, calibrator | None]]]`.

### Story 6.4 — Stratification edge case (SPEC.md §6.6's Resolved note)
- [ ] Decide and implement a concrete fallback for barcodes with fewer than `n_folds` cells (StratifiedKFold will otherwise raise). SPEC.md sketches "collapse rare barcodes into a shared stratum for fold-assignment purposes only" as one option — pick one, document the choice, and cover it with a unit test using a deliberately rare barcode.
- [ ] Confirm real per-variant/per-barcode cell counts from an actual experiment before finalizing `min_cells`/`n_folds` defaults; adjust `params.yaml` if the defaults turn out wrong at scale.

### Story 6.5 — Integration with Epic 4
- [ ] Takes `embeddings.parquet` + `filtered_keys.parquet` + `normalizer.parquet` as three inputs; constructs `filtered_lf` via Epic 4's `load_filtered_embeddings()` before calling `ovwt_batchwise()`.

---

## Epic 7 — Global Variant Embeddings: `GLOBAL_VARIANT_EMBEDDINGS` (SPEC.md §6.7)

*Goal: cross-experiment median pooling, then PCA. Runs once, unconditionally, over every experiment (SPEC.md §3 decision 8 — no `global_channels`-style scoping).*

### Story 7.1 — `global_variant_embeddings()`
- [ ] `median_across_batches()` vendored unchanged from `globalfeatureselect.py`.
- [ ] `compute_pca()` called with `random_state=cfg.random_seed` (Epic 0's added parameter).
- [ ] `GlobalVariantEmbeddingsConfig.n_components: int = 50` (SPEC.md's Resolved default).
- [ ] Output: `median_aggregate.parquet`, `pca_scores.parquet` (`meta_aa_changes`, `meta_pc_1..meta_pc_{n}`), `pca_components.parquet`.
- [ ] Unit test: PCA over a small synthetic multi-experiment aggregate table, confirming `n_components <= min(n_variants, n_retained_dims)` is enforced with a clear error rather than a cryptic sklearn one.

---

## Epic 8 — Global Variant Distinguish-ability Scores: `GLOBAL_VARIANT_DISTINGUISHABILITY` (SPEC.md §6.8)

*Goal: two-step pooling (SPEC.md §3 decision 9) — per-experiment synonymous z-score, then cross-experiment median. Not a direct median of raw AUROC.*

### Story 8.1 — `global_variant_distinguishability()`
- [ ] Per experiment: `variant_classification()` + `Normalizer.from_lazyframe(fit_only_on_control=True)` fit **fresh per experiment** on that experiment's own `results.parquet` (not reused from Epic 4/6's per-cell normalizer — different population, different unit).
- [ ] `Normalizer.apply()` called unmodified — confirm it only touches `auroc_pooled`/`auroc_median_barcode` and leaves `meta_n_barcodes`/`meta_n_cells` alone (this only works if Epic 6 actually named those columns `meta_*` — cross-check).
- [ ] Cross-experiment median of the **z-scored** values (not raw AUROC) into `meta_median_auroc_pooled`/`meta_median_auroc_median_barcode`, plus `meta_num_experiments`.
- [ ] Unit test covering the graceful-degradation path: an experiment with a single (or zero-variance) synonymous population should null out that column rather than raising, and polars' `.median()` should silently exclude the null from the pooled result (SPEC.md's Resolved note) — write a test that actually exercises this, since it's easy to get the null-propagation wrong.
- [ ] Unit test: synonymous variants' own z-scored values land near 0 (the built-in sanity check SPEC.md calls out).

---

## Epic 9 — Nextflow orchestration (SPEC.md §7)

*Goal: wire every module above into a working end-to-end `nextflow run`.*

### Story 9.1 — `main.nf` / `workflows/embeddings.nf`
- [ ] `configsDir` validation (`pipeline_dir` required, at least one `configs/*.yaml`) implemented with clear `error` messages.
- [ ] `config_ch` construction from `config_files` (left as a TODO in the scaffold — needs a real implementation, e.g. mapping each YAML's filename stem to `batch_stem`).
- [ ] Full per-batch chain wired: `BUILD_DATASET -> {QC_FILTER, EMBED_CELLS} -> FILTER_EMBEDDINGS -> {AGGREGATE_EMBEDDINGS, OVWT_BATCHWISE}` exactly as the scaffold's `workflows/embeddings.nf` sketches, with real (not glob-string) path channels feeding the two global processes (the `AGENTS.md`-cited `fisseq-data-pipeline` resume-cache gotcha).
- [ ] `-resume` verified to actually skip unchanged per-batch stages on a second run (this is the whole point of the WebDataset/no-copy redesign — verify it actually delivers the promised incrementality).

### Story 9.2 — Remaining `modules/local/*.nf`
- [ ] `build_dataset.nf`, `qc_filter.nf` finalized against Epics 1-2's real config field names (the scaffold's CLI arg lists are best-guess placeholders).
- [ ] `filter_embeddings.nf`, `aggregate_embeddings.nf`, `ovwt_batchwise.nf` finalized against Epics 4-6.
- [ ] `global_variant_embeddings.nf`, `global_variant_distinguishability.nf` finalized against Epics 7-8.
- [ ] Every module: `errorStrategy 'ignore'`, `container "${params.container_image}"`, `publishDir`, trailing `random_seed=${params.random_seed}` — confirmed present on all 8.

### Story 9.3 — End-to-end smoke run
- [ ] A full `nextflow run . --pipeline_dir <tiny-synthetic-experiment> -params-file params.yaml` completes with exit code 0 against the real (not stubbed) modules.
- [ ] Output directory tree matches SPEC.md §8 exactly (spot-check every file listed there exists).

---

## Epic 10 — Configuration & containerization (SPEC.md §9.1, §9.2)

### Story 10.1 — `params.yaml` completeness & fail-fast
- [ ] Every Hydra config field introduced in Epics 1-8 has a corresponding `params.yaml` entry (cross-check against the scaffold's starter file — it may be incomplete).
- [ ] Required-with-no-default params (`pipeline_dir`, `cell_dino_checkpoint`) fail with a clear, specific error message when unset — **not** Nextflow's generic "no such property" (SPEC.md §9.1's Resolved note) — write a test exercising this.

### Story 10.2 — Docker image
- [ ] `Dockerfile` builds successfully (`docker build -t fisseq-embeddings-pipeline:latest .`).
- [ ] Image actually runs each stage's `python -m fisseq_embeddings_pipeline.<module>` command successfully, GPU stage included (verify `--gpus all` / `nvidia-smi` inside the container on a GPU host).
- [ ] `dinov2` vendoring/installation resolved inside the Dockerfile (currently a TODO in the scaffold — SPEC.md doesn't decide packaging strategy for it).
- [ ] Decide and document image versioning/publishing (registry, tag scheme, CI build) — explicitly out of scope in SPEC.md §9.2's Resolved note, needs a decision here.
- [ ] Revisit (not required for v1) whether a CPU-only image is worth splitting out, once real pull-time cost is measured.

---

## Epic 11 — Testing (SPEC.md §9.3)

### Story 11.1 — Unit test coverage
- [ ] One test module per stage exists (`test_dataset.py`, `test_qcfilter.py`, `test_embed.py`, `test_filter.py`, `test_aggregate.py`, `test_ovwt.py`, `test_global_embeddings.py`, `test_global_distinguishability.py`), mirroring `fisseq-data-pipeline/tests/unit/`'s one-module-per-stage convention.
- [ ] Every "Resolved" note across `SPEC.md` §6 that describes specific behavior (dropped downsampling, no per-barcode pooling, graceful null-on-zero-variance, etc.) has at least one test asserting that behavior, not just the happy path.

### Story 11.2 — Integration suite
- [ ] `tests/integration/test_integration.py` implements the session-scoped `pipeline_outputs` fixture (synthetic `phenotyping_dir` + experiment config, real `nextflow run` via `subprocess`).
- [ ] Synthetic fixture generator (`_write_synthetic_tiles`) produces valid tiny crops/masks/cell tables matching `starcall-workflow`'s `make_cell_images` output shape closely enough for `BUILD_DATASET` to ingest them without special-casing.
- [ ] `test_pipeline_exits_cleanly`, `test_filter_embeddings_has_no_embedding_columns`, `test_ovwt_results_have_both_auroc_columns` implemented per the scaffold's sketch.
- [ ] `test_rerunning_with_same_seed_reproduces_ovwt_scores` implemented for real (a stub `...` in the sketch) — this is the test that actually proves SPEC.md §3 decision 11's reproducibility claim, not just that the field exists.
- [ ] `EMBED_CELLS`' CI/GPU-checkpoint problem (SPEC.md §9.3's Resolved note) decided: tiny stub checkpoint vs. test-mode fake — implement whichever was chosen, and document the choice in `AGENTS.md`'s testing section.
- [ ] CI actually runs `tests/integration` (with `nextflow` installed) on some cadence, even if not every PR (GPU/nextflow-runtime cost permitting) — decide and document.

---

## Epic 12 — Documentation & release readiness

### Story 12.1 — Real docs, replacing the placeholders
- [ ] `docs/index.md`, `docs/architecture.md`, `docs/nextflow.md`, `docs/configuration.md` rewritten with real content (not the "see SPEC.md" placeholders this scaffold ships with), once the corresponding epics are done.
- [ ] `README.md`'s "Status: pre-implementation" note removed/updated once the pipeline actually runs end to end.
- [ ] `mkdocs` site builds and publishes (mirroring `fisseq-data-pipeline`'s `mkdocs.yml`/`site/` setup) if that's still the desired docs delivery mechanism.

### Story 12.2 — Housekeeping
- [ ] `LICENSE.txt` added (not yet decided — flagged in `README.md`).
- [ ] `.github/` CI workflows added (lint + unit tests at minimum; integration tests per Epic 11's cadence decision).
- [ ] `SPEC.md` reconciled against final implementation — any place implementation legitimately diverged from the spec (beyond the two items §10 always expected to change) should be called out in `docs/architecture.md`, not left as a silent mismatch between the two documents.
