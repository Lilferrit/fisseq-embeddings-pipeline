# AGENTS.md — fisseq-embeddings-pipeline

> **This repo is pre-implementation.** `SPEC.md` (repo root) is the
> authoritative design reference — every architecture decision, data
> contract, per-stage config/function sketch, Nextflow wiring, and output
> layout is there. `docs/` is a placeholder tree, not yet written for real —
> do not treat it as current (same caution `fisseq-data-pipeline`'s own
> AGENTS.md gives about its `docs/`, just pre-emptively here since there's no
> implementation to document yet). `IMPLEMENTATION_CHECKLIST.md` is the work
> plan: an Agile-style backlog of epics/stories/acceptance criteria, in
> implementation order, with checkboxes to mark off as you go. Work through
> it top to bottom; each story names the exact `SPEC.md` section(s) it
> implements.

---

## Project overview

`fisseq-embeddings-pipeline` is the embedding-space sibling of
`fisseq-data-pipeline` — same overall shape (Nextflow DSL2 orchestrating
Python/Hydra/polars stages, per-experiment batches, a QC → normalize →
one-vs-wildtype → aggregate → global-pool structure), but scores genetic
variants against a pretrained **Cell-DINO** vision transformer's learned
embeddings instead of hand-engineered CellProfiler features. See `SPEC.md`
§1 for the full picture and its ASCII DAG.

**Sibling repos** (referenced throughout `SPEC.md`, and the source of every
piece of vendored code below):
- `fisseq-data-pipeline` — the CellProfiler-feature version of this same
  analysis. Most of this pipeline's Python is either vendored unchanged from
  it or adapted from it — `SPEC.md` §3 decision 2 lists exactly what, and
  each stub module under `src/fisseq_embeddings_pipeline/` says what it
  vendors/adapts and from which file.
- `starcall-workflow` — the Snakemake pipeline whose `origin/devel` branch
  produces this pipeline's two inputs (Cell Info Table, Cell Images). See
  `SPEC.md` §5.

If you have read access to both sibling repos (e.g. checked out alongside
this one), use them directly rather than re-deriving vendored code from
`SPEC.md`'s illustrative sketches — the sketches are a design aid, not
guaranteed-correct source to copy verbatim. Where `SPEC.md` says "vendored
unchanged," pull the real file across; where it says "vendored with N things
different," diff against the real file and apply exactly those changes.

## Working through the checklist

- Follow `IMPLEMENTATION_CHECKLIST.md`'s epic order — later epics assume
  earlier ones exist (e.g. every stage's Hydra config extends `AppConfig`
  from Epic 0; `AGGREGATE_EMBEDDINGS`/`OVWT_BATCHWISE` in Epics 5-6 both call
  `filter.py`'s `load_filtered_embeddings()` from Epic 4).
- Check off each item as you complete it (`- [x]`) and commit the checked-off
  checklist alongside the code it corresponds to, so the checklist stays a
  true record of progress for whoever picks this up next (including a future
  you).
- Each story's acceptance criteria are meant to be concretely verifiable —
  a specific output file, column set, or test — not vibes. If an acceptance
  criterion turns out to be wrong or ambiguous once you're actually
  implementing it, fix the checklist item to say what's actually correct
  (and note why in the commit message) rather than silently diverging from
  it.
- Two items are explicitly *not* resolved in `SPEC.md` (§10) and need your
  judgment once you're in the real code/data: Cell-DINO's actual inference
  API (§6.3, Epic 3), and WebDataset shard byte-sizing (§6.1, Epic 1). Don't
  treat the spec's sketches for those as settled.

## Repo conventions

- **Python stages**: each `src/fisseq_embeddings_pipeline/<stage>.py` is a
  Hydra entry point invoked as `python -m fisseq_embeddings_pipeline.<stage>`
  (see any `modules/local/*.nf` for the exact CLI shape), with a
  `@dataclasses.dataclass class <Stage>Config(AppConfig)` registered via
  `ConfigStore`, matching `fisseq-data-pipeline`'s pattern exactly.
- **Every config extends `AppConfig`** (`config/app.py`), which carries the
  one shared `random_seed` field every stochastic stage reads from — never
  add a stage-local `random_state`/seed field (`SPEC.md` §3 decision 11).
- **polars, not pandas**, for all tabular data except where `SPEC.md`
  explicitly uses pandas (`BUILD_DATASET`'s tile manifest, §6.1 — matching
  `starcall-workflow`'s own CSV-reading convention there).
- **`meta_*` column convention**: metadata columns are prefixed `meta_*`;
  `FEATURE_SELECTOR` (`cs.exclude("^meta_.*$")`) and the new
  `EMBEDDING_SELECTOR` (`cs.matches(r"^emb_\d+$")`) key off this — see
  `SPEC.md` §6.3's Output note before adding any new non-`meta_*` column.
- **No stage copies another stage's data wholesale** — see `SPEC.md` §3
  decision 10. If you're about to write a full copy of another stage's
  table to disk (rather than a join key + something new), stop and check
  whether that violates the no-copy principle.
- **Nextflow modules** (`modules/local/*.nf`): `errorStrategy 'ignore'`,
  `container "${params.container_image}"`, `publishDir`, a `python -m
  <pkg>.<module>` script block ending in `random_seed=${params.random_seed}`
  — see `embed_cells.nf` for the fully-worked example, `SPEC.md` §7.3.
- **Config**: defaults belong in `params.yaml` (repo root), never in
  `nextflow.config`'s `params {}` block — see `SPEC.md` §9.1.

## Testing

```bash
uv sync --group dev
uv run pytest tests/unit                      # fast, no nextflow/GPU needed
uv run pytest tests/integration                # needs `nextflow` on PATH; see SPEC.md §9.3
uv run ruff check --fix . && uv run ruff format .
uv run pre-commit run --all-files
```

`tests/unit/` mirrors `fisseq-data-pipeline`'s layout (one test module per
pipeline stage). `tests/integration/test_integration.py` is modeled directly
on that repo's own integration suite — a synthetic fixture, a
`subprocess`-driven end-to-end `nextflow run`, and output-file/column
assertions. See `SPEC.md` §9.3 for the sketch and the one open design
problem it flags (stubbing `EMBED_CELLS`' GPU/checkpoint dependency for CI).

## Docker / devcontainer

`Dockerfile` (repo root) is the single image every Nextflow process runs in
(`SPEC.md` §9.2) — build it locally with `docker build -t
fisseq-embeddings-pipeline:latest .` and point `params.yaml`'s
`container_image` at wherever you publish it. `.devcontainer/` is a
Claude-Code-in-a-sandbox dev environment (mirrors `fisseq-data-pipeline`'s
own `.devcontainer/`) — not the pipeline's runtime container, just where you
edit code.
