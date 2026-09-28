# AGENTS.md — fisseq-embeddings-pipeline

> **This repo is implemented.** `docs/` (built with mkdocs, published to
> GitHub Pages on every push to `main` — see **CI** below) is the
> authoritative reference: architecture decisions, data contracts,
> per-stage config/usage, Nextflow wiring, and output layout. Read it
> before `SPEC.md`/`IMPLEMENTATION_CHECKLIST.md`, which no longer exist —
> their content was folded into `docs/` once implementation caught up to
> the design.

---

## Project overview

`fisseq-embeddings-pipeline` is the embedding-space sibling of
`fisseq-data-pipeline` — same overall shape (a workflow engine orchestrating
Python/Hydra/polars stages, per-experiment batches, a QC → normalize →
one-vs-wildtype → aggregate → global-pool structure), but scores genetic
variants against a pretrained **Cell-DINO** vision transformer's learned
embeddings instead of hand-engineered CellProfiler features. See
[`docs/architecture.md`](docs/architecture.md) for the full picture and
its ASCII DAG.

**Sibling repos** (the source of every piece of vendored code in this
pipeline):
- `fisseq-data-pipeline` — the CellProfiler-feature version of this same
  analysis. Most of this pipeline's Python is either vendored unchanged
  from it or adapted from it — each module under
  `src/fisseq_embeddings_pipeline/` says what it vendors/adapts and from
  which file in its own docstring; `docs/architecture.md` has the full
  terminology map.
- `starcall-workflow` — the Snakemake pipeline whose `origin/devel` branch
  produces this pipeline's two inputs (Cell Info Table, Cell Images). See
  `docs/architecture.md`'s Data contracts section.

`.devcontainer/devcontainer.json` bind-mounts both sibling repos read-only
into this sandbox at `/workspaces/fisseq-data-pipeline` and
`/workspaces/starcall-workflow` — use them directly from there when
tracing vendored code back to its source. They're mounted read-only on
purpose — don't write into them; if a vendored file needs a fix upstream,
note that in your commit message instead of editing the sibling repo in
place.

**`starcall-workflow` gotcha:** this pipeline tracks `starcall-workflow`'s
`origin/devel` branch, not `master` — check
`/workspaces/starcall-workflow`'s checked-out branch before trusting
anything you read from it (`git -C /workspaces/starcall-workflow branch
--show-current`); if it's on `master`, the phenotyping layout this pipeline
depends on (`workflow/rules/phenotyping.smk`'s whole-tile outputs under
`phenotyping_dir`) won't be there at all. Its own `make_cell_images` is
broken against its cell table, which is why `BUILD_DATASET` does the
cropping itself (`docs/architecture.md` decision 17).

## Repo conventions

- **Python stages**: each `src/fisseq_embeddings_pipeline/<stage>.py` is a
  Hydra entry point invoked as `python -m fisseq_embeddings_pipeline.<stage>`
  (see any module in `modules/local/` for the exact CLI shape), with a
  `@dataclasses.dataclass class <Stage>Config(AppConfig)` registered via
  `ConfigStore`, matching `fisseq-data-pipeline`'s pattern exactly.
- **Every config extends `AppConfig`** (`config/app.py`), which carries the
  one shared `random_seed` field every stochastic stage reads from — never
  add a stage-local `random_state`/seed field.
- **polars, not pandas**, for all tabular data except where the pipeline
  explicitly uses pandas (`build_cell_images_table.py`'s per-tile CSV
  reads — matching `starcall-workflow`'s own CSV-reading convention there;
  it still writes its final `cell_table.parquet` via polars, though, to
  keep everything downstream of `BUILD_CELL_IMAGES` in the usual
  convention).
- **`meta_*` column convention**: metadata columns are prefixed `meta_*`;
  `FEATURE_SELECTOR` (`cs.exclude("^meta_.*$")`) and `EMBEDDING_SELECTOR`
  (`cs.matches(r"^emb_\d+$")`) key off this — see
  [`docs/api/utils.md`](docs/api/utils.md) before adding any new
  non-`meta_*` column.
- **No stage copies another stage's data wholesale.** If you're about to
  write a full copy of another stage's table to disk (rather than a join
  key + something new), stop and check whether that violates the no-copy
  principle — see `docs/architecture.md`'s architecture decisions.
- **Nextflow processes**: one per stage, in
  `modules/local/<name>/main.nf`, wired together in
  `workflows/embeddings.nf`. Each carries `errorStrategy 'ignore'`, a
  `process_*` label, `container "${params.container_image}"` and a
  `publishDir ..., mode: 'copy'` into `pipeline_dir`, and a `script:` of
  `${threadEnv(task.cpus)}` (`modules/local/functions.nf`) plus one
  `python -m <pkg>.<module>` invocation with `output_dir=.` and a trailing
  `random_seed=${params.random_seed}` — see `EMBED_CELLS` for the
  fully-worked example, and `BUILD_CELL_IMAGES` for the one genuine
  exception (a nested starcall `snakemake`). `PLAN_EXPERIMENTS` runs first
  and owns validation/routing (`config/experiments.py`) — add new
  per-experiment routing there, in Python, not in Groovy. See
  [`docs/nextflow.md`](docs/nextflow.md#modules).
- **No scheduler-specific code.** Cluster settings are the user's: a
  `-c site.config` for Nextflow and a snakemake 7 `starcall_profile` for
  the nested starcall run. Only the generic image re-entry jobscript
  (`render_starcall_jobscript`) lives here.
- **Config**: defaults belong in `params.yaml` (repo root), never in
  `nextflow.config` or a profile — see
  [`docs/configuration.md`](docs/configuration.md).

## Git workflow

`main` is this repo's default/integration branch, with a GitHub remote —
see **CI** below for what runs where.

- **One branch per unit of work.** Branch off `main`:
  `git checkout -b <short-slug>`. Merge back to `main` when the work is
  done and tested (see below), then delete the branch.
- **Before every commit**, run the relevant tests and linters and don't
  commit a red state:
  ```bash
  uv run pytest tests/unit
  uv run ruff check --fix . && uv run ruff format .
  ```
  (Full commands in **Testing** below — `tests/integration` doesn't need
  to pass for every single commit, but do run it before merging a branch
  back to `main`.)
- **Finishing a branch:** once `tests/unit` (plus `tests/integration` if
  the branch touches workflow wiring) pass, merge it back to `main`
  (`git checkout main && git merge --no-ff <slug>`; `--no-ff` keeps the
  branch boundary visible in `git log --graph`). Delete the branch after
  merging.
- **Never rewrite history already merged into `main`** (no
  `push --force`/rebase of a merged branch) — even solo, this keeps
  `git log` a trustworthy record of what happened and when.
- If `docs/` diverges from what actually ended up on disk, fix the docs in
  the same commit as the code change that caused the divergence —
  `docs/` should always reflect what's actually implemented, never what's
  planned or in progress.

## Testing

```bash
uv run pytest tests/unit                      # fast, no GPU needed
uv run pytest tests/integration                # real `nextflow run -profile local`; needs nextflow + java on PATH
uv run pytest tests/integration --container    # real starcall instead; needs docker + testing_data/lmna_t3_mini
uv run pre-commit run --all-files
```

`tests/unit/` mirrors `fisseq-data-pipeline`'s layout (one test module per
pipeline stage). `tests/integration/test_integration.py` is modeled
directly on that repo's own integration suite — a synthetic fixture, a
`subprocess`-driven end-to-end `nextflow run -profile local`, and
output-file/column assertions. BUILD_CELL_IMAGES' nested `snakemake` is a
stub on PATH that records its argv; the fixture pre-writes the
starcall-shaped outputs it would have produced.

`EMBED_CELLS`' GPU/checkpoint dependency is handled in the integration
fixture by building a tiny, from-scratch, randomly-initialized
`vit_small` checkpoint (`_write_tiny_checkpoint`) and running
`EMBED_CELLS` against it with `device=cpu`, rather than stubbing
`load_cell_dino` out entirely. No GPU or real checkpoint is needed to run
`tests/integration` anywhere, including CI.

`--container` is mutually exclusive with the synthetic suite (see
`tests/integration/conftest.py`) and runs only the `container`-marked
tests: real starcall-workflow inside a real build of the root Dockerfile
(under Docker), on the tiny `testing_data/lmna_t3_mini/` fixture
(`uv run python scripts/prepare_real_starcall_test_data.py --minimal`),
stopping after `EMBED_CELLS`. `test_real_starcall_local` runs the nested
starcall in local mode; `test_real_starcall_profile_mode` runs it through
a throwaway `starcall_profile` with a fake cluster and fake container
runtime — the real `apptainer` re-entry on a compute node is only
verifiable on a real cluster. Self-skips without the fixture or `docker`;
`FISSEQ_TEST_IMAGE=<tag>` skips the image build; not run in CI. Docker
Desktop's file-sharing allowlist can silently block bind-mounting this
repo's temp directories (see the test module's container-section comment).

## CI

Three workflows under `.github/workflows/` — `pr-checks.yml`,
`docker.yml`, `docs.yml`; read them for their exact triggers. The image
tagging convention is in
[`docs/configuration.md`](docs/configuration.md#docker-image-versioning-publishing).

## Docker / devcontainer

`Dockerfile` (repo root) is the single image every task runs
in — build it locally with `docker build -t
fisseq-embeddings-pipeline:latest .` and point `params.yaml`'s
`container_image` at wherever you publish it (see
[`docs/configuration.md`](docs/configuration.md)). `.devcontainer/` is a
Claude-Code-in-a-sandbox dev environment (mirrors `fisseq-data-pipeline`'s
own `.devcontainer/`) — not the pipeline's runtime container, just where
you edit code.
