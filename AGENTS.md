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
`fisseq-data-pipeline` — same overall shape (Nextflow DSL2 orchestrating
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
--show-current`); if it's on `master`, the phenotyping rules this pipeline
depends on (`make_cell_images`, `extract_embeddings` in
`workflow/rules/phenotyping.smk`) won't be there at all.

## Repo conventions

- **Python stages**: each `src/fisseq_embeddings_pipeline/<stage>.py` is a
  Hydra entry point invoked as `python -m fisseq_embeddings_pipeline.<stage>`
  (see any `modules/local/*.nf` for the exact CLI shape), with a
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
- **Nextflow modules** (`modules/local/*.nf`): `errorStrategy 'ignore'`,
  `container "${params.container_image}"`, `publishDir`, a `python -m
  <pkg>.<module>` script block ending in `random_seed=${params.random_seed}`
  — see `embed_cells.nf` for the fully-worked example.
- **Config**: defaults belong in `params.yaml` (repo root), never in
  `nextflow.config`'s `params {}` block — see
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
  the branch touches Nextflow wiring) pass, merge it back to `main`
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
uv sync --group dev
uv run pytest tests/unit                      # fast, no nextflow/GPU needed
uv run pytest tests/integration                # needs `nextflow` on PATH
uv run ruff check --fix . && uv run ruff format .
uv run pre-commit run --all-files
```

`tests/unit/` mirrors `fisseq-data-pipeline`'s layout (one test module per
pipeline stage). `tests/integration/test_integration.py` is modeled
directly on that repo's own integration suite — a synthetic fixture, a
`subprocess`-driven end-to-end `nextflow run`, and output-file/column
assertions.

`EMBED_CELLS`' GPU/checkpoint dependency is handled in the integration
fixture by building a tiny, from-scratch, randomly-initialized
`vit_small` checkpoint (`_write_tiny_checkpoint`) and running
`EMBED_CELLS` against it with `device=cpu`, rather than stubbing
`load_cell_dino` out entirely. This exercises the wrapper's real control
flow (weight loading, forward pass, shape handling) — not Cell-DINO's
actual pretrained-checkpoint output quality — at the cost of a few
seconds of real (CPU) compute per run. No GPU or real checkpoint is
needed to run `tests/integration` anywhere, including CI.

`tests/integration/test_integration_real_starcall.py` is the one
exception to all of the above: it invokes a **real** `snakemake` run
against real starcall-workflow data (every other test fakes that step
with a stub `snakemake` on PATH), through a real build of the root
Dockerfile. Opt-in only — self-skips unless `testing_data/lmna_t3/` has
been populated (`uv run python scripts/prepare_real_starcall_test_data.py`,
downloads ~6.3GB) and `docker`/`nextflow` are on PATH; not run in CI.
Needs a Docker daemon that can bind-mount this repo's own temp
directories — Docker Desktop's file-sharing allowlist can silently block
that on some local setups (see that test module's own docstring for the
exact failure signature and how to tell it apart from a real bug).

## CI

Three workflows under `.github/workflows/`:

- **`pr-checks.yml`** — every pull request against `main`: unit tests,
  the integration suite, and lint.
- **`docker.yml`** — builds the Docker image on every PR (build-only, no
  push); on every push to `main` (and on a pushed `v*` tag) it also
  pushes to `ghcr.io/<owner>/<repo>`. See
  [`docs/configuration.md`](docs/configuration.md#docker-image-versioning-publishing)
  for the tagging convention.
- **`docs.yml`** — builds the mkdocs site and deploys it to the
  `gh-pages` branch on every push to `main` that touches `docs/`,
  `mkdocs.yml`, `src/`, or `README.md`. `gh-pages` holds only the
  generated site, never the source.

## Docker / devcontainer

`Dockerfile` (repo root) is the single image every Nextflow process runs
in — build it locally with `docker build -t
fisseq-embeddings-pipeline:latest .` and point `params.yaml`'s
`container_image` at wherever you publish it (see
[`docs/configuration.md`](docs/configuration.md)). `.devcontainer/` is a
Claude-Code-in-a-sandbox dev environment (mirrors `fisseq-data-pipeline`'s
own `.devcontainer/`) — not the pipeline's runtime container, just where
you edit code.
