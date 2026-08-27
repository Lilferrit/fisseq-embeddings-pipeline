# Dockerfile -- single image every Nextflow process runs in (SPEC.md §9.2 /
# §3 decision 13). One CUDA-capable base serves CPU-only stages too (simpler
# to build/publish as one artifact); revisit splitting into a CPU + GPU image
# later if the pull cost matters in practice -- see IMPLEMENTATION_CHECKLIST.md
# Epic 10 Story 10.2.
#
# Build-verified for real (docker build + docker run against every stage's
# CLI entry point) once this devcontainer got docker-outside-of-docker
# access to the host's real Docker daemon -- see the Epic 10 commit
# messages for the bugs a real `docker build`/`docker run` pass caught
# (beyond the two found earlier by hand-simulating the COPY/RUN layering
# without a daemon). Still not run with `--gpus all` on an actual GPU host
# -- this devcontainer's host has none.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv as a standalone static binary (Astral's own published image), not via
# `pip install uv` -- this needs no system Python/pip pre-installed at all,
# since uv provisions its own Python interpreter itself (confirmed
# empirically: `uv sync` pulls a managed CPython even with only an
# incompatible system Python on PATH). ubuntu22.04 (jammy)'s own apt
# archives only carry Python 3.10 by default (3.11 needs a backport/PPA,
# and neither satisfies pyproject.toml's `requires-python = ">=3.13"`
# regardless) -- rather than fight that, let uv manage the interpreter
# version explicitly, which is what actually determines what runs at
# container run time anyway (see the ENV PATH line below).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /opt/fisseq-embeddings-pipeline

# .python-version (pins 3.13, not just pyproject.toml's open-ended
# `>=3.13`) has to be copied in before the first `uv sync` -- caught by a
# real `docker build`: without it, uv has nothing here to prefer 3.13 over
# any other >=3.13 interpreter, so it silently resolved the newest one it
# could fetch instead (3.14 as of this build) -- which turned out to be
# genuinely incompatible: Hydra 1.3.x's `get_args_parser()` crashes on
# Python 3.14's stricter argparse `_check_help` (`TypeError: argument of
# type 'LazyCompletionHelp' is not a container or iterable`), breaking
# every single stage's `python -m fisseq_embeddings_pipeline.<module>`
# invocation outright, not just `--help`. Confirmed fixed by copying
# .python-version in before either `uv sync` call below, restoring 3.13.
#
# Dependencies-only layer, cached independently of source changes. `uv sync`
# always also builds/installs the local project itself by default, which
# fails here (no src/ or README.md yet -- hatchling's own metadata
# validation needs both present, confirmed empirically) -- so restrict this
# pass to third-party deps only. --no-dev excludes the mkdocs/pytest/ruff/
# pre-commit dev-group tooling (AGENTS.md's `uv sync --group dev`) from the
# runtime image; every process below runs `python -m
# fisseq_embeddings_pipeline.<module>` directly, never `pytest`/`ruff`.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-dev

# Now install the project itself -- the deps layer above stays cached across
# source-only changes. dinov2 needs no separate vendoring/installation step
# here (this comment used to carry an Epic 3/SPEC.md §6.3 TODO to that
# effect): it's vendored as pure-torch source directly under
# src/fisseq_embeddings_pipeline/vendor/dinov2/ (see that directory's
# VENDORED_FROM.md for why and what), so it's part of this package and
# installs with it -- no xformers/cuml/git-dependency wrangling needed.
COPY README.md ./
COPY src/ src/
RUN uv sync --frozen --no-dev

# `uv sync` installs into a project-local .venv/, not any system Python --
# put it on PATH so the bare `python -m fisseq_embeddings_pipeline.<module>`
# every modules/local/*.nf script block invokes actually resolves there.
# Confirmed empirically this matters: without it, `python`/`python3` would
# fall through to whatever's first on the base image's own PATH (nothing,
# now that no system Python is installed above -- but even with one, none
# of the deps installed above would be on its sys.path), and every process
# would fail with ModuleNotFoundError/`python: command not found`.
ENV PATH="/opt/fisseq-embeddings-pipeline/.venv/bin:${PATH}"

ENTRYPOINT []
