# Dockerfile -- single image every Nextflow process runs in (SPEC.md §9.2 /
# §3 decision 13). One CUDA-capable base serves CPU-only stages too (simpler
# to build/publish as one artifact); revisit splitting into a CPU + GPU image
# later if the pull cost matters in practice -- see IMPLEMENTATION_CHECKLIST.md
# Epic 10 Story 10.2.
#
# NOT build-verified in this sandbox (no docker daemon available here --
# `dockerd` fails to start at all: no CAP_NET_ADMIN/iptables access even
# under sudo, so there's no way around it, unlike Epic 9's nextflow/java gap
# which a plain package install fixed). Everything below the base-image
# `apt-get`/CUDA layer was instead verified by literally reproducing this
# file's COPY/RUN ordering by hand against a scratch directory with `uv`
# (installed in this sandbox already) -- see the Epic 10 commit message for
# exactly what that caught. Still needs a real `docker build` +
# `docker run --gpus all` pass on a GPU host before this is trusted blind.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# uv as a standalone static binary (Astral's own published image), not via
# `pip install uv` -- this needs no system Python/pip pre-installed at all,
# since uv provisions its own Python interpreter satisfying pyproject.toml's
# `requires-python = ">=3.13"` itself (confirmed empirically: `uv sync` pulls
# a managed CPython even with only an incompatible system Python on PATH).
# ubuntu22.04 (jammy)'s own apt archives only carry Python 3.10 by default
# (3.11 needs a backport/PPA, and neither satisfies >=3.13 regardless) --
# rather than fight that, let uv manage the interpreter version explicitly,
# which is what actually determines what runs at container run time anyway
# (see the ENV PATH line below).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /opt/fisseq-embeddings-pipeline

# Dependencies-only layer, cached independently of source changes. `uv sync`
# always also builds/installs the local project itself by default, which
# fails here (no src/ or README.md yet -- hatchling's own metadata
# validation needs both present, confirmed empirically) -- so restrict this
# pass to third-party deps only. --no-dev excludes the mkdocs/pytest/ruff/
# pre-commit dev-group tooling (AGENTS.md's `uv sync --group dev`) from the
# runtime image; every process below runs `python -m
# fisseq_embeddings_pipeline.<module>` directly, never `pytest`/`ruff`.
COPY pyproject.toml uv.lock ./
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
