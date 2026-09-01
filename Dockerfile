# Dockerfile -- single image every Nextflow process runs in, including
# BUILD_CELL_IMAGES (modules/local/build_cell_images.nf). One CUDA-capable
# base serves CPU-only stages too (simpler to build/publish as one
# artifact); revisit splitting into a CPU + GPU image later if the pull
# cost matters in practice.
#
# Build-verified for real (docker build + docker run against every stage's
# CLI entry point) against the host's real Docker daemon. Not run with
# `--gpus all` on an actual GPU host.
#
# BUILD_CELL_IMAGES used to run in a wholly separate image
# (docker/starcall.Dockerfile, since deleted) because it's the one stage
# that invokes starcall-workflow's own Snakemake pipeline, whose dependency
# stack (tensorflow/stardist/cellpose) was assumed likely-incompatible with
# this repo's own torch/polars stack. That assumption was never actually
# tested against a real merged build -- the risk was really a Python
# version mismatch (this repo: 3.13, starcall-workflow's own baseline:
# 3.10), which a second, isolated conda env inside this one image resolves
# just as well as a second image would. See the `ops` env block below --
# folded in here almost verbatim from the deleted Dockerfile, including all
# its build-failure-derived rationale comments, which are still true. One
# concrete benefit of merging: this repo's own CI (.github/workflows/
# docker.yml) only ever built this file, never docker/starcall.Dockerfile
# -- the `ops` env now gets real CI build coverage for the first time.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# ca-certificates/git: needed by both the uv project install below and the
# `ops` env's starcall-workflow clone. curl: fetches the Miniforge
# installer (this base image ships no conda). build-essential: the `ops`
# env's snakemake dependency `datrie` has no prebuilt wheel for this
# platform and fails to compile ("error: [Errno 2] No such file or
# directory: 'gcc'") without a C compiler present -- confirmed via a real
# `docker build` of the pre-merge docker/starcall.Dockerfile.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git curl build-essential \
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

# ── starcall-workflow's `ops` conda env ─────────────────────────────────────
# Placed here -- after the uv deps layer, before the project-source layer
# -- so a routine source-only commit (far more common than an `ops`-env
# dependency bump) doesn't force Docker to redo this whole block. Installed
# to /opt/conda, deliberately NOT added to ENV PATH: bare `python`/
# `python3` must always resolve to the uv-managed 3.13 venv (ENV PATH line
# near the bottom), never ambiguously to this env's Python 3.10. Every
# process that needs something from `ops` (just the one `snakemake ...`
# line in modules/local/build_cell_images.nf) reaches it by absolute path,
# /opt/conda/envs/ops/bin/<binary>, instead.
# Architecture-detected, not hardcoded to x86_64 -- confirmed via a real
# `docker build` that a hardcoded x86_64 installer run under arm64 (native
# or emulated) fails opaquely ("rosetta error: failed to open elf"/a
# SIGTRAP from the self-extracting installer trying to run x86_64 code),
# not with an obvious "wrong architecture" message. `uname -m` maps
# `x86_64`->`x86_64` and `aarch64`->`aarch64`, matching Miniforge's own
# release asset naming (`Miniforge3-Linux-<arch>.sh`) exactly.
RUN miniforge_arch="$(uname -m)" \
    && curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${miniforge_arch}.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -afy

# python 3.10, matching starcall-workflow's own readme.md exactly (its
# `pip3 install -r requirements.txt` step assumes this interpreter
# version -- not independently re-verified against every pinned dependency
# in requirements.txt, e.g. tensorflow==2.13.0/stardist==0.8.5/
# cellpose==2.2.2, but matching the documented baseline is the safest
# starting point).
RUN /opt/conda/bin/conda create -y -n ops python=3.10 && /opt/conda/bin/conda clean -afy

# Activate the `ops` env for every subsequent RUN in this block -- conda's
# own `conda activate` doesn't persist across Docker RUN layers without
# this SHELL trick. Reset back to plain bash once the `ops`-specific steps
# are done (below), so it doesn't leak into the uv-facing layers that
# follow.
SHELL ["/opt/conda/bin/conda", "run", "--no-capture-output", "-n", "ops", "/bin/bash", "-c"]

# snakemake>=7 (starcall-workflow's own requirement) plus conda-frontend
# support for `--use-conda` (materializing workflow/envs/cp4.yaml's
# CellProfiler env on demand, the first time a cp_features: true
# experiment actually needs it -- see modules/local/build_cell_images.nf's
# `--use-conda --conda-frontend conda` invocation). mamba is already on
# PATH via Miniforge, and would be faster than conda's own solver for
# `--conda-frontend`, but build_cell_images.nf's own invocation uses
# --conda-frontend conda for the widest compatibility; switch it there if
# mamba is confirmed to work once real rule execution is validated.
RUN pip install --no-cache-dir "snakemake>=7"

# Cloned at build time (no vendored copy lives in this repo) at the
# origin/devel ref this pipeline tracks -- NOT master, which has a
# differently-shaped make_cell_images and no extract_embeddings rule at
# all (see docs/architecture.md's "A note on branches"). --recursive pulls
# both submodules (packages/starcall, packages/constitch) in the same
# step. Cloned into a fixed, well-known path purely to install its Python
# *dependencies* (requirements.txt + the two submodules' own setup.py
# packages) into the `ops` env -- NOT as the checkout any experiment's
# starcall_workflow_dir should point at; that always points at a real,
# already-populated experiment-specific checkout mounted from outside the
# container (starcall_workflow_dir is per-experiment, not baked into this
# image -- see modules/local/build_cell_images.nf's own comment on why).
# Removed once its pip installs succeed (below) -- nothing at runtime reads
# this checkout, only the `ops` env's now-installed site-packages.
ARG STARCALL_WORKFLOW_GIT_URL=https://github.com/FowlerLab/starcall-workflow.git
ARG STARCALL_WORKFLOW_REF=origin/devel
RUN git clone --recursive "${STARCALL_WORKFLOW_GIT_URL}" /opt/starcall-workflow-deps \
    && cd /opt/starcall-workflow-deps \
    && git fetch origin devel \
    && git checkout "${STARCALL_WORKFLOW_REF}" \
    && git submodule update --init --recursive

# SETUPTOOLS_USE_DISTUTILS=stdlib is required -- confirmed via a real
# `docker build`: stardist's legacy numpy.distutils-based build imports
# `distutils.msvccompiler` (a Windows-only compiler shim) unconditionally
# at module load time; setuptools' own vendored `_distutils` copy (the
# default resolution since setuptools>=60) doesn't ship that submodule at
# all, so the build fails with `ModuleNotFoundError: No module named
# 'distutils.msvccompiler'` even on Linux, where it's never actually used.
# Forcing the real stdlib `distutils` (which does ship that submodule,
# platform-conditional behavior notwithstanding) is the standard fix for
# this well-documented numpy.distutils/setuptools incompatibility.
ENV SETUPTOOLS_USE_DISTUTILS=stdlib
# NOT editable (-e) installs for the two submodules -- confirmed via a real
# `docker build` + `docker run` that `pip install -e` fails at import time
# here: both packages/starcall/setup.py and packages/constitch/setup.py
# declare `packages=['starcall/']`/`packages=['constitch/']` (note the
# stray trailing slash -- a real bug in starcall-workflow's own setup.py,
# out of scope to fix here, it's a read-only sibling repo), which makes
# setuptools' auto-generated editable-install finder's own MAPPING key
# ('starcall/', with the slash) never match `fullname` at actual import
# time ('starcall', without it) -- `import starcall` then raises
# ModuleNotFoundError despite `pip show starcall` reporting it installed. A
# plain (non-editable) install doesn't hit this path at all -- it copies
# the package directory once at install time using setuptools' own legacy
# `packages=[...]`-driven file discovery, which tolerates the trailing
# slash typo fine, and does not need to remain "live" against the checkout
# afterward (this checkout is deleted right after, below).
RUN pip install --no-cache-dir -r /opt/starcall-workflow-deps/requirements.txt \
    && pip install --no-cache-dir /opt/starcall-workflow-deps/packages/constitch \
    && pip install --no-cache-dir /opt/starcall-workflow-deps/packages/starcall

# Build-time-only checkout, no longer needed once the pip installs above
# succeed -- trims a meaningful chunk of image size (a full starcall-
# workflow + submodules git history) at zero runtime cost.
RUN rm -rf /opt/starcall-workflow-deps

# Reset SHELL back to plain bash -- the layers below (uv-facing) should
# not run inside the `ops` conda env.
SHELL ["/bin/bash", "-c"]

# Now install the project itself -- the deps layer above stays cached across
# source-only changes. dinov2 needs no separate vendoring/installation step
# here: it's vendored as pure-torch source directly under
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
# would fail with ModuleNotFoundError/`python: command not found`. The
# `ops` conda env above is deliberately never added here, for the same
# "no ambiguous bare python" reason in reverse -- see that block's own
# comment.
ENV PATH="/opt/fisseq-embeddings-pipeline/.venv/bin:${PATH}"

ENTRYPOINT []
