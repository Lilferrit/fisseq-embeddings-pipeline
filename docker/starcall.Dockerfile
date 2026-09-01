# docker/starcall.Dockerfile -- the SEPARATE image BUILD_CELL_IMAGES runs in
# (modules/local/build_cell_images.nf), distinct from the repo-root
# Dockerfile every other process uses. BUILD_CELL_IMAGES is the only stage
# that invokes starcall-workflow's own Snakemake pipeline, whose dependency
# stack (tensorflow/stardist/cellpose) is very likely incompatible with
# this repo's own torch/Cell-DINO/polars image -- see docs/architecture.md.
#
# BUILD- AND IMPORT-VERIFIED, RULE EXECUTION NOT YET VERIFIED: this file has
# actually been built (`docker build`, real Docker daemon, arm64) and every
# dependency confirmed importable inside the built image at the pinned
# versions requirements.txt asks for (tensorflow 2.13.0, stardist 0.8.5,
# cellpose 2.2.2, snakemake 7.32.4, plus starcall/constitch themselves and
# this pipeline's own build_cell_images_glue.py) -- not merely assumed.
# Two real, reproduced build failures were found and fixed this way (see
# inline comments below): a missing `build-essential` (snakemake's own
# `datrie` dependency fails to compile without a C compiler present), and
# `pip install -e` failing for `packages/starcall`/`packages/constitch`
# specifically (their own `setup.py`s declare `packages=['starcall/']` /
# `packages=['constitch/']` -- note the stray trailing slash, a real bug in
# starcall-workflow's own setup.py, not fixable here -- which breaks
# setuptools' auto-generated editable-install import hook; switched to a
# plain, non-editable install instead, which doesn't hit that path).
#
# What's still NOT verified (no real starcall-workflow-managed experiment
# tree was available to test against): an actual `snakemake --cores N ...`
# run of `stitch_tile_pt`/`stitch_tile_segmentation`/`split_grid_table`/
# `merge_final_tables` (and `run_cellprofiler` via `--use-conda`, if any
# experiment sets `cp_features: true`) against real data. Also unconfirmed:
# whether the lab's real deployment needs anything from `run.sh`'s
# SGE-specific wiring (a conda env literally named `ops3`, `qsub`/`qdel`
# cluster submission) beyond what this Dockerfile captures -- `run.sh`
# itself is NOT portable and is not used here; only its generic
# `ops`-env-creation + `pip install -r requirements.txt` baseline (from
# starcall-workflow's own readme.md) is followed.
#
# Baseline: starcall-workflow's readme.md "Installation" section (conda
# env named `ops`, python 3.10, `pip3 install -r requirements.txt`) --
# there is no Dockerfile/CI/registry image anywhere in starcall-workflow
# or its two submodules (packages/starcall, packages/constitch) to adapt
# instead (confirmed by searching the full git history of all three, all
# branches). CellProfiler itself is deliberately NOT installed here -- it
# lives in a SEPARATE per-rule conda env (workflow/envs/cp4.yaml) that
# Snakemake's own `--use-conda` materializes on demand the first time a
# cp_features: true experiment actually needs it (see
# modules/local/build_cell_images.nf's `--use-conda --conda-frontend conda`
# invocation) -- baking cp4 into this base image isn't needed and would
# only slow down every build for experiments that never set cp_features.
FROM condaforge/miniforge3:latest

# build-essential is required -- confirmed via a real `docker build`:
# snakemake's own dependency `datrie` has no prebuilt wheel for this
# platform and fails ("error: [Errno 2] No such file or directory: 'gcc'")
# without a C compiler present.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Per-experiment starcall-workflow checkout(s) ────────────────────────────
# starcall_workflow_dir is per-experiment (params.yaml's own comment
# explains why -- Snakemake keeps invocation-scoped .snakemake/ lock state
# in its own working directory), so this image does NOT bake in one fixed
# checkout at a single path the way a typical "clone your app into the
# image" Dockerfile would -- each experiment's params.yaml entry points
# `starcall_workflow_dir` at wherever that experiment's own checkout
# lives on the host/shared filesystem, mounted into the container the same
# way phenotyping_dir/segmentation_dir/sequencing_dir already are (see
# docs/nextflow.md's Singularity/Apptainer note on host-path visibility).
# This image therefore only needs the *runtime environment* baked in
# (conda env + snakemake + starcall-workflow's own pip dependencies +
# its two submodules' packages), not a specific checkout's source tree.

# ── Runtime environment ──────────────────────────────────────────────────
# python 3.10, matching starcall-workflow's own readme.md exactly (its
# `pip3 install -r requirements.txt` step assumes this interpreter
# version -- not independently re-verified against every pinned dependency
# in requirements.txt, e.g. tensorflow==2.13.0/stardist==0.8.5/
# cellpose==2.2.2, but matching the documented baseline is the safest
# starting point).
RUN conda create -y -n ops python=3.10 && conda clean -afy

# Activate the `ops` env for every subsequent RUN/CMD in this build --
# conda's own `conda activate` doesn't persist across Docker RUN layers
# without this SHELL trick.
SHELL ["conda", "run", "--no-capture-output", "-n", "ops", "/bin/bash", "-c"]

# snakemake>=7 (this repo's own dependency, per requirements.txt) plus
# conda-frontend support for `--use-conda` (materializing workflow/envs/
# cp4.yaml's CellProfiler env on demand -- see module header above).
# mamba is already on PATH via the condaforge/miniforge3 base image, used
# as --conda-frontend mamba would be faster than conda's own solver, but
# build_cell_images.nf's own invocation uses --conda-frontend conda for
# the widest compatibility; switch it there if mamba is confirmed to work
# once this image is actually validated.
RUN pip install --no-cache-dir "snakemake>=7"

# ── starcall-workflow's own dependencies ────────────────────────────────────
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
# container (see the per-experiment note above).
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
# afterward (this image never edits `/opt/starcall-workflow-deps` post-build).
RUN pip install --no-cache-dir -r /opt/starcall-workflow-deps/requirements.txt \
    && pip install --no-cache-dir /opt/starcall-workflow-deps/packages/constitch \
    && pip install --no-cache-dir /opt/starcall-workflow-deps/packages/starcall

# ── build_cell_images_glue.py's one extra dependency ────────────────────────
# Not part of starcall-workflow's own requirements.txt: this pipeline's
# glue script (modules/local/build_cell_images_glue.py) writes
# cell_table.parquet via polars, matching this repo's own parquet-writing
# convention (AGENTS.md), rather than adding a pyarrow dependency just for
# pandas.to_parquet(). See that script's own module docstring.
RUN pip install --no-cache-dir "polars>=1.32.3"

# Put the `ops` env on PATH for every container invocation (not just
# build-time RUN steps) -- modules/local/build_cell_images.nf's script
# block calls `python3`/`snakemake` directly, with no `conda run`/`conda
# activate` wrapper of its own.
ENV PATH="/opt/conda/envs/ops/bin:${PATH}"

ENTRYPOINT []
