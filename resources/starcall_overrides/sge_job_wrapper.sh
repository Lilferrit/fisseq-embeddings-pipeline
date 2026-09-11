#!/usr/bin/env bash
# The per-rule job body SGE actually runs, on a bare exec node, for every
# starcall rule BUILD_CELL_IMAGES' phase 2 submits (see sge_submit.sh, which
# qsub's this).
#
# Invoked as: sge_job_wrapper.sh <jobscript> <cuda>
#
# This runs OUTSIDE any container -- SGE started it directly -- and its whole
# job is to re-enter the pipeline image so the jobscript's snakemake runs with
# starcall-workflow's dependency stack available. That re-entry is not
# optional: starcall's rules are overwhelmingly `run:` blocks (78 `run:` vs 6
# `shell:` across workflow/rules/*.smk on the pinned devel checkout), and a
# `run:` body executes IN-PROCESS inside the child snakemake, importing
# numpy/tifffile/pandas/starcall/constitch/tensorflow. snakemake never
# containerizes a `run:` body, so the child's own interpreter has to be the
# image's `ops` env -- there is no arrangement where the heavy stack lives in
# the container but the rule bodies run outside it.
#
# The jobscript itself is run unmodified. That works only because the
# SUBMITTER also runs inside this same image: snakemake bakes its own
# `sys.executable` into every generated jobscript (ClusterExecutor.
# get_python_executable is `sys.executable if self.assume_shared_fs else
# "python"`, and there is no jobscript-template hook for it -- confirmed
# against a real snakemake 7.32.4 source tree), so that absolute path is
# /opt/conda/envs/ops/bin/python3.10, which resolves identically here inside
# the container. Moving the submitter out of the container -- onto a host-side
# "thin" snakemake env -- would bake a host path in instead and require
# rewriting every jobscript before exec'ing it. Don't do that without also
# adding the rewrite; see docs/nextflow.md.
set -euo pipefail

jobscript="$1"
cuda="${2:-0}"

: "${STARCALL_SIF:?sge_job_wrapper.sh: STARCALL_SIF not exported}"
: "${STARCALL_BINDS:?sge_job_wrapper.sh: STARCALL_BINDS not exported}"

# Snakemake builds its SourceCache -- os.makedirs($XDG_CACHE_HOME/snakemake,
# falling back to $HOME/.cache) -- inside Workflow.__init__, i.e. before it
# parses a single rule and with no CLI flag to relocate it. That applies to
# the CHILD snakemake in this jobscript exactly as it does to the submitter,
# so the same read-only-$HOME failure the module works around
# (params.snakemake_cache_dir; "OSError: [Errno 30] Read-only file system:
# '/net/noble'") is waiting here too, N times over.
#
# sge_submit.sh forwards the submitter's already-redirected HOME/
# XDG_CACHE_HOME via `qsub -v`, but Apptainer/Singularity resets $HOME to the
# invoking user's real home on entry unless told otherwise -- so re-inject
# both through the container boundary. APPTAINERENV_*/SINGULARITYENV_* are the
# same mechanism under the two names: Apptainer honours both, but
# Singularity 3.x-era builds (what `singularity` is a symlink to on some of
# these nodes) honour only the SINGULARITYENV_ spelling, so set both rather
# than guessing which binary is on this node's PATH.
export APPTAINERENV_HOME="$HOME"
export SINGULARITYENV_HOME="$HOME"
export APPTAINERENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"
export SINGULARITYENV_XDG_CACHE_HOME="$XDG_CACHE_HOME"

# --nv only for the rules that asked for a GPU (sge_submit.sh passes the same
# value it used to decide `-l cuda=1`). Unconditionally passing it fails
# outright on a node with no GPU/driver, exactly as `docker run --gpus all`
# does on a GPU-less host -- see nextflow.config's process_gpu comment for
# that same failure on the Nextflow side.
nv=()
if [ "${cuda}" -eq 1 ]; then
    nv=(--nv)
fi

# STARCALL_BINDS is a comma-separated src:dest list built once by the module's
# script block, covering every host path this job can touch: the starcall
# checkout (which holds .snakemake/tmp.*/ where this very jobscript lives),
# the resolved phenotyping/segmentation/sequencing dirs, the snakemake cache
# dir, and -- easy to miss -- the Nextflow TASK WORK DIR, because snakemake
# prefixes every jobscript with `cd <workdir_init>`, the directory the
# submitter was launched from rather than its --directory. A missing bind
# there makes the job die on its very first `cd`.
#
# Paths are bound at their own unchanged locations (src == dest) for the same
# reason nextflow.config's containerOptions closures do it: starcall's rules
# build every output path by literal string concatenation onto
# phenotyping_dir/segmentation_dir/sequencing_dir, so an in-container path
# that merely *reaches* the data isn't enough -- it has to match the host path
# exactly.
exec "${STARCALL_APPTAINER_BIN:-singularity}" exec \
    "${nv[@]}" \
    --bind "${STARCALL_BINDS}" \
    "${STARCALL_SIF}" \
    /bin/sh "${jobscript}"
