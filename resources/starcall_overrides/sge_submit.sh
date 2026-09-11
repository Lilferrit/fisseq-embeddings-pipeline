#!/usr/bin/env bash
# Snakemake --cluster submit command for BUILD_CELL_IMAGES' phase 2.
#
# Runs on the SUBMITTER side -- i.e. inside the Nextflow task's own container,
# on whatever node SGE gave that task -- once per starcall rule job. Snakemake
# invokes it as, literally (executors/__init__.py's `'{submitcmd} "{jobscript}"'`,
# confirmed against a real snakemake 7.32.4 source tree):
#
#   sge_submit.sh <rule> <jobid> <threads> <mem_mb> <cuda> "<jobscript>"
#
# i.e. the five --cluster format placeholders this repo passes (see
# scratch/nextflow.config's ext.snakemake_cluster_args) followed by the
# generated jobscript path, which snakemake always appends LAST. It must print
# the scheduler's job id as the first line of stdout -- hence `qsub -terse`.
#
# This lives in a checked-in script rather than inline in the --cluster string
# because that string has to survive Groovy interpolation, Nextflow's own bash
# heredoc, snakemake's str.format() over {rule}/{threads}/{resources.*}, and
# then `shell=True` -- four quoting layers. Anything non-trivial written inline
# there (as starcall-workflow's own run.sh does, with nested $( ... realpath
# ... ) substitutions) is effectively unreviewable. Keeping it here also means
# the only braces in the --cluster string are real snakemake placeholders;
# a stray literal { or } would make str.format() raise.
set -euo pipefail

rule="$1"
jobid="$2"
threads="$3"
mem_mb="$4"
cuda="$5"
jobscript="$6"

# Exported by the module's script block (modules/local/build_cell_images/main.nf).
# Fail loudly here rather than emitting a malformed qsub line: a qsub that fails
# to submit surfaces as a snakemake WorkflowError naming this script, whereas a
# qsub that submits something wrong surfaces N minutes later as a dead job.
: "${STARCALL_LOG_DIR:?sge_submit.sh: STARCALL_LOG_DIR not exported}"
: "${STARCALL_JOB_WRAPPER:?sge_submit.sh: STARCALL_JOB_WRAPPER not exported}"
: "${STARCALL_JOB_TAG:?sge_submit.sh: STARCALL_JOB_TAG not exported}"
: "${STARCALL_SGE_PROJECT:?sge_submit.sh: STARCALL_SGE_PROJECT not exported}"
: "${STARCALL_SGE_QUEUE:?sge_submit.sh: STARCALL_SGE_QUEUE not exported}"
: "${STARCALL_SGE_RUNTIME:?sge_submit.sh: STARCALL_SGE_RUNTIME not exported}"

# SGE's mfree is a PER-SLOT limit, not a per-job one, so a rule asking for
# mem_mb=64000 across `-pe serial 8` needs mfree=8000M, not 64000M -- request
# the latter and you reserve 8x the memory you meant. starcall-workflow's own
# run.sh does the same division (`expr {resources.mem_mb} / {threads}`); this
# rounds UP instead of truncating, so the total stays >= what the rule asked
# for. The floor keeps a low-memory rule on a 1-slot job from asking for an
# mfree so small the node's own overhead trips it.
mem_per_slot=$(( (mem_mb + threads - 1) / threads ))
if [ "$mem_per_slot" -lt 1024 ]; then
    mem_per_slot=1024
fi

# SGE does not create -o/-e directories and fails the job outright if they're
# missing, so make them here rather than relying on the submitter having done
# it. One subdirectory per rule keeps a few hundred jobs' logs navigable.
#
# Deliberately NOT starcall-workflow run.sh's `logs/{output[0]}.out` naming:
# {output[0]} is an absolute, multi-hundred-character path into the starcall
# tree, which both blows past path length limits and requires the fragile
# inline `$( (test -f ...); mkdir -p ...; realpath ...)` substitution that
# makes that run.sh's --cluster string so hard to follow. snakemake itself
# logs the rule/jobid -> external jobid mapping ("Submitted job N with
# external jobid 'X'"), so <rule>/<jobid> loses no correlation.
log_dir="${STARCALL_LOG_DIR}/${rule}"
mkdir -p "$log_dir"

# Only ask the scheduler for a GPU when the rule actually wants one. Which
# rules those are is decided by --set-resources in ext.snakemake_cluster_args,
# NOT by the rules' own `resources:` blocks -- starcall-workflow's devel branch
# has `cuda = 1` commented out on every segmentation rule (segmentation.smk),
# so relying on the declared value would put every job on a CPU node. See
# that config's own comment.
cuda_arg=()
if [ "${cuda:-0}" -eq 1 ]; then
    cuda_arg=(-l cuda=1)
fi

# -N ties every child job to this one snakemake invocation. It is the handle
# the module's cleanup trap (and the documented manual `qstat | qdel`
# one-liner) uses to find orphans when the submitter dies without running
# snakemake's own --cluster-cancel. SGE truncates long job names and rejects
# some characters, so STARCALL_JOB_TAG is kept short and alnum at the export
# site.
#
# -v (not -V): forward only the variables the wrapper actually reads, so a
# child job doesn't silently inherit the submitter's whole environment --
# which, inside the Nextflow task container, includes container-only PATH
# entries that mean nothing on a bare exec node.
external_jobid="$(
    qsub -terse \
        -N "${STARCALL_JOB_TAG}-${rule}" \
        -P "${STARCALL_SGE_PROJECT}" \
        -q "${STARCALL_SGE_QUEUE}" \
        -pe serial "${threads}" \
        -l "mfree=${mem_per_slot}M" \
        -l "h_rt=${STARCALL_SGE_RUNTIME}" \
        "${cuda_arg[@]}" \
        -o "${log_dir}/${jobid}.out" \
        -e "${log_dir}/${jobid}.err" \
        -v STARCALL_SIF,STARCALL_BINDS,STARCALL_APPTAINER_BIN,XDG_CACHE_HOME,HOME \
        "${STARCALL_JOB_WRAPPER}" "${jobscript}" "${cuda}" | head -1
)"

# Record the id so the module's cleanup trap can qdel this job by ID rather
# than by name. Name matching via `qstat` is the obvious alternative and is
# wrong: plain `qstat` truncates the job-name column, so a
# `${STARCALL_JOB_TAG}-segment_cells_bases` name comes back cut off and a
# prefix match silently misses (or, worse, over-matches another run's jobs).
# Snakemake submits serially from a single process, so appending here needs
# no locking.
if [ -n "${STARCALL_JOBID_FILE:-}" ]; then
    printf '%s\n' "$external_jobid" >> "$STARCALL_JOBID_FILE"
fi

# Must be the first line of stdout -- snakemake parses it as the external job
# id it will then poll for.
printf '%s\n' "$external_jobid"
