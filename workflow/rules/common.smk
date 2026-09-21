# Shared parse-time setup: validation, the per-experiment plan, container
# binds, and the small helpers every rule file uses.
#
# Everything with real logic lives in importable, unit-tested Python
# (config/experiments.py, config/binds.py) rather than here -- this file is
# the thin Snakemake-facing layer over it.

import hashlib
import os
import sys

from fisseq_embeddings_pipeline.config.binds import apply_container_env
from fisseq_embeddings_pipeline.config.experiments import (
    cell_images_overrides,
    cp_features_overrides,
    dataset_overrides,
    hydra_overrides,
    validate_config,
)

# Fails fast with a specific message for every required-with-no-default key,
# rather than letting a missing-key error surface deep inside a rule -- the
# job the deleted workflows/embeddings.nf did at the top of its workflow{}.
EXPERIMENTS = validate_config(config)
BATCHES = [entry["batch_stem"] for entry in EXPERIMENTS]
CP_BATCHES = [e["batch_stem"] for e in EXPERIMENTS if e.get("cp_features")]
BY_STEM = {entry["batch_stem"]: entry for entry in EXPERIMENTS}

PIPELINE_DIR = os.path.abspath(str(config["pipeline_dir"]))

# Repo-relative paths have to be resolved BEFORE workdir() moves us into
# pipeline_dir (workflow.basedir is workflow/, so the repo root is its
# parent). starcall_overrides_dir defaults to this repo's own copy; the
# apptainer profile points it at the one baked into the image instead.
REPO_ROOT = os.path.dirname(workflow.basedir)
STARCALL_OVERRIDES_DIR = str(
    config.get("starcall_overrides_dir")
    or os.path.join(REPO_ROOT, "resources", "starcall_overrides")
)

# Bind every host path a containerized rule can reach, via $APPTAINER_BIND
# (see config/binds.py). Inert unless --software-deployment-method apptainer
# is actually in play.
apply_container_env(config, EXPERIMENTS)

# The reproducibility-filtering fan-out (cellDINO track only): one split per
# bootstrap replicate, two halves per split, one aggregation method per half.
# See docs/architecture.md and params.yaml's own comments.
REPS = list(range(1, int(config["reproducibility_bootstrap_reps"]) + 1))
HALVES = [1, 2]
AGG_METHODS = list(config["aggregate_methods"])
PASSTHROUGH_METHODS = list(config["aggregate_methods_passthrough"] or [])

# Bare emb_0000 columns only when aggregate_methods is EXACTLY ["median"] --
# aggregate_embeddings' own rule. AGGREGATE_HALF runs one method per job and so
# cannot work this out for itself, but its column names have to match
# aggregate.parquet's exactly or FILTER_AGGREGATE's blocklist finds nothing to
# drop. Hence deciding it here, from the whole list, and passing it in.
BARE_COLUMNS = str(AGG_METHODS == ["median"]).lower()

# Only ever a known batch_stem -- stops a wildcard from matching across a
# '/' and silently inventing a rule match for a path we never meant. Same
# reasoning for method: constrained to the methods this run actually asked
# for, so a stray path can't conjure an AGGREGATE_HALF job for one it didn't.
wildcard_constraints:
    batch="|".join(re.escape(b) for b in BATCHES) if BATCHES else "^$",
    rep=r"\d+",
    half="[12]",
    method=(
        "|".join(re.escape(m) for m in sorted(set(AGG_METHODS) | set(PASSTHROUGH_METHODS)))
        if (AGG_METHODS or PASSTHROUGH_METHODS)
        else "^$"
    ),


# Replaces scratch/nextflow.config's `beforeScript`, which exported these
# from task.cpus for every process. Prefixed onto every rule's shell: body,
# so it now applies to local runs too, not just the cluster profile -- each
# library otherwise defaults to "all cores on the machine" and oversubscribes
# badly when several rules run at once.
THREAD_ENV = """
export POLARS_MAX_THREADS={threads}
export OMP_NUM_THREADS={threads}
export OPENBLAS_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}
export NUMEXPR_NUM_THREADS={threads}
export VECLIB_MAXIMUM_THREADS={threads}
"""


def cell_images_args(batch):
    """BUILD_CELL_IMAGES' Hydra overrides for one experiment."""
    return hydra_overrides(cell_images_overrides(BY_STEM[batch], config))


def dataset_args(batch):
    """BUILD_DATASET's Hydra overrides for one experiment."""
    return hydra_overrides(dataset_overrides(BY_STEM[batch], config))


def cp_features_args(batch):
    """BUILD_CP_FEATURES' Hydra overrides for one experiment."""
    return hydra_overrides(cp_features_overrides(BY_STEM[batch], config))


def hydra_list(key, values):
    """Render an ordered list as one shell-safe Hydra override token."""
    return hydra_overrides({key: list(values)})


def starcall_child_image():
    """The .sif each per-rule cluster job re-enters the image with.

    An explicit `starcall_child_image` always wins. Otherwise: a
    `container_image` that is already a local file is used as-is (Snakemake
    does the same -- it never pulls a local path), and anything else is a
    URI Snakemake will have pulled and converted under its --apptainer-prefix,
    so reconstruct that path.

    Reconstructing rather than asking is unavoidable -- the deployment
    setting isn't readable from inside a rule, hence apptainer_prefix being
    mirrored in config. The naming rule is Snakemake's own
    `md5(url).hexdigest() + ".simg"` (deployment/singularity.py's Image),
    which is an implementation detail, not an API. The risk is bounded: if a
    Snakemake upgrade changes it the path stops existing, and the rule's own
    `-s` guard then fails that one job immediately with the candidate named,
    rather than letting hundreds of child jobs die on the nodes. Set
    starcall_child_image explicitly to opt out of the guesswork.
    """
    explicit = config.get("starcall_child_image")
    if explicit:
        return str(explicit), "config starcall_child_image"

    image = str(config.get("container_image") or "")
    if image and "://" not in image:
        return image, "container_image (already a local file)"

    prefix = config.get("apptainer_prefix")
    if image and prefix:
        digest = hashlib.md5(image.encode(), usedforsecurity=False).hexdigest()
        return os.path.join(str(prefix), f"{digest}.simg"), (
            "derived from container_image + apptainer_prefix"
        )
    return "", "no candidate (set starcall_child_image or apptainer_prefix)"


def starcall_cluster_preamble(batch):
    """The bash the build_cell_images rule emits before its nested snakemake
    when, and only when, a profile has opted into per-rule cluster submission
    by setting snakemake_cluster_args.

    Empty by default, so the command that rule runs locally is exactly what
    it would have been with none of this machinery present.
    """
    if not config.get("snakemake_cluster_args"):
        return ""

    child_image, origin = starcall_child_image()

    # A blank value here means the profile interpolated something that was
    # never set. Left alone it exports the literal string "null" and the
    # failure surfaces much later, as an unintelligible scheduler or
    # bind-mount error on every child job. Caught generically so the repo
    # side stays scheduler-agnostic.
    #
    # The literal strings count as unset, not just None: `--config
    # 'starcall_cluster_env={"SGE_ROOT": null}'` parses the value as the
    # STRING "null" (confirmed against snakemake.cli.parse_config), which is
    # exactly the shape of the mistake this guard exists for -- a YAML null
    # that has already been stringified somewhere upstream.
    cluster_env = config.get("starcall_cluster_env") or {}
    unset = sorted(
        k
        for k, v in cluster_env.items()
        if v is None or str(v).strip().lower() in ("", "null", "none", "~")
    )
    if unset:
        raise ValueError(
            f"starcall_cluster_env has no value for {', '.join(unset)} -- the "
            "profile references something that was never set (check the keys "
            "its config: block interpolates)."
        )
    exports = "\n".join(f"    export {k}='{v}'" for k, v in cluster_env.items())

    cache_dir = config.get("snakemake_cache_dir") or f"{PIPELINE_DIR}/.snakemake_cache"
    starcall_dir = BY_STEM[batch]["starcall_workflow_dir"]
    # Short, alnum-only and unique per batch: it prefixes every child job's
    # SGE name, and SGE truncates long names and rejects some characters.
    # Only ever used for human identification in qstat -- the cleanup trap
    # matches on recorded job IDs, not on this.
    job_tag = "sc" + re.sub(r"[^A-Za-z0-9]", "", batch)[:8]

    return f"""
    # ---- cluster submission preamble (profile opted in) ------------------
    # Every child rule job re-enters this image on its own node, so it needs a
    # real .sif FILE. container_image is a docker:// URI on the cluster, and
    # hundreds of jobs each re-resolving that against one shared cache, with
    # registry credentials, is exactly what starcall_child_image avoids. Fail
    # loudly here rather than letting every child job fail identically later.
    export STARCALL_SIF='{child_image}'
    if [ ! -s "$STARCALL_SIF" ]; then
        echo "build_cell_images: no usable container image for the per-rule cluster jobs." >&2
        echo "  tried ({origin}): '$STARCALL_SIF'" >&2
        echo "  Set starcall_child_image to a pre-built .sif, or make sure apptainer_prefix" >&2
        echo "  mirrors your profile's own apptainer-prefix." >&2
        exit 1
    fi
    export STARCALL_APPTAINER_BIN='{config.get("starcall_apptainer_bin") or "singularity"}'
    # The two helper scripts live on opposite sides of the container boundary
    # and so are addressed by DIFFERENT paths:
    #   - sge_submit.sh is invoked BY snakemake, i.e. inside this job's own
    #     container, so it takes the in-image path.
    #   - sge_job_wrapper.sh is what the SCHEDULER runs, on a bare exec node
    #     with no container at all, so it must be a HOST path on shared
    #     storage -- the in-image /opt/... path does not exist there.
    export STARCALL_SUBMIT_SCRIPT='{STARCALL_OVERRIDES_DIR}/sge_submit.sh'
    export STARCALL_JOB_WRAPPER='{config.get("starcall_host_overrides_dir") or ""}/sge_job_wrapper.sh'
    if [ ! -x "$STARCALL_JOB_WRAPPER" ]; then
        echo "build_cell_images: starcall_host_overrides_dir must point at this repo's" >&2
        echo "  resources/starcall_overrides on SHARED STORAGE the exec nodes can read --" >&2
        echo "  the per-rule job wrapper runs outside any container (got: '$STARCALL_JOB_WRAPPER')" >&2
        exit 1
    fi
    export STARCALL_LOG_DIR='{PIPELINE_DIR}/logs/starcall/{batch}'
    export STARCALL_JOB_TAG='{job_tag}'
    export STARCALL_JOBID_FILE="$PWD/cluster_jobids.txt"
{exports}
    mkdir -p "$STARCALL_LOG_DIR"
    : > "$STARCALL_JOBID_FILE"

    # Every host path a child job can touch, bound at its own unchanged
    # location (src == dest) -- starcall's rules concatenate strings onto
    # phenotyping_dir/segmentation_dir/sequencing_dir, so reaching the data
    # under some other in-container path is not enough. $PWD is NOT optional:
    # snakemake prefixes every generated jobscript with
    # `cd <the directory the submitter was launched from>`.
    STARCALL_BINDS="$(printf '%s\\n' \\
        '{starcall_dir}' \\
        "$phenotyping_dir" "$segmentation_dir" "$sequencing_dir" \\
        '{cache_dir}' "$PWD" \\
        | sort -u | sed 's|.*|&:&|' | paste -sd, -)"
    export STARCALL_BINDS

    # qdel whatever is still queued if this job dies. snakemake's own
    # --cluster-cancel only fires on a graceful shutdown, and SGE's default
    # terminate is SIGKILL -- which this trap does NOT catch either, so this
    # is defence in depth, not a guarantee. The other two layers are the
    # bounded -l h_rt on every child and the documented manual sweep.
    starcall_cancel_children() {{
        if [ -s "$STARCALL_JOBID_FILE" ]; then
            xargs -r qdel < "$STARCALL_JOBID_FILE" >/dev/null 2>&1 || true
        fi
    }}
    trap starcall_cancel_children EXIT INT TERM

    # A SIGKILLed submitter leaves a lock in <starcall_workflow_dir>/.snakemake/
    # locks/ and the NEXT run then dies with "Directory cannot be locked"
    # before doing any work. Safe to clear unconditionally only because each
    # experiment has its own starcall_workflow_dir and concurrent invocations
    # against one tree are already forbidden.
    {config["snakemake_bin"]} \\
        --snakefile "{STARCALL_OVERRIDES_DIR}/wrapper.smk" \\
        --directory "{starcall_dir}" \\
        --unlock \\
        --config phenotyping_dir="$phenotyping_dir/" segmentation_dir="$segmentation_dir/" sequencing_dir="$sequencing_dir/" starcall_workflow_dir="{starcall_dir}" \\
        || true
    # ---- end cluster submission preamble ---------------------------------
"""
