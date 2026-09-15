"""Host paths every containerized rule must be able to see, and the
environment that makes Apptainer bind them.

Replaces the three per-task ``containerOptions`` closures the deleted
``nextflow.config`` carried. Nextflow bind-mounted only a task's own work
directory, so each closure had to enumerate, per task, whatever host paths
that one process would reach; Snakemake has no per-rule equivalent, only a
single global ``--apptainer-args``.

It doesn't need one. Every path those closures computed is derivable from
``params.yaml`` before the run starts, so this module computes the whole
set once and exports it through ``$APPTAINER_BIND``/``$SINGULARITY_BIND``,
which Apptainer honours for every ``apptainer exec`` Snakemake spawns --
exactly equivalent to passing ``-B src:dest`` on each one.

Every path is bound at its own, unchanged absolute location (``src:src``).
That is mandatory, not tidiness: ``wrapper.smk`` and starcall-workflow's own
rules build every output path by literal string concatenation onto
``phenotyping_dir``/``segmentation_dir``/``sequencing_dir``, so reaching the
data under some other in-container path is not enough.

One footgun to know about, documented next to ``extra_bind_paths`` in
``params.yaml``: Apptainer's own ``--bind`` flag *replaces*
``$APPTAINER_BIND`` rather than merging with it, so site-specific storage
roots belong in ``extra_bind_paths``, never in ``--apptainer-args``.
"""

import os
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence

from ..build_cell_images_enumerate import resolve_data_dir

#: The three starcall-workflow data trees, each resolved per experiment.
_DATA_DIR_KEYS = ("phenotyping_dir", "segmentation_dir", "sequencing_dir")


def container_bind_paths(
    config: Mapping[str, Any], experiments: Sequence[Mapping[str, Any]]
) -> List[str]:
    """
    Every host path a containerized rule can reach, deduplicated and sorted.

    Parameters
    ----------
    config : Mapping[str, Any]
        The loaded ``params.yaml`` (plus any ``--config`` overrides).
    experiments : Sequence[Mapping[str, Any]]
        The validated ``experiments`` list.

    Returns
    -------
    list[str]
        Absolute host paths. ``pipeline_dir`` is in here where it never was
        under Nextflow: outputs now land there directly, rather than being
        published host-side after the fact, so every rule needs it visible.

    Notes
    -----
    The three data dirs are resolved with
    :func:`~fisseq_embeddings_pipeline.build_cell_images_enumerate.resolve_data_dir`
    -- the same pure, already-unit-tested function that writes
    ``resolved_dirs.env`` during ``build_cell_images``' phase 1. Calling it
    here means the bind set is correct before any rule has run, which is
    what lets this replace the old closure that had to read
    ``resolved_dirs.env`` back off disk mid-run.
    """
    paths: List[str] = []

    checkpoint = config.get("cell_dino_checkpoint")
    if checkpoint:
        paths.append(str(checkpoint))

    pipeline_dir = config.get("pipeline_dir")
    if pipeline_dir:
        paths.append(str(pipeline_dir))
        paths.append(
            str(config.get("snakemake_cache_dir") or f"{pipeline_dir}/.snakemake_cache")
        )

    for entry in experiments:
        starcall_dir = entry.get("starcall_workflow_dir")
        if not starcall_dir:
            continue
        paths.append(str(starcall_dir))
        for key in _DATA_DIR_KEYS:
            paths.append(resolve_data_dir(str(starcall_dir), key, entry.get(key)))

    paths.extend(str(p) for p in config.get("extra_bind_paths") or [])

    return sorted({os.path.abspath(p) for p in paths if p})


def container_env(
    config: Mapping[str, Any], experiments: Sequence[Mapping[str, Any]]
) -> Dict[str, str]:
    """
    The Apptainer environment variables for this run.

    ``APPTAINER_BIND``/``SINGULARITY_BIND`` carry :func:`container_bind_paths`
    as a comma-separated ``src:dest`` list; ``APPTAINER_NV``/``SINGULARITY_NV``
    request the GPU when either GPU-capable stage asks for one
    (``cell_dino_device`` for EMBED_CELLS, ``starcall_gpu`` for
    BUILD_CELL_IMAGES' nested segmentation rules).

    Both spellings are set because Apptainer is frequently installed behind a
    ``singularity`` symlink, and each binary reads only its own prefix.

    The GPU flag is necessarily global here, where ``nextflow.config`` gated
    it per process. That is safe in a way the Docker version was not:
    ``apptainer exec --nv`` on a GPU-less host warns and proceeds, whereas
    ``docker run --gpus all`` failed outright before the container's
    entrypoint ran -- which is the only reason the per-process gating existed.
    """
    env = {}
    binds = container_bind_paths(config, experiments)
    if binds:
        bind_arg = ",".join(f"{p}:{p}" for p in binds)
        env["APPTAINER_BIND"] = bind_arg
        env["SINGULARITY_BIND"] = bind_arg

    wants_gpu = config.get("cell_dino_device") != "cpu" or bool(
        config.get("starcall_gpu")
    )
    if wants_gpu:
        env["APPTAINER_NV"] = "1"
        env["SINGULARITY_NV"] = "1"

    return env


def apply_container_env(
    config: Mapping[str, Any],
    experiments: Sequence[Mapping[str, Any]],
    environ: MutableMapping[str, str] = os.environ,
) -> Dict[str, str]:
    """
    Export :func:`container_env` into ``environ`` and return what was set.

    Called once from ``workflow/Snakefile`` at parse time. A no-op in effect
    when no container engine runs -- the variables are simply never read.
    """
    env = container_env(config, experiments)
    environ.update(env)
    return env
