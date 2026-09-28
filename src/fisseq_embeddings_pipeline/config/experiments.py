"""Validation and per-experiment field routing for ``params.yaml``.

Ported from ``config/experiments.py``'s Groovy (deleted in the Snakemake
rewrite), which validated ``params.experiments`` inline and routed each
entry's keys to the stage(s) that own them via three disjoint
include/exclude sets. Living here instead of inside a ``.smk`` file makes
that logic ordinary, unit-testable Python -- it never was, in Groovy.

The routing contract, unchanged from the Nextflow version:

- ``BUILD_CELL_IMAGES`` is the only stage that touches starcall-workflow's
  tree, so every starcall-facing key (:data:`CELL_IMAGES_FIELDS`) routes
  to it and to nothing else.
- ``BUILD_DATASET`` and ``BUILD_CP_FEATURES`` each get whatever keys are
  left after excluding the starcall-facing set plus ``batch_stem`` and
  ``cp_features`` (and, for ``BUILD_CP_FEATURES``, ``window``). ``cell_images_dir`` is
  injected by the workflow from ``BUILD_CELL_IMAGES``' own output, never
  set by the user.
- ``window``, ``cellprofiler_pipeline`` and ``cellprofiler_cycle`` each
  have a pipeline-wide default in ``params.yaml``; an entry that doesn't
  set its own value inherits it. An entry's own value always wins.
  ``window`` is ``BUILD_DATASET``'s alone: it's the crop size that stage
  cuts each cell at.
"""

from typing import Any, Dict, List, Mapping

#: Keys routed to ``BUILD_CELL_IMAGES`` only -- the starcall-workflow-facing
#: fields plus the three ``cp_features``-related ones it folds into
#: ``cell_table.parquet``. Mirrors ``cell_images_field_includes`` in the
#: deleted ``config/experiments.py``.
CELL_IMAGES_FIELDS = frozenset(
    {
        "starcall_workflow_dir",
        "phenotyping_dir",
        "segmentation_dir",
        "sequencing_dir",
        "wells",
        "grid_size",
        "segmentation_type",
        "use_corrected",
        "sequencing_reads_params",
        "cp_features",
        "cellprofiler_pipeline",
        "cellprofiler_cycle",
    }
)

#: Keys never passed through to ``BUILD_DATASET``/``BUILD_CP_FEATURES`` as
#: Hydra overrides: ``batch_stem`` is passed explicitly and ``cp_features``
#: is a track selector rather than a stage field.
_NON_STAGE_FIELDS = frozenset({"batch_stem", "cp_features"})

#: Keys only ``BUILD_DATASET`` reads -- ``window`` is the crop size it cuts
#: each cell at, which ``CpFeaturesConfig`` has no field for.
_DATASET_ONLY_FIELDS = frozenset({"window"})

#: Global ``params.yaml`` defaults an ``experiments:`` entry inherits when it
#: doesn't set the key itself, per stage. See the module docstring.
_CELL_IMAGES_FALLBACKS = ("cellprofiler_pipeline", "cellprofiler_cycle")
_DATASET_FALLBACKS = ("window",)


def validate_config(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate a loaded ``params.yaml`` and return its ``experiments`` list.

    Fails fast with a specific message for every required-with-no-default
    param, rather than letting a missing-key error surface deep inside a
    rule. Mirrors ``config/experiments.py``'s own validation block.

    Parameters
    ----------
    config : Mapping[str, Any]
        Snakemake's ``config`` dict (``params.yaml`` plus any ``--config``
        overrides).

    Returns
    -------
    list[dict]
        The validated ``experiments`` list, unchanged.

    Raises
    ------
    ValueError
        If ``pipeline_dir`` or ``cell_dino_checkpoint`` is unset, if
        ``experiments`` is missing/empty/not a list, if any entry is not a
        mapping, if any entry lacks a non-blank string ``batch_stem``, if
        any entry's ``cp_features`` is not a boolean, if two entries
        share a ``batch_stem``, if ``aggregate_methods`` /
        ``aggregate_methods_passthrough`` name an unknown aggregator or
        overlap each other, or if ``reproducibility_bootstrap_reps`` is
        below 2.
    """
    if config.get("pipeline_dir") is None:
        raise ValueError("pipeline_dir is required (--config pipeline_dir=...).")
    if config.get("cell_dino_checkpoint") is None:
        raise ValueError(
            "cell_dino_checkpoint is required (path to a Cell-DINO .pth checkpoint)."
        )

    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError(
            "experiments must be a non-empty list of experiment maps (see params.yaml)."
        )

    for i, entry in enumerate(experiments):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"experiments[{i}] must be a map, got {type(entry).__name__}."
            )
        batch_stem = entry.get("batch_stem")
        if not isinstance(batch_stem, str) or not batch_stem.strip():
            raise ValueError(
                f"experiments[{i}] is missing a required, non-empty 'batch_stem' field."
            )
        if "cp_features" in entry and not isinstance(entry["cp_features"], bool):
            raise ValueError(
                f"experiments[{i}].cp_features must be a boolean (true/false), "
                f"got {type(entry['cp_features']).__name__}."
            )

    stems = [entry["batch_stem"] for entry in experiments]
    duplicates = sorted({s for s in stems if stems.count(s) > 1})
    if duplicates:
        raise ValueError(
            f"experiments has duplicate batch_stem value(s): {', '.join(duplicates)}. "
            "Every experiment's batch_stem must be unique."
        )

    _validate_aggregate_methods(config)
    _validate_reproducibility(config)

    return [dict(entry) for entry in experiments]


def _validate_aggregate_methods(config: Mapping[str, Any]) -> None:
    """
    Check both aggregator lists name real aggregators and stay disjoint.

    Validated here, at parse time, rather than being left to each stage's
    own Hydra config: the method names are interpolated straight into a
    rule's shell command and into its output paths, so a bad entry
    otherwise surfaces as a confusing Snakemake wildcard or a shell-level
    failure long before any Python validation could fire. This is the
    Python counterpart of the ``aggregatorKeys()`` check
    fisseq-data-pipeline does in Groovy.
    """
    from ..aggregate import _AGGREGATORS

    known = sorted(_AGGREGATORS)
    lists = {
        "aggregate_methods": config.get("aggregate_methods") or [],
        "aggregate_methods_cp_features": config.get("aggregate_methods_cp_features")
        or [],
        "aggregate_methods_passthrough": config.get("aggregate_methods_passthrough")
        or [],
    }
    for key, value in lists.items():
        if not isinstance(value, list):
            raise ValueError(
                f"{key} must be a list of aggregator names, got {value!r}."
            )
        unknown = sorted(set(value) - set(known))
        if unknown:
            raise ValueError(
                f"{key} has unrecognized entry/entries: {', '.join(unknown)}. "
                f"Choose from: {', '.join(known)}."
            )
        duplicates = sorted({m for m in value if value.count(m) > 1})
        if duplicates:
            raise ValueError(
                f"{key} has duplicate entry/entries: {', '.join(duplicates)}."
            )

    if not lists["aggregate_methods"]:
        raise ValueError("aggregate_methods must name at least one aggregator.")

    overlap = sorted(
        set(lists["aggregate_methods"]) & set(lists["aggregate_methods_passthrough"])
    )
    if overlap:
        raise ValueError(
            "aggregate_methods_passthrough overlaps aggregate_methods: "
            f"{', '.join(overlap)}. A method is either reproducibility-filtered "
            "or passed through, never both."
        )


def _validate_reproducibility(config: Mapping[str, Any]) -> None:
    """
    Check the reproducibility-filtering knobs.

    ``reproducibility_bootstrap_reps`` must be at least 2: BLOCKLIST takes
    a median across replicates, and a median of one value is that value --
    a single replicate would make the whole verdict hostage to one random
    split.
    """
    reps = config.get("reproducibility_bootstrap_reps")
    if not isinstance(reps, int) or isinstance(reps, bool) or reps < 2:
        raise ValueError(
            "reproducibility_bootstrap_reps must be an integer >= 2 (got "
            f"{reps!r}); BLOCKLIST medians across replicates, so one "
            "replicate is not a reproducibility test."
        )

    min_corr = config.get("reproducibility_min_correlation")
    if not isinstance(min_corr, (int, float)) or isinstance(min_corr, bool):
        raise ValueError(
            f"reproducibility_min_correlation must be a number, got {min_corr!r}."
        )
    if not (-1.0 <= float(min_corr) <= 1.0):
        raise ValueError(
            "reproducibility_min_correlation must be a Pearson r in [-1, 1], got "
            f"{min_corr!r}."
        )

    min_batches = config.get("reproducibility_global_min_batches_ok")
    if min_batches is not None and (
        not isinstance(min_batches, int)
        or isinstance(min_batches, bool)
        or min_batches < 1
    ):
        raise ValueError(
            "reproducibility_global_min_batches_ok must be null or an integer "
            f">= 1, got {min_batches!r}."
        )


def _with_fallbacks(
    overrides: Dict[str, Any], config: Mapping[str, Any], keys: "tuple[str, ...]"
) -> Dict[str, Any]:
    """Fill each of ``keys`` from the global ``config`` default when
    ``overrides`` doesn't already carry it and the default isn't ``None``."""
    for key in keys:
        if key not in overrides and config.get(key) is not None:
            overrides[key] = config[key]
    return overrides


def cell_images_overrides(
    entry: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """
    The ``BUILD_CELL_IMAGES``-bound Hydra overrides for one experiment.

    :data:`CELL_IMAGES_FIELDS` only, with the ``cellprofiler_pipeline``/
    ``cellprofiler_cycle`` global fallbacks applied.
    """
    overrides = {k: v for k, v in entry.items() if k in CELL_IMAGES_FIELDS}
    return _with_fallbacks(overrides, config, _CELL_IMAGES_FALLBACKS)


def dataset_overrides(
    entry: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """
    The ``BUILD_DATASET``-bound Hydra overrides for one experiment.

    Everything the starcall-facing set and :data:`_NON_STAGE_FIELDS` don't
    claim, with the ``window`` global fallback applied. ``cell_images_dir``
    is injected by the rule, not here.
    """
    excluded = CELL_IMAGES_FIELDS | _NON_STAGE_FIELDS
    overrides = {k: v for k, v in entry.items() if k not in excluded}
    return _with_fallbacks(overrides, config, _DATASET_FALLBACKS)


def cp_features_overrides(
    entry: Mapping[str, Any], config: Mapping[str, Any]
) -> Dict[str, Any]:
    """
    The ``BUILD_CP_FEATURES``-bound Hydra overrides for one experiment.

    Same exclusion set as :func:`dataset_overrides` plus ``window`` --
    ``CpFeaturesConfig`` has no ``window`` field (it reads
    ``cell_table.parquet``'s already-materialized CellProfiler columns).
    """
    excluded = CELL_IMAGES_FIELDS | _NON_STAGE_FIELDS | _DATASET_ONLY_FIELDS
    return {k: v for k, v in entry.items() if k not in excluded}


def hydra_overrides(mapping: Mapping[str, Any]) -> str:
    """
    Render a mapping as a space-separated Hydra CLI override string.

    Lists become Hydra's bracket syntax, single-quoted so the shell
    doesn't split or glob them; booleans are lowercased to match YAML/Hydra
    spelling (Python's ``True`` is not a valid Hydra boolean). Scalars are
    passed through bare, matching the Groovy idiom this replaces.

    Examples
    --------
    >>> hydra_overrides({"wells": ["w1", "w2"], "grid_size": 8})
    "'wells=[w1,w2]' grid_size=8"
    """
    parts = []
    for key, value in mapping.items():
        if isinstance(value, (list, tuple)):
            joined = ",".join(str(v) for v in value)
            parts.append(f"'{key}=[{joined}]'")
        elif isinstance(value, bool):
            parts.append(f"{key}={str(value).lower()}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)
