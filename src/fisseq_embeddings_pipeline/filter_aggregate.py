"""FILTER_AGGREGATE -- aggregates -> filtered aggregates.

Stage 4 of the reproducibility-filtering chain, and the step the whole
chain exists to feed: it turns AGGREGATE_EMBEDDINGS'
``aggregate.parquet`` into the ``filtered_aggregate.parquet`` that
GLOBAL_VARIANT_EMBEDDINGS' cross-experiment median + PCA runs on.

Adapted from the blocklist-dropping and passthrough-joining halves of
fisseq-data-pipeline's ``featureselect.py`` (its pycytominer
variance/correlation filtering is not ported -- the reproducibility
verdict is the only selection this pipeline applies).

Two outputs, and the split between them is load-bearing:

- ``filtered_aggregate.parquet`` -- the blocklist applied, nothing else
  added. This is PCA's input.
- ``aggregate_with_passthrough.parquet`` -- the same, plus the
  passthrough aggregates joined on last. Terminal: nothing in this
  pipeline reads it.

fisseq-data-pipeline keeps passthrough columns out of selection and PCA
simply by joining them after those steps, within one process. That is not
enough here, because GLOBAL_VARIANT_EMBEDDINGS is a *separate* stage that
re-reads the file from disk and picks its feature columns with
``FEATURE_SELECTOR`` (exclude ``meta_*``) -- which happily matches a
stat-suffixed ``emb_0000_KSnegLogP``. Hence two files rather than one: a
passthrough column that never enters ``filtered_aggregate.parquet``
cannot leak into the PCA no matter what a future consumer does with the
selector.

The point of the passthrough list itself is unchanged from the sibling
repo: it is for statistics you want reported per variant but that must
not influence which dimensions are kept -- the ``KSnegLogP`` /
``AUROCnegLogP`` p-values above all. A median-correlation reproducibility
threshold is not a meaningful test for a p-value column, and blocking a
dimension's p-value while keeping its statistic (or vice versa) would
make the output incoherent.
"""

import dataclasses
import glob
import logging
import pathlib
from typing import Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.log import setup_logging


def blocked_features(blocklist_df: pl.DataFrame) -> set[str]:
    """
    The set of column names a blocklist marks as not reproducible.

    Parameters
    ----------
    blocklist_df : pl.DataFrame
        COMBINE_BLOCKLISTS output: ``feature``, ``median_r``,
        ``feature_ok``.

    Returns
    -------
    set of str
        Every ``feature`` whose ``feature_ok`` is false. A null
        ``feature_ok`` counts as blocked (``compute_blocklist`` already
        fills those, so this is belt and braces).
    """
    return set(
        blocklist_df.filter(~pl.col("feature_ok").fill_null(False))["feature"].to_list()
    )


def apply_blocklist(agg_df: pl.DataFrame, blocklist_df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop every blocked feature column from an aggregate table.

    Columns named in the blocklist but absent from ``agg_df`` are ignored
    (a method may have been removed from ``aggregate_methods`` since the
    blocklist was built); columns present but *unmentioned* are kept, so
    a partial blocklist can never silently delete data.
    """
    blocked = blocked_features(blocklist_df)
    to_drop = [c for c in agg_df.columns if c in blocked]
    logging.info(
        "Dropping %d non-reproducible column(s) of %d blocklisted; %d column(s) remain",
        len(to_drop),
        len(blocked),
        agg_df.width - len(to_drop),
    )
    return agg_df.drop(to_drop)


def join_passthrough(
    filtered_df: pl.DataFrame, pattern: Optional[str], label_column: str
) -> pl.DataFrame:
    """
    Left-join every passthrough aggregate onto the filtered table.

    Parameters
    ----------
    filtered_df : pl.DataFrame
        :func:`apply_blocklist`'s output.
    pattern : str or None
        Glob matching the AGGREGATE_PASSTHROUGH outputs. ``None``, or a
        pattern matching nothing, returns ``filtered_df`` unchanged --
        an empty ``aggregate_methods_passthrough`` is the default, so
        this is a warning, not the ``ValueError`` an empty blocklist glob
        would raise.
    label_column : str
        Join key.

    Returns
    -------
    pl.DataFrame
        A LEFT join, so a variant missing from a passthrough aggregate
        surfaces as nulls rather than vanishing from the output.

    Raises
    ------
    ValueError
        If a passthrough file carries a non-label column that already
        exists in ``filtered_df`` -- that would mean a method is in both
        ``aggregate_methods`` and ``aggregate_methods_passthrough``, which
        ``validate_config`` rejects up front, so reaching this is a bug.
    """
    if not pattern:
        return filtered_df

    paths = sorted(glob.glob(pattern))
    if not paths:
        logging.warning(
            "No files matched passthrough glob pattern %r; no passthrough "
            "columns will be joined",
            pattern,
        )
        return filtered_df

    logging.info("Joining passthrough aggregates from %d file(s)", len(paths))
    result = filtered_df
    for path in paths:
        pt_df = pl.read_parquet(path)
        collisions = (set(pt_df.columns) & set(result.columns)) - {label_column}
        if collisions:
            raise ValueError(
                f"Passthrough aggregate {path!r} collides with existing "
                f"column(s) {sorted(collisions)}. A method is either selected "
                "on or passed through, never both."
            )
        result = result.join(pt_df, on=label_column, how="left")
    return result


@dataclasses.dataclass
class FilterAggregateConfig(AppConfig):
    """
    Hydra structured configuration for FILTER_AGGREGATE.

    Attributes
    ----------
    aggregate_file : str
        Path to AGGREGATE_EMBEDDINGS' aggregate.parquet. Required.
    blocklist_file : str
        Path to COMBINE_BLOCKLISTS' blocklist.parquet. Required.
    passthrough_files : str or None
        Glob matching this experiment's AGGREGATE_PASSTHROUGH outputs.
        ``None`` (the default) or an empty match means no passthrough
        columns -- the default, since ``aggregate_methods_passthrough``
        defaults to ``[]``.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    """

    aggregate_file: str = MISSING
    blocklist_file: str = MISSING
    passthrough_files: Optional[str] = None
    label_column: str = "meta_aa_changes"


_cs = ConfigStore.instance()
_cs.store(name="filter_aggregate_main", node=FilterAggregateConfig)


@hydra.main(version_base=None, config_path=None, config_name="filter_aggregate_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: apply the blocklist, then attach the passthroughs.

    Output files
    ------------
    - ``{prefix}filtered_aggregate.parquet`` -- reproducible columns only.
      GLOBAL_VARIANT_EMBEDDINGS' input.
    - ``{prefix}aggregate_with_passthrough.parquet`` -- the same plus the
      passthrough columns. Terminal; nothing in-pipeline reads it. Do not
      assume ``FEATURE_SELECTOR`` over THIS file yields selected features.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.filter_aggregate \\
            output_dir=./out \\
            aggregate_file=aggregate.parquet \\
            blocklist_file=blocklist.parquet \\
            'passthrough_files=./passthrough_aggregates/*.parquet'
    """
    fa_cfg: FilterAggregateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(fa_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fa_cfg.output_dir = str(output_dir)
    setup_logging(fa_cfg, "filter_aggregate")

    prefix = f"{fa_cfg.output_root}." if fa_cfg.output_root is not None else ""

    logging.info("Reading aggregate from %s", fa_cfg.aggregate_file)
    agg_df = pl.read_parquet(fa_cfg.aggregate_file)
    logging.info("Reading blocklist from %s", fa_cfg.blocklist_file)
    blocklist_df = pl.read_parquet(fa_cfg.blocklist_file)

    filtered_df = apply_blocklist(agg_df, blocklist_df)
    filtered_path = output_dir / f"{prefix}filtered_aggregate.parquet"
    logging.info("Writing %s", filtered_path)
    filtered_df.write_parquet(filtered_path)

    with_pt_df = join_passthrough(
        filtered_df, fa_cfg.passthrough_files, fa_cfg.label_column
    )
    with_pt_path = output_dir / f"{prefix}aggregate_with_passthrough.parquet"
    logging.info("Writing %s (%d column(s))", with_pt_path, with_pt_df.width)
    with_pt_df.write_parquet(with_pt_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
