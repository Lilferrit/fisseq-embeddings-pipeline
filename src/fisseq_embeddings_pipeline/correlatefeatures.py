"""CORRELATE_FEATURES -- per-dimension agreement between two split halves.

Stage 2c of the reproducibility-filtering chain, adapted from
fisseq-data-pipeline's ``correlatefeatures.py``.

For one bootstrap replicate and one aggregation method, joins the two
AGGREGATE_HALF outputs on the variant label -- so each row of the join
pairs a variant's half-1 aggregate with its half-2 aggregate -- and takes
the Pearson correlation of those two columns **across variants**, one
correlation per embedding dimension.

That is what "reproducible" means here: a dimension is reproducible if
the variant-to-variant pattern it reports from one random half of the
cells is the pattern it reports from the other half. A dimension that
mostly measures noise correlates near zero however large its values are.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import FEATURE_SELECTOR
from .utils.log import setup_logging


def compute_feature_correlations(
    df1: pl.DataFrame, df2: pl.DataFrame, label_col: str
) -> pl.DataFrame:
    """
    Per-dimension Pearson correlation between two half-aggregates.

    Parameters
    ----------
    df1, df2 : pl.DataFrame
        The two halves' aggregates for one method: ``label_col`` plus one
        column per embedding dimension. Same schema.
    label_col : str
        Column the two frames are aligned on (one row per variant).

    Returns
    -------
    pl.DataFrame
        One row per dimension: ``feature``, ``r``, ``r_squared``. ``r`` is
        **null** for a dimension that is constant in either half -- Polars'
        ``corr`` of a zero-variance column is NaN, which is normalized to
        null here on purpose. NaN would propagate through BLOCKLIST's
        median and condemn a dimension outright on the strength of one
        degenerate replicate; null is skipped by ``median``, so the
        dimension is judged on the replicates that actually produced a
        number. A dimension that was degenerate in *every* replicate gets a
        null median, which fails the threshold comparison and is blocked --
        the right verdict, since it carried no variant-to-variant signal at
        all.

    Notes
    -----
    Uses ``FEATURE_SELECTOR`` (exclude ``meta_*``), not
    ``EMBEDDING_SELECTOR``: AGGREGATE_HALF's columns are stat-suffixed
    (``emb_0000_KS``), which ``EMBEDDING_SELECTOR``'s ``^emb_\\d+$`` does
    not match.
    """
    df1 = df1.select(FEATURE_SELECTOR, pl.col(label_col))
    df2 = df2.select(FEATURE_SELECTOR, pl.col(label_col))
    df_joined = df1.join(df2, on=label_col, suffix="_right")

    features = [c for c in df1.columns if c != label_col]
    if not features:
        return pl.DataFrame(
            schema={"feature": pl.String, "r": pl.Float64, "r_squared": pl.Float64}
        )
    corrs = df_joined.select(
        # nan_to_null: a zero-variance column makes pl.corr return NaN, and
        # NaN survives BLOCKLIST's median where null is skipped. See the
        # Returns section above.
        pl.corr(f, f"{f}_right").fill_nan(None).alias(f)
        for f in features
    ).row(0)

    return pl.DataFrame(
        [
            {
                "feature": feature,
                "r": corr,
                "r_squared": None if corr is None else corr**2,
            }
            for feature, corr in zip(features, corrs)
        ],
        schema=["feature", "r", "r_squared"],
    )


@dataclasses.dataclass
class CorrelateFeaturesConfig(AppConfig):
    """
    Hydra structured configuration for CORRELATE_FEATURES.

    Attributes
    ----------
    half1_file : str
        Path to the first half's AGGREGATE_HALF output. Required.
    half2_file : str
        Path to the second half's AGGREGATE_HALF output, same method.
        Required.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    output_name : str
        Basename (without ``.parquet``) of the single output file. The rule
        sets it to the aggregation method, so one experiment's methods can
        share a per-replicate directory. Defaults to ``"correlations"``.
    """

    half1_file: str = MISSING
    half2_file: str = MISSING
    label_column: str = "meta_aa_changes"
    output_name: str = "correlations"


_cs = ConfigStore.instance()
_cs.store(name="correlate_features_main", node=CorrelateFeaturesConfig)


@hydra.main(version_base=None, config_path=None, config_name="correlate_features_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: correlate two aggregated pseudo-replicate halves.

    Output file
    ------------
    - ``{prefix}{output_name}.parquet`` -- ``feature``, ``r``, ``r_squared``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.correlatefeatures \\
            output_dir=./out \\
            half1_file=./half1/KS.parquet \\
            half2_file=./half2/KS.parquet
    """
    corr_cfg: CorrelateFeaturesConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(corr_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corr_cfg.output_dir = str(output_dir)
    setup_logging(corr_cfg, "correlate_features")

    prefix = f"{corr_cfg.output_root}." if corr_cfg.output_root is not None else ""

    logging.info("Reading halves %s / %s", corr_cfg.half1_file, corr_cfg.half2_file)
    df1 = pl.read_parquet(corr_cfg.half1_file)
    df2 = pl.read_parquet(corr_cfg.half2_file)
    corr_df = compute_feature_correlations(df1, df2, corr_cfg.label_column)

    out_path = output_dir / f"{prefix}{corr_cfg.output_name}.parquet"
    logging.info("Writing %s (%d dimension(s))", out_path, corr_df.height)
    corr_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
