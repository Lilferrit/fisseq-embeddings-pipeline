"""AGGREGATE_HALF / AGGREGATE_PASSTHROUGH -- lean single-method aggregation.

One module, two Snakemake rules, mirroring fisseq-data-pipeline's
``aggregatefeaturetype.py`` (which its Nextflow workflow includes twice
under two aliases, ``AGGREGATE_HALF`` and ``AGGREGATE_FEATURE_TYPE``):

- **AGGREGATE_HALF** (``split_file`` set) -- stage 2b of the
  reproducibility chain. Aggregates one pseudo-replicate half with one
  aggregator, so CORRELATE_FEATURES can correlate the two halves
  dimension by dimension.
- **AGGREGATE_PASSTHROUGH** (``split_file`` unset) -- aggregates *every*
  QC-passed cell with one aggregator, for a method listed in
  ``params.aggregate_methods_passthrough``. Nothing downstream of it but
  FILTER_AGGREGATE's final join: a passthrough method never reaches the
  bootstrap halves, the correlation or the blocklist, which is the whole
  point of the second list.

Output is **lean** -- ``[label_column] + <this method's stat columns>``,
via ``aggregate_embeddings(..., include_metadata=False)``. No normalizer
is fitted or saved, no metadata is joined, no impact score is computed:
those happen once, upstream in AGGREGATE_EMBEDDINGS, and a half's
``meta_num_cells`` would be actively misleading anyway.

Both rules run one aggregator per job rather than all of them at once.
That is the fan-out fisseq-data-pipeline uses, and it matters most for
the reference-based aggregators (KS/AUROC and the two ``*negLogP``
variants), whose peak memory is the reason
:data:`~fisseq_embeddings_pipeline.aggregate.DEFAULT_FEATURE_CHUNK_SIZE`
exists at all.
"""

import dataclasses
import logging
import pathlib
from typing import Optional

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .aggregate import DEFAULT_FEATURE_CHUNK_SIZE, aggregate_embeddings
from .config import AppConfig
from .filter import JOIN_KEYS, load_filtered_embeddings
from .utils.log import setup_logging
from .utils.normalizer import Normalizer
from .utils.splits import filter_by_split_file


@dataclasses.dataclass
class AggregateHalfConfig(AppConfig):
    """
    Hydra structured configuration for AGGREGATE_HALF / AGGREGATE_PASSTHROUGH.

    Extends AppConfig; ``random_seed`` is inherited uniformly but unused
    (every aggregator is deterministic, and which cells are in this half
    was already decided by GENERATE_SPLIT).

    Attributes
    ----------
    embeddings_file : str
        Path to EMBED_CELLS' embeddings.parquet. Required.
    filtered_keys_file : str
        Path to FILTER_EMBEDDINGS' filtered_keys.parquet. Required.
    normalizer_file : str
        Path to FILTER_EMBEDDINGS' normalizer.parquet. Required.
    aggregator : str
        The single aggregation method to run -- a key in
        :data:`~fisseq_embeddings_pipeline.aggregate._AGGREGATORS`.
        Required.
    split_file : str or None
        Path to one GENERATE_SPLIT half (a parquet of
        :data:`~fisseq_embeddings_pipeline.filter.JOIN_KEYS` rows).
        ``None`` aggregates every QC-passed cell, which is what
        AGGREGATE_PASSTHROUGH wants. Defaults to ``None``.
    label_column : str
        Name of the variant label column. Defaults to ``"meta_aa_changes"``.
    feature_chunk_size : int or None
        Embedding dimensions evaluated per Polars query -- a memory dial
        only. Defaults to
        :data:`~fisseq_embeddings_pipeline.aggregate.DEFAULT_FEATURE_CHUNK_SIZE`.
    bare_columns : bool
        Whether an ``aggregator=median`` job strips the ``_median``
        suffix. Set from the run's full ``aggregate_methods``, not from
        this job's aggregator -- see :func:`main`. Defaults to ``False``.
    output_name : str
        Basename (without ``.parquet``) of the single output file.
        Defaults to ``"aggregate"``.
    """

    embeddings_file: str = MISSING
    filtered_keys_file: str = MISSING
    normalizer_file: str = MISSING
    aggregator: str = MISSING
    split_file: Optional[str] = None
    label_column: str = "meta_aa_changes"
    feature_chunk_size: Optional[int] = DEFAULT_FEATURE_CHUNK_SIZE
    bare_columns: bool = False
    output_name: str = "aggregate"


_cs = ConfigStore.instance()
_cs.store(name="aggregate_half_main", node=AggregateHalfConfig)


@hydra.main(version_base=None, config_path=None, config_name="aggregate_half_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: aggregate one method over one cell subset, leanly.

    Reconstructs the QC-passed, synonymous-corrected embedding table via
    :func:`fisseq_embeddings_pipeline.filter.load_filtered_embeddings`,
    restricts it to ``split_file``'s cells (a no-op when unset), and runs
    the single configured aggregator.

    ``bare_columns`` decides whether a ``median`` job writes bare
    ``emb_0000`` or suffixed ``emb_0000_median`` columns. The rule sets it
    from the run's full ``aggregate_methods`` (bare only when that list is
    exactly ``["median"]``), NOT from this job's single aggregator -- the
    blocklist keys features by column name, so a half's column names have
    to match AGGREGATE_EMBEDDINGS' ``aggregate.parquet`` exactly or
    FILTER_AGGREGATE would find nothing to drop.

    Output file
    ------------
    - ``{prefix}{output_name}.parquet``

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.aggregate_half \\
            output_dir=./out \\
            embeddings_file=embeddings.parquet \\
            filtered_keys_file=filtered_keys.parquet \\
            normalizer_file=normalizer.parquet \\
            aggregator=KS \\
            split_file=./splits/rep1/half1.parquet
    """
    half_cfg: AggregateHalfConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(half_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    half_cfg.output_dir = str(output_dir)
    setup_logging(half_cfg, "aggregate_half")

    prefix = f"{half_cfg.output_root}." if half_cfg.output_root is not None else ""

    logging.info("Reading embeddings from %s", half_cfg.embeddings_file)
    embeddings_lf = pl.scan_parquet(half_cfg.embeddings_file)
    logging.info("Reading filtered keys from %s", half_cfg.filtered_keys_file)
    filtered_keys_lf = pl.scan_parquet(half_cfg.filtered_keys_file)
    logging.info("Loading normalizer from %s", half_cfg.normalizer_file)
    normalizer = Normalizer.load(half_cfg.normalizer_file)

    logging.info("Reconstructing QC-passed, synonymous-corrected embeddings")
    filtered_lf = load_filtered_embeddings(embeddings_lf, filtered_keys_lf, normalizer)
    filtered_lf = filter_by_split_file(filtered_lf, half_cfg.split_file, JOIN_KEYS)

    logging.info(
        "Aggregating via %s (feature_chunk_size=%s, lean output)",
        half_cfg.aggregator,
        half_cfg.feature_chunk_size,
    )
    agg_df = aggregate_embeddings(
        filtered_lf,
        half_cfg.label_column,
        [half_cfg.aggregator],
        feature_chunk_size=half_cfg.feature_chunk_size,
        include_metadata=False,
        bare_median=half_cfg.bare_columns,
    )

    out_path = output_dir / f"{prefix}{half_cfg.output_name}.parquet"
    logging.info("Writing %s (%d row(s))", out_path, agg_df.height)
    agg_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
