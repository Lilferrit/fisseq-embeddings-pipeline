"""GENERATE_SPLIT -- one bootstrap pseudo-replicate 50/50 cell split.

Stage 2a of the reproducibility-filtering chain (GENERATE_SPLIT ->
AGGREGATE_HALF -> CORRELATE_FEATURES -> BLOCKLIST -> COMBINE_BLOCKLISTS ->
FILTER_AGGREGATE), adapted from fisseq-data-pipeline's
``generatesplit.py``.

"Replicate" here is a *pseudo*-replicate: there is only one cell
population per experiment, so reproducibility is measured by splitting it
in half, aggregating each half independently, and correlating the two
per-variant aggregates dimension by dimension. Splitting is stratified on
the variant label so both halves see every variant in proportion.

Reads FILTER_EMBEDDINGS' ``filtered_keys.parquet`` and nothing else. That
file already carries the composite cell key, ``meta_is_control`` and the
label column (see filter.py's ``filter_and_fit_normalizer``), which is
everything a split needs -- there is no reason to touch the much larger
``embeddings.parquet`` just to decide which cells go where.

Per-replicate seeding follows this repo's one-seed rule (AGENTS.md: no
stage-local seed field): the Snakemake rule passes
``random_seed={config[random_seed]}`` and ``bootstrap_idx={rep}``, and the
split is drawn at ``random_seed + bootstrap_idx`` so each replicate gets a
distinct but fully reproducible split off the one pipeline-wide seed.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
import sklearn.model_selection
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .filter import JOIN_KEYS
from .utils.log import setup_logging
from .utils.splits import write_split


@dataclasses.dataclass
class GenerateSplitConfig(AppConfig):
    """
    Hydra structured configuration for GENERATE_SPLIT.

    Attributes
    ----------
    filtered_keys_file : str
        Path to FILTER_EMBEDDINGS' filtered_keys.parquet. Required.
    label_column : str
        Name of the variant label column, stratified on. Defaults to
        ``"meta_aa_changes"``.
    bootstrap_idx : int
        Which bootstrap replicate this split is for. Offsets
        ``random_seed`` (see the module docstring) -- it is NOT a
        stage-local seed field. Defaults to ``1``.
    """

    filtered_keys_file: str = MISSING
    label_column: str = "meta_aa_changes"
    bootstrap_idx: int = 1


_cs = ConfigStore.instance()
_cs.store(name="generate_split_main", node=GenerateSplitConfig)


@hydra.main(version_base=None, config_path=None, config_name="generate_split_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: generate one pseudo-replicate 50/50 split.

    Output files
    ------------
    - ``{prefix}half1.parquet``
    - ``{prefix}half2.parquet``

    each a parquet of :data:`~fisseq_embeddings_pipeline.filter.JOIN_KEYS`
    rows naming that half's cells, and nothing else.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.generatesplit \\
            output_dir=./out \\
            filtered_keys_file=filtered_keys.parquet \\
            bootstrap_idx=3 \\
            random_seed=0

    Raises
    ------
    ValueError
        If any variant label has only one QC-passed cell -- stratified
        splitting cannot place it in both halves, and silently dropping it
        from one would bias that half's aggregate.
    """
    split_cfg: GenerateSplitConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(split_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_cfg.output_dir = str(output_dir)
    setup_logging(split_cfg, "generate_split")

    prefix = f"{split_cfg.output_root}." if split_cfg.output_root is not None else ""
    seed = split_cfg.random_seed + split_cfg.bootstrap_idx

    logging.info("Reading filtered keys from %s", split_cfg.filtered_keys_file)
    keys_df = pl.read_parquet(split_cfg.filtered_keys_file).select(
        JOIN_KEYS + [split_cfg.label_column]
    )

    singletons = (
        keys_df.group_by(split_cfg.label_column)
        .len()
        .filter(pl.col("len") < 2)[split_cfg.label_column]
        .to_list()
    )
    if singletons:
        raise ValueError(
            f"Cannot stratify a 50/50 split: {len(singletons)} label(s) have "
            f"fewer than 2 QC-passed cells, e.g. {sorted(singletons)[:5]}. "
            "Raise QC_FILTER's barcode/variant thresholds so every retained "
            "variant has at least two cells."
        )

    labels = keys_df[split_cfg.label_column].to_list()
    idx = list(range(keys_df.height))
    half1_idx, half2_idx = sklearn.model_selection.train_test_split(
        idx, stratify=labels, random_state=seed, test_size=0.5
    )

    logging.info(
        "Bootstrap replicate %d (seed %d): %d + %d cells",
        split_cfg.bootstrap_idx,
        seed,
        len(half1_idx),
        len(half2_idx),
    )
    key_only = keys_df.select(JOIN_KEYS)
    for name, rows in (("half1", half1_idx), ("half2", half2_idx)):
        out_path = output_dir / f"{prefix}{name}.parquet"
        logging.info("Writing %s", out_path)
        write_split(key_only[sorted(rows)], out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
