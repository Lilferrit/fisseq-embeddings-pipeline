"""QC_FILTER -- SPEC.md §6.2 (Epic 2).

Vendored close to verbatim from fisseq-data-pipeline's
src/fisseq_data_pipeline/qcfilter.py -- same edit-distance / barcode-count /
variant-barcode-count filters, same n_variants variant-selection cap, same
QcFilterConfig fields (bc_threshold=10, variant_bc_threshold=4,
edit_distance_threshold=1 defaults), with the
downsample_amounts/downsample_classes/downsample_seed pseudo-variant
machinery dropped (SPEC.md §6.2's Resolved note). Hydra entry point
(`python -m fisseq_embeddings_pipeline.qcfilter`), backing the Nextflow
process QC_FILTER (modules/local/qc_filter.nf). Reads BUILD_DATASET's
metadata.parquet (Epic 1) as `cell_files` instead of the raw CSV the
upstream source expects -- see the two deviations below.

Two deviations from the upstream source, beyond the dropped downsample
machinery (see IMPLEMENTATION_CHECKLIST.md Epic 2 for the full writeup):

1. `barcode_col_name`/`aa_changes_col_name`/`edit_distance_col_name` default
   to `"meta_barcode"`/`"meta_aa_changes"`/`"meta_edit_distance"` instead of
   the upstream raw-CSV names (`"upBarcode"`/`"aaChanges"`/`"editDistance"`),
   since `metadata.parquet` already writes those columns under their
   canonical `meta_*` names (see dataset.py's write_dataset_shards) -- this
   pipeline's only real `cell_files` input is never the raw, unrenamed cell
   table. `filter_columns`'s rename-then-select logic is otherwise
   unaffected: renaming a column to its own existing name is a harmless
   no-op in Polars.
2. `select_variants`'s `mode="random"` seed comes from `cfg.random_seed`
   (inherited from AppConfig, SPEC.md §3 decision 11) rather than the
   dropped `downsample_seed` field.

`read_file` also gains an explicit `else: raise ValueError(...)` for an
unrecognized file suffix -- the upstream `if`/`elif` has no `else`, so an
unrecognized suffix raises an opaque `UnboundLocalError` deep inside a
later `.collect()` call.
"""

import dataclasses
import logging
import pathlib
from os import PathLike
from typing import Any, Iterable, List, Optional, Tuple

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import (
    META_BARCODE_COL,
    META_EDIT_DISTANCE_COL,
    META_VARIANT_TAG_COL,
)
from .utils.log import setup_logging
from .utils.variant import classify_variant

VARIANT_DOWNSAMPLE_CLASSES = ("Single Missense",)
VARIANT_DOWNSAMPLE_MODES = ("top", "random")


@dataclasses.dataclass
class QcFilterConfig(AppConfig):
    """
    Hydra structured configuration for QC_FILTER.

    Extends AppConfig (output_dir, output_root, log_level, random_seed --
    SPEC.md §3 decision 11); `select_variants`'s "random" mode is the only
    place this stage consumes random_seed.

    Attributes
    ----------
    cell_files : Any
        Path or list of paths to cell data files (CSV or Parquet) -- in
        practice, BUILD_DATASET's metadata.parquet (Epic 1).
    bc_threshold : int
        Minimum number of cells required for a barcode to pass QC.
        Defaults to ``10``.
    variant_bc_threshold : int
        Minimum number of unique barcodes required for a variant to pass QC.
        Defaults to ``4``.
    edit_distance_threshold : int
        Maximum edit distance allowed for a cell to pass QC. Defaults to
        ``1``.
    barcode_col_name : str
        Name of the barcode column in the input data. Defaults to
        ``"meta_barcode"`` -- metadata.parquet's own column name (see the
        module docstring's Deviation 1).
    aa_changes_col_name : str
        Name of the amino-acid changes column in the input data. Defaults
        to ``"meta_aa_changes"``.
    edit_distance_col_name : str
        Name of the edit distance column in the input data. Defaults to
        ``"meta_edit_distance"``.
    label_column : str
        Name of the output label column after renaming. Defaults to
        ``"meta_aa_changes"``.
    n_variants : Optional[int]
        If set, restricts variants whose classified label is in
        ``variant_downsample_classes`` to at most this many distinct
        variants (see :func:`select_variants`); every other class passes
        through untouched. Runs before QC thresholding. Defaults to
        ``None`` (disabled).
    variant_downsample_classes : List[str]
        Classes (from :func:`fisseq_embeddings_pipeline.utils.variant.classify_variant`)
        eligible for the ``n_variants`` restriction. Defaults to
        ``["Single Missense"]``.
    variant_downsample_mode : str
        ``"top"`` keeps the ``n_variants`` variants with the highest cell
        count (ties broken alphabetically); ``"random"`` keeps a seeded
        random sample of ``n_variants`` variants (seeded by
        ``AppConfig.random_seed``). Defaults to ``"top"``.
    variant_allow_list_file : Optional[str]
        Optional path to a Parquet file with a ``label_column`` column of
        variants that bypass the ``n_variants`` cap entirely and aren't
        counted against it (see :func:`select_variants`). Entries not
        present in the data are silently ignored. Meaningless if
        ``n_variants`` is unset -- in that case it is ignored with a
        warning, since the two fields are set independently. Defaults to
        ``None`` (disabled).
    """

    cell_files: Any = MISSING
    bc_threshold: int = 10
    variant_bc_threshold: int = 4
    edit_distance_threshold: int = 1
    barcode_col_name: str = "meta_barcode"
    aa_changes_col_name: str = "meta_aa_changes"
    edit_distance_col_name: str = "meta_edit_distance"
    label_column: str = "meta_aa_changes"
    n_variants: Optional[int] = None
    variant_downsample_classes: List[str] = dataclasses.field(
        default_factory=lambda: list(VARIANT_DOWNSAMPLE_CLASSES)
    )
    variant_downsample_mode: str = "top"
    variant_allow_list_file: Optional[str] = None


# --- read_file / combine_cell_files (Story 2.1) ---


def read_file(cell_file_path: pathlib.Path) -> pl.LazyFrame:
    """
    Read a single cell file into a lazy frame.

    Supports CSV and Parquet formats (extensions ``.csv``, ``.parquet``,
    ``.parq``, ``.pq``). Adds two metadata columns: ``meta_source_file``
    (file path as a string) and ``meta_source_file_idx`` (row index within
    the source file).

    Parameters
    ----------
    cell_file_path : pathlib.Path
        Path to the cell data file.

    Returns
    -------
    pl.LazyFrame
        Lazy frame of the file contents with metadata columns.

    Raises
    ------
    ValueError
        If ``cell_file_path``'s suffix isn't a recognized CSV/Parquet
        extension.
    """
    logging.info("Scanning file %s", cell_file_path)

    if cell_file_path.suffix == ".csv":
        logging.warning(
            "Lazy evaluation is much slower for CSV files."
            " Consider converting to Parquet for better performance."
        )
        lf = pl.scan_csv(cell_file_path)
    elif cell_file_path.suffix in [".parquet", ".parq", ".pq"]:
        lf = pl.scan_parquet(cell_file_path)
    else:
        raise ValueError(
            f"Unrecognized cell_files suffix {cell_file_path.suffix!r} for "
            f"{cell_file_path} -- expected one of .csv, .parquet, .parq, .pq"
        )

    lf = lf.with_columns(
        pl.lit(str(cell_file_path)).alias("meta_source_file"),
        pl.row_index().alias("meta_source_file_idx"),
    )

    return lf


def combine_cell_files(cell_files: Iterable[PathLike]) -> pl.LazyFrame:
    """
    Read and concatenate multiple cell files into one lazy frame.

    Parameters
    ----------
    cell_files : Iterable[PathLike]
        Iterable of paths to cell data files.

    Returns
    -------
    pl.LazyFrame
        Concatenated lazy frame of all input files.
    """
    return pl.concat([read_file(pathlib.Path(cell_file)) for cell_file in cell_files])


# --- filter_columns (Story 2.1) ---


def filter_columns(lf: pl.LazyFrame, cfg: QcFilterConfig) -> pl.LazyFrame:
    """
    Rename input columns to canonical meta names and retain only QC-relevant columns.

    Renames ``cfg.barcode_col_name`` -> ``META_BARCODE_COL`` and
    ``cfg.edit_distance_col_name`` -> ``META_EDIT_DISTANCE_COL``.
    ``cfg.aa_changes_col_name`` is split on the first ``":"`` into a base
    label (``cfg.label_column``) and an optional tag (``META_VARIANT_TAG_COL``,
    ``null`` when no tag is present). Then drops all columns except meta
    columns (``meta_`` prefix) and CellProfiler feature columns (starting
    with an uppercase letter and containing ``_``) -- the latter are a
    no-op against metadata.parquet, which never has such columns, but are
    kept for parity with cell_files inputs that do (e.g. a raw upstream
    cell table).

    Parameters
    ----------
    lf : pl.LazyFrame
        Lazy frame containing all raw input columns.
    cfg : QcFilterConfig
        Supplies column name mappings.

    Returns
    -------
    pl.LazyFrame
        Lazy frame retaining only the necessary columns with canonical names.
    """
    aa_changes_split = pl.col(cfg.aa_changes_col_name).str.splitn(":", 2)
    lf = lf.with_columns(
        aa_changes_split.struct.field("field_0").alias(cfg.label_column),
        aa_changes_split.struct.field("field_1").alias(META_VARIANT_TAG_COL),
        pl.col(cfg.edit_distance_col_name).alias(META_EDIT_DISTANCE_COL),
        pl.col(cfg.barcode_col_name).alias(META_BARCODE_COL),
    )

    schema_names = lf.collect_schema().names()
    cell_profiler_columns = [
        col for col in schema_names if len(col) > 0 and col[0].isupper() and "_" in col
    ]
    meta_columns = [col for col in schema_names if col.startswith("meta_")]

    return lf.select(pl.col(meta_columns + cell_profiler_columns))


# --- get_barcode_counts / get_barcodes_per_variant / add_qc_queries (Story 2.1) ---


def get_barcode_counts(lf: pl.LazyFrame, cfg: QcFilterConfig) -> pl.LazyFrame:
    """
    Count cells per barcode and flag barcodes meeting the threshold.

    Groups by ``META_BARCODE_COL``, counts occurrences, and adds a
    ``barcode_ok`` column (non-null when count >= ``cfg.bc_threshold``).
    Retains the first ``cfg.label_column`` per barcode group.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level lazy frame containing ``META_BARCODE_COL`` and
        ``cfg.label_column`` (as produced by :func:`filter_columns`).
    cfg : QcFilterConfig
        Supplies ``bc_threshold`` and ``label_column``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with one row per barcode, including ``count``,
        ``cfg.label_column``, and ``barcode_ok``.
    """
    return (
        lf.group_by(META_BARCODE_COL)
        .agg(
            [
                pl.len().alias("count"),
                pl.col(cfg.label_column).first(),
            ]
        )
        .with_columns(
            pl.when(pl.col("count") >= cfg.bc_threshold)
            .then(pl.col("count"))
            .otherwise(None)
            .alias("barcode_ok")
        )
    )


def get_barcodes_per_variant(
    cells_lf: pl.LazyFrame, cfg: QcFilterConfig
) -> pl.LazyFrame:
    """
    Count distinct barcodes per variant and flag variants meeting threshold.

    Groups by ``cfg.label_column``, counts barcodes, and adds a
    ``variant_barcode_count_ok`` column (non-null when barcode count
    >= ``cfg.variant_bc_threshold``).

    Parameters
    ----------
    cells_lf : pl.LazyFrame
        Cell-level lazy frame containing ``META_BARCODE_COL`` and
        ``cfg.label_column`` (as produced by :func:`filter_columns`).
    cfg : QcFilterConfig
        Supplies ``variant_bc_threshold`` and ``label_column``.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with one row per variant, including ``barcode_count``
        and ``variant_barcode_count_ok``.
    """
    return (
        cells_lf.group_by(cfg.label_column)
        .agg(
            [
                pl.col(META_BARCODE_COL).n_unique().alias("barcode_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("barcode_count") >= cfg.variant_bc_threshold)
            .then(pl.col("barcode_count"))
            .otherwise(None)
            .alias("variant_barcode_count_ok")
        )
    )


def add_qc_queries(
    lf: pl.LazyFrame, cfg: QcFilterConfig
) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """
    Apply edit-distance, barcode-level, and variant-level QC filters.

    Filters are applied sequentially:

    1. Retain rows with ``META_EDIT_DISTANCE_COL`` <= ``cfg.edit_distance_threshold``.
    2. Remove barcodes below ``cfg.bc_threshold`` cell count.
    3. Remove variants below ``cfg.variant_bc_threshold`` barcode count.

    Parameters
    ----------
    lf : pl.LazyFrame
        Cell-level lazy frame to filter.
    cfg : QcFilterConfig
        Supplies QC thresholds and column names.

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]
        ``(filtered_lf, barcode_count_lf, variants_per_barcode_lf)`` where
        the latter two contain the intermediate QC summary frames.
    """
    logging.info("Adding edit distance QC query")
    lf = lf.filter(pl.col(META_EDIT_DISTANCE_COL) <= cfg.edit_distance_threshold)

    logging.info("Adding Barcode Level QC query")
    barcode_count_lf = get_barcode_counts(lf, cfg)
    lf = lf.join(
        barcode_count_lf.filter(pl.col("barcode_ok").is_not_null()).select(
            META_BARCODE_COL
        ),
        on=META_BARCODE_COL,
        how="inner",
    )

    logging.info("Adding Variant Level QC query")
    variants_per_barcode_lf = get_barcodes_per_variant(lf, cfg)
    lf = lf.join(
        variants_per_barcode_lf.filter(
            pl.col("variant_barcode_count_ok").is_not_null()
        ).select(cfg.label_column),
        on=cfg.label_column,
        how="inner",
    )

    return lf, barcode_count_lf, variants_per_barcode_lf


# --- select_variants (Story 2.1) ---


def select_variants(
    lf: pl.LazyFrame,
    cfg: QcFilterConfig,
    variant_downsample_classes: Tuple[str, ...],
    n_variants: int,
    mode: str,
    seed: int,
    variant_allow_list_file: Optional[str] = None,
) -> pl.LazyFrame:
    """
    Restrict rows whose classified ``cfg.label_column`` value is in
    `variant_downsample_classes` to at most `n_variants` distinct variants.
    Rows in any other class are left untouched (not filtered at all).

    `mode` is one of:

    - ``"top"``: keep the `n_variants` variants (within
      `variant_downsample_classes`) with the highest cell count, ties broken
      alphabetically ascending on ``cfg.label_column``. Fully deterministic.
    - ``"random"``: keep a seeded-random sample of `n_variants` distinct
      variants from the eligible pool. Deterministic given `seed`.

    If `variant_allow_list_file` is set, it must point to a Parquet file
    with a ``cfg.label_column`` column. The eligible pool is partitioned
    before the `mode` logic runs: rows whose ``cfg.label_column`` value
    appears in that file pass through unconditionally and are *not* counted
    against `n_variants`; the remaining eligible rows go through the
    existing top/random selection, capped at `n_variants`. Allow-list
    entries absent from the data are silently ignored (a left-semi join
    handles this naturally). If every eligible variant is allow-listed,
    `n_variants` has no effect for this run and a warning is logged.

    Runs upstream of :func:`add_qc_queries`, so ``barcode_counts``/
    ``variants_per_barcode`` reflect the post-selection population.

    Raises
    ------
    ValueError
        If `mode` isn't one of ``VARIANT_DOWNSAMPLE_MODES``.
    """
    if mode not in VARIANT_DOWNSAMPLE_MODES:
        raise ValueError(
            f"variant_downsample_mode must be one of {VARIANT_DOWNSAMPLE_MODES}, "
            f"got {mode!r}"
        )

    logging.info(
        "Restricting classes %s to %d variant(s) (mode=%s, seed=%d)",
        variant_downsample_classes,
        n_variants,
        mode,
        seed,
    )
    classified = lf.with_columns(
        pl.col(cfg.label_column)
        .map_elements(classify_variant, return_dtype=pl.String)
        .alias("_variant_class")
    )
    non_eligible = classified.filter(
        ~pl.col("_variant_class").is_in(variant_downsample_classes)
    )
    eligible = classified.filter(
        pl.col("_variant_class").is_in(variant_downsample_classes)
    )

    if variant_allow_list_file is not None:
        allow_list_lf = (
            pl.read_parquet(variant_allow_list_file)
            .select(cfg.label_column)
            .unique()
            .lazy()
        )
        allow_listed = eligible.join(allow_list_lf, on=cfg.label_column, how="semi")
        selection_pool = eligible.join(allow_list_lf, on=cfg.label_column, how="anti")

        n_eligible_variants = (
            eligible.select(pl.col(cfg.label_column).n_unique()).collect().item()
        )
        n_pool_variants = (
            selection_pool.select(pl.col(cfg.label_column).n_unique()).collect().item()
        )
        if n_eligible_variants > 0 and n_pool_variants == 0:
            logging.warning(
                "All %d eligible variant(s) matched variant_allow_list_file; "
                "n_variants=%d has no effect for this run",
                n_eligible_variants,
                n_variants,
            )
    else:
        allow_listed = None
        selection_pool = eligible

    counts = selection_pool.group_by(cfg.label_column).agg(pl.len().alias("_n_cells"))
    if mode == "top":
        selected = (
            counts.sort(["_n_cells", cfg.label_column], descending=[True, False])
            .head(n_variants)
            .select(cfg.label_column)
        )
    else:
        selected = (
            counts.with_columns(pl.col(cfg.label_column).hash(seed=seed).alias("_rand"))
            .sort("_rand")
            .head(n_variants)
            .select(cfg.label_column)
        )

    selection_kept = selection_pool.join(selected, on=cfg.label_column, how="inner")

    parts = [non_eligible]
    if allow_listed is not None:
        parts.append(allow_listed)
    parts.append(selection_kept)

    return pl.concat(parts, how="vertical_relaxed").drop("_variant_class")


_cs = ConfigStore.instance()
_cs.store(name="qc_filter_main", node=QcFilterConfig)


@hydra.main(version_base=None, config_path=None, config_name="qc_filter_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: QC filtering of cell-level data files.

    Steps
    -----
    1. Read and concatenate all ``cell_files`` via :func:`combine_cell_files`.
    2. Retain only QC-relevant columns via :func:`filter_columns`.
    3. If ``n_variants`` is set, restrict ``variant_downsample_classes`` to
       at most ``n_variants`` distinct variants via :func:`select_variants`
       (``"top"`` or ``"random"`` mode); otherwise skip this step entirely.
       If ``variant_allow_list_file`` is also set, variants it lists bypass
       the ``n_variants`` cap entirely and aren't counted against it.
    4. Apply edit-distance, barcode-count, and variant-count filters via
       :func:`add_qc_queries`.
    5. Write three output Parquet files to ``output_dir``.

    Output files
    ------------
    - ``{prefix}filtered_cells.parquet``
    - ``{prefix}barcode_counts.parquet``
    - ``{prefix}variants_per_barcode.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.qcfilter \\
            output_dir=./out \\
            'cell_files=[data/metadata.parquet]' \\
            bc_threshold=10 \\
            random_seed=0
    """
    qc_cfg: QcFilterConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(qc_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_cfg.output_dir = str(output_dir)
    setup_logging(qc_cfg, "qc_filter")

    prefix = f"{qc_cfg.output_root}." if qc_cfg.output_root is not None else ""

    cell_files = (
        [qc_cfg.cell_files]
        if isinstance(qc_cfg.cell_files, str)
        else list(qc_cfg.cell_files)
    )

    combined_lf = filter_columns(combine_cell_files(cell_files), qc_cfg)

    if qc_cfg.n_variants is not None:
        combined_lf = select_variants(
            combined_lf,
            qc_cfg,
            variant_downsample_classes=tuple(qc_cfg.variant_downsample_classes),
            n_variants=qc_cfg.n_variants,
            mode=qc_cfg.variant_downsample_mode,
            seed=qc_cfg.random_seed,
            variant_allow_list_file=qc_cfg.variant_allow_list_file,
        )
    else:
        logging.info("n_variants not set; skipping variant-level selection")
        if qc_cfg.variant_allow_list_file is not None:
            logging.warning(
                "variant_allow_list_file is set but n_variants is None; there "
                "is no n_variants cap to bypass, so the allow-list is ignored"
            )

    combined_lf, barcode_count_lf, variants_per_barcode_lf = add_qc_queries(
        combined_lf, qc_cfg
    )

    logging.info("Writing output files to %s", output_dir)
    for name, lf in [
        ("filtered_cells", combined_lf),
        ("barcode_counts", barcode_count_lf),
        ("variants_per_barcode", variants_per_barcode_lf),
    ]:
        logging.info("Writing %s", name)
        lf.sink_parquet(output_dir / f"{prefix}{name}.parquet")

    logging.info("Done")


if __name__ == "__main__":
    main()
