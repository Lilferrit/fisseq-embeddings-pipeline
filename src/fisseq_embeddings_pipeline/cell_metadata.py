"""BUILD_CELL_METADATA.

Projects BUILD_CELL_IMAGES' ``cell_table.parquet`` down to the seven
``meta_*`` columns QC_FILTER reads, as one per-experiment
``metadata.parquet`` -- no images, no feature columns, no
starcall-workflow tree access.

This stage exists to make QC_FILTER the pipeline's shared fan-out point
instead of BUILD_DATASET. Before it, ``qcfilter.py``'s ``cell_files``
input was BUILD_DATASET's own ``metadata.parquet``, written inside that
stage's WebDataset shard-writing loop (``dataset.py``'s
``write_dataset_shards``) -- which made the expensive, image-reading
cellDINO dataset build a hard dependency of the CellProfiler track,
whose ``FILTER_CP_FEATURES`` consumes the very same QC output. Both
tracks now hang off QC independently, and QC itself depends only on the
cell table. It also means QC thresholds can be retuned (and the whole QC
report regenerated) without touching the shard build at all.

Structurally the analogue of ``fisseq-data-pipeline``'s own ``INPUT``
stage (``src/fisseq_data_pipeline/input.py``, ``modules/local/input.nf``):
the cheap read-and-normalize step that turns whatever the upstream
produced into the one per-batch parquet QC_FILTER consumes.

QC_FILTER can't simply read ``cell_table.parquet`` itself:
``qcfilter.py``'s ``filter_columns`` does rename the barcode/edit-
distance/amino-acid-changes columns to their canonical ``meta_*`` names,
but its closing ``select`` keeps only ``meta_``-prefixed (and
CellProfiler-looking) columns -- so the cell table's unprefixed ``well``/
``tile``/``tile_cell_index`` would be dropped, leaving ``filter.py``'s
``JOIN_KEYS`` with nothing to join on downstream. The projection this
stage applies is shared with ``cp_features.py`` via
``utils/cell_table.py`` so those key columns can't drift between the two.

Note this stage sees every row of ``cell_table.parquet``, whereas
BUILD_DATASET's ``metadata.parquet`` only ever held cells that made it
into a shard (``dataset.py`` skips empty tiles and needs each tile's crop
stacks to be readable). ``filtered_cells.parquet`` can therefore cover
strictly more cells than it used to; every downstream consumer joins it
back on ``JOIN_KEYS`` with an inner join, so the extra rows drop out
where they don't apply.
"""

import dataclasses
import logging
import pathlib

import hydra
import polars as pl
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.cell_table import CELL_METADATA_SCHEMA, cell_metadata_exprs
from .utils.log import setup_logging


@dataclasses.dataclass
class CellMetadataConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CELL_METADATA.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    this stage's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    cell_table : str
        Path to BUILD_CELL_IMAGES' ``cell_table.parquet``. Unlike
        ``CpFeaturesConfig.cell_images_dir``, this is the file itself,
        not the directory holding it: the Nextflow module stages it as a
        real ``path`` input (modules/local/build_cell_metadata/main.nf),
        which is what keeps this stage out of the container-visibility
        problem the other cell_images_dir consumers need bind mounts for.
    batch_stem : str
        This experiment's identifier, written into every row as
        ``meta_batch`` -- one run covers exactly one experiment, matching
        BUILD_DATASET's and BUILD_CP_FEATURES' convention.
    barcode_col_name : str
        Name of the barcode column in cell_table.parquet. Defaults to
        ``"upBarcode"`` -- same default as ``BuildDatasetConfig`` and
        ``CpFeaturesConfig``, since all three read the same table.
    aa_changes_col_name : str
        Name of the amino-acid changes column in cell_table.parquet.
        Defaults to ``"aaChanges"``.
    edit_distance_col_name : str
        Name of the edit distance column in cell_table.parquet. Defaults
        to ``"editDistance"``.
    """

    cell_table: str = MISSING
    batch_stem: str = MISSING
    barcode_col_name: str = "upBarcode"
    aa_changes_col_name: str = "aaChanges"
    edit_distance_col_name: str = "editDistance"


def build_cell_metadata(cfg: CellMetadataConfig) -> pl.DataFrame:
    """
    Project one experiment's cell table down to its ``meta_*`` columns.

    Parameters
    ----------
    cfg : CellMetadataConfig
        Supplies ``cell_table``, ``batch_stem``, and the three column-name
        overrides.

    Returns
    -------
    pl.DataFrame
        One row per cell, with exactly ``CELL_METADATA_SCHEMA``'s columns:
        ``meta_batch``, ``meta_well``, ``meta_tile``, ``meta_cell_index``,
        ``meta_barcode``, ``meta_aa_changes``, ``meta_edit_distance``.
        An empty frame carrying that schema if the cell table has no rows
        (nothing to infer column types from otherwise).
    """
    cell_table_path = pathlib.Path(cfg.cell_table)
    table = pl.read_parquet(cell_table_path)

    if table.height == 0:
        logging.info("cell table at %s has no rows", cell_table_path)
        return pl.DataFrame(schema=CELL_METADATA_SCHEMA)

    return table.select(
        *cell_metadata_exprs(
            cfg.batch_stem,
            cfg.barcode_col_name,
            cfg.aa_changes_col_name,
            cfg.edit_distance_col_name,
        )
    )


_cs = ConfigStore.instance()
_cs.store(name="cell_metadata_main", node=CellMetadataConfig)


@hydra.main(version_base=None, config_path=None, config_name="cell_metadata_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: build one experiment's QC metadata table.

    Steps
    -----
    1. Create ``output_dir``.
    2. Read ``cell_table`` and project it via :func:`build_cell_metadata`.
    3. Write ``{prefix}metadata.parquet`` to ``output_dir``.

    Output file
    ------------
    - ``{prefix}metadata.parquet``

    where ``prefix`` is ``{output_root}.`` when ``output_root`` is set,
    otherwise empty.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.cell_metadata \\
            output_dir=./out \\
            cell_table=/pipeline/cell_images/experiment1/cell_table.parquet \\
            batch_stem=experiment1 \\
            random_seed=0
    """
    meta_cfg: CellMetadataConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(meta_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_cfg.output_dir = str(output_dir)
    setup_logging(meta_cfg, "cell_metadata")

    logging.info(
        "Building cell metadata for batch %s from %s",
        meta_cfg.batch_stem,
        meta_cfg.cell_table,
    )
    metadata_df = build_cell_metadata(meta_cfg)

    prefix = f"{meta_cfg.output_root}." if meta_cfg.output_root is not None else ""
    out_path = output_dir / f"{prefix}metadata.parquet"
    logging.info("Writing %s (%d cells)", out_path, metadata_df.height)
    metadata_df.write_parquet(out_path)

    logging.info("Done")


if __name__ == "__main__":
    main()
