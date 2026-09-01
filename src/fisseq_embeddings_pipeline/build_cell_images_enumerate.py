"""BUILD_CELL_IMAGES, phase 1: enumerate.

Hydra entry point (`python -m
fisseq_embeddings_pipeline.build_cell_images_enumerate`), backing the first
of BUILD_CELL_IMAGES' three phases (modules/local/build_cell_images.nf).
Resolves each well's tile grid size (explicit override or auto-detected)
and enumerates existing tile directories directly against
starcall-workflow's own `phenotyping_dir` tree, then writes:

- `targets_out`: one Snakemake target file path per line, forcing the
  whole-tile phenotype image, segmentation mask, segmentation cell table,
  and sequencing reads table to exist for every discovered tile (plus the
  CellProfiler CSV, if `cp_features` is set) -- consumed by the Nextflow
  module's own `snakemake ... $(cat targets.txt)` invocation.
- `manifest_out`: a CSV (`well,tile,segmentation_csv,reads_csv,
  cellprofiler_csv,pt_tif,mask_tif`) driving phase 3
  (`build_cell_images_table.py`).
- `symlinks_out`: a TSV (`relative_path<TAB>absolute_path`) of just the two
  per-tile image files, for the Nextflow module's own symlink-collection
  loop (phase 2's tail end).

Until this stage's Docker image merged starcall-workflow's own `ops` conda
env into this repo's main image (see the root `Dockerfile`), this logic
lived in a standalone `modules/local/build_cell_images_glue.py` that
deliberately avoided importing `fisseq_embeddings_pipeline`, because it ran
inside a wholly separate container. That constraint no longer applies --
this module runs like every other stage, via this repo's own installed
package -- only the Snakemake invocation between this phase and
`build_cell_images_table.py` still needs the separate `ops` env, and that's
a plain shell step in `build_cell_images.nf`, not Python.

`resolve_grid_size`/`enumerate_tile_names` are conceptually the same
grid-size-and-tile-discovery problem `dataset.py`'s `discover_tiles`
solves, but operate on a different tree at a different point in the
pipeline (starcall-workflow's raw, not-yet-normalized `phenotyping_dir`,
here, vs. BUILD_CELL_IMAGES' own already-normalized `cell_images_dir`
there) with different return shapes, so they're kept as separate
implementations -- only the tile-directory-naming regex (`TILE_DIR_RE`,
`utils/constants.py`) is shared between the two, since that's the one
piece that must not silently drift.

`resolve_data_dir` resolves `phenotyping_dir`/`segmentation_dir`/
`sequencing_dir` themselves, each independently optional: an explicit
value always wins; otherwise starcall-workflow's own project config
(`{starcall_workflow_dir}/config.yaml`, or `default-config.yaml` if that's
absent -- the exact same file, in the exact same precedence, `workflow/
Snakefile` itself consults, confirmed by reading it directly) is read for
that key, so a project that remaps these paths is still resolved
correctly; only once neither file sets it does this fall back to a
subdirectory of `starcall_workflow_dir` matching starcall-workflow's own
documented default (`phenotyping/`, `segmentation/`, `sequencing/`).
`segmentation_dir` is resolved here too even though this phase's own logic
never reads it (only `build_cell_images.nf`'s own `snakemake` invocation
does) -- see `resolved_dirs_out` below -- so there's exactly one place
that knows how to find these three directories, not two.
"""

import csv
import dataclasses
import glob
import logging
import os
import pathlib
import re
from typing import Any, Dict, List, Optional

import hydra
import yaml
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.constants import TILE_DIR_RE
from .utils.log import setup_logging

# Matches a well's grid directory name (e.g. "well1_grid4") -- used only by
# resolve_grid_size's auto-detection scan.
_GRID_DIR_RE = re.compile(r"_grid(\d+)$")

# starcall-workflow's own default-config.yaml default for each directory
# key, relative to its own working directory -- see resolve_data_dir.
_DEFAULT_SUBDIR = {
    "phenotyping_dir": "phenotyping",
    "segmentation_dir": "segmentation",
    "sequencing_dir": "sequencing",
}

# The two project-config filenames workflow/Snakefile itself consults, in
# the same either/or precedence order (never both -- confirmed by reading
# that file directly: `if os.path.exists('config.yaml'): configfile:
# 'config.yaml' elif os.path.exists('default-config.yaml'): configfile:
# 'default-config.yaml'`).
_PROJECT_CONFIG_FILENAMES = ("config.yaml", "default-config.yaml")

_MANIFEST_FIELDNAMES = [
    "well",
    "tile",
    "segmentation_csv",
    "reads_csv",
    "cellprofiler_csv",
    "pt_tif",
    "mask_tif",
]

_RESOLVED_DIR_KEYS = ("phenotyping_dir", "segmentation_dir", "sequencing_dir")


@dataclasses.dataclass
class BuildCellImagesEnumerateConfig(AppConfig):
    """
    Hydra structured configuration for BUILD_CELL_IMAGES' enumerate phase.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    this stage's own logic doesn't consume random_seed itself, but every
    stage config inherits it uniformly.

    Attributes
    ----------
    starcall_workflow_dir : str
        The Snakemake checkout/working directory this experiment's
        `phenotyping_dir`/`segmentation_dir`/`sequencing_dir` default
        under, and whose own project config (`config.yaml`/
        `default-config.yaml`) is consulted first -- see
        :func:`resolve_data_dir`.
    phenotyping_dir, segmentation_dir, sequencing_dir : str or None
        starcall-workflow's own per-experiment output trees (real,
        unredirected paths -- see build_cell_images.nf's module docstring
        on why). All optional: ``None`` (the default) resolves via
        :func:`resolve_data_dir` against `starcall_workflow_dir`; set one
        explicitly only when that tree isn't colocated under
        `starcall_workflow_dir`.
    wells : list[str]
        Wells to enumerate.
    grid_size : int or None
        Explicit override, or ``None`` to auto-detect per well (see
        :func:`resolve_grid_size`).
    segmentation_type : str
        Defaults to ``"cells"``.
    use_corrected : bool
        Whether to target `corrected_pt.tif` instead of `raw_pt.tif`.
        Defaults to ``False``.
    sequencing_reads_params : str
        Suffix threaded into the reads CSV filename
        (`{segmentation_type}_reads{sequencing_reads_params}.csv`).
        Defaults to ``""``.
    cp_features : bool
        Whether to also target this experiment's CellProfiler CSV.
        Defaults to ``False``.
    cellprofiler_cycle, cellprofiler_pipeline : str
        Threaded into the CellProfiler CSV filename when `cp_features` is
        set. Default to ``""``.
    targets_out, manifest_out, symlinks_out : str
        Output filenames, written under `output_dir`.
    resolved_dirs_out : str
        Output filename (under `output_dir`) for the fully-resolved
        `phenotyping_dir`/`segmentation_dir`/`sequencing_dir` -- a
        shell-sourceable `key='value'` file, one line per key, so
        `build_cell_images.nf`'s own `snakemake` invocation (phase 2) uses
        the exact same resolved paths as this phase, without duplicating
        this module's own resolution logic in Groovy.
    """

    starcall_workflow_dir: str = MISSING
    phenotyping_dir: Optional[str] = None
    segmentation_dir: Optional[str] = None
    sequencing_dir: Optional[str] = None
    wells: List[str] = MISSING
    grid_size: Optional[int] = None
    segmentation_type: str = "cells"
    use_corrected: bool = False
    sequencing_reads_params: str = ""
    cp_features: bool = False
    cellprofiler_cycle: str = ""
    cellprofiler_pipeline: str = ""
    targets_out: str = "targets.txt"
    manifest_out: str = "tiles_manifest.csv"
    symlinks_out: str = "symlinks.txt"
    resolved_dirs_out: str = "resolved_dirs.env"


def resolve_grid_size(
    phenotyping_dir: str, well: str, grid_size: Optional[int]
) -> int:
    """Resolve one well's tile grid size, auto-detecting it when omitted.

    An explicit ``grid_size`` is returned as-is; otherwise this scans
    ``phenotyping_dir`` for a ``{well}_grid<N>`` directory. Exactly one
    distinct grid size is required.

    Raises
    ------
    ValueError
        If auto-detection finds zero or more than one distinct grid size.
    """
    if grid_size is not None:
        return grid_size

    matches: List[str] = []
    sizes = set()
    for candidate in sorted(glob.glob(f"{phenotyping_dir}/{well}_grid*")):
        if not os.path.isdir(candidate):
            continue
        m = _GRID_DIR_RE.search(os.path.basename(candidate))
        if m is not None:
            sizes.add(int(m.group(1)))
            matches.append(candidate)

    if not sizes:
        raise ValueError(
            f"Could not auto-detect grid_size for well {well!r}: no "
            f"'{well}_grid<N>' directory found under {phenotyping_dir!r}. "
            "Fix phenotyping_dir/wells, or set grid_size explicitly."
        )
    if len(sizes) > 1:
        raise ValueError(
            f"Could not auto-detect grid_size for well {well!r}: found "
            f"multiple candidate directories with different grid sizes "
            f"({', '.join(matches)}). Set grid_size explicitly to "
            "disambiguate."
        )
    return sizes.pop()


def resolve_data_dir(
    starcall_workflow_dir: str, dir_key: str, explicit: Optional[str]
) -> str:
    """Resolve one of phenotyping_dir/segmentation_dir/sequencing_dir.

    Precedence, matching `workflow/Snakefile`'s own resolution exactly (so
    this stays correct even when a project's own config.yaml remaps these
    paths -- "handles cases where the output looks different for whatever
    reason", not just the documented-default case):

    1. ``explicit``, if given.
    2. ``dir_key`` read from ``starcall_workflow_dir``'s own project
       config -- ``config.yaml`` if present, else ``default-config.yaml``
       (never both; a project's ``config.yaml`` existing at all stops
       ``default-config.yaml`` from being consulted, exactly like
       `workflow/Snakefile`'s own `if os.path.exists('config.yaml'): ...
       elif os.path.exists('default-config.yaml'): ...`). A relative value
       there (starcall-workflow's own convention, e.g. ``'phenotyping/'``)
       is resolved against ``starcall_workflow_dir`` -- the same directory
       Snakemake itself runs relative to (`--directory`); an absolute
       value is used as-is.
    3. ``{starcall_workflow_dir}/{bare_name}`` (e.g. ``.../phenotyping``),
       matching starcall-workflow's own ``default-config.yaml`` default
       for the common case where a project doesn't override this key at
       all (confirmed by reading `rules/config.smk`'s own
       ``config.get(dir_key, '<bare_name>/')`` -- this fallback is exactly
       that default, resolved against the same working directory).

    Parameters
    ----------
    dir_key : str
        One of ``"phenotyping_dir"``, ``"segmentation_dir"``,
        ``"sequencing_dir"``.
    """
    if explicit:
        return explicit

    bare_name = _DEFAULT_SUBDIR[dir_key]
    for config_filename in _PROJECT_CONFIG_FILENAMES:
        config_path = os.path.join(starcall_workflow_dir, config_filename)
        if not os.path.isfile(config_path):
            continue
        with open(config_path) as f:
            project_config = yaml.safe_load(f) or {}
        value = project_config.get(dir_key)
        if value:
            value = str(value).rstrip("/")
            return value if os.path.isabs(value) else os.path.join(
                starcall_workflow_dir, value
            )
        break  # config.yaml exists (even without dir_key set) -- don't
        # also fall through to default-config.yaml, matching the
        # Snakefile's own either/or (never both) precedence.

    return os.path.join(starcall_workflow_dir, bare_name)


def enumerate_tile_names(phenotyping_dir: str, well: str, grid_size: int) -> List[str]:
    """List tile directory names (e.g. ``tile0x0y``) for one well/grid_size.

    Globs ``{phenotyping_dir}/{well}_grid{grid_size}/tile*x*y`` and keeps
    only directory names matching ``TILE_DIR_RE``, sorted lexically
    (numeric tile ordering is handled by the caller, which iterates
    wells/tiles in a fixed, deterministic order for target-list/manifest
    generation -- exact tile order doesn't affect correctness here, only
    reproducibility of file ordering).
    """
    tiles = []
    pattern = f"{phenotyping_dir}/{well}_grid{grid_size}/tile*x*y"
    for tile_dir in sorted(glob.glob(pattern)):
        name = os.path.basename(tile_dir)
        if TILE_DIR_RE.match(name):
            tiles.append(name)
    return tiles


def build_enumeration(
    phenotyping_dir: str,
    sequencing_dir: str,
    wells: List[str],
    grid_size: Optional[int],
    segmentation_type: str,
    use_corrected: bool,
    sequencing_reads_params: str,
    cp_features: bool,
    cellprofiler_cycle: str,
    cellprofiler_pipeline: str,
) -> Dict[str, Any]:
    """Build the full target list + tile manifest for one experiment.

    Returns
    -------
    dict
        ``{"targets": [...], "manifest_rows": [...], "symlinks": [...]}``
        -- see :func:`main`'s docstring for what each becomes on disk.
        ``symlinks`` entries are ``(relative_path, absolute_path)`` pairs
        for just the two per-tile image files (not the CSVs -- those are
        read directly by ``build_cell_images_table.py``, never re-exposed
        as files of their own; see ``build_cell_images.nf``'s Phase 3
        comment).
    """
    pt_filename = "corrected_pt.tif" if use_corrected else "raw_pt.tif"
    targets: List[str] = []
    manifest_rows: List[Dict[str, str]] = []
    symlinks: List[tuple] = []

    for well in wells:
        resolved_grid_size = resolve_grid_size(phenotyping_dir, well, grid_size)
        for tile in enumerate_tile_names(phenotyping_dir, well, resolved_grid_size):
            grid_dir = f"{well}_grid{resolved_grid_size}"
            tile_dir = f"{phenotyping_dir}/{grid_dir}/{tile}"
            seq_tile_dir = f"{sequencing_dir}/{grid_dir}/{tile}"

            pt_path = f"{tile_dir}/{pt_filename}"
            mask_path = f"{tile_dir}/{segmentation_type}_mask.tif"
            seg_csv = f"{tile_dir}/{segmentation_type}.csv"
            reads_csv = (
                f"{seq_tile_dir}/{segmentation_type}_reads"
                f"{sequencing_reads_params}.csv"
            )

            targets.extend([pt_path, mask_path, seg_csv, reads_csv])
            symlinks.append((f"{grid_dir}/{tile}/{pt_filename}", pt_path))
            symlinks.append(
                (f"{grid_dir}/{tile}/{segmentation_type}_mask.tif", mask_path)
            )

            cp_csv = ""
            if cp_features:
                cp_csv = (
                    f"{tile_dir}/cellprofiler{cellprofiler_cycle}_"
                    f"{cellprofiler_pipeline}.csv"
                )
                targets.append(cp_csv)

            manifest_rows.append(
                {
                    "well": well,
                    "tile": tile,
                    "segmentation_csv": seg_csv,
                    "reads_csv": reads_csv,
                    "cellprofiler_csv": cp_csv,
                    "pt_tif": pt_path,
                    "mask_tif": mask_path,
                }
            )

    return {"targets": targets, "manifest_rows": manifest_rows, "symlinks": symlinks}


_cs = ConfigStore.instance()
_cs.store(name="build_cell_images_enumerate_main", node=BuildCellImagesEnumerateConfig)


@hydra.main(
    version_base=None,
    config_path=None,
    config_name="build_cell_images_enumerate_main",
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: resolve grid sizes, enumerate tiles, write
    targets/manifest/symlinks.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.build_cell_images_enumerate \\
            output_dir=./out \\
            starcall_workflow_dir=/data/experiment1 \\
            'wells=[well1,well2]' \\
            segmentation_type=cells
    """
    enum_cfg: BuildCellImagesEnumerateConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(enum_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    enum_cfg.output_dir = str(output_dir)
    setup_logging(enum_cfg, "build_cell_images_enumerate")

    resolved_dirs = {
        dir_key: resolve_data_dir(
            enum_cfg.starcall_workflow_dir, dir_key, getattr(enum_cfg, dir_key)
        )
        for dir_key in _RESOLVED_DIR_KEYS
    }
    resolved_dirs_path = output_dir / enum_cfg.resolved_dirs_out
    with open(resolved_dirs_path, "w") as f:
        for dir_key in _RESOLVED_DIR_KEYS:
            f.write(f"{dir_key}='{resolved_dirs[dir_key]}'\n")
    logging.info("Resolved starcall-workflow directories: %s", resolved_dirs)

    result = build_enumeration(
        phenotyping_dir=resolved_dirs["phenotyping_dir"],
        sequencing_dir=resolved_dirs["sequencing_dir"],
        wells=enum_cfg.wells,
        grid_size=enum_cfg.grid_size,
        segmentation_type=enum_cfg.segmentation_type,
        use_corrected=enum_cfg.use_corrected,
        sequencing_reads_params=enum_cfg.sequencing_reads_params,
        cp_features=enum_cfg.cp_features,
        cellprofiler_cycle=enum_cfg.cellprofiler_cycle,
        cellprofiler_pipeline=enum_cfg.cellprofiler_pipeline,
    )

    targets_path = output_dir / enum_cfg.targets_out
    with open(targets_path, "w") as f:
        for target in result["targets"]:
            f.write(target + "\n")

    manifest_path = output_dir / enum_cfg.manifest_out
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(result["manifest_rows"])

    symlinks_path = output_dir / enum_cfg.symlinks_out
    with open(symlinks_path, "w") as f:
        for rel_path, abs_path in result["symlinks"]:
            f.write(f"{rel_path}\t{abs_path}\n")

    logging.info(
        "Enumerated %d tile(s) across %d well(s); wrote %d Snakemake "
        "target(s) to %s",
        len(result["manifest_rows"]),
        len(enum_cfg.wells),
        len(result["targets"]),
        targets_path,
    )


if __name__ == "__main__":
    main()
