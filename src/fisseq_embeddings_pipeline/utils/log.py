"""Shared logging setup for every Hydra entry point.

Vendored unchanged from fisseq-data-pipeline's
src/fisseq_data_pipeline/utils/log.py (SPEC.md §3 decision 2). Defines
:func:`setup_logging`, which configures a console handler plus a per-run
log file under ``output_dir`` (optionally prefixed by ``output_root``),
called at the start of every entry point's ``main()``.
"""

import logging
import pathlib

from ..config import AppConfig


def setup_logging(cfg: AppConfig, name: str) -> None:
    """
    Configure logging for the pipeline.

    A timestamped log file and a console stream are set up simultaneously.
    The log file is named ``{name}.{cfg.output_root}.log``. Its location follows
    the same convention used for other output files:

    Parameters
    ----------
    cfg : AppConfig
        Application configuration supplying ``output_dir`` and optionally
        ``output_root``.
    name : str
        Base name for the log file, typically the calling module or command
        (e.g. ``"normalize"``).
    """
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    if cfg.output_root is not None:
        name = f"{cfg.output_root}.{name}"

    name = f"{name}.log"
    log_path = pathlib.Path(cfg.output_dir) / name
    log_level = log_levels.get(cfg.log_level, logging.INFO)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="w")]

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
