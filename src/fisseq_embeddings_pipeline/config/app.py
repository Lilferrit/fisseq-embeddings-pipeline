"""Base Hydra structured config shared by every pipeline entry point.

Defines :class:`AppConfig` -- vendored from fisseq-data-pipeline's
config/app.py with one added field, `random_seed` (SPEC.md §3 decision 11).
See SPEC.md's architecture-decisions section for the full docstring this
was copied from.
"""

import dataclasses
from typing import Optional

from omegaconf import MISSING


@dataclasses.dataclass
class AppConfig:
    """
    Shared application-level configuration.

    output_dir : str
        Directory for outputs produced by the current run. Required.
    output_root : str or None
        If set, every output file is prefixed ``{output_root}.{name}``
        instead of being placed under ``output_dir``. Defaults to ``None``.
    log_level : str
        Logging verbosity. Defaults to ``"info"``.
    random_seed : int
        Shared seed for every stochastic pipeline stage (StratifiedKFold
        shuffling, XGBoost's own `seed` param, calibration's inner split,
        PCA's solver -- see SPEC.md §6.6/§6.7). Defaults to ``0``.
    """

    output_dir: str = MISSING
    output_root: Optional[str] = None
    log_level: str = "info"
    random_seed: int = 0
