"""EMBED_CELLS -- SPEC.md §6.3 (Epic 3).

Hydra entry point (`python -m fisseq_embeddings_pipeline.embed`), backing
the Nextflow process EMBED_CELLS (modules/local/embed_cells.nf, the
pipeline's only GPU-bound stage). Streams every cell in a BUILD_DATASET
WebDataset through a pretrained Cell-DINO checkpoint (Meta's dinov2,
bag-of-channels mode) and writes one row per cell to embeddings.parquet.
Not gated by QC_FILTER -- matches the diagram, and the whole point of
building the WebDataset up front (SPEC.md §6.1): this GPU pass runs once
per experiment regardless of how many times QC thresholds get retuned.

This module (config + dataloader only so far -- the model wrapper and
Hydra entry point land in later Story 3.3/3.4 commits) also corrects one
thing found while implementing `load_embedding_dataloader()`: SPEC.md's
dataclass docstring for `shard_pattern` claims `webdataset.WebDataset`
accepts a bare glob (`"dataset-*.tar"`) or a brace pattern
(`"dataset-{000000..000042}.tar"`) interchangeably. Verified against the
installed `webdataset==1.0.2`: only brace patterns are expanded internally
(`webdataset.shardlists.expand_urls`); a bare `*` glob is passed through as
a literal, unmatched filename. `load_embedding_dataloader()` expands a
non-brace `shard_pattern` via `glob.glob()` itself before handing shard
paths to `wds.WebDataset`.
"""

import dataclasses
import glob

import torch
import webdataset as wds
from omegaconf import MISSING

from .config import AppConfig


@dataclasses.dataclass
class EmbedCellsConfig(AppConfig):
    """
    Hydra structured configuration for EMBED_CELLS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed --
    SPEC.md §3 decision 11); EMBED_CELLS's own logic doesn't consume
    random_seed itself (inference is deterministic given a fixed
    checkpoint), but every stage config inherits it uniformly.

    Attributes
    ----------
    shard_pattern : str
        Path/brace pattern for this experiment's BUILD_DATASET shards, e.g.
        ``"dataset-{000000..000042}.tar"``. A bare glob (``"dataset-*.tar"``)
        also works -- ``load_embedding_dataloader`` expands it itself (see
        the module docstring; real ``webdataset`` doesn't do this for you).
    checkpoint_path : str
        Path to the Cell-DINO teacher checkpoint (``.pth``).
    arch : str
        Backbone architecture, one of ``vit_small``/``vit_base``/
        ``vit_large``/``vit_giant2``. Defaults to ``"vit_large"``
        (embed dim 1024).
    patch_size : int
        ViT patch size. Defaults to ``16``.
    crop_size : int
        Expected crop size -- must match BUILD_DATASET's ``window``.
        Defaults to ``224``.
    channel_pool : str
        How per-channel CLS embeddings are pooled into one per-cell
        embedding: ``"mean"`` or ``"max"``. Defaults to ``"mean"``.
    mask_mode : str
        ``"zero_background"`` zeroes every pixel outside the target cell
        (using ``mask.npy``) before embedding; ``"none"`` passes crops
        through untouched. Defaults to ``"none"``.
    device : str
        torch device string. Defaults to ``"cuda"``.
    batch_size : int
        Cells per dataloader batch. Defaults to ``256``.
    num_workers : int
        ``webdataset``/``DataLoader`` worker processes. Defaults to ``4``.
    """

    shard_pattern: str = MISSING
    checkpoint_path: str = MISSING
    arch: str = "vit_large"
    patch_size: int = 16
    crop_size: int = 224
    channel_pool: str = "mean"
    mask_mode: str = "none"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 4


def load_embedding_dataloader(cfg: EmbedCellsConfig) -> "torch.utils.data.DataLoader":
    """Stream ``(key, crop, mask, meta)`` batches from BUILD_DATASET's shards.

    ``webdataset.WebDataset(...).decode().to_tuple(...)`` is the standard
    reader side of the shards ``write_dataset_shards()`` (dataset.py, Epic
    1) writes; batching via ``.batched()``/``DataLoader(batch_size=None)``
    keeps shard-order batches (webdataset's usual pattern) rather than a
    random-access ``Dataset``, which a tar-shard format doesn't support
    efficiently. ``mask.npy`` is always fetched -- whether it's applied is
    decided in :func:`embed_batch` (Story 3.3) via ``cfg.mask_mode``, not
    here.

    A non-brace ``shard_pattern`` (no ``{``) is expanded via ``glob.glob``
    first -- see the module docstring for why: ``webdataset``'s own URL
    expansion only understands brace patterns, not bare globs.

    Parameters
    ----------
    cfg : EmbedCellsConfig
        Supplies ``shard_pattern``, ``batch_size``, ``num_workers``.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(keys, crops, masks, metas)`` per batch: ``keys`` is a
        ``list[str]``, ``crops``/``masks`` are stacked ``torch.Tensor``
        (``(B, C, H, W)`` ``uint16`` / ``(B, H, W)`` ``uint8``), ``metas``
        is a ``list[dict]``.

    Raises
    ------
    ValueError
        If ``shard_pattern`` has no brace expression and its glob expansion
        matches no files.
    """
    if "{" in cfg.shard_pattern:
        urls = cfg.shard_pattern
    else:
        urls = sorted(glob.glob(cfg.shard_pattern))
        if not urls:
            raise ValueError(
                f"shard_pattern={cfg.shard_pattern!r} matched no files "
                "(not a brace pattern, and glob.glob found nothing)"
            )

    dataset = (
        wds.WebDataset(urls, shardshuffle=False)
        .decode()
        .to_tuple("__key__", "crop.npy", "mask.npy", "meta.json")
        .batched(cfg.batch_size)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=cfg.num_workers
    )
