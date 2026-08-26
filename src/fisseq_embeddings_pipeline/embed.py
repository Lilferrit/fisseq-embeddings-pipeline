"""EMBED_CELLS -- SPEC.md §6.3 (Epic 3).

Hydra entry point (`python -m fisseq_embeddings_pipeline.embed`), backing
the Nextflow process EMBED_CELLS (modules/local/embed_cells.nf, the
pipeline's only GPU-bound stage). Streams every cell in a BUILD_DATASET
WebDataset through a pretrained Cell-DINO checkpoint (Meta's dinov2,
bag-of-channels mode) and writes one row per cell to embeddings.parquet.
Not gated by QC_FILTER -- matches the diagram, and the whole point of
building the WebDataset up front (SPEC.md §6.1): this GPU pass runs once
per experiment regardless of how many times QC thresholds get retuned.

SPEC.md §6.3's `load_cell_dino()`/`embed_batch()` sketch was an admitted
best guess pending real `dinov2` source review (§10 item 1). Story 3.1
(IMPLEMENTATION_CHECKLIST.md, see docs/architecture.md for the full
writeup) verified/corrected it against the real `facebookresearch/dinov2`
source:

1. Model construction goes through the architecture factory function
   directly (`vision_transformer.vit_large(...)`, dict-dispatched by
   `cfg.arch`) rather than `dinov2.eval.setup`/`build_model_from_cfg`,
   which needs a full training-style cfg this pipeline doesn't have.
2. Checkpoint loading ports the real `dinov2.utils.utils.
   load_pretrained_weights()` logic: index by checkpoint key ("teacher")
   if present, strip `module.`/`backbone.` prefixes, load non-strict --
   not SPEC.md's stricter strict-load placeholder.
3. Pooling: `DinoVisionTransformer.forward()` returns the CLS token
   straight through an identity head, so SPEC.md's mean/max-pool-over-
   per-channel-CLS-tokens sketch is correct as written.

The vendored model (`.vendor.dinov2`) is not installed as the `dinov2`
package -- see `vendor/dinov2/VENDORED_FROM.md`.

One further correction, found while implementing `load_embedding_
dataloader()`: SPEC.md's dataclass docstring for `shard_pattern` claims
`webdataset.WebDataset` accepts a bare glob (`"dataset-*.tar"`) or a brace
pattern (`"dataset-{000000..000042}.tar"`) interchangeably. Verified
against the installed `webdataset==1.0.2`: only brace patterns are
expanded internally (`webdataset.shardlists.expand_urls`); a bare `*` glob
is passed through as a literal, unmatched filename. `load_embedding_
dataloader()` expands a non-brace `shard_pattern` via `glob.glob()` itself
before handing shard paths to `wds.WebDataset`.
"""

import dataclasses
import glob
import logging
from typing import Callable, Dict

import torch
import webdataset as wds
from omegaconf import MISSING

from .config import AppConfig
from .vendor.dinov2.models.vision_transformer import (
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)

# Named architectures this pipeline supports -- matches dinov2's own
# vits.__dict__[arch] dispatch pattern (dinov2/models/__init__.py's
# build_model, not vendored -- see vendor/dinov2/VENDORED_FROM.md).
_ARCHS: Dict[str, Callable[..., torch.nn.Module]] = {
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_giant2": vit_giant2,
}

# The checkpoint key real Cell-DINO/dinov2 teacher checkpoints are stored
# under (SPEC.md §6.3, verified against dinov2.utils.utils.
# load_pretrained_weights -- see the module docstring's Story 3.1 note).
_CHECKPOINT_KEY = "teacher"


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


def load_cell_dino(cfg: EmbedCellsConfig) -> torch.nn.Module:
    """Build a channel-adaptive (in_chans=1) DINOv2 ViT and load the teacher checkpoint.

    Construction goes through the named architecture factory
    (``vits.__dict__[cfg.arch]``-equivalent dict dispatch, see ``_ARCHS``)
    with ``in_chans=1, channel_adaptive=True`` baked in -- not
    ``dinov2.eval.setup``/``build_model_from_cfg``, which needs a full
    training-style config this pipeline doesn't have (see the module
    docstring's Story 3.1 note / docs/architecture.md).

    Checkpoint loading ports ``dinov2.utils.utils.load_pretrained_weights``'s
    real logic: index into the checkpoint dict by ``"teacher"`` if present,
    strip ``module.``/``backbone.`` prefixes from every key (real
    checkpoints commonly carry these from the training-time multicrop/DDP
    wrapper), then a **non-strict** ``load_state_dict`` -- a backbone-only
    checkpoint legitimately won't have ``head``/EMA-only keys.

    Parameters
    ----------
    cfg : EmbedCellsConfig
        Supplies ``arch``, ``patch_size``, ``crop_size``, ``checkpoint_path``,
        ``device``.

    Returns
    -------
    torch.nn.Module
        The loaded backbone, on ``cfg.device`` and in ``eval()`` mode.

    Raises
    ------
    ValueError
        If ``cfg.arch`` isn't one of ``_ARCHS``.
    """
    if cfg.arch not in _ARCHS:
        raise ValueError(f"Unknown arch {cfg.arch!r}, expected one of {sorted(_ARCHS)}")

    model = _ARCHS[cfg.arch](
        patch_size=cfg.patch_size,
        in_chans=1,
        channel_adaptive=True,
        img_size=cfg.crop_size,
    )

    state = torch.load(cfg.checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and _CHECKPOINT_KEY in state:
        logging.info("Taking key %r from checkpoint dict", _CHECKPOINT_KEY)
        state = state[_CHECKPOINT_KEY]
    state = {
        k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()
    }
    msg = model.load_state_dict(state, strict=False)
    logging.info(
        "Loaded checkpoint %s (arch=%s) with msg: %s",
        cfg.checkpoint_path,
        cfg.arch,
        msg,
    )
    return model.to(cfg.device).eval()


@torch.no_grad()
def embed_batch(
    model: torch.nn.Module,
    crops: torch.Tensor,
    masks: torch.Tensor,
    cfg: EmbedCellsConfig,
) -> torch.Tensor:
    """
    Embed a batch of multi-channel cell crops in bag-of-channels mode.

    Parameters
    ----------
    model : torch.nn.Module
        A channel-adaptive (``in_chans=1``) ViT, as built by
        :func:`load_cell_dino` (or an equivalent, e.g. a shrunk test model).
    crops : torch.Tensor
        Shape ``(B, C, crop_size, crop_size)``, any real dtype (typically
        ``uint16`` straight off the dataloader) -- cast to ``float32``
        before the forward pass.
    masks : torch.Tensor
        Shape ``(B, crop_size, crop_size)``, ``uint8`` label mask --
        nonzero where a pixel belongs to the target cell. Only consulted
        when ``cfg.mask_mode == "zero_background"``.
    cfg : EmbedCellsConfig
        Supplies ``mask_mode`` and ``channel_pool``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, D)`` -- one pooled embedding per cell, on the model's
        device. ``D`` = backbone width (1024 for ViT-L).

    Raises
    ------
    ValueError
        If ``cfg.mask_mode`` or ``cfg.channel_pool`` isn't recognized.
    """
    device = next(model.parameters()).device
    crops = crops.to(device=device, dtype=torch.float32)

    if cfg.mask_mode == "zero_background":
        masks = masks.to(device=device)
        crops = crops * (masks > 0).unsqueeze(
            1
        )  # zero every pixel not belonging to this cell
    elif cfg.mask_mode != "none":
        raise ValueError(f"Unknown mask_mode {cfg.mask_mode!r}")

    b, c, h, w = crops.shape
    per_channel = crops.reshape(b * c, 1, h, w)  # bag: each channel is its own "image"
    tokens = model(per_channel)  # (B*C, D) CLS embeddings
    tokens = tokens.reshape(b, c, -1)
    if cfg.channel_pool == "mean":
        return tokens.mean(dim=1)
    elif cfg.channel_pool == "max":
        return tokens.max(dim=1).values
    raise ValueError(f"Unknown channel_pool {cfg.channel_pool!r}")
