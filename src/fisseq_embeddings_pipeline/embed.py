"""EMBED_CELLS.

Hydra entry point (`python -m fisseq_embeddings_pipeline.embed`), backing
the Nextflow process EMBED_CELLS (modules/local/embed_cells.nf, the
pipeline's only GPU-bound stage). Streams every cell in a BUILD_DATASET
WebDataset through a pretrained Cell-DINO checkpoint (Meta's dinov2) and
writes one row per cell to embeddings.parquet. Not gated by QC_FILTER: this
GPU pass runs once per experiment regardless of how many times QC
thresholds get retuned afterward.

`load_cell_dino()` builds the backbone via dinov2's architecture factory
functions directly (`vision_transformer.vit_large(...)`, dict-dispatched by
`cfg.arch`) and ports `dinov2.utils.utils.load_pretrained_weights()`'s real
checkpoint-loading logic: index by checkpoint key ("teacher") if present,
strip `module.`/`backbone.` prefixes, then a non-strict `load_state_dict`.
The vendored model (`.vendor.dinov2`) is not installed as the `dinov2`
package -- see `vendor/dinov2/VENDORED_FROM.md`.

Real Cell-DINO checkpoints are not all shaped the same way, so
`load_cell_dino()` doesn't hardcode a single architecture: it inspects the
checkpoint's own state dict to recover `in_chans` (bag-of-channels,
`in_chans=1`, vs. a fixed-channel-count backbone that ingests several
stacked channels jointly), `img_size` (from `pos_embed`'s patch count --
decoupled from `cfg.crop_size`, since `interpolate_pos_encoding`
reconciles any difference at inference time), `block_chunks` (chunked vs.
flat transformer-block keys), and whether LayerScale is present, rather
than assuming one fixed shape. `embed_batch()` branches on the loaded
model's actual `patch_embed.in_chans`: `1` gets the per-channel
split-and-pool treatment (`cfg.channel_pool`), anything else is fed to the
model jointly in one forward pass, requiring the crop's channel count to
match exactly. See docs/architecture.md for a comparison of the two real
checkpoints this was verified against.

`EmbedCellsConfig.channels` selects and orders which of the crop's channel
indices actually get fed to the model (a crop may carry more channels than
the model should see, e.g. multiple imaging cycles); `channel_apply_mask`
independently controls, per selected channel, whether the shared per-cell
segmentation mask gets applied before embedding. Both are consulted inside
`embed_batch()`, before the bag-of-channels-vs-joint branch above.

`load_embedding_dataloader()` expands a non-brace `shard_pattern` (no
`"dataset-{000000..000042}.tar"`-style brace expression) via `glob.glob()`
itself before handing shard paths to `wds.WebDataset` -- the installed
`webdataset` only expands brace patterns internally
(`webdataset.shardlists.expand_urls`); a bare `*` glob would otherwise pass
through as a literal, unmatched filename.
"""

import dataclasses
import glob
import logging
import pathlib
import re
from typing import Callable, Dict, List, Optional

import hydra
import polars as pl
import torch
import webdataset as wds
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from .config import AppConfig
from .utils.log import setup_logging
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
# under (verified against dinov2.utils.utils.load_pretrained_weights).
_CHECKPOINT_KEY = "teacher"


@dataclasses.dataclass
class EmbedCellsConfig(AppConfig):
    """
    Hydra structured configuration for EMBED_CELLS.

    Extends AppConfig (output_dir, output_root, log_level, random_seed);
    EMBED_CELLS's own logic doesn't consume random_seed itself (inference
    is deterministic given a fixed checkpoint), but every stage config
    inherits it uniformly.

    Attributes
    ----------
    shard_pattern : str
        Path/brace pattern for this experiment's BUILD_DATASET shards, e.g.
        ``"dataset-{000000..000042}.tar"``. A bare glob (``"dataset-*.tar"``)
        also works -- ``load_embedding_dataloader`` expands it itself (see
        the module docstring; real ``webdataset`` doesn't do this for you).
    checkpoint_path : str
        Path to the Cell-DINO checkpoint (``.pth``). The real, verified-
        compatible checkpoint this repo has on disk is
        ``weights/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth``
        -- a bag-of-channels ``vit_large``/patch-16 checkpoint matching
        this config's own defaults exactly (see docs/architecture.md's
        "Real checkpoint" section). Not defaulted here, or in
        ``params.yaml``: a checkpoint path is inherently deployment-specific
        and ``weights/`` is gitignored, so a hardcoded default would
        silently be wrong -- or simply absent -- in any other checkout.
    arch : str
        Backbone architecture, one of ``vit_small``/``vit_base``/
        ``vit_large``/``vit_giant2``. Defaults to ``"vit_large"``
        (embed dim 1024).
    patch_size : int
        ViT patch size. Defaults to ``16``.
    crop_size : int
        Expected crop size -- must match BUILD_DATASET's ``window``.
        Defaults to ``224``. See :func:`load_cell_dino`'s docstring: this
        is only a fallback for model construction (used when the
        checkpoint's own ``pos_embed`` can't be inspected), not something
        that constrains what crop size can actually be embedded.
    channels : list[int]
        Which of the crop's channel indices to feed into the model, in
        this order -- lets a crop carry more channels (e.g. multiple
        imaging cycles, flattened cycle-major per dataset.py) than the
        model should actually see, or reorders/subsets them. Defaults to
        ``[0, 1, 2, 3]`` (the first imaging cycle's four phenotyping
        channels, per BUILD_DATASET/starcall-workflow's own 4-channel
        default -- see docs/configuration.md).
    channel_apply_mask : list[bool]
        One entry per ``channels`` entry (same order, same length): whether
        that selected channel gets ``mask.npy``-based background zeroing
        before embedding. Every cell in this pipeline's data model has
        exactly one shared segmentation mask (BUILD_DATASET writes a
        single ``mask.npy`` per cell, not one per channel), so this is
        "apply the shared mask to this channel or not," not a claim that
        different channels have their own distinct masks.
        Defaults to ``[True, True, True, True]``.
    channel_pool : str
        How per-channel CLS embeddings are pooled into one per-cell
        embedding: ``"mean"`` or ``"max"``. Only consulted when the loaded
        model is bag-of-channels (``patch_embed.in_chans == 1`` -- see
        :func:`embed_batch`); ignored for a fixed-channel-count backbone.
        Defaults to ``"mean"``.
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
    channels: List[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3])
    channel_apply_mask: List[bool] = dataclasses.field(
        default_factory=lambda: [True, True, True, True]
    )
    channel_pool: str = "mean"
    device: str = "cuda"
    batch_size: int = 256
    num_workers: int = 4


def load_embedding_dataloader(cfg: EmbedCellsConfig) -> "torch.utils.data.DataLoader":
    """Stream ``(key, crop, mask, meta)`` batches from BUILD_DATASET's shards.

    ``webdataset.WebDataset(...).decode().to_tuple(...)`` is the standard
    reader side of the shards ``write_dataset_shards()`` (dataset.py)
    writes; batching via ``.batched()``/``DataLoader(batch_size=None)``
    keeps shard-order batches (webdataset's usual pattern) rather than a
    random-access ``Dataset``, which a tar-shard format doesn't support
    efficiently. ``mask.npy`` is always fetched -- whether/where it's
    applied is decided in :func:`embed_batch` via ``cfg.channel_apply_mask``,
    not here.

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


# Matches a chunked-block state-dict key, e.g. "blocks.1.3.norm1.weight"
# (dinov2's block_chunks>0 layout -- see vendor's DinoVisionTransformer.
# __init__: self.blocks is a ModuleList of BlockChunk, so the first digit
# group is the chunk index and the second is the position *within* the
# original (unchunked) block sequence).
_CHUNKED_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.(\d+)\.")
# Matches a flat (block_chunks=0) state-dict key, e.g. "blocks.3.norm1.weight".
_FLAT_BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.")
# LayerScale (vendor/dinov2/layers/block.py) only registers these params
# when a block was built with a truthy init_values -- their presence in a
# checkpoint is how we know to pass a (placeholder, checkpoint-overwritten)
# nonzero init_values ourselves, instead of silently dropping every
# block's learned residual scale to nn.Identity().
_LAYERSCALE_KEY_SUFFIXES = (".ls1.gamma", ".ls2.gamma")


def _infer_patch_embed_shape(
    state: dict, fallback_patch_size: int
) -> "tuple[int, Optional[int]]":
    """Recover ``(in_chans, patch_size)`` from ``patch_embed.proj.weight``'s
    shape (``(embed_dim, in_chans, patch_size, patch_size)``), the most
    direct source of truth for what the checkpoint actually expects --
    cheaper and more reliable than trusting a config field to have been
    set correctly. Returns ``patch_size=None`` (skip the cross-check) if
    the key is missing rather than guessing.
    """
    weight = state.get("patch_embed.proj.weight")
    if weight is None:
        logging.warning(
            "checkpoint has no patch_embed.proj.weight key -- assuming "
            "bag-of-channels (in_chans=1); this may be wrong"
        )
        return 1, None
    return weight.shape[1], weight.shape[-1]


def _infer_img_size(state: dict, patch_size: int, fallback_crop_size: int) -> int:
    """Recover the square ``img_size`` the checkpoint's ``pos_embed`` was
    trained at from its patch-token count. This is *not* the same as
    ``cfg.crop_size`` (the real per-experiment crop window) -- it only
    needs to reproduce ``pos_embed``'s exact shape so ``load_state_dict``
    doesn't raise; ``interpolate_pos_encoding`` (vendor/dinov2's own
    forward path) reconciles any difference against the real input size at
    inference time, so passing the checkpoint's native size here doesn't
    constrain what crop size can actually be embedded later.
    """
    pos_embed = state.get("pos_embed")
    if pos_embed is None:
        logging.warning(
            "checkpoint has no pos_embed key -- falling back to "
            "cfg.crop_size=%d for model construction",
            fallback_crop_size,
        )
        return fallback_crop_size
    num_patches = pos_embed.shape[1] - 1
    grid = round(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(
            f"checkpoint's pos_embed has {num_patches} patch positions, not "
            "a perfect square -- can't recover a square img_size from it"
        )
    return grid * patch_size


def _infer_block_chunks(state: dict) -> int:
    """Recover ``block_chunks`` from whether block keys are chunked
    (``blocks.<chunk>.<pos>....``) or flat (``blocks.<pos>....``) -- the
    distinct chunk-index count *is* ``block_chunks`` (see
    ``DinoVisionTransformer.__init__``'s own chunking loop). Defaults to
    ``1`` (matching that class's own default) if no block keys are found
    at all, since that's the shape a from-scratch construction would take.
    """
    chunk_ids = {int(m.group(1)) for k in state if (m := _CHUNKED_BLOCK_KEY.match(k))}
    if chunk_ids:
        return max(chunk_ids) + 1
    flat_ids = {int(m.group(1)) for k in state if (m := _FLAT_BLOCK_KEY.match(k))}
    return 0 if flat_ids else 1


def load_cell_dino(cfg: EmbedCellsConfig) -> torch.nn.Module:
    """Build a DINOv2 ViT matching the checkpoint's own architecture and load it.

    Construction goes through the named architecture factory
    (``vits.__dict__[cfg.arch]``-equivalent dict dispatch, see ``_ARCHS``)
    rather than ``dinov2.eval.setup``/``build_model_from_cfg``, which needs
    a full training-style cfg this pipeline doesn't have.

    Everything ``_ARCHS[cfg.arch]`` can't fix (``embed_dim``/``depth``/
    ``num_heads``/``mlp_ratio``) is inferred from the checkpoint's own
    state dict instead of assumed -- ``in_chans`` (bag-of-channels
    ``in_chans=1`` vs. a fixed-channel-count backbone), ``img_size`` (from
    ``pos_embed``'s patch count), ``block_chunks`` (chunked vs. flat block
    keys), and whether LayerScale is present -- rather than hardcoding one
    fixed shape as the only supported case. See docs/architecture.md for a
    comparison of the two real checkpoints this was verified against, one
    of which is *not* bag-of-channels at all.

    Checkpoint loading otherwise still ports ``dinov2.utils.utils.
    load_pretrained_weights``'s real logic: index into the checkpoint dict
    by ``"teacher"`` if present, strip ``module.``/``backbone.`` prefixes
    from every key (real checkpoints commonly carry these from the
    training-time multicrop/DDP wrapper), then a **non-strict**
    ``load_state_dict`` -- a backbone-only checkpoint legitimately won't
    have ``head``/EMA-only keys (which show up as harmless
    ``unexpected_keys``, only logged). ``missing_keys`` is different: since
    this pipeline's own ``head`` is always ``nn.Identity()`` (no
    parameters to ever be missing), any non-empty ``missing_keys`` here
    means the constructed architecture didn't actually match the
    checkpoint -- silently leaving real backbone weights at their random
    initialization, exactly the kind of quiet correctness bug the
    inference above exists to prevent -- so that case raises instead of
    logging.

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
        If ``cfg.arch`` isn't one of ``_ARCHS``, if ``cfg.patch_size``
        doesn't match the checkpoint's own patch_embed kernel size, or if
        the checkpoint's ``pos_embed`` can't be reduced to a square grid.
    RuntimeError
        If any parameter the constructed model needs is missing from the
        (non-strict-loaded) checkpoint.
    """
    if cfg.arch not in _ARCHS:
        raise ValueError(f"Unknown arch {cfg.arch!r}, expected one of {sorted(_ARCHS)}")

    state = torch.load(cfg.checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and _CHECKPOINT_KEY in state:
        logging.info("Taking key %r from checkpoint dict", _CHECKPOINT_KEY)
        state = state[_CHECKPOINT_KEY]
    state = {
        k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()
    }

    in_chans, checkpoint_patch_size = _infer_patch_embed_shape(state, cfg.patch_size)
    if checkpoint_patch_size is not None and checkpoint_patch_size != cfg.patch_size:
        raise ValueError(
            f"cfg.patch_size={cfg.patch_size} but checkpoint_path="
            f"{cfg.checkpoint_path!r}'s patch_embed kernel is "
            f"{checkpoint_patch_size}x{checkpoint_patch_size}"
        )
    img_size = _infer_img_size(state, cfg.patch_size, cfg.crop_size)
    block_chunks = _infer_block_chunks(state)
    has_layerscale = any(k.endswith(_LAYERSCALE_KEY_SUFFIXES) for k in state)

    model = _ARCHS[cfg.arch](
        patch_size=cfg.patch_size,
        in_chans=in_chans,
        channel_adaptive=(in_chans == 1),
        img_size=img_size,
        block_chunks=block_chunks,
        init_values=(1.0 if has_layerscale else None),
    )

    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys:
        raise RuntimeError(
            f"checkpoint_path={cfg.checkpoint_path!r} left "
            f"{len(msg.missing_keys)} backbone parameter(s) unloaded -- "
            f"arch/patch_size/in_chans/block_chunks mismatch? first few: "
            f"{msg.missing_keys[:10]}"
        )
    logging.info(
        "Loaded checkpoint %s (arch=%s, in_chans=%d, img_size=%d, "
        "block_chunks=%d, layerscale=%s), %d unexpected (ignored) key(s): %s",
        cfg.checkpoint_path,
        cfg.arch,
        in_chans,
        img_size,
        block_chunks,
        has_layerscale,
        len(msg.unexpected_keys),
        msg.unexpected_keys,
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
    Embed a batch of multi-channel cell crops.

    First selects/reorders ``cfg.channels`` out of the crop's full channel
    axis (a crop may carry more channels than the model should see -- e.g.
    multiple imaging cycles, per dataset.py's cycle-major flattening), then
    per-channel-optionally applies the shared cell mask
    (``cfg.channel_apply_mask``), then branches on the loaded model's
    *actual* ``patch_embed.in_chans`` (not a config flag) --
    ``load_cell_dino()`` sets this from whatever the checkpoint's own
    ``patch_embed.proj.weight`` says, since not every real Cell-DINO
    checkpoint is bag-of-channels:

    - ``in_chans == 1`` (bag-of-channels/channel-adaptive -- the mode
      ``weights/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth``
      actually is): the (now channel-selected) crop is split into ``C``
      single-channel images, each run through the *same* shared-weight
      backbone, and the resulting per-channel CLS tokens are pooled
      (``cfg.channel_pool``) into one embedding per cell.
    - any other ``in_chans``: the checkpoint expects exactly that many
      stacked channels jointly (one plain forward pass, no split, no
      pooling) -- ``len(cfg.channels)`` must match exactly.

    Parameters
    ----------
    model : torch.nn.Module
        As built by :func:`load_cell_dino` (or an equivalent, e.g. a
        shrunk test model) -- its ``patch_embed.in_chans`` selects the mode
        above.
    crops : torch.Tensor
        Shape ``(B, C, crop_size, crop_size)`` -- ``C`` may exceed
        ``len(cfg.channels)``, since channel selection happens here, not
        upstream. Any real dtype (typically ``uint16`` straight off the
        dataloader) -- cast to ``float32`` before the forward pass.
    masks : torch.Tensor
        Shape ``(B, crop_size, crop_size)``, ``uint8`` label mask --
        nonzero where a pixel belongs to the target cell. This pipeline's
        data model has exactly one shared mask per cell (not one per
        channel), so this same tensor is what ``cfg.channel_apply_mask``
        selectively applies to each selected channel. Only read at all
        when at least one entry of
        ``cfg.channel_apply_mask`` is ``True``.
    cfg : EmbedCellsConfig
        Supplies ``channels``, ``channel_apply_mask``, and (bag-of-channels
        mode only) ``channel_pool``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, D)`` -- one embedding per cell, on the model's device.
        ``D`` = backbone width.

    Raises
    ------
    ValueError
        If ``cfg.channels``/``cfg.channel_apply_mask`` don't match 1:1, if
        ``cfg.channels`` has an index out of range for the crop, if
        ``cfg.channel_pool`` isn't recognized, or if the model expects a
        fixed channel count ``cfg.channels`` doesn't provide.
    """
    if len(cfg.channel_apply_mask) != len(cfg.channels):
        raise ValueError(
            f"cfg.channel_apply_mask has {len(cfg.channel_apply_mask)} "
            f"entries but cfg.channels selects {len(cfg.channels)} "
            "channel(s) -- must be the same length, in the same order"
        )
    if any(idx < 0 or idx >= crops.shape[1] for idx in cfg.channels):
        raise ValueError(
            f"cfg.channels={cfg.channels} out of range for a crop with "
            f"{crops.shape[1]} channel(s)"
        )

    device = next(model.parameters()).device
    crops = crops.to(device=device, dtype=torch.float32)
    crops = crops[:, cfg.channels, :, :]  # select/reorder the configured channels

    if any(cfg.channel_apply_mask):
        masks = masks.to(device=device)
        mask_bin = (masks > 0).unsqueeze(1)  # (B, 1, H, W)
        apply = torch.tensor(
            cfg.channel_apply_mask, dtype=torch.bool, device=device
        ).view(1, -1, 1, 1)
        # zero every pixel not belonging to this cell, but only on the
        # channels cfg.channel_apply_mask actually flags
        crops = torch.where(apply, crops * mask_bin, crops)

    b, c, h, w = crops.shape
    in_chans = model.patch_embed.in_chans

    if in_chans != 1:
        if c != in_chans:
            raise ValueError(
                f"model expects in_chans={in_chans} (not bag-of-channels) "
                f"but cfg.channels selects {c} channel(s)"
            )
        return model(crops)  # (B, D) CLS embeddings, one joint forward pass

    per_channel = crops.reshape(b * c, 1, h, w)  # bag: each channel is its own "image"
    tokens = model(per_channel)  # (B*C, D) CLS embeddings
    tokens = tokens.reshape(b, c, -1)
    if cfg.channel_pool == "mean":
        return tokens.mean(dim=1)
    elif cfg.channel_pool == "max":
        return tokens.max(dim=1).values
    raise ValueError(f"Unknown channel_pool {cfg.channel_pool!r}")


_cs = ConfigStore.instance()
_cs.store(name="embed_main", node=EmbedCellsConfig)


@hydra.main(version_base=None, config_path=None, config_name="embed_main")
def main(cfg: DictConfig) -> None:
    """
    Hydra entry point: embed every cell in a Cell Dataset via Cell-DINO.

    Steps
    -----
    1. Create ``output_dir``.
    2. Build the dataloader (:func:`load_embedding_dataloader`) and model
       (:func:`load_cell_dino`).
    3. For each batch, embed via :func:`embed_batch` and accumulate one row
       per cell: passthrough ``meta.json`` fields plus zero-padded
       ``emb_0000``..``emb_{D-1}`` columns (``EMBEDDING_SELECTOR``,
       ``utils/constants.py``, matches these).
    4. Write ``embeddings.parquet``.

    Configuration
    -------------
    Override any field on the command line, e.g.::

        python -m fisseq_embeddings_pipeline.embed \\
            output_dir=./out \\
            'shard_pattern=./dataset-*.tar' \\
            checkpoint_path=/data/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth \\
            device=cpu \\
            'channels=[0,1,2,3]' \\
            'channel_apply_mask=[true,true,true,true]' \\
            random_seed=0
    """
    embed_cfg: EmbedCellsConfig = OmegaConf.to_object(cfg)

    output_dir = pathlib.Path(embed_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embed_cfg.output_dir = str(output_dir)
    setup_logging(embed_cfg, "embed")

    logging.info(
        "Embedding cells from %s (arch=%s, checkpoint=%s, device=%s)",
        embed_cfg.shard_pattern,
        embed_cfg.arch,
        embed_cfg.checkpoint_path,
        embed_cfg.device,
    )

    dataloader = load_embedding_dataloader(embed_cfg)
    model = load_cell_dino(embed_cfg)

    rows = []
    n_cells = 0
    for keys, crops, masks, metas in dataloader:
        embeddings = embed_batch(model, crops, masks, embed_cfg).cpu().numpy()
        n_dims = embeddings.shape[1]
        emb_cols = [f"emb_{i:04d}" for i in range(n_dims)]
        for meta, emb_row in zip(metas, embeddings):
            rows.append({**meta, **dict(zip(emb_cols, emb_row.tolist()))})
        n_cells += len(keys)

    logging.info("Writing embeddings.parquet (%d cells)", n_cells)
    pl.DataFrame(rows).write_parquet(output_dir / "embeddings.parquet")
    logging.info("Done")


if __name__ == "__main__":
    main()
