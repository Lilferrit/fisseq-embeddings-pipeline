"""Unit tests for EMBED_CELLS (embed.py, SPEC.md §6.3, Epic 3).

No real Cell-DINO checkpoint is available anywhere in this environment (see
docs/architecture.md) -- every test here either builds a real vendored ViT
at a deliberately shrunk size (fast on CPU, no GPU/pretrained-weights
dependency) or a small deterministic stub standing in for the backbone, to
exercise embed_batch()'s reshape/mask/pool control flow independent of real
transformer numerics.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import webdataset as wds

from fisseq_embeddings_pipeline.embed import (
    EmbedCellsConfig,
    embed_batch,
    load_cell_dino,
    load_embedding_dataloader,
)
from fisseq_embeddings_pipeline.vendor.dinov2.models.vision_transformer import (
    DinoVisionTransformer,
    vit_small,
)

CROP = 32  # small enough to keep real vendored-model tests fast on CPU


def _base_cfg(tmp_path: Path, **overrides) -> EmbedCellsConfig:
    cfg = EmbedCellsConfig(
        output_dir=str(tmp_path),
        shard_pattern="unused",
        checkpoint_path="unused",
        crop_size=CROP,
    )
    return dataclasses.replace(cfg, **overrides)


def _write_shard(
    tmp_path: Path,
    n_cells: int = 5,
    channels: int = 3,
    crop_size: int = CROP,
    maxcount: int = 100,
) -> Path:
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    with wds.ShardWriter(
        str(shard_dir / "dataset-%06d.tar"), maxcount=maxcount
    ) as sink:
        for i in range(n_cells):
            rng = np.random.default_rng(i)
            crop = rng.integers(0, 1000, size=(channels, crop_size, crop_size)).astype(
                np.uint16
            )
            mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
            mask[: crop_size // 2, :] = 1  # top half "belongs to" the cell
            meta = {
                "meta_batch": "batch1",
                "meta_well": "well1",
                "meta_tile": "tile0x0y",
                "meta_cell_index": i,
            }
            sink.write(
                {
                    "__key__": f"cell{i}",
                    "crop.npy": crop,
                    "mask.npy": mask,
                    "meta.json": meta,
                }
            )
    return shard_dir


# ---------------------------------------------------------------------------
# load_embedding_dataloader
# ---------------------------------------------------------------------------


def test_load_embedding_dataloader_expands_glob_and_streams_batches(tmp_path: Path):
    shard_dir = _write_shard(tmp_path, n_cells=5)
    cfg = _base_cfg(
        tmp_path,
        shard_pattern=str(shard_dir / "dataset-*.tar"),
        batch_size=2,
        num_workers=0,
    )

    batches = list(load_embedding_dataloader(cfg))

    assert sum(len(keys) for keys, *_ in batches) == 5
    keys, crops, masks, metas = batches[0]
    assert crops.shape == (2, 3, CROP, CROP)
    assert crops.dtype == torch.uint16
    assert masks.shape == (2, CROP, CROP)
    assert masks.dtype == torch.uint8
    assert metas[0]["meta_batch"] == "batch1"
    assert keys[0] == "cell0"


def test_load_embedding_dataloader_accepts_brace_pattern(tmp_path: Path):
    shard_dir = _write_shard(tmp_path, n_cells=3, maxcount=1)  # 3 one-cell shards
    n_shards = len(list(shard_dir.glob("dataset-*.tar")))
    pattern = str(shard_dir / f"dataset-{{000000..{n_shards - 1:06d}}}.tar")
    cfg = _base_cfg(tmp_path, shard_pattern=pattern, batch_size=10, num_workers=0)

    batches = list(load_embedding_dataloader(cfg))

    assert sum(len(keys) for keys, *_ in batches) == 3


def test_load_embedding_dataloader_raises_on_unmatched_glob(tmp_path: Path):
    cfg = _base_cfg(tmp_path, shard_pattern=str(tmp_path / "nonexistent-*.tar"))

    with pytest.raises(ValueError, match="matched no files"):
        load_embedding_dataloader(cfg)


# ---------------------------------------------------------------------------
# load_cell_dino
# ---------------------------------------------------------------------------


def test_load_cell_dino_raises_on_unknown_arch(tmp_path: Path):
    cfg = _base_cfg(tmp_path, arch="not_a_real_arch")

    with pytest.raises(ValueError, match="Unknown arch"):
        load_cell_dino(cfg)


def test_load_cell_dino_strips_prefixes_and_loads_nonstrict(tmp_path: Path):
    """Real checkpoints commonly wrap weights under module./backbone.
    prefixes and a "teacher" top-level key (dinov2.utils.utils.
    load_pretrained_weights, see the module docstring's Story 3.1 note) --
    confirm load_cell_dino()'s ported version of that logic actually
    strips them and loads the real values, not just constructs fresh
    random weights silently."""
    reference = vit_small(
        patch_size=16, in_chans=1, channel_adaptive=True, img_size=CROP
    )
    wrapped_state = {
        f"module.backbone.{k}": v for k, v in reference.state_dict().items()
    }
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"teacher": wrapped_state}, checkpoint_path)

    cfg = _base_cfg(
        tmp_path,
        arch="vit_small",
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )
    model = load_cell_dino(cfg)

    assert model.training is False
    loaded = dict(model.named_parameters())
    for name, ref_param in reference.named_parameters():
        assert torch.allclose(loaded[name], ref_param), name


# ---------------------------------------------------------------------------
# embed_batch -- deterministic stub model (isolates reshape/mask/pool logic
# from real transformer numerics)
# ---------------------------------------------------------------------------


class _MeanPixelStub(nn.Module):
    """Stand-in backbone: returns each single-channel image's mean pixel
    value broadcast across embed_dim, so embed_batch()'s bag-of-channels
    reshape and mask/pool logic can be checked against hand-computed
    expected values."""

    def __init__(self, embed_dim: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self._unused_param = nn.Parameter(torch.zeros(1))  # gives the module a device

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, 1, H, W)
        means = x.mean(dim=(1, 2, 3))
        return means.unsqueeze(-1).expand(-1, self.embed_dim).clone()


def test_embed_batch_output_shape():
    model = _MeanPixelStub(embed_dim=4)
    cfg = _base_cfg(Path("."), channel_pool="mean", mask_mode="none")
    crops = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    masks = torch.ones(2, 4, 4, dtype=torch.uint8)

    out = embed_batch(model, crops, masks, cfg)

    assert out.shape == (2, 4)


def test_embed_batch_zero_background_zeroes_masked_out_pixels():
    model = _MeanPixelStub(embed_dim=1)
    crops = torch.full((1, 2, 4, 4), 10.0)
    mask = torch.zeros(1, 4, 4, dtype=torch.uint8)
    mask[:, :2, :] = 1  # 8 of 16 pixels "belong to" the cell

    masked_cfg = _base_cfg(Path("."), channel_pool="mean", mask_mode="zero_background")
    masked_out = embed_batch(model, crops, mask, masked_cfg)
    # mean over 16 pixels, 8 zeroed out: (8*10 + 8*0) / 16 = 5.0
    assert torch.allclose(masked_out, torch.full((1, 1), 5.0))

    none_cfg = _base_cfg(Path("."), channel_pool="mean", mask_mode="none")
    none_out = embed_batch(model, crops, mask, none_cfg)
    assert torch.allclose(none_out, torch.full((1, 1), 10.0))


def test_embed_batch_channel_pool_mean_vs_max():
    model = _MeanPixelStub(embed_dim=1)
    # channel 0 is uniformly 1.0, channel 1 is uniformly 9.0
    crops = torch.stack([torch.full((4, 4), 1.0), torch.full((4, 4), 9.0)]).unsqueeze(0)
    mask = torch.ones(1, 4, 4, dtype=torch.uint8)

    max_out = embed_batch(model, crops, mask, _base_cfg(Path("."), channel_pool="max"))
    assert torch.allclose(max_out, torch.full((1, 1), 9.0))

    mean_out = embed_batch(
        model, crops, mask, _base_cfg(Path("."), channel_pool="mean")
    )
    assert torch.allclose(mean_out, torch.full((1, 1), 5.0))


def test_embed_batch_raises_on_unknown_mask_mode():
    model = _MeanPixelStub()
    cfg = _base_cfg(Path("."), mask_mode="bogus")

    with pytest.raises(ValueError, match="Unknown mask_mode"):
        embed_batch(
            model, torch.zeros(1, 1, 4, 4), torch.zeros(1, 4, 4, dtype=torch.uint8), cfg
        )


def test_embed_batch_raises_on_unknown_channel_pool():
    model = _MeanPixelStub()
    cfg = _base_cfg(Path("."), channel_pool="bogus")

    with pytest.raises(ValueError, match="Unknown channel_pool"):
        embed_batch(
            model, torch.zeros(1, 1, 4, 4), torch.zeros(1, 4, 4, dtype=torch.uint8), cfg
        )


def test_embed_batch_against_random_weight_real_model():
    """SPEC.md/IMPLEMENTATION_CHECKLIST.md Story 3.3's acceptance
    criterion: embed_batch() against a random-weight (not pretrained) model
    of the same real architecture, confirming output shape (B, D)."""
    model = DinoVisionTransformer(
        img_size=CROP,
        patch_size=16,
        in_chans=1,
        embed_dim=8,
        depth=2,
        num_heads=2,
        channel_adaptive=True,
        block_chunks=0,
    )
    model.eval()
    cfg = _base_cfg(Path("."), channel_pool="mean", mask_mode="zero_background")
    crops = torch.randint(0, 1000, (2, 3, CROP, CROP), dtype=torch.uint16)
    masks = (torch.rand(2, CROP, CROP) > 0.5).to(torch.uint8)

    out = embed_batch(model, crops, masks, cfg)

    assert out.shape == (2, 8)
