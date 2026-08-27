"""Unit tests for EMBED_CELLS (embed.py).

Two real Cell-DINO checkpoints are available locally under weights/ (see
docs/architecture.md's "Real checkpoint" section) -- their tests are
skipped automatically when the file isn't present (e.g. in CI, since
weights/ is gitignored). Everything else either builds a real vendored ViT
at a deliberately shrunk size (fast on CPU, no pretrained-weights
dependency) or a small deterministic stub standing in for the backbone, to
exercise embed_batch()'s reshape/mask/pool control flow independent of real
transformer numerics.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import torch.nn as nn
import webdataset as wds

from fisseq_embeddings_pipeline.embed import (
    EmbedCellsConfig,
    embed_batch,
    load_cell_dino,
    load_embedding_dataloader,
    main,
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
    load_pretrained_weights) -- confirm load_cell_dino()'s ported version
    of that logic actually
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


def test_load_cell_dino_infers_chunked_blocks_and_layerscale(tmp_path: Path):
    """The real checkpoint that motivated this (weights/cell_dino_vits8_
    pretrain_cp-37d20e9c.pth, see the module docstring) stores its 12
    transformer blocks chunked (block_chunks=4) and has LayerScale
    (ls1.gamma/ls2.gamma) -- neither of which the original hardcoded
    in_chans=1/no-layerscale construction accounted for. Reproduce that
    shape with a small reference model and confirm load_cell_dino() infers
    both correctly (a real-numbered LayerScale gamma, not left at its
    default, proves the checkpoint's values -- not just freshly
    initialized ones -- actually made it into the loaded model)."""
    reference = vit_small(
        patch_size=8,
        in_chans=1,
        channel_adaptive=True,
        img_size=CROP,
        block_chunks=4,
        init_values=1e-5,
    )
    with torch.no_grad():
        for p in reference.parameters():
            if p.ndim == 1 and p.numel() == reference.embed_dim:
                p.uniform_(
                    0.1, 0.2
                )  # give ls*.gamma (among others) a distinctive value
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(reference.state_dict(), checkpoint_path)

    cfg = _base_cfg(
        tmp_path,
        arch="vit_small",
        patch_size=8,
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )
    model = load_cell_dino(cfg)

    assert model.blocks[3][11].ls1.gamma.shape == (reference.embed_dim,)
    loaded = dict(model.named_parameters())
    for name, ref_param in reference.named_parameters():
        assert torch.allclose(loaded[name], ref_param), name


def test_load_cell_dino_infers_joint_multichannel_in_chans(tmp_path: Path):
    """The real checkpoint's patch_embed.proj.weight is (384, 5, 8, 8) --
    a fixed 5-channel backbone, not bag-of-channels. Reproduce that shape
    via the same vit_small factory load_cell_dino() itself dispatches
    through, and confirm it infers in_chans=5 (not the old hardcoded 1)
    and builds a non-channel-adaptive model."""
    reference = vit_small(
        patch_size=8, in_chans=5, channel_adaptive=False, img_size=CROP, block_chunks=0
    )
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(reference.state_dict(), checkpoint_path)

    cfg = _base_cfg(
        tmp_path,
        arch="vit_small",
        patch_size=8,
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )

    model = load_cell_dino(cfg)

    assert model.patch_embed.in_chans == 5
    assert model.bag_of_channels is False
    loaded = dict(model.named_parameters())
    for name, ref_param in reference.named_parameters():
        assert torch.allclose(loaded[name], ref_param), name


def test_load_cell_dino_raises_on_patch_size_mismatch(tmp_path: Path):
    reference = vit_small(
        patch_size=16, in_chans=1, channel_adaptive=True, img_size=CROP
    )
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(reference.state_dict(), checkpoint_path)

    cfg = _base_cfg(
        tmp_path,
        arch="vit_small",
        patch_size=8,  # checkpoint is actually patch_size=16
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )

    with pytest.raises(ValueError, match="patch_size"):
        load_cell_dino(cfg)


def test_load_cell_dino_raises_on_incompatible_checkpoint(tmp_path: Path):
    """A checkpoint architecturally incompatible with cfg.arch (here:
    vit_small's depth=12 vs. a 2-block checkpoint) must not silently leave
    most of the backbone at its random initialization -- see the "Correction
    found once a real checkpoint became available" module-docstring note on
    why missing_keys is escalated to a raise instead of just logged."""
    tiny = DinoVisionTransformer(
        img_size=CROP,
        patch_size=16,
        in_chans=1,
        embed_dim=384,
        depth=2,
        num_heads=6,
        channel_adaptive=True,
        block_chunks=0,
    )
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(tiny.state_dict(), checkpoint_path)

    cfg = _base_cfg(
        tmp_path, arch="vit_small", checkpoint_path=str(checkpoint_path), device="cpu"
    )

    with pytest.raises(RuntimeError, match="unloaded"):
        load_cell_dino(cfg)


# ---------------------------------------------------------------------------
# load_cell_dino -- against the real Cell-DINO checkpoint, when present
# ---------------------------------------------------------------------------

_WEIGHTS_DIR = Path(__file__).parents[2] / "weights"
_REAL_CHECKPOINT_VITS8 = _WEIGHTS_DIR / "cell_dino_vits8_pretrain_cp-37d20e9c.pth"
_REAL_CHECKPOINT_VITL16 = (
    _WEIGHTS_DIR / "channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth"
)


@pytest.mark.skipif(
    not _REAL_CHECKPOINT_VITS8.exists(), reason="real Cell-DINO checkpoint not present"
)
def test_load_cell_dino_and_embed_batch_against_real_vits8_checkpoint(tmp_path: Path):
    """weights/cell_dino_vits8_pretrain_cp-37d20e9c.pth: ViT-Small/patch8, a
    fixed 5-channel backbone (not bag-of-channels), chunked blocks,
    LayerScale -- see the module docstring. Confirms load_cell_dino()/
    embed_batch() actually load and run it end to end, not just against
    synthetic stand-ins reproducing its shape."""
    cfg = _base_cfg(
        tmp_path,
        arch="vit_small",
        patch_size=8,
        checkpoint_path=str(_REAL_CHECKPOINT_VITS8),
        channels=[0, 1, 2, 3, 4],
        channel_apply_mask=[True, True, True, True, True],
        device="cpu",
    )

    model = load_cell_dino(cfg)

    assert model.patch_embed.in_chans == 5
    assert model.embed_dim == 384

    crops = torch.randint(0, 4096, (2, 5, CROP, CROP), dtype=torch.uint16)
    masks = (torch.rand(2, CROP, CROP) > 0.5).to(torch.uint8)
    out = embed_batch(model, crops, masks, cfg)

    assert out.shape == (2, 384)
    assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not _REAL_CHECKPOINT_VITL16.exists(),
    reason="real channel-adaptive Cell-DINO checkpoint not present",
)
def test_load_cell_dino_and_embed_batch_against_real_vitl16_checkpoint(tmp_path: Path):
    """weights/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth --
    a bag-of-channels vit_large/patch-16/224 checkpoint (see the module
    docstring): loads with this
    module's plain defaults (arch/patch_size/crop_size unchanged), and
    embed_batch()'s default cfg.channels=[0,1,2,3]/channel_apply_mask
    selects/masks 4 of a 6-channel crop."""
    cfg = _base_cfg(
        tmp_path,
        arch="vit_large",
        patch_size=16,
        checkpoint_path=str(_REAL_CHECKPOINT_VITL16),
        device="cpu",
    )

    model = load_cell_dino(cfg)

    assert model.patch_embed.in_chans == 1
    assert model.bag_of_channels is True
    assert model.embed_dim == 1024

    crops = torch.randint(0, 4096, (2, 6, CROP, CROP), dtype=torch.uint16)
    masks = (torch.rand(2, CROP, CROP) > 0.5).to(torch.uint8)
    out = embed_batch(
        model, crops, masks, cfg
    )  # cfg.channels=[0,1,2,3] selects 4 of the 6

    assert out.shape == (2, 1024)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# embed_batch -- deterministic stub model (isolates reshape/mask/pool logic
# from real transformer numerics)
# ---------------------------------------------------------------------------


class _MeanPixelStub(nn.Module):
    """Stand-in backbone: returns each image's mean pixel value broadcast
    across embed_dim, so embed_batch()'s reshape/mask/pool logic can be
    checked against hand-computed expected values, independent of real
    transformer numerics. `patch_embed.in_chans` is the real
    `load_cell_dino()`-built model's own dispatch point for embed_batch()'s
    bag-of-channels-vs-joint branch (see the module docstring's real-
    checkpoint note) -- mocked here so this stub is dispatched the same way."""

    def __init__(self, embed_dim: int = 4, in_chans: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = types.SimpleNamespace(in_chans=in_chans)
        self._unused_param = nn.Parameter(torch.zeros(1))  # gives the module a device

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, in_chans, H, W)
        means = x.mean(dim=(1, 2, 3))
        return means.unsqueeze(-1).expand(-1, self.embed_dim).clone()


def test_embed_batch_output_shape():
    model = _MeanPixelStub(embed_dim=4)
    cfg = _base_cfg(
        Path("."),
        channel_pool="mean",
        channels=[0, 1, 2],
        channel_apply_mask=[False] * 3,
    )
    crops = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    masks = torch.ones(2, 4, 4, dtype=torch.uint8)

    out = embed_batch(model, crops, masks, cfg)

    assert out.shape == (2, 4)


def test_embed_batch_zero_background_zeroes_masked_out_pixels():
    model = _MeanPixelStub(embed_dim=1)
    crops = torch.full((1, 2, 4, 4), 10.0)
    mask = torch.zeros(1, 4, 4, dtype=torch.uint8)
    mask[:, :2, :] = 1  # 8 of 16 pixels "belong to" the cell

    masked_cfg = _base_cfg(
        Path("."), channel_pool="mean", channels=[0, 1], channel_apply_mask=[True, True]
    )
    masked_out = embed_batch(model, crops, mask, masked_cfg)
    # mean over 16 pixels, 8 zeroed out: (8*10 + 8*0) / 16 = 5.0
    assert torch.allclose(masked_out, torch.full((1, 1), 5.0))

    none_cfg = _base_cfg(
        Path("."),
        channel_pool="mean",
        channels=[0, 1],
        channel_apply_mask=[False, False],
    )
    none_out = embed_batch(model, crops, mask, none_cfg)
    assert torch.allclose(none_out, torch.full((1, 1), 10.0))


def test_embed_batch_channel_apply_mask_is_per_channel():
    """The whole point of channel_apply_mask being a list, not a single
    bool: one selected channel can get the shared mask applied while
    another doesn't, in the same call."""
    model = _MeanPixelStub(embed_dim=1)
    crops = torch.full((1, 2, 4, 4), 10.0)
    mask = torch.zeros(1, 4, 4, dtype=torch.uint8)
    mask[:, :2, :] = 1  # 8 of 16 pixels "belong to" the cell

    cfg = _base_cfg(
        Path("."),
        channel_pool="mean",
        channels=[0, 1],
        channel_apply_mask=[True, False],
    )
    out = embed_batch(model, crops, mask, cfg)

    # channel 0 (masked): mean = 5.0 (see test above); channel 1 (unmasked):
    # mean = 10.0; mean-pooled across the two channels' CLS tokens = 7.5
    assert torch.allclose(out, torch.full((1, 1), 7.5))


def test_embed_batch_selects_and_reorders_configured_channels():
    """cfg.channels picks out (and can reorder) a subset of the crop's
    full channel axis -- e.g. a crop with more channels than the model
    should see (multiple imaging cycles, per dataset.py's cycle-major
    flattening)."""
    model = _MeanPixelStub(embed_dim=1)
    # 4 channels, each a distinct uniform value -- channel_pool="mean"
    # over just the selected two makes the selection observable.
    crops = torch.stack(
        [torch.full((4, 4), v) for v in (1.0, 2.0, 100.0, 200.0)]
    ).unsqueeze(0)
    mask = torch.ones(1, 4, 4, dtype=torch.uint8)

    cfg = _base_cfg(
        Path("."),
        channel_pool="mean",
        channels=[3, 0],
        channel_apply_mask=[False, False],
    )
    out = embed_batch(model, crops, mask, cfg)

    # only channels 3 (200.0) and 0 (1.0) selected; channels 1/2 ignored
    assert torch.allclose(out, torch.full((1, 1), (200.0 + 1.0) / 2))


def test_embed_batch_raises_on_channels_apply_mask_length_mismatch():
    model = _MeanPixelStub()
    cfg = _base_cfg(Path("."), channels=[0, 1], channel_apply_mask=[True])

    with pytest.raises(ValueError, match="channel_apply_mask"):
        embed_batch(
            model, torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 4, dtype=torch.uint8), cfg
        )


def test_embed_batch_raises_on_channels_out_of_range():
    model = _MeanPixelStub()
    cfg = _base_cfg(Path("."), channels=[0, 4], channel_apply_mask=[False, False])

    with pytest.raises(ValueError, match="out of range"):
        embed_batch(
            model, torch.zeros(1, 2, 4, 4), torch.zeros(1, 4, 4, dtype=torch.uint8), cfg
        )


def test_embed_batch_channel_pool_mean_vs_max():
    model = _MeanPixelStub(embed_dim=1)
    # channel 0 is uniformly 1.0, channel 1 is uniformly 9.0
    crops = torch.stack([torch.full((4, 4), 1.0), torch.full((4, 4), 9.0)]).unsqueeze(0)
    mask = torch.ones(1, 4, 4, dtype=torch.uint8)
    channels_cfg = dict(channels=[0, 1], channel_apply_mask=[False, False])

    max_out = embed_batch(
        model, crops, mask, _base_cfg(Path("."), channel_pool="max", **channels_cfg)
    )
    assert torch.allclose(max_out, torch.full((1, 1), 9.0))

    mean_out = embed_batch(
        model, crops, mask, _base_cfg(Path("."), channel_pool="mean", **channels_cfg)
    )
    assert torch.allclose(mean_out, torch.full((1, 1), 5.0))


def test_embed_batch_raises_on_unknown_channel_pool():
    model = _MeanPixelStub()
    cfg = _base_cfg(
        Path("."), channel_pool="bogus", channels=[0], channel_apply_mask=[False]
    )

    with pytest.raises(ValueError, match="Unknown channel_pool"):
        embed_batch(
            model, torch.zeros(1, 1, 4, 4), torch.zeros(1, 4, 4, dtype=torch.uint8), cfg
        )


def test_embed_batch_joint_multichannel_skips_split_and_pool():
    """model.patch_embed.in_chans != 1 (a fixed-channel-count backbone, per
    the real checkpoint this repo has -- see the module docstring) must be
    fed all its channels jointly in one forward pass, not split per-channel
    like the bag-of-channels path. _MeanPixelStub's mean-over-everything
    forward makes the two paths numerically distinguishable: joint mode
    means one mean over all C*H*W pixels; bag-of-channels would mean each
    channel separately, then pool -- different numbers whenever channel
    means differ."""
    model = _MeanPixelStub(embed_dim=1, in_chans=2)
    # channel 0 uniformly 1.0, channel 1 uniformly 9.0 -- bag-of-channels
    # mean-pool would give 5.0; one joint forward over both channels'
    # pixels together gives (1.0 + 9.0) / 2 = 5.0 too by coincidence here,
    # so also assert dispatch never reshaped batch*channels into the batch
    # dimension (which raise_on_unknown_channel_pool below would trigger
    # for an unset channel_pool if the bag-of-channels branch ran instead).
    crops = torch.stack([torch.full((4, 4), 1.0), torch.full((4, 4), 9.0)]).unsqueeze(0)
    mask = torch.ones(1, 4, 4, dtype=torch.uint8)

    out = embed_batch(
        model,
        crops,
        mask,
        _base_cfg(
            Path("."),
            channel_pool="bogus",
            channels=[0, 1],
            channel_apply_mask=[False, False],
        ),
    )  # channel_pool is irrelevant/unused in joint mode -- an invalid value must not raise

    assert out.shape == (1, 1)
    assert torch.allclose(out, torch.full((1, 1), 5.0))


def test_embed_batch_raises_on_channel_count_mismatch_in_joint_mode():
    model = _MeanPixelStub(embed_dim=1, in_chans=5)
    crops = torch.zeros(1, 3, 4, 4)  # cfg.channels selects 3, model expects exactly 5
    mask = torch.ones(1, 4, 4, dtype=torch.uint8)
    cfg = _base_cfg(Path("."), channels=[0, 1, 2], channel_apply_mask=[False] * 3)

    with pytest.raises(ValueError, match="in_chans=5"):
        embed_batch(model, crops, mask, cfg)


def test_embed_batch_against_random_weight_real_model():
    """embed_batch() against a random-weight (not pretrained) model of the
    same real architecture, confirming output shape (B, D)."""
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
    cfg = _base_cfg(
        Path("."),
        channel_pool="mean",
        channels=[0, 1, 2],
        channel_apply_mask=[True, True, True],
    )
    crops = torch.randint(0, 1000, (2, 3, CROP, CROP), dtype=torch.uint16)
    masks = (torch.rand(2, CROP, CROP) > 0.5).to(torch.uint8)

    out = embed_batch(model, crops, masks, cfg)

    assert out.shape == (2, 8)


# ---------------------------------------------------------------------------
# main() -- end-to-end CLI smoke test
# ---------------------------------------------------------------------------


def test_main_runs_end_to_end_via_cli(tmp_path: Path):
    # 4 channels to match EmbedCellsConfig's own default channels=[0,1,2,3]
    # -- this test doesn't override channels/channel_apply_mask, exercising
    # those defaults through the real CLI path.
    shard_dir = _write_shard(tmp_path, n_cells=4, channels=4, crop_size=CROP)
    reference = vit_small(
        patch_size=16, in_chans=1, channel_adaptive=True, img_size=CROP
    )
    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save({"teacher": reference.state_dict()}, checkpoint_path)
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fisseq_embeddings_pipeline.embed",
            f"output_dir={output_dir}",
            f"shard_pattern={shard_dir}/dataset-*.tar",
            f"checkpoint_path={checkpoint_path}",
            "arch=vit_small",
            f"crop_size={CROP}",
            "device=cpu",
            "batch_size=2",
            "num_workers=0",
            "random_seed=0",
        ],
        capture_output=True,
        text=True,
        # Hydra's own working-directory management writes outputs/<date>/
        # <time>/ under the process cwd -- run from tmp_path, matching
        # test_dataset.py's own CLI smoke test.
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    out = pl.read_parquet(output_dir / "embeddings.parquet")
    assert out.height == 4
    assert "emb_0000" in out.columns
    assert "emb_0383" in out.columns  # vit_small embed_dim=384
    assert out["meta_batch"].unique().to_list() == ["batch1"]


def test_main_is_hydra_entry_point():
    """Sanity check that `main` is importable and hydra-wrapped (the real
    invocation path is exercised via subprocess above -- hydra.main-wrapped
    functions parse sys.argv, so they aren't meant to be called directly
    from a test process)."""
    assert callable(main)
