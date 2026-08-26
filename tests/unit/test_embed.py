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
import webdataset as wds

from fisseq_embeddings_pipeline.embed import EmbedCellsConfig, load_embedding_dataloader

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
