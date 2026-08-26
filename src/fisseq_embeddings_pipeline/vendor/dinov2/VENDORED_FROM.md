# Vendored from `facebookresearch/dinov2`

**Source:** https://github.com/facebookresearch/dinov2
**Commit:** `7764ea0f912e53c92e82eb78a2a1631e92725fc8` (`main` branch, fetched 2026-08-26)
**License:** Apache License, Version 2.0 (original per-file headers preserved below).

## Why vendored instead of installed as a dependency (SPEC.md §6.3/§10 item 1)

`dinov2` is not published on PyPI, and installing it from GitHub (e.g. a pinned
`dinov2 @ git+https://github.com/facebookresearch/dinov2@<sha>` entry in `pyproject.toml`) drags
in its own `requirements.txt`, which pins `torch==2.0.0`, `torchvision==0.15.0`, `xformers==0.0.18`,
and `cuml-cu11` -- incompatible with this repo's `torch>=2.4.0`, and `xformers`/`cuml` are
GPU-toolchain-specific builds that don't resolve in a CPU sandbox and aren't actually required:
every `xformers` import in the files below is wrapped in a `try/except ImportError` that falls
back to plain `torch.nn.functional.scaled_dot_product_attention`, confirmed by reading
`attention.py`/`block.py`/`swiglu_ffn.py` directly.

Rather than install the whole package, only the minimal pure-`torch` subset needed to construct a
channel-adaptive ViT and run a forward pass is vendored here, traced by following
`vision_transformer.py`'s own imports transitively:

- `models/vision_transformer.py` -- `DinoVisionTransformer` + the `vit_small`/`vit_base`/
  `vit_large`/`vit_giant2` factory functions.
- `layers/__init__.py`, `attention.py`, `block.py`, `mlp.py`, `patch_embed.py`, `swiglu_ffn.py`,
  `layer_scale.py`, `dino_head.py`, `drop_path.py` -- everything `layers/__init__.py` re-exports,
  which is everything `vision_transformer.py` needs.

**Not vendored** (and not needed for inference): `dinov2/eval/`, `dinov2/train/`, `dinov2/data/`,
`dinov2/run/`, `dinov2/utils/config.py`, `dinov2/utils/utils.py`'s `load_pretrained_weights` (its
checkpoint-loading *logic* -- `torch.load` -> index by checkpoint key -> strip `module.`/
`backbone.` prefixes -> `load_state_dict(strict=False)` -- is replicated directly in
`fisseq_embeddings_pipeline.embed.load_cell_dino()` instead of importing the file, since that file
otherwise pulls in `dinov2.utils.config`'s `omegaconf`-based `setup()` machinery this pipeline
doesn't use).

**Modifications versus upstream:** exactly one import line, in `models/vision_transformer.py`
(`from dinov2.layers import ...` -> `from ..layers import ...`, since this vendored subtree isn't
installed as the top-level `dinov2` package) -- flagged inline at that file's top. Every other
vendored file is byte-for-byte upstream, including its original Apache-2.0 header.

## What this vendoring does *not* resolve

No Cell-DINO checkpoint (`.pth`) is available in this environment, so the checkpoint's actual
state-dict key (`"teacher"` vs. top-level) and shape compatibility with the vendored
`vit_large(in_chans=1, channel_adaptive=True, patch_size=16)` construction are unverified against
real weights -- see `docs/architecture.md` and `SPEC.md` §10 item 1. Everything in this
vendored subtree, and the wrapper in `embed.py`, is verified against the *real* `dinov2` source
(not `SPEC.md`'s placeholder sketch), but not yet against a real checkpoint.
