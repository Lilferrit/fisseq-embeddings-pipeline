# Architecture

Placeholder -- see SPEC.md §1-§4 (Overview, Terminology map, Architecture
decisions, Repository layout) until this is written for real
post-implementation, epic by epic. This page currently only has real
content for the piece Epic 3 resolved: `EMBED_CELLS`'s Cell-DINO inference
internals, the single item SPEC.md §10 flagged as unresolved pending real
`dinov2` source review.

## `EMBED_CELLS` / Cell-DINO inference internals (SPEC.md §6.3, §10 item 1)

SPEC.md §6.3's `load_cell_dino()`/`embed_batch()` sketch was an admitted
best guess -- `dinov2`'s own docs don't publish a documented inference API.
This section records what was actually verified against the real
`facebookresearch/dinov2` source (fetched from GitHub, commit
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`, `main` branch -- neither the
real `dinov2` repo nor a Cell-DINO checkpoint is mounted in this sandbox,
unlike `fisseq-data-pipeline`/`starcall-workflow`).

**Cell-DINO is real, not a fictional stand-in invented for this spec** --
the public `dinov2` repo really does ship `docs/README_CELL_DINO.md`,
`docs/README_CHANNEL_ADAPTIVE_DINO.md`, `LICENSE_CELL_DINO_CODE`, and a
`channel_adaptive` constructor flag on `DinoVisionTransformer`.

### 1. Model construction

SPEC.md's sketch assumed a bare `build_model_from_cfg(arch=..., patch_size=...,
in_chans=1)` call. The real construction path
(`dinov2/eval/setup.py::build_model_for_eval`) instead goes through
`build_model_from_cfg(cfg, only_teacher=True)` -> `dinov2/models/__init__.py::build_model`,
which needs a full training-style config object (`cfg.student.{arch,
patch_size, layerscale, ffn_layer, block_chunks, qkv_bias, proj_bias,
ffn_bias, num_register_tokens, interpolate_offset, interpolate_antialias,
in_chans, channel_adaptive}` plus `cfg.crops.global_crops_size`) -- more
than `EmbedCellsConfig` exposes, and more than is available here (no
training config for the actual Cell-DINO checkpoint is available in this
environment).

**Decision:** skip `build_model_from_cfg`/`dinov2.eval.setup` entirely and
call the architecture factory function directly --
`vision_transformer.vit_large(patch_size=cfg.patch_size, in_chans=1,
channel_adaptive=True, img_size=cfg.crop_size)` -- which is what
`build_model` does internally anyway once its argparse-namespace plumbing
is stripped away. This is a deliberate simplification given the config
this pipeline actually has, not an oversight.

### 2. Checkpoint loading

SPEC.md's sketch: `state["teacher"] if "teacher" in state else state`, then
a **strict** `model.load_state_dict(state)`. The real loader
(`dinov2/utils/utils.py::load_pretrained_weights(model, path,
checkpoint_key)`) does more:

1. `torch.load(path, map_location="cpu")`.
2. If `checkpoint_key` (`"teacher"`) is a key in the loaded dict, index into it.
3. Strip `module.` and `backbone.` prefixes from every state-dict key
   (real checkpoints commonly carry these from the training-time
   multicrop/DDP wrapper).
4. `model.load_state_dict(state_dict, strict=False)` -- not strict, since a
   backbone-only checkpoint legitimately won't have `head`/EMA-only keys.

`load_cell_dino()` ports this exact logic rather than SPEC.md's stricter
placeholder.

### 3. Pooling / forward path

Confirmed `DinoVisionTransformer.forward(x, is_training=False)` returns
`self.head(x_norm_clstoken)`, and `head` defaults to `nn.Identity()` -- so
plain `model(x)` on an `(N, 1, H, W)` batch already returns `(N, D)` CLS
embeddings directly (verified directly: a shrunk 2-layer/32-dim vendored
model returns exactly that shape). This matches SPEC.md's `embed_batch()`
sketch's assumption, so that part of the sketch is correct as written:
reshape `(B, C, H, W) -> (B*C, 1, H, W)`, call `model(x)`, reshape back to
`(B, C, D)`, then mean/max-pool over the channel dimension.

One thing worth recording: the model's own `channel_adaptive=True`
constructor flag (what the real repo calls "bag of channels") only changes
behavior inside `get_intermediate_layers()` -- used by the paper's own
*linear-probe* eval scripts (`--bag-of-channels --avgpool --n-last-blocks
4`), which concatenate several transformer blocks' tokens and avgpool
patch tokens, not just the final CLS token. That's a heavier protocol built
for training a linear classifier, not for producing one fixed-length
embedding per cell. Since this pipeline only wants a single per-cell
embedding for downstream median-pooling/distinguishability scoring (not a
classifier), the simpler plain-`forward()` CLS-token path is used instead,
and `channel_adaptive=True` is passed at construction time only so the
checkpoint's own state dict (trained with that flag) lines up -- not
because `get_intermediate_layers`'s bag-of-channels branch is invoked.
This is a deliberate divergence from the paper's own linear-eval protocol,
not a gap.

### 4. Vendoring `dinov2`, not installing it

`dinov2`'s own `requirements.txt` pins `torch==2.0.0`, `xformers==0.0.18`,
`cuml-cu11` -- incompatible with this repo's `torch>=2.4.0`, and
`xformers`/`cuml` are GPU-toolchain-specific and unneeded: every
`xformers` import in the real source is wrapped in `try/except
ImportError`, falling back to plain `torch.nn.functional.
scaled_dot_product_attention` (confirmed directly in `attention.py`,
`block.py`, `swiglu_ffn.py`). Only the minimal pure-`torch` file subset
needed for inference is vendored into
`src/fisseq_embeddings_pipeline/vendor/dinov2/` -- see that directory's
`VENDORED_FROM.md` for the exact commit and file list.

### Still unresolved

No Cell-DINO checkpoint (`.pth`) exists anywhere in this sandbox, so the
checkpoint's real state-dict key (`"teacher"` vs. top-level) and shape
compatibility with `vit_large(in_chans=1, channel_adaptive=True,
patch_size=16)` remain unverified against real weights. Everything above
is verified against the real `dinov2` *source*, not a real *checkpoint* --
matching SPEC.md §10 item 1's own framing of what needed implementation-time
judgment. Revisit once a real checkpoint is available.
