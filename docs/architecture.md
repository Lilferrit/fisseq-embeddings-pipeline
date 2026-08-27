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

### 5. Real checkpoint: `weights/cell_dino_vits8_pretrain_cp-37d20e9c.pth`

A real Cell-DINO checkpoint is now present locally (`weights/`, gitignored
via the repo's blanket `*.pth` rule -- not the same "no checkpoint exists
anywhere" situation the rest of this page was written against). Inspecting
its state dict directly (`torch.load(..., map_location="cpu")`) resolved
the item above **and surfaced a real architecture mismatch** the
`in_chans=1`/`channel_adaptive=True` assumption above got wrong for this
specific file:

| Property | Assumed (this page, §1 above) | This real checkpoint |
|---|---|---|
| Checkpoint key | `"teacher"` | top-level (no wrapper key at all) |
| Architecture | `vit_large`, patch 16 | `vit_small` (embed_dim 384), patch **8** |
| `in_chans` | `1` (bag of channels) | **`5`** -- a fixed 5-channel backbone |
| `channel_adaptive` | `True` | `False` (structurally -- `patch_embed.proj.weight` is `(384, 5, 8, 8)`, so each patch is convolved jointly over all 5 channels, not one at a time) |
| `pos_embed` / native `img_size` | 224 (`global_crops_size`) | 128 (`pos_embed` has 257 = 256+1 entries → 16×16 patches × patch 8) |
| `block_chunks` | unchunked (implicit) | **4** -- block keys are `blocks.<chunk 0-3>.<original position 0-11>.*`, not flat `blocks.<0-11>.*` |
| LayerScale | not considered | present (`ls1.gamma`/`ls2.gamma` on every block, non-trivial values ~1e-3 to 1e-1 -- not safely droppable) |

None of this is guessable from the filename or from `dinov2`'s public
per-variant docs alone -- `README_CELL_DINO.md` (plain, fixed-channel-count
Cell-DINO) and `README_CHANNEL_ADAPTIVE_DINO.md` (bag-of-channels) describe
two genuinely different model families, and this checkpoint (going by its
own structure) is the former, not the latter this pipeline's SPEC.md §6.3
assumed as its only supported case.

**Fix:** `load_cell_dino()` no longer hardcodes `in_chans=1`/
`channel_adaptive=True`/`block_chunks` implicit-default/no-LayerScale. It
inspects the (prefix-stripped) checkpoint state dict itself and derives:
- `in_chans` and a cross-checked `patch_size` from `patch_embed.proj.
  weight`'s shape (`(embed_dim, in_chans, patch, patch)`) -- raising a
  clear `ValueError` if `cfg.patch_size` disagrees with the checkpoint's
  own kernel size, rather than silently building the wrong grid.
- `img_size` (for a matching `pos_embed` parameter shape only -- *not* a
  claim about what crop size can be embedded later) from `pos_embed`'s
  patch-token count. This is genuinely decoupled from `cfg.crop_size`:
  `DinoVisionTransformer.interpolate_pos_encoding` already reconciles any
  difference between the checkpoint's native grid and the actual input
  size on every forward call, so construction only needs to reproduce the
  checkpoint's own shape for `load_state_dict` to succeed.
- `block_chunks` from whether block keys match the chunked
  `blocks.<chunk>.<pos>.` pattern (chunk count = `block_chunks`) or the
  flat `blocks.<pos>.` pattern (`block_chunks=0`).
- Whether to pass a (placeholder, checkpoint-overwritten) nonzero
  `init_values` at all, from whether any `ls1.gamma`/`ls2.gamma` key
  exists in the checkpoint.

`embed_batch()` correspondingly branches on the *loaded model's own*
`patch_embed.in_chans` rather than assuming bag-of-channels
unconditionally: `in_chans == 1` still gets the original per-channel
split-and-pool treatment; anything else is fed to the model jointly in one
plain forward pass (`model(crops)`, no split, no `channel_pool`), raising
a clear error if the crop's channel count doesn't match exactly.

One more correction this forced: the original code's `model.load_state_
dict(state, strict=False)` only ever *logged* `missing_keys`/
`unexpected_keys`, on the theory that a backbone-only checkpoint
legitimately lacks head/EMA keys. That's true for `unexpected_keys`
(checkpoint has keys the model doesn't need), but before this fix, the
old `in_chans=1`/no-`block_chunks` construction against *this* checkpoint
would have produced **9 of 12 transformer blocks worth of `missing_keys`**
(only chunk 0's 3 blocks would have keys the un-chunked model recognized)
-- silently leaving three-quarters of the backbone at its random
initialization while `embed_batch()` ran anyway, no crash, no warning
above INFO level, just quietly wrong embeddings. `load_cell_dino()` now
raises a `RuntimeError` on any non-empty `missing_keys` (never on
`unexpected_keys`, which stays informational) -- since this pipeline's own
`head` is always `nn.Identity()` (no parameters that could ever
legitimately be missing), any missing key really does mean the
constructed architecture doesn't match the checkpoint.

Verified end to end against the real file: `load_cell_dino()` loads it
with **zero** missing keys and zero unexpected keys once `arch="vit_small"`/
`patch_size=8` are set; `embed_batch()` on a synthetic 5-channel, 128×128
crop batch returns finite, non-degenerate `(B, 384)` embeddings (mean/std
per dimension well within a normal transformer's output range -- not NaN,
not exploding). See `tests/unit/test_embed.py::
test_load_cell_dino_and_embed_batch_against_real_vits8_checkpoint` (skipped
automatically when the checkpoint file isn't present, e.g. in CI) and the
synthetic-shape regression tests alongside it that lock in each inferred
property (`test_load_cell_dino_infers_chunked_blocks_and_layerscale`,
`test_load_cell_dino_infers_joint_multichannel_in_chans`) independent of
the real file's availability.

### 6. Second real checkpoint: `weights/channel_adaptive_dino_vitl16_pretrain_cells-ef7c17ff.pth`

A second real checkpoint was added later. Unlike the one above, **this one
matches SPEC.md §6.3's originally-assumed shape exactly** -- inspecting its
state dict directly confirms:

| Property | SPEC.md's assumption | This checkpoint |
|---|---|---|
| Checkpoint key | `"teacher"` | top-level (no wrapper key) |
| Architecture | `vit_large` (embed_dim 1024), patch 16 | matches |
| `in_chans` / `channel_adaptive` | `1` / `True` (bag of channels) | matches (`patch_embed.proj.weight` is `(1024, 1, 16, 16)`) |
| `pos_embed` / native `img_size` | 224 | matches (197 = 196+1 → 14×14 patches × patch 16 = 224) |
| `block_chunks` | not specified | 4 (24 blocks / 4 chunks = 6 each) |
| LayerScale | not specified | present |

`load_cell_dino()` loads this file with zero missing/zero unexpected keys
using `EmbedCellsConfig`'s existing defaults (`arch="vit_large",
patch_size=16, crop_size=224`) completely unchanged -- no code correction
was needed for this file the way the first one needed one; the
already-generalized checkpoint-introspection logic (§5 above) simply
handles it, since a from-scratch construction path built specifically for
*this* checkpoint's shape would have looked identical to what already
existed. See `tests/unit/test_embed.py::
test_load_cell_dino_and_embed_batch_against_real_vitl16_checkpoint`.

### 7. Configurable input channels and per-channel masking

Added alongside the second checkpoint: `EmbedCellsConfig.channels` (a
`list[int]`, default `[0, 1, 2, 3]`) selects and orders which of the
crop's channel indices actually get embedded -- a crop may legitimately
carry more channels than the model should see (e.g. multiple imaging
cycles, per `dataset.py`'s cycle-major flattening), and different
checkpoints/experiments may want a different subset or order.
`EmbedCellsConfig.channel_apply_mask` (a `list[bool]`, same length as
`channels`, same default-length default `[True, True, True, True]`)
independently controls, per selected channel, whether that channel gets
the shared per-cell segmentation mask applied before embedding. "Shared"
matters here: `BUILD_DATASET` (§6.1) writes exactly one `mask.npy` per
cell, not one per channel, so this is "apply the one mask to this channel
or not," not a claim that different channels carry their own distinct
masks -- the pipeline's data model has no such thing today.

This replaces the earlier single `mask_mode: "none" | "zero_background"`
enum (SPEC.md §6.3's original sketch) -- a plain boolean split whether
*any* masking happened at all, uniformly across every channel.
`channel_apply_mask` subsumes both of its states (all-`False` ≡ `"none"`,
all-`True` ≡ `"zero_background"`) while additionally allowing a per-channel
mix, so keeping both knobs around would have meant two overlapping,
occasionally-contradictory ways to say the same thing.

`embed_batch()` applies channel selection and per-channel masking *before*
the bag-of-channels-vs-joint-multichannel branch (§3 above) -- so, for a
bag-of-channels model, `channel_pool` only ever pools over the selected
channels, and for a fixed-channel-count model, `len(cfg.channels)` must
equal that model's `in_chans` exactly (channel selection happens here in
Python, not upstream in `dataset.py`, so a crop with more raw channels
than the model needs is completely normal, not an error).

**Not resolved by this work:** whether the `vit_small`/patch-8/5-channel
checkpoint (§5 above) or this `vit_large`/patch-16/bag-of-channels one is
the pipeline's actual intended production checkpoint is a product
decision, not a code question -- `EmbedCellsConfig.checkpoint_path` and
`params.yaml`'s `cell_dino_checkpoint` are deliberately left required-with-
no-default (SPEC.md §9.1's Resolved note): a checkpoint path is inherently
deployment-specific, and `weights/` is gitignored, so hardcoding either
local path as *the* default would silently be wrong (or simply absent) in
any other checkout. `load_cell_dino()`/`embed_batch()` now work correctly
with either real checkpoint shape; picking which one a real run should
point at, and what `channels`/`channel_apply_mask` that run should use, is
left to whoever configures it.
