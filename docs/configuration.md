# Configuration reference

Placeholder -- see SPEC.md §6 (per-stage Hydra configs) and params.yaml
(repo root, every default parameter) until this is written for real
post-implementation, epic by epic. This page currently has real content for
the pieces Epic 1 and Epic 10 resolved: `BUILD_DATASET`'s WebDataset shard
sizing, and the Docker image's versioning/publishing policy.

## Docker image versioning & publishing (SPEC.md §9.2's Resolved note, Epic 10 Story 10.2)

SPEC.md deliberately left this undecided ("out of scope for the design spec
itself, revisit alongside CI setup"). Decided here, implemented in
`.github/workflows/docker.yml`:

- **Registry:** GitHub Container Registry, `ghcr.io/<owner>/<repo>` (derived
  from the repo's own `${{ github.repository }}` at build time, not
  hardcoded) -- matches SPEC.md §9.2's own `nextflow.config` sketch, and
  needs no separate registry account since this is a GitHub-hosted repo.
- **Tags, on every push to `master`:** `:latest` (moving -- convenience/dev
  use) and `:<short-sha>` (exact, 7-char commit SHA -- what
  `params.yaml`'s `container_image` should point at for anything that
  needs to pin a specific build instead of floating on `:latest`, e.g. a
  reproducibility-sensitive run).
- **Tags, on a pushed `v*` git tag** (a real release): additionally
  `:<version>` (the tag with its `v` prefix stripped, e.g. `v0.1.0` ->
  `0.1.0`) -- not tied to `pyproject.toml`'s own `version` field
  automatically; bump that field and push a matching `vX.Y.Z` tag together
  when cutting a release.
- **Every PR:** build-only, no push, no registry credentials needed -- a
  smoke test against Dockerfile regressions. This exact check would have
  caught two real bugs found (and fixed) while writing this Dockerfile
  before any real `docker build`/`docker run` pass ever ran against it (see
  the Epic 10 commit message: `uv sync`'s dependencies-only layer needing
  `--no-install-project`, and the missing `.venv/bin` `PATH` entry every
  `modules/local/*.nf` script block's bare `python -m
  fisseq_embeddings_pipeline.<module>` call depends on).

**Still open, not resolved by this decision:**

- **Not build-verified in this sandbox at all** -- there is no Docker
  daemon available here (`dockerd` fails to start even under `sudo`: no
  `CAP_NET_ADMIN`/iptables access, unlike Epic 9's `nextflow`/`java` gap,
  which a plain package install fixed). Everything below the
  `apt-get`/CUDA base-image layer in the `Dockerfile` was instead verified
  by reproducing its exact `COPY`/`RUN` ordering by hand against a scratch
  directory with `uv` (already installed in this sandbox) -- real, but not
  a substitute for an actual `docker build`.
- **GPU stage unverified on a real GPU host** -- `docker run --gpus all` /
  `nvidia-smi` inside the built image, and `EMBED_CELLS` actually running
  against a real checkpoint inside the container, both still need a real
  GPU host + Docker, neither available here.
- **`.github/workflows/docker.yml` is inert until this repo has a remote**
  (same situation as `.github/workflows/ci.yml` -- AGENTS.md's Git workflow
  section) -- added now so the decision above is in place the moment one
  exists, not run anywhere yet.
- **CPU-only image split** -- explicitly deferred, not required for v1
  (SPEC.md §9.2's own note); revisit once real image-pull cost at
  CPU-only-stage runtime is actually measured.

## `BUILD_DATASET` shard sizing (SPEC.md §10, item 2)

`shard_maxcount` (default `2000`, `BuildDatasetConfig.shard_maxcount`)
controls how many cells `write_dataset_shards()` packs into each
`dataset-*.tar` shard. No real experiment was available to measure this
against directly in this environment (only the pipeline source repos are
mounted, not phenotyping data) -- the number below is a computed estimate
from real defaults elsewhere in the stack, not a real measurement, and
should be re-checked against one real experiment's actual `stitch_tile_pt`
output byte size once available.

`BUILD_DATASET` crops `crop.npy`/`mask.npy` itself from `stitch_tile_pt`'s
stitched tile image and `stitch_tile_from_well_segmentation`'s mask (SPEC.md
§5.2/§6.1), rather than reading a pre-cropped file, but the estimate below
is unaffected by that -- it's the same underlying phenotype image either
way, only the file it's cropped from changed.

**Inputs to the estimate:**

- Channel count: **4**, from `starcall-workflow`'s (`origin/devel`)
  `default-config.yaml`: `phenotyping_channels: ['DAPI', 'GFP', 'Ph+WGA', 'Mito']`.
  This assumes the default single phenotype cycle (`phenotype_cycles: ['PT']`)
  -- `crop.npy`'s actual channel dimension is
  `num_phenotyping_cycles × num_channels` (cycle-major flattened), so a
  deployment configuring more than one phenotyping cycle scales this
  estimate proportionally and should re-check `shard_maxcount`.
- Crop window: **224** (`window`/`crop_size`), from Cell-DINO's
  channel-adaptive eval config (`global_crops_size: 224`, SPEC.md §6.3).
- Crop dtype: **uint16**, the standard bit depth for fluorescence
  microscopy TIFFs -- assumed, not confirmed against a real
  `raw_pt.tif`/`corrected_pt.tif` (`stitch_tile_pt` writes the stitched
  image in the source phenotype image's own dtype, whatever that turns out
  to be for a real acquisition).
- Mask dtype: **uint8** label mask (fixed -- `_crop_cell()` always writes
  `uint8`, not memory-mappable `bool`, matching `make_cell_images`'s own
  convention for the same reason).

**Per-sample size:**

| Component | Formula | Size |
| --- | --- | --- |
| `crop.npy` | 4 × 224 × 224 × 2 bytes | ≈ 392 KB |
| `mask.npy` | 224 × 224 × 1 byte | ≈ 49 KB |
| `meta.json` + tar per-file headers (3 files/sample) | -- | a few KB |
| **Total** | | **≈ 440 KB/sample** |

**Per-shard size** at the default `shard_maxcount=2000`:
440 KB × 2000 ≈ **~880 MB/shard** -- within the "hundreds of MB to ~1GB"
band `SPEC.md`/`IMPLEMENTATION_CHECKLIST.md` call reasonable for a
webdataset shard, so **the `2000` default is kept as-is**.

**Still open:** re-run this estimate (or better, measure directly) once a
real experiment's `phenotyping_dir` is available -- channel count, crop
dtype, and window are all assumption-flagged above and could shift the
number meaningfully (e.g. a channel count above ~9 or a `uint32`/float
crop dtype would push a shard past 1GB at the current default).
