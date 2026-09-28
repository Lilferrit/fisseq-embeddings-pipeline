# testing_data/

Gitignored contents (this file and the directory itself are the only
tracked things here -- see `.gitignore`).

## `lmna_t3_mini/` -- what `--container` runs on

A tiny slice of the Fowler lab's public LMNA_T3 starcall-workflow testing
image set
(`https://visseq.gs.washington.edu/static/LMNA_T3_testing_image_set.tar.gz`),
used by `tests/integration/test_integration.py`'s `--container` tests --
the ones that run starcall-workflow **for real** (background correction,
stitching, stardist/cellpose segmentation, base calling) inside the
pipeline image. Every other integration test fakes that step with a stub
`snakemake` on `PATH`.

Generate it with:

```bash
uv run python scripts/prepare_real_starcall_test_data.py --minimal
```

That builds `lmna_t3/` first if it isn't there yet (below), then keeps all
12 sequencing cycles (the barcodes are 12 bases long) of one sequencing
tile plus one phenotype tile, every cycle cropped to 512 px -- about 40MB,
into `lmna_t3_mini/starcall_input/`. The crops are aligned by phase
correlation so starcall's initial stitching layout (tile index x tile size)
is already right; tiny unaligned crops never register, and the stitched
phenotype tile balloons to gigabytes. See `crop_to_minimal`. `tests/integration/fixtures/lmna_t3_mini_config.yaml`
is the matching starcall config (all grid sizes 1, so exactly one tile,
`tile00x00y`).

The `--container` tests are opt-in twice over: they run only under
`uv run pytest tests/integration --container`, and even then skip unless
this fixture exists and `docker` is on `PATH`. The first run builds the
root `Dockerfile` (the slow part); set `FISSEQ_TEST_IMAGE=<tag>` to reuse
an image you've already built. Both tests stop after `EMBED_CELLS`
(`--embeddings_only true`), share one starcall run between them, and take
minutes rather than the hour-plus the old full-size fixture needed:

- `test_real_starcall_local` -- the nested starcall run in local mode.
- `test_real_starcall_profile_mode` -- the same through a throwaway
  `starcall_profile` whose "cluster" backgrounds each jobscript, with a
  fake container runtime standing in for `apptainer`. That exercises real
  snakemake 7 profile/`--jobscript` handling and the jobscript's bind
  list; only a real cluster can check `apptainer` itself re-entering the
  `.sif` on a node.

The starcall-workflow `origin/devel` checkout the tests run against is
cloned once into `_starcall_workflow_checkout/` (or reused from
`lmna_t3/_starcall_workflow_checkout/` if an older run left one there).

## `lmna_t3/`

The intermediate the minimal fixture is cut from: the source tarball's
`well1_subset3` cropped to a single tile per sequencing cycle, into
`lmna_t3/starcall_input/` (well under 1GB). Built by the same script
without `--minimal`:

```bash
uv run python scripts/prepare_real_starcall_test_data.py
```

This downloads the ~6.3GB source tarball (cached under
`_download_cache/`, resumable if interrupted); see the script's own
docstring for exactly how and why it crops. Nothing runs on `lmna_t3/`
directly any more.
