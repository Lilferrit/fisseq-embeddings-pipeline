# testing_data/

Gitignored contents (this file and the directory itself are the only
tracked things here -- see `.gitignore`).

## `lmna_t3/`

A shrunk fixture derived from the Fowler lab's public LMNA_T3
starcall-workflow testing image set
(`https://visseq.gs.washington.edu/static/LMNA_T3_testing_image_set.tar.gz`),
used by `tests/integration/test_integration_real_starcall.py` -- the one
integration test that invokes a **real** `snakemake` run against real
starcall-workflow data (every other integration test fakes that step with
a stub `snakemake` on PATH; see `AGENTS.md`/`docs/architecture.md` for why
that gap existed).

Generate it with:

```bash
uv run python scripts/prepare_real_starcall_test_data.py
```

This downloads the ~6.3GB source tarball (cached under
`_download_cache/`, resumable if interrupted), then crops it down to a
single tile per sequencing cycle (see that script's own docstring for
exactly how and why) into `lmna_t3/starcall_input/`, well under 1GB.

`test_integration_real_starcall.py` skips automatically if this hasn't
been generated (or if Docker isn't available) -- it's opt-in, not run by
default, since it needs the real `ops`-env-bearing Docker image and takes
meaningfully longer than the rest of the suite.
