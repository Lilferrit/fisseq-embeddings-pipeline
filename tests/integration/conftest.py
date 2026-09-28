"""Integration-suite mode switch.

`tests/integration/test_integration.py` holds two kinds of end-to-end test
that cannot run in the same invocation, so a CLI flag picks between them:

- **default** (`pytest tests/integration`): the synthetic suite, driving
  real `nextflow run` subprocesses under `-profile local` -- no container,
  no Docker image build, `snakemake` faked by a stub prepended onto PATH.
  Fast enough for every PR, and what CI runs.
- **`--container`**: the real-starcall test, driving the pipeline through
  the default (containerized) profile against real microscopy data, with a
  real `snakemake` inside a real build of the root Dockerfile.

They are mutually exclusive rather than additive because the synthetic
fixture's stub `snakemake` stops applying in container mode: under the
default profile `process.ext.snakemake_bin` is the absolute in-image
`/opt/conda/envs/ops/bin/snakemake` (nextflow.config), and the host's
`stub_bin` directory isn't bind-mounted into the task container, so those
tests would silently invoke the real, heavy `ops`-env Snakemake. Running
the synthetic suite containerized would need that stub solved first;
until then `--container` means "the real-starcall test instead", not "as
well".

The `container` marker (registered in pyproject.toml) is the mechanism.
Selecting by marker here, rather than a gate inside each test, also
centralizes the `nextflow`-on-PATH check that used to be repeated
verbatim in seven places.
"""

from __future__ import annotations

import shutil

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--container",
        action="store_true",
        default=False,
        help=(
            "run the containerized real-starcall integration test (needs "
            "docker + a populated testing_data/) instead of the synthetic "
            "-profile local suite"
        ),
    )


def pytest_collection_modifyitems(config, items):
    container_mode = config.getoption("--container")
    no_nextflow = shutil.which("nextflow") is None

    skip_nextflow = pytest.mark.skip(
        reason="nextflow not on PATH -- see tests/integration/test_integration.py"
    )
    skip_synthetic = pytest.mark.skip(
        reason="--container given: running the real-starcall test instead of "
        "the synthetic -profile local suite"
    )
    skip_container = pytest.mark.skip(
        reason="real-starcall test is opt-in -- pass --container to run it"
    )

    for item in items:
        wants_container = "container" in item.keywords
        if no_nextflow:
            item.add_marker(skip_nextflow)
        elif wants_container and not container_mode:
            item.add_marker(skip_container)
        elif container_mode and not wants_container:
            item.add_marker(skip_synthetic)
