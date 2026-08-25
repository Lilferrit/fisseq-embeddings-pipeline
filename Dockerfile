# Dockerfile -- single image every Nextflow process runs in (SPEC.md §9.2 /
# §3 decision 13). One CUDA-capable base serves CPU-only stages too (simpler
# to build/publish as one artifact); revisit splitting into a CPU + GPU image
# later if the pull cost matters in practice -- see IMPLEMENTATION_CHECKLIST.md
# Epic 10.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fisseq-embeddings-pipeline
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --frozen

COPY src/ src/
RUN uv pip install -e .

# TODO(Epic 3, SPEC.md §6.3): vendor/install dinov2 here once its exact
# construction path is verified against a real checkpoint -- it is not on
# PyPI (see pyproject.toml's torch/dinov2 comment).

ENTRYPOINT []
