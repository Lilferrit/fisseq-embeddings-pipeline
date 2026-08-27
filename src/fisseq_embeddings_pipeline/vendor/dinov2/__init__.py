"""Minimal vendored subset of facebookresearch/dinov2.

Only the pure-``torch`` model-definition code needed to construct a
channel-adaptive ViT and run inference is vendored here -- not the full
``dinov2`` package (training loops, data pipelines, eval harness). See
``VENDORED_FROM.md`` in this directory for the exact commit, file list, and
why this is vendored rather than installed as a dependency.
"""
