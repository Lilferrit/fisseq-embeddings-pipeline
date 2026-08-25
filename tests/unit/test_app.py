"""Tests for AppConfig (SPEC.md §3, IMPLEMENTATION_CHECKLIST.md Epic 0 Story 0.2).

AppConfig itself needed no code change from the scaffold -- it already
matches SPEC.md's dataclass exactly. This module covers the story's
acceptance criterion: a ConfigStore-registered stage config's default
random_seed is 0 and is overridable via Hydra CLI override syntax (SPEC.md
§3 decision 11 -- one shared seed field, reused by every stage).
"""

from __future__ import annotations

import dataclasses

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from fisseq_embeddings_pipeline.config import AppConfig


@dataclasses.dataclass
class _DummyStageConfig(AppConfig):
    """A minimal stage config, standing in for a real Epic 1+ stage config,
    to confirm AppConfig's random_seed default/override behavior end to end
    through Hydra's own config-composition machinery rather than just
    dataclass defaults."""

    pass


def test_appconfig_default_random_seed_is_zero():
    assert AppConfig().random_seed == 0


def test_dummy_stage_config_inherits_random_seed_default():
    assert _DummyStageConfig(output_dir="/tmp/out").random_seed == 0


def test_random_seed_overridable_via_omegaconf_merge():
    """Stands in for a Hydra CLI override (`random_seed=42`) -- OmegaConf.merge
    is what Hydra's CLI-override parsing ultimately calls into."""
    base = OmegaConf.structured(_DummyStageConfig(output_dir="/tmp/out"))
    overridden = OmegaConf.merge(base, {"random_seed": 42})
    assert overridden.random_seed == 42
    assert base.random_seed == 0  # base config untouched


def test_random_seed_overridable_via_hydra_compose():
    """The real end-to-end path: a ConfigStore-registered node, composed via
    Hydra with a CLI-override-style string, the way every stage's `main()`
    actually runs."""
    cs = ConfigStore.instance()
    cs.store(name="dummy_stage_test", node=_DummyStageConfig)

    with initialize(version_base=None, config_path=None):
        cfg = compose(
            config_name="dummy_stage_test",
            overrides=["random_seed=42", "output_dir=/tmp/out"],
        )
    assert cfg.random_seed == 42
