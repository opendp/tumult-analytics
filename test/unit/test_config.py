"""Tests for :mod:`tmlt.analytics.config`."""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

from unittest.mock import patch

import pytest

import tmlt.analytics.config as config_module
from tmlt.analytics.config import Config, FeatureFlag, config


def test_config_singleton():
    """Verify that the config object acts as a singleton."""
    assert Config() is Config()
    assert config is Config()


@pytest.fixture
def _with_example_features():
    # pylint: disable=protected-access
    """Add some example feature flags for testing."""

    class _Features(Config.Features):
        ff1 = FeatureFlag("Flag1", default=False)
        ff2 = FeatureFlag("Flag2", default=True)

    orig_features = Config.Features
    orig_instance = Config._instance
    Config.Features = _Features  # type: ignore[misc]
    Config._instance = None

    yield

    Config.Features = orig_features  # type: ignore[misc]
    Config._instance = orig_instance

    # Ensure that we haven't messed up anything visible outside the test using
    # this fixture.
    for ff in ("ff1", "ff2"):
        assert not hasattr(config.features, ff)
        assert not hasattr(config_module.config.features, ff)
        assert not hasattr(Config().features, ff)


@pytest.mark.usefixtures("_with_example_features")
def test_config_feature_flag_values():
    """Feature flags have expected defaults and can be enabled/disabled."""
    cfg = Config()

    assert not cfg.features.ff1  # type: ignore[attr-defined]
    assert cfg.features.ff2  # type: ignore[attr-defined]

    cfg.features.ff1 = True
    cfg.features.ff2 = False

    assert cfg.features.ff1  # type: ignore[attr-defined]
    assert not cfg.features.ff2  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_with_example_features")
def test_config_feature_flag_raise_if_disabled():
    # pylint: disable=no-member
    """Feature flags' raise_if_disabled raises when expected."""
    cfg = Config()

    with pytest.raises(RuntimeError):
        cfg.features.ff1.raise_if_disabled()  # type: ignore[attr-defined]
    cfg.features.ff2.raise_if_disabled()  # type: ignore[attr-defined]

    cfg.features.ff1 = True
    cfg.features.ff2 = False

    cfg.features.ff1.raise_if_disabled()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        cfg.features.ff2.raise_if_disabled()  # type: ignore[attr-defined]


@pytest.mark.usefixtures("_with_example_features")
def test_config_feature_flag_raise_if_disabled_snippet():
    # pylint: disable=no-member,protected-access
    """Feature flags' raise_if_disabled produces example code that enables flag."""
    cfg = Config()
    cfg.features.ff1 = False
    cfg.features.ff2 = False

    # Extract the error message from raise_if_disabled(), find the code
    # snippet to enable the flag, and then exec it and check that the flag
    # actually gets enabled.
    for ff in (cfg.features.ff1, cfg.features.ff2):  # type: ignore[attr-defined]
        assert not ff
        with pytest.raises(RuntimeError) as exc_info:
            ff.raise_if_disabled()  # type: ignore[attr-defined]
        error_message = str(exc_info.value)
        enable_snippet_idx = error_message.find("from tmlt.analytics")
        assert (
            enable_snippet_idx != -1
        ), "No snippet to enable flag found in exception message"
        enable_snippet = error_message[enable_snippet_idx:]
        with patch("tmlt.analytics.config.config", cfg):
            exec(enable_snippet, {}, {})  # pylint: disable=exec-used
        assert ff, (
            f"Flag {ff._name} did not get set "  # type: ignore[attr-defined]
            "by snippet from exception message"
        )
