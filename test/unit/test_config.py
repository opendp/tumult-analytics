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


# Adding feature flags for use in the tests is necessary because the collection
# of feature flags existing at any given time is not stable. Unfortunately doing
# so makes mypy and pylint very unhappy, so we're ignoring errors related to the
# existence of an attribute on a class for the rest of this file.

# mypy: disable-error-code=attr-defined
# pylint: disable=no-member


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
    """Feature flags have expected defaults and can be enabled/disabled/reset."""
    cfg = Config()

    assert not cfg.features.ff1
    assert cfg.features.ff2

    cfg.features.ff1.enable()
    cfg.features.ff2.disable()
    assert cfg.features.ff1
    assert not cfg.features.ff2

    cfg.features.ff1.reset()
    cfg.features.ff2.reset()
    assert not cfg.features.ff1
    assert cfg.features.ff2


@pytest.mark.usefixtures("_with_example_features")
def test_config_feature_flag_context_managers():
    """Feature flag context managers work as expected."""
    cfg = Config()

    assert not cfg.features.ff1
    assert cfg.features.ff2

    with cfg.features.ff1.enabled():
        assert cfg.features.ff1
        with cfg.features.ff1.disabled():
            assert not cfg.features.ff1
        assert cfg.features.ff1
        cfg.features.ff1.disable()
        assert not cfg.features.ff1
        cfg.features.ff1.enable()
        assert cfg.features.ff1
    assert not cfg.features.ff1


@pytest.mark.usefixtures("_with_example_features")
def test_config_feature_flag_raise_if_disabled():
    """Feature flags' raise_if_disabled raises when expected."""
    cfg = Config()

    with pytest.raises(RuntimeError):
        cfg.features.ff1.raise_if_disabled()
    cfg.features.ff2.raise_if_disabled()

    cfg.features.ff1.enable()
    cfg.features.ff2.disable()

    cfg.features.ff1.raise_if_disabled()
    with pytest.raises(RuntimeError):
        cfg.features.ff2.raise_if_disabled()


@pytest.mark.usefixtures("_with_example_features")
def test_config_feature_flag_raise_if_disabled_snippet():
    # pylint: disable=protected-access
    """Feature flags' raise_if_disabled produces example code that enables flag."""
    cfg = Config()

    # Extract the error message from raise_if_disabled(), find the code
    # snippet to enable the flag, and then exec it and check that the flag
    # actually gets enabled.
    for ff in (cfg.features.ff1, cfg.features.ff2):
        ff.disable()
        with pytest.raises(RuntimeError) as exc_info:
            ff.raise_if_disabled()
        error_message = str(exc_info.value)
        enable_snippet_idx = error_message.find("from tmlt.analytics")
        assert (
            enable_snippet_idx != -1
        ), "No snippet to enable flag found in exception message"
        enable_snippet = error_message[enable_snippet_idx:]
        with patch("tmlt.analytics.config.config", cfg):
            exec(enable_snippet, {}, {})  # pylint: disable=exec-used
        assert ff, f"Flag {ff._name} did not get set by snippet from exception message"
        ff.disable()
