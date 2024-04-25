"""Configuration for Tumult Analytics.

This module contains various execution options for Tumult Analytics, controlling
experimental features and other behavior. Most users will not need to use the
options available in this module.
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright Tumult Labs 2024

import textwrap
from contextlib import contextmanager
from typing import Any, Optional


class FeatureFlag:
    """A flag for enabling an individual feature.

    An instance of :class:`~FeatureFlag` can be used as a bool, or
    :meth:`~raise_if_disabled` can raise an appropriate exception if the
    corresponding feature is used while the flag is disabled.
    """

    def __init__(self, summary: str, default: bool):
        """@nodoc."""
        self._summary = summary
        self._default = default
        self._enabled: Optional[bool] = None
        # _name gets filled in by the Features class when it is initialized.
        self._name: Optional[str] = None

    def __bool__(self) -> bool:
        """@nodoc."""
        return self._enabled if self._enabled is not None else self._default

    def __str__(self) -> str:
        """@nodoc."""
        return f"{self._name}: {'enabled' if self else 'disabled'}"

    def enable(self):
        """Enable the features controlled by this feature flag."""
        self._enabled = True

    def disable(self):
        """Disable the features controlled by this feature flag."""
        self._enabled = False

    def reset(self):
        """Reset this feature flag to its base state."""
        # Note: The default value is returned if self.enabled is None.
        # The error message is different if self.enabled is None vs False though.
        self._enabled = None

    @contextmanager
    def enabled(self):
        """A context manager inside which this feature flag will be enabled.

        When the context manager exits, the feature flag will be reset to the
        state it was in before entering the context manager, ignoring any
        changes that occurred inside the context.
        """
        original_state = self._enabled
        self._enabled = True
        try:
            yield
        finally:
            self._enabled = original_state

    @contextmanager
    def disabled(self):
        """A context manager inside which this feature flag will be disabled.

        When the context manager exits, the feature flag will be reset to the
        state it was in before entering the context manager, ignoring any
        changes that occurred inside the context.
        """
        original_state = self._enabled
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = original_state

    def raise_if_disabled(self):
        """Raise a RuntimeError if this feature flag is not enabled."""
        if self:
            return

        # Note that checking against False explicitly is required, as None means
        # something else but is also false-y.
        if self._enabled is False:
            raise RuntimeError(
                textwrap.dedent(
                    f"""
                    {self._summary}, and has been disabled.

                    To use this feature, you must enable the {self._name} feature flag:
                    from tmlt.analytics.config import config
                    config.features.{self._name}.enable()
                    """
                ).strip()
            )

        raise RuntimeError(
            textwrap.dedent(
                f"""
                {self._summary}, and is disabled by default.

                To use this feature, you must enable the {self._name} feature flag:
                from tmlt.analytics.config import config
                config.features.{self._name}.enable()
                """
            ).strip()
        )


class Config:
    """Configuration for Tumult Analytics."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Enforces that Config is a singleton. @nodoc."""
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    class Features:
        """Toggles for experimental features in Tumult Analytics.

        Most users should not need to modify these. Any features that are
        disabled by default are experimental and should not be used in
        production. Their APIs may change at any time.

        A particular feature can be enabled or disabled by using the methods on
        its :class:`~FeatureFlag` instance, for example:

        .. code-block::

            config.features.example_feature.enable()
        """

        # Add Feature Flags here to list them as experimental:
        # ex. "new_feature = FeatureFlag('Description of new feature', default=False)"
        auto_partition_selection = FeatureFlag(
            "Automatic partition selection is experimental", default=False
        )

        def __init__(self):
            """@nodoc."""
            attrs = {
                k: v
                for k, v in Config.Features.__dict__.items()
                if not k.startswith("_")
            }
            for k, v in attrs.items():
                if not isinstance(v, FeatureFlag):
                    raise RuntimeError(
                        "Attributes of Config.Features must be instances of FeatureFlag"
                    )
                v._name = k

        def __setattr__(self, name: str, value: Any):
            """@nodoc."""
            raise RuntimeError(
                "Features cannot be assigned to, use their enable()/disable() methods"
            )

    def __init__(self):
        """@nodoc."""
        self.features = Config.Features()


config = Config()
"""The current configuration of Tumult Analytics."""
