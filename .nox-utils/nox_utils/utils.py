"""Utility functions."""

from nox import Session as NoxSession
from nox_poetry.sessions import Session as PoetrySession
from typing import Union


def get_session(args) -> Union[NoxSession, PoetrySession]:
    """Given an argument list, get the Nox session from it.

    This is a hack to allow using decorators interchangeably on normal functions
    and class methods, as it effectively allows skipping the `self` argument to
    class methods.
    """
    if len(args) > 0 and isinstance(args[0], (NoxSession, PoetrySession)):
        return args[0]
    elif len(args) > 1 and isinstance(args[1], (NoxSession, PoetrySession)):
        return args[1]
    else:
        raise AssertionError("Couldn't find Nox session")
