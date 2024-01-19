import datetime
import os
import re
import subprocess
import tempfile
from functools import wraps
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

from .environment import in_ci
from .utils import get_session

def install(*decorator_args, **decorator_kwargs):
    """Install packages into the test virtual environment.

    Installs one or more given packages, if the current environment supports
    installing packages. Parameters to the decorator are passed directly to
    nox's session.install, so anything valid there can be passed to the
    decorator.

    The difference between using this decorator and using a normal
    session.install call is that this decorator will automatically skip
    installation when nox is not running tests in a virtual environment, rather
    than raising an error. This is helpful for writing sessions that can be used
    either in sandboxed environments in the CI or directly in developers'
    working environments.
    """

    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            session = get_session(args)
            if session.virtualenv.is_sandboxed:
                session.install(*decorator_args, **decorator_kwargs)
            else:
                session.log("Skipping package installation, non-sandboxed environment")
            return f(*args, **kwargs)

        return inner

    return decorator


def show_installed(f):
    """Show a list of installed packages in the active environment for debugging.

    By default, the package list is only shown when running in the CI, as that
    is where it is most difficult to debug. However, the show_installed option
    can be passed to any function with this decorator to force showing or not
    showing it.
    """

    @wraps(f)
    def inner(*args, show_installed: Optional[bool] = None, **kwargs):
        session = get_session(args)
        show_installed = show_installed if show_installed is not None else in_ci()
        if show_installed:
            session.run("pip", "freeze")
        return f(*args, **kwargs)

    return inner
