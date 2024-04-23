from functools import wraps
import os, subprocess
import tempfile
from .utils import get_session



def in_ci() -> bool:
    """Return whether nox is running in a CI pipeline."""
    return bool(os.environ.get("CI"))


def with_clean_workdir(f):
    """If in a sandboxed virtualenv, execute session from an empty tempdir.

    This decorator works around an issue with the tests where they will try to
    use the code (and thus the shared libraries) from the repository rather than
    the wheel that should be used. By moving to a temporary directory before
    running the tests, the repository is not in the Python load path, so the
    problem is resolved.
    """

    @wraps(f)
    def inner(*args, **kwargs):
        session = get_session(args)
        if session.virtualenv.is_sandboxed:
            with tempfile.TemporaryDirectory() as workdir, session.cd(workdir):
                return f(*args, **kwargs)
        else:
            return f(*args, **kwargs)

    return inner
