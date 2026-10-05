#!/usr/bin/env python3
"""Find when each file with a Tumult Labs copyright notice was first written.

Writes a tab-separated table with one row per file:

    path    first_added (YYYY-MM)    commit    message    note

The first-added date follows the file across renames and moves (using
``git log --follow``). Git's rename detection misses some moves, such as one
file being split into several, or code being moved into a new file while
it is rewritten. To catch these, if a new file shares enough lines with a
file deleted in the same commit (ignoring imports and comments), the date is
taken from that deleted file instead (recursively), and the ``note`` column
says so. The ``message`` column is the full message of the commit that first
added the file, with whitespace (including newlines) collapsed to single
spaces so it fits on one line. Review the table, edit ``first_added`` by hand if needed, then pass
it to ``update_copyright.py``.

Usage:
    python scripts/copyright_dates.py > copyright_dates.tsv
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright the Tumult Analytics Contributors

import re
import subprocess
import sys
from functools import cache

# Matches both "Copyright Tumult Labs 2026" and "Copyright 2026 Tumult Labs",
# possibly followed by the end of an HTML or C-style comment. The notice must
# end the line, so notices that have already been updated don't match.
NOTICE_PATTERN = r"Copyright (Tumult Labs [0-9]{4}|[0-9]{4} Tumult Labs)( -->| \*/)?$"

# Number of lines a new file must share with a deleted file for the new file
# to be treated as (partly) moved from it. In this repository's history, real
# splits share 6 or more lines and unrelated files share at most 3.
MIN_SHARED_LINES = 5

# Lines shorter than this (after stripping) are too generic to compare.
MIN_LINE_LENGTH = 20

# Lines too common to show that content was moved: imports, lists of imported
# names, closing parentheses, and comments (including license headers).
BOILERPLATE = re.compile(r"^(from \S+ import|import \S|#|\.\.|[\w.]+,?$|\)$)")


def git(*args: str) -> str:
    """Run a git command and return its stdout."""
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def files_with_notice() -> list[str]:
    """Return tracked files containing a Tumult Labs copyright notice."""
    result = subprocess.run(
        ["git", "grep", "-lE", NOTICE_PATTERN],
        check=False,
        capture_output=True,
        text=True,
    )
    # git grep exits 1 when there are no matches.
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return result.stdout.split()


@cache
def content_lines(rev: str, path: str) -> frozenset[str]:
    """Return the distinctive lines of a file at a revision."""
    try:
        text = git("show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return frozenset()
    return frozenset(
        line.strip()
        for line in text.splitlines()
        if len(line.strip()) >= MIN_LINE_LENGTH and not BOILERPLATE.match(line.strip())
    )


@cache
def deleted_in(commit: str) -> tuple[str, ...]:
    """Return files deleted (not renamed) by the given commit."""
    status = git("show", "--format=", "--name-status", "-M", commit)
    deleted = []
    for line in status.splitlines():
        fields = line.split("\t")
        if fields[0] == "D":
            deleted.append(fields[1])
    return tuple(deleted)


def git_first_added(path: str, rev: str) -> tuple[str, str, str]:
    """Return (YYYY-MM, commit, path then) for the commit that added a file.

    Follows renames that git detects, starting from ``path`` at ``rev``.
    """
    log = git(
        "log",
        "--follow",
        "--diff-filter=A",
        "--date=format:%Y-%m",
        "--format=%x00%ad %h",
        "--name-only",
        rev,
        "--",
        path,
    )
    entries = [entry.split() for entry in log.split("\0") if entry.strip()]
    if not entries:
        raise RuntimeError(f"No commit found adding {path} at {rev}")
    date, commit, path_then = entries[-1]
    return date, commit, path_then


def commit_message(commit: str) -> str:
    """Return a commit's full message, joined onto one line."""
    message = git("log", "-1", "--format=%B", commit)
    return " ".join(message.split())


def moved_from(commit: str, path: str) -> tuple[str, int] | None:
    """Return the deleted file this new file's content came from, if any.

    Returns the deleted file sharing the most lines with the new file, and
    how many lines they share.
    """
    new_lines = content_lines(commit, path)
    best, best_shared = None, 0
    for old_path in deleted_in(commit):
        shared = len(new_lines & content_lines(f"{commit}^", old_path))
        if shared > best_shared:
            best, best_shared = old_path, shared
    if best is None or best_shared < MIN_SHARED_LINES:
        return None
    return best, best_shared


def first_added(path: str) -> tuple[str, str, str]:
    """Return (YYYY-MM, commit, note) for when this file was first written."""
    date, commit, path_then = git_first_added(path, "HEAD")
    moves = []
    while (moved := moved_from(commit, path_then)) is not None:
        source, shared = moved
        moves.append(
            f"{path_then} shares {shared} lines with {source}, deleted in {commit}"
        )
        date, commit, path_then = git_first_added(source, f"{commit}^")
    return date, commit, "; ".join(moves)


def main() -> None:
    """Print the first-added table for all files with a notice."""
    for path in files_with_notice():
        date, commit, note = first_added(path)
        message = commit_message(commit)
        print(f"{path}\t{date}\t{commit}\t{message}\t{note}")
        if note:
            print(f"MOVED {path}: {note}", file=sys.stderr)


if __name__ == "__main__":
    main()
