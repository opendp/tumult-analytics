#!/usr/bin/env python3
"""Rewrite Tumult Labs copyright notices based on when each file was written.

Reads the table produced by ``copyright_dates.py``. For each file:

* If it was first written in or before June 2025, every notice becomes
  "Copyright Tumult Labs <year> - 2025, and the Tumult Analytics Contributors
  2025-present", where <year> is the year the file was first written. If
  <year> is 2025, the range "<year> - 2025" is just "2025".
* Otherwise, every notice becomes "Copyright the Tumult Analytics
  Contributors".

Files not in the table are left alone.

Usage:
    python scripts/update_copyright.py copyright_dates.tsv
"""

# SPDX-License-Identifier: Apache-2.0
# Copyright the Tumult Analytics Contributors

import re
import sys
from collections import Counter

# Matches both "Copyright Tumult Labs 2026" and "Copyright 2026 Tumult Labs",
# possibly followed by the end of an HTML or C-style comment. The notice must
# end the line, so notices that have already been updated are left alone and
# running this script twice is safe.
NOTICE = re.compile(
    r"Copyright (Tumult Labs [0-9]{4}|[0-9]{4} Tumult Labs)(?=( -->| \*/)?$)",
    re.MULTILINE,
)

# Files first written in or before this month keep Tumult Labs in the notice.
CUTOFF = "2025-06"


def new_notice(first_added: str) -> str:
    """Return the notice for a file first written in the given YYYY-MM."""
    if first_added <= CUTOFF:
        year = first_added[:4]
        years = "2025" if year == "2025" else f"{year}-2025"
        return (
            f"Copyright Tumult Labs {years}, "
            "and the Tumult Analytics Contributors 2025-present"
        )
    return "Copyright the Tumult Analytics Contributors"


def main(table_path: str) -> None:
    """Update notices in every file listed in the table."""
    counts: Counter[str] = Counter()
    with open(table_path) as table:
        for line in table:
            if not line.strip():
                continue
            path, first_added = line.rstrip("\n").split("\t")[:2]
            if not re.fullmatch(r"[0-9]{4}-[0-9]{2}", first_added):
                raise ValueError(f"Bad date {first_added!r} for {path}")
            notice = new_notice(first_added)
            with open(path) as f:
                text = f.read()
            text, n = NOTICE.subn(notice, text)
            if n == 0:
                print(f"WARNING {path}: no notice found", file=sys.stderr)
                continue
            with open(path, "w") as f:
                f.write(text)
            counts[notice] += 1
    for notice, count in sorted(counts.items()):
        print(f"{count:4} files: {notice}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
