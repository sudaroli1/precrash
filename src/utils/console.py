"""
Console output that cannot crash on a Windows terminal.

WHY THIS EXISTS
---------------
`print()` encodes with the console's codec. On Windows that is cp1252 by
default, which has no arrow, no delta and no em dash. A table header containing
"rank L->C" written with a real arrow therefore raises

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'

*after* the first rows have already been printed, so the run dies partway
through a table that looked fine a moment earlier. This happened on the first
local run of this repository.

Two rules follow, and both are applied here rather than left to discipline:

  1. Anything that reaches a console is ASCII. Alignment matters in a table, and
     a replacement character is as bad as a crash if it shifts a column.
  2. Even so, encoding is never allowed to raise. A stray non-ASCII character in
     a filename, a caption from the corpus, or an exception message must not end
     a long run.

Files written to disk are a separate matter: those are opened with an explicit
`encoding="utf-8"`, so the markdown tables keep their arrows and dashes.
"""

from __future__ import annotations

import sys


def safe_console() -> None:
    """Make stdout and stderr survive characters the console codec lacks.

    Idempotent, and a no-op on Python versions or streams without
    `reconfigure`. Call once at the start of `main()`.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
