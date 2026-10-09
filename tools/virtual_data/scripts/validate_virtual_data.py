#!/usr/bin/env python3
"""Validate a virtual-data output directory (exit code 1 on failure)."""

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "taehyeon"))
import team_paths  # noqa: E402

team_paths.bootstrap()

from virtual_data.validation import validate_output  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--recheck-rows", type=int, default=30)
    args = parser.parse_args()
    failures = validate_output(args.output, args.recheck_rows)
    for failure in failures:
        print("FAIL:", failure)
    print("OK" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
