#!/usr/bin/env python3
"""Copy public working-tree files into a standalone directory without Git history.

Run only from a Git checkout. Both tracked and untracked files are included,
subject to the current ignore rules, including rules for already tracked files.
The source checkout is unchanged. This is a working-tree export, not a rewrite
of earlier commits or removal of anything already published to a remote.
Review the exported contents before publishing; ignore rules are the boundary,
not an automatic detector of confidential text. Symlinks/submodules are refused.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from verify_reproduction_data import ROOT, public_worktree_files


def export_repository(root: Path, output: Path) -> int:
    root = root.resolve()
    if output.is_symlink():
        raise ValueError("The export destination must not be a symlink")
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("The export destination must be outside the source repository")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("The export destination must be absent or an empty directory")
    files = public_worktree_files(root)
    if not files:
        raise ValueError("The checkout contains no public files to export")
    output.mkdir(parents=True, exist_ok=True)
    for relative in files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Absent or empty directory outside this checkout")
    parser.add_argument("--list", action="store_true", help="List export paths without copying")
    args = parser.parse_args()
    if not args.list and args.output is None:
        parser.error("--output is required unless --list is used")
    if args.list and args.output is not None:
        parser.error("Choose --list or --output, not both")
    try:
        if args.list:
            for path in public_worktree_files(ROOT):
                print(path.as_posix())
        else:
            count = export_repository(ROOT, args.output)
            print(f"Exported {count} public files to {args.output.resolve()}")
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
