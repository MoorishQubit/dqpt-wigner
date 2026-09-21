#!/usr/bin/env python3
"""Check the SHA-256 inventory of public scientific data and configurations.

Verification works in an exported directory without Git. Creating an inventory
requires a Git checkout and includes working-tree files under results/ and
configs/, applying current ignore rules even to already tracked files. Hashes
establish byte identity, not scientific correctness or completeness of a study.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("reproducibility/data_manifest.sha256")
DATA_DIRECTORIES = {"results", "configs"}


def _contains_symlink(root: Path, relative: PurePosixPath) -> bool:
    return any(
        (root / Path(*relative.parts[:end])).is_symlink()
        for end in range(1, len(relative.parts) + 1)
    )


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args], input=input_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def public_worktree_files(root: Path) -> list[Path]:
    """List existing public files; ignore rules also override Git's tracked list.

    Deleted working-tree files are omitted. Symlinks and submodules are rejected
    instead of following them into potentially private or external directories.
    """
    root = root.resolve()
    checkout = _git(root, "rev-parse", "--show-toplevel")
    if checkout.returncode or Path(checkout.stdout.decode().strip()).resolve() != root:
        raise ValueError(f"A Git checkout rooted at {root} is required")
    listed = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    if listed.returncode:
        raise ValueError(listed.stderr.decode(errors="replace").strip())
    names = sorted(set(name for name in listed.stdout.split(b"\0") if name))
    if not names:
        return []
    ignored = _git(
        root, "check-ignore", "--no-index", "--stdin", "-z",
        input_bytes=b"\0".join(names) + b"\0",
    )
    if ignored.returncode not in (0, 1):
        raise ValueError(ignored.stderr.decode(errors="replace").strip())
    excluded = set(ignored.stdout.split(b"\0"))
    files = []
    for name in names:
        if name in excluded:
            continue
        relative = Path(name.decode("utf-8"))
        path = root / relative
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise ValueError(f"Unsafe public path: {relative}")
        if _contains_symlink(root, relative):
            raise ValueError(f"Public symlinks are unsupported: {relative}")
        if not path.exists():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Public path escapes the repository: {relative}")
        if not path.is_file():
            raise ValueError(f"Public path is not a regular file (submodule?): {relative}")
        files.append(relative)
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _data_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (
        not name or relative.is_absolute() or ".." in relative.parts
        or not relative.parts or relative.parts[0] not in DATA_DIRECTORIES
        or "\\" in name or "\n" in name or "\r" in name
        or relative.as_posix() != name
    ):
        raise ValueError(f"Invalid scientific data path: {name!r}")
    path = root / relative
    if _contains_symlink(root, relative) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Scientific data path escapes the repository or is a symlink: {name}")
    return path


def write_manifest(root: Path, manifest: Path) -> int:
    files = [p for p in public_worktree_files(root) if p.parts[0] in DATA_DIRECTORIES]
    if not files:
        raise ValueError("No public scientific data or configurations found")
    entries = []
    for relative in files:
        path = _data_path(root, relative.as_posix())
        entries.append(f"{sha256(path)}  {relative.as_posix()}\n")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(entries), encoding="utf-8")
    return len(entries)


def verify_manifest(root: Path, manifest: Path) -> tuple[int, list[str]]:
    entries = []
    seen = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise ValueError(f"Malformed SHA-256 entry on manifest line {number}")
        expected, name = match.groups()
        if name in seen:
            raise ValueError(f"Duplicate manifest entry: {name}")
        seen.add(name)
        entries.append((expected, name, _data_path(root, name)))
    if not entries:
        raise ValueError("The data manifest contains no entries")
    failures = []
    for expected, name, path in entries:
        if not path.is_file():
            failures.append(f"MISSING {name}")
        elif sha256(path) != expected:
            failures.append(f"CHANGED {name}")
    return len(entries), failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST)
    parser.add_argument("--write-manifest", action="store_true", help="Inventory current public data")
    args = parser.parse_args()
    try:
        if args.write_manifest:
            count = write_manifest(ROOT, args.manifest)
            print(f"Wrote {count} scientific data/configuration hashes to {args.manifest}")
            return 0
        count, failures = verify_manifest(ROOT, args.manifest)
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    for failure in failures:
        print(failure)
    print(f"Checked {count} scientific data/configuration files; {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
