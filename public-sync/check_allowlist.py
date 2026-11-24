#!/usr/bin/env python3
"""Check the public-sync allowlist of nvs-observation (check only; never edits anything).

(i)   every file in allowlist.txt exists under the package root;
(ii)  with --staged DIR: no file in the staged (to-be-published) tree is missing from the
      allowlist; and no allowlist entry points at a private location;
(iii) every sha256 in sync-manifest.json matches the file on disk, and the manifest lists
      exactly the allowlisted files.

Standard library only. Exit status 0 = clean, 1 = problems (listed on stderr).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
DEFAULT_PACKAGE_ROOT = HERE.parent  # public-sync/ sits at the package (= public repository) root
PRIVATE_PARTS = {"tests_private", "__pycache__", ".git", ".mypy_cache", ".ruff_cache"}
PRIVATE_NAMES = {".coverage"}


def read_allowlist(path: Path) -> list[str]:
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _staged_files(staged: Path) -> set[str]:
    found: set[str] = set()
    for p in staged.rglob("*"):
        if p.is_file():
            rel = p.relative_to(staged)
            if ".git" in rel.parts:
                continue
            found.add(rel.as_posix())
    return found


def _relative_to(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def check(
    package_root: Path,
    allowlist: Path,
    manifest: Path,
    staged: Path | None = None,
) -> list[str]:
    problems: list[str] = []
    entries = read_allowlist(allowlist)
    allowed = set(entries)
    if len(allowed) != len(entries):
        problems.append("allowlist has duplicate entries")

    for entry in entries:
        pure = PurePosixPath(entry)
        if pure.is_absolute() or ".." in pure.parts:
            problems.append(f"allowlist entry is not a relative path inside the package: {entry}")
            continue
        if PRIVATE_PARTS & set(pure.parts) or pure.name in PRIVATE_NAMES:
            problems.append(f"allowlist entry points at a private location: {entry}")
        if not (package_root / entry).is_file():  # (i)
            problems.append(f"allowlisted file does not exist: {entry}")

    if staged is not None:  # (ii)
        for extra in sorted(_staged_files(staged) - allowed):
            problems.append(f"staged file is not in the allowlist: {extra}")

    data = json.loads(manifest.read_text(encoding="utf-8"))  # (iii)
    files: dict[str, str] = data.get("files", {})
    own = _relative_to(manifest, package_root)
    if own is not None:
        if own in files:
            problems.append(f"manifest must not list itself: {own}")
            files = {k: v for k, v in files.items() if k != own}
        allowed = allowed - {own}  # the manifest is not hashed by itself
    for entry in sorted(allowed - set(files)):
        problems.append(f"manifest has no sha256 for allowlisted file: {entry}")
    for entry in sorted(set(files) - allowed):
        problems.append(f"manifest lists a file that is not allowlisted: {entry}")
    for entry, expected in sorted(files.items()):
        target = package_root / entry
        if target.is_file() and sha256_of(target) != expected:
            problems.append(f"sha256 differs from the manifest: {entry}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    ap.add_argument("--allowlist", type=Path, default=HERE / "allowlist.txt")
    ap.add_argument("--manifest", type=Path, default=HERE / "sync-manifest.json")
    ap.add_argument("--staged", type=Path, default=None, help="tree about to be published")
    args = ap.parse_args(argv)
    problems = check(args.package_root, args.allowlist, args.manifest, args.staged)
    for p in problems:
        print(f"check_allowlist: {p}", file=sys.stderr)
    if not problems:
        print("check_allowlist: ok")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
