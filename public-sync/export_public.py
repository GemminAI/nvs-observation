#!/usr/bin/env python3
"""Export the public tree of nvs-observation into a new folder (no git history).

* The files are the ones in ``allowlist.txt``. They are read from the commit ``HEAD`` (not from
  the working tree); the package folder must be clean, and the target folder must be new (or
  empty) and outside the repository. Nothing is committed and no git data is copied.
* ``public-sync/sync-manifest.json`` is written into the export: the sha256 of every exported
  file (the manifest cannot hold its own), and the sha256 of the audit rules and of the audit
  tool that are exported with it. It holds no commit id. Nothing in the export names the commit
  it was made from: the export fails, and writes nothing, if any file (or the manifest) contains
  the source commit or the kernel commit, in full or as its first 7 characters.
* ``--internal-record PATH`` writes, outside the export and outside the repository, a JSON
  record of the export: the source commit, the kernel commit (``--kernel-commit REV``, or null),
  the branch, the number of exported files, the sha256 of the public manifest and the time (UTC).
  PATH inside either folder is refused.
* ``--audit`` applies the audit rules (``--rules``, repeatable; the exported public rules by
  default) to the exported files and prints the number of hits by layer and rule. Layer-3 hits
  are named by file and line only.

Standard library only. Exit status 0 = done, 1 = a problem (on stderr), 2 = bad rules.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_artifacts  # noqa: E402
import check_allowlist  # noqa: E402

PACKAGE = "nvs-observation"
MANIFEST_REL = "public-sync/sync-manifest.json"
RULES_REL = "public-sync/audit-rules.toml"
TOOL_REL = "public-sync/audit_artifacts.py"


def _git(cwd: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True).stdout


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def names_commit(data: bytes, commit: str) -> bool:
    """Whether ``data`` contains ``commit`` in full or as its first 7 characters."""
    text = data.lower()
    return commit.lower().encode() in text or commit[:7].lower().encode() in text


def _inside(path: Path, folder: Path) -> bool:
    return path == folder or folder in path.parents


def export(
    root: Path,
    out: Path,
    allowlist: Path,
    internal_record: Path | None = None,
    kernel_commit: str | None = None,
) -> dict[str, object]:
    root = root.resolve()
    out = out.resolve()
    repo = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if _inside(out, repo) or _inside(out, root):
        raise ValueError("the target folder must be outside the repository")
    if internal_record is not None:
        record = internal_record.resolve()
        for folder, name in ((repo, "the repository"), (out, "the export")):
            if _inside(record, folder):
                raise ValueError(f"the internal record must be outside {name}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("the target folder must be new or empty")
    if _git(root, "status", "--porcelain", "--", ".").strip():
        raise ValueError("the package folder is not clean: commit first (files are read from HEAD)")
    commit = _git(repo, "rev-parse", "HEAD").decode().strip()
    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").decode().strip()
    kernel = (
        _git(repo, "rev-parse", "--verify", f"{kernel_commit}^{{commit}}").decode().strip()
        if kernel_commit is not None
        else None
    )
    entries = check_allowlist.read_allowlist(allowlist)
    problems: list[str] = []
    if len(set(entries)) != len(entries):
        problems.append("allowlist has duplicate entries")
    for entry in entries:
        pure = PurePosixPath(entry)
        if pure.is_absolute() or ".." in pure.parts:
            problems.append(f"not a relative path inside the package: {entry}")
        elif check_allowlist.PRIVATE_PARTS & set(pure.parts) or (
            pure.name in check_allowlist.PRIVATE_NAMES
        ):
            problems.append(f"private location: {entry}")
        elif entry == MANIFEST_REL:
            problems.append(f"the manifest is written by the export, not listed: {entry}")
    if problems:
        raise ValueError("allowlist problems: " + "; ".join(problems))
    contents: dict[str, bytes] = {rel: _git(root, "show", f"HEAD:./{rel}") for rel in entries}
    files = {rel: _sha256(data) for rel, data in sorted(contents.items())}
    manifest: dict[str, object] = {
        "schema": 1,
        "package": PACKAGE,
        "audit_rules_sha256": files.get(RULES_REL),
        "audit_tool_sha256": files.get(TOOL_REL),
        "files": files,
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    contents[MANIFEST_REL] = manifest_bytes
    named = sorted(
        rel
        for rel, data in contents.items()
        if any(names_commit(data, c) for c in (commit, kernel) if c is not None)
    )
    if named:
        raise ValueError("a commit id is in the export (nothing written): " + ", ".join(named))
    for rel, data in contents.items():
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if internal_record is not None:
        data_out = {
            "package": PACKAGE,
            "source_commit": commit,
            "kernel_commit": kernel,
            "branch": branch,
            "file_count": len(contents),
            "manifest_sha256": _sha256(manifest_bytes),
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        internal_record.parent.mkdir(parents=True, exist_ok=True)
        internal_record.write_text(json.dumps(data_out, indent=2) + "\n", encoding="utf-8")
    return manifest


def audit(
    out: Path, rule_files: list[Path]
) -> dict[tuple[int, str], tuple[int, int, list[tuple[str, int]]]]:
    """``(layer, rule id) -> (files, lines, [(file, line)])`` for the files under ``out``."""
    rules: list[audit_artifacts.Rule] = []
    for path in rule_files:
        rules.extend(audit_artifacts.load_rules(path))
    result: dict[tuple[int, str], tuple[int, int, list[tuple[str, int]]]] = {}
    paths = sorted(p for p in out.rglob("*") if p.is_file())
    binary = []
    for path in paths:
        try:
            path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            binary.append((path.relative_to(out).as_posix(), 0))
    if binary:
        result[(3, "binary.not-decodable")] = (len(binary), len(binary), binary)
    for rule in rules:
        hits: list[tuple[str, int]] = []
        for path in paths:
            rel = path.relative_to(out).as_posix()
            if rule.scope in ("name", "both") and audit_artifacts._matches(
                rule,
                audit_artifacts.normalize(rel, rule.fold_separators) if rule.normalize else rel,
            ):
                hits.append((rel, 0))
            if rule.scope in ("content", "both"):
                try:
                    text = path.read_bytes().decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if rule.normalize:
                    subject, line_of = audit_artifacts.normalize_with_lines(
                        text, rule.fold_separators
                    )
                    found = [line_of[start] for start, _ in audit_artifacts._matches(rule, subject)]
                else:
                    found = [
                        text.count("\n", 0, start) + 1
                        for start, _ in audit_artifacts._matches(rule, text)
                    ]
                hits.extend((rel, line) for line in sorted(set(found)))
        if hits:
            result[(rule.layer, rule.id)] = (len({f for f, _ in hits}), len(hits), hits)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=HERE.parent)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allowlist", type=Path, default=HERE / "allowlist.txt")
    ap.add_argument("--internal-record", type=Path, default=None)
    ap.add_argument("--kernel-commit", default=None)
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--rules", type=Path, action="append", default=None)
    args = ap.parse_args(argv)
    try:
        manifest = export(
            args.root, args.out, args.allowlist, args.internal_record, args.kernel_commit
        )
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"export_public: {error}", file=sys.stderr)
        return 1
    files = manifest["files"]
    assert isinstance(files, dict)
    print(f"export_public: {len(files)} file(s) of {PACKAGE}")
    if args.audit:
        rules = args.rules or [args.out / RULES_REL]
        try:
            found = audit(args.out, rules)
        except audit_artifacts.RulesError as error:
            print(f"export_public: bad rules: {error}", file=sys.stderr)
            return 2
        for (layer, rule_id), (n_files, n_lines, hits) in sorted(found.items()):
            print(f"audit L{layer} {rule_id}: {n_files} file(s), {n_lines} line(s)")
            if layer == 3:  # file and line only, never the matched text
                by_file: dict[str, list[int]] = collections.defaultdict(list)
                for rel, line in hits:
                    by_file[rel].append(line)
                for rel, lines in sorted(by_file.items()):
                    print(f"  {rel}: {len(lines)} line(s), first {lines[:5]}")
        if not found:
            print("audit: no hit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
