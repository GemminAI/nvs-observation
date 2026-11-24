"""Tests of export_public.py: what the export holds, and that no commit id is in it."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import export_public  # noqa: E402


def _commit(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", message],
        cwd=root,
        check=True,
    )
    return _head(root)


def _head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    """A repository whose package folder ``pkg`` holds the files of a small public tree."""
    repo = tmp_path / "repo"
    pkg = repo / "pkg"
    (pkg / "src").mkdir(parents=True)
    (pkg / "public-sync").mkdir()
    (pkg / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "private.txt").write_text("private\n", encoding="utf-8")
    (pkg / "public-sync" / "audit-rules.toml").write_text('visibility = "public"\n')
    (pkg / "public-sync" / "audit_artifacts.py").write_text("# tool\n", encoding="utf-8")
    (pkg / "public-sync" / "allowlist.txt").write_text(
        "# c\nsrc/a.py\npublic-sync/audit-rules.toml\npublic-sync/audit_artifacts.py\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    _commit(repo, "c")
    return repo, pkg


def _list(pkg: Path) -> Path:
    return pkg / "public-sync" / "allowlist.txt"


def test_the_export_holds_only_the_allowlist_and_a_manifest_without_a_commit(
    tmp_path: Path,
) -> None:
    repo, pkg = _repo(tmp_path)
    out = tmp_path / "out"
    manifest = export_public.export(pkg, out, _list(pkg))
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert "private.txt" not in names and "src/a.py" in names
    assert export_public.MANIFEST_REL in names and not (out / ".git").exists()
    written = json.loads((out / export_public.MANIFEST_REL).read_text(encoding="utf-8"))
    assert written == manifest
    assert set(written) == {"schema", "package", "audit_rules_sha256", "audit_tool_sha256", "files"}
    assert export_public.MANIFEST_REL not in written["files"]
    assert written["files"]["src/a.py"] == hashlib.sha256(b"x = 1\n").hexdigest()
    assert _head(repo) not in (out / export_public.MANIFEST_REL).read_text(encoding="utf-8")


def test_an_internal_record_inside_the_export_or_the_repository_is_refused(
    tmp_path: Path,
) -> None:
    repo, pkg = _repo(tmp_path)
    out = tmp_path / "out"
    for record in (out / "record.json", pkg / "record.json", repo / "record.json", out, repo):
        with pytest.raises(ValueError, match="internal record must be outside"):
            export_public.export(pkg, out, _list(pkg), record)
    assert not out.exists()


def test_the_internal_record_holds_the_commits_and_the_manifest_digest(tmp_path: Path) -> None:
    repo, pkg = _repo(tmp_path)
    first = _head(repo)
    (pkg / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    _commit(repo, "d")
    out, record = tmp_path / "out", tmp_path / "elsewhere" / "record.json"
    export_public.export(pkg, out, _list(pkg), record, first)
    data = json.loads(record.read_text(encoding="utf-8"))
    assert set(data) == {
        "package",
        "source_commit",
        "kernel_commit",
        "branch",
        "file_count",
        "manifest_sha256",
        "created_utc",
    }
    assert data["source_commit"] == _head(repo) and data["kernel_commit"] == first
    manifest_file = out / export_public.MANIFEST_REL
    assert data["manifest_sha256"] == hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    assert data["file_count"] == len([p for p in out.rglob("*") if p.is_file()])
    assert data["created_utc"].endswith("Z") and data["branch"]


def test_a_file_that_names_a_commit_stops_the_export_before_anything_is_written(
    tmp_path: Path,
) -> None:
    repo, pkg = _repo(tmp_path)
    older = _head(repo)
    (pkg / "src" / "a.py").write_text(f"# checked against {older[:7]}\n", encoding="utf-8")
    _commit(repo, "d")
    out = tmp_path / "out"
    # the older commit is given as the kernel commit: its first 7 characters are in a.py
    with pytest.raises(ValueError, match="commit id is in the export.*src/a.py"):
        export_public.export(pkg, out, _list(pkg), None, older)
    assert not out.exists()
    # without it, the export of the same tree is made
    export_public.export(pkg, out, _list(pkg))
    assert (out / "src" / "a.py").is_file()


def test_a_commit_id_is_found_in_full_or_as_its_first_seven_characters() -> None:
    commit = "0123456789abcdef" * 2 + "01234567"
    assert export_public.names_commit(b"x 0123456 y", commit)
    assert export_public.names_commit(commit.upper().encode(), commit)
    assert not export_public.names_commit(b"x 012345 y", commit)


def test_a_dirty_package_a_listed_manifest_or_a_target_inside_is_refused(tmp_path: Path) -> None:
    repo, pkg = _repo(tmp_path)
    with pytest.raises(ValueError, match="outside the repository"):
        export_public.export(pkg, pkg / "out", _list(pkg))
    with pytest.raises(ValueError, match="outside the repository"):
        export_public.export(pkg, repo / "other", _list(pkg))
    (pkg / "src" / "a.py").write_text("x = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not clean"):
        export_public.export(pkg, tmp_path / "out", _list(pkg))
    _commit(repo, "e")
    _list(pkg).write_text(_list(pkg).read_text() + export_public.MANIFEST_REL + "\n")
    _commit(repo, "f")
    with pytest.raises(ValueError, match="written by the export"):
        export_public.export(pkg, tmp_path / "out2", _list(pkg))
