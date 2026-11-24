"""Tests of check_allowlist.py: clean, staged tree with a non-allowlisted file, sha256 mismatch."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_allowlist


def _make(tmp_path: Path) -> tuple[Path, Path, Path]:
    pkg = tmp_path / "pkg"
    (pkg / "src").mkdir(parents=True)
    (pkg / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "LICENSE").write_text("license\n", encoding="utf-8")
    allow = tmp_path / "allowlist.txt"
    allow.write_text("# c\nsrc/a.py\nLICENSE\n", encoding="utf-8")
    files = {
        rel: hashlib.sha256((pkg / rel).read_bytes()).hexdigest() for rel in ("src/a.py", "LICENSE")
    }
    manifest = tmp_path / "sync-manifest.json"
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    return pkg, allow, manifest


def test_clean(tmp_path: Path) -> None:
    pkg, allow, manifest = _make(tmp_path)
    assert check_allowlist.check(pkg, allow, manifest) == []
    staged = tmp_path / "staged"
    (staged / "src").mkdir(parents=True)
    (staged / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (staged / "LICENSE").write_text("license\n", encoding="utf-8")
    assert check_allowlist.check(pkg, allow, manifest, staged) == []
    assert check_allowlist.main(["--package-root", str(pkg), "--allowlist", str(allow),
                                 "--manifest", str(manifest)]) == 0


def test_file_not_in_allowlist_is_reported(tmp_path: Path) -> None:
    pkg, allow, manifest = _make(tmp_path)
    staged = tmp_path / "staged"
    (staged / "src").mkdir(parents=True)
    (staged / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (staged / "tests_private").mkdir()
    (staged / "tests_private" / "t.py").write_text("secret\n", encoding="utf-8")
    problems = check_allowlist.check(pkg, allow, manifest, staged)
    assert problems == ["staged file is not in the allowlist: tests_private/t.py"]


def test_sha256_mismatch_is_reported(tmp_path: Path) -> None:
    pkg, allow, manifest = _make(tmp_path)
    (pkg / "src" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert check_allowlist.check(pkg, allow, manifest) == [
        "sha256 differs from the manifest: src/a.py"
    ]
    assert check_allowlist.main(["--package-root", str(pkg), "--allowlist", str(allow),
                                 "--manifest", str(manifest)]) == 1


def test_manifest_inside_the_package_is_not_hashed_by_itself(tmp_path: Path) -> None:
    pkg, allow, _ = _make(tmp_path)
    (pkg / "public-sync").mkdir()
    manifest = pkg / "public-sync" / "sync-manifest.json"
    files = {
        rel: hashlib.sha256((pkg / rel).read_bytes()).hexdigest() for rel in ("src/a.py", "LICENSE")
    }
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    allow.write_text(
        "src/a.py\nLICENSE\npublic-sync/sync-manifest.json\n", encoding="utf-8"
    )
    # The manifest is allowlisted and present, but has no sha256 of itself: that is correct.
    assert check_allowlist.check(pkg, allow, manifest) == []
    # A manifest that lists itself is reported (its own hash could never be right).
    files["public-sync/sync-manifest.json"] = "0" * 64
    manifest.write_text(json.dumps({"files": files}), encoding="utf-8")
    assert check_allowlist.check(pkg, allow, manifest) == [
        "manifest must not list itself: public-sync/sync-manifest.json"
    ]
