"""The sdist include list in pyproject.toml must be exactly the public allowlist."""

from __future__ import annotations

import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]  # package (= public repository) root


def test_sdist_include_equals_allowlist() -> None:
    config = tomllib.loads((PACKAGE / "pyproject.toml").read_text(encoding="utf-8"))
    include = config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    in_sdist = {item.lstrip("/") for item in include} | {"pyproject.toml"}
    text = (PACKAGE / "public-sync" / "allowlist.txt").read_text(encoding="utf-8")
    allowlist = {
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert in_sdist == allowlist
