"""Tests of audit_artifacts.py: public rules, normalization, dummy rule files, error handling.

No test here contains a word of the audit rules in plain form: the public words are put
together while the test runs, and the other rules are dummies written to a temporary file.
"""

from __future__ import annotations

import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_artifacts as aa  # noqa: E402

WHEEL = "demo_pkg-0.1.0-py3-none-any.whl"
DIST = "demo_pkg-0.1.0.dist-info"
GOOD_METADATA = (
    "Metadata-Version: 2.4\nName: demo-pkg\nVersion: 0.1.0\nLicense: Apache-2.0\n"
    "Requires-Python: >=3.12\nRequires-Dist: numpy>=1.26\n\nA demo.\n"
)
ALLOWLIST = ["LICENSE", "NOTICE", "README.md", "src/demo_pkg/__init__.py", "src/demo_pkg/py.typed"]
PUBLIC_RULES = aa.DEFAULT_RULES


def _join(*parts: str) -> str:
    return "".join(parts)


def _fullwidth(text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)


NAME_ = _join("nvs", "_", "kernel")  # a public rule word, put together at run time


def _wheel(
    tmp_path: Path, extra: dict[str, str | bytes] | None = None, metadata: str = GOOD_METADATA
) -> Path:
    files: dict[str, str | bytes] = {
        "demo_pkg/__init__.py": "X = 1\n",
        "demo_pkg/py.typed": "",
        f"{DIST}/METADATA": metadata,
        f"{DIST}/WHEEL": "Wheel-Version: 1.0\n",
        f"{DIST}/RECORD": "demo_pkg/__init__.py,sha256=" + "A" * 43 + ",6\n",
        f"{DIST}/licenses/LICENSE": "Apache License\n",
        f"{DIST}/licenses/NOTICE": "demo-pkg\n",
    }
    files.update(extra or {})
    path = tmp_path / WHEEL
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


def _rules_file(tmp_path: Path, body: str, visibility: str = "public") -> Path:
    path = tmp_path / "rules.toml"
    path.write_text(f'visibility = "{visibility}"\n{body}', encoding="utf-8")
    return path


def _dummy_rule(rid: str, pattern: str, aliases: str = "", extra: str = "") -> str:
    return (
        f'[[rule]]\nid = "{rid}"\nlayer = 2\nkind = "literal"\npattern = "{pattern}"\n'
        f'{aliases}scope = "both"\ndescription = "dummy"\n{extra}\n'
    )


# ---- the wheel itself ------------------------------------------------------------------------


def test_clean_wheel_passes(tmp_path: Path) -> None:
    path = _wheel(tmp_path)
    rules = aa.load_rules(PUBLIC_RULES)
    assert aa.audit(path, ALLOWLIST, rules) == []
    allow = tmp_path / "allow.txt"
    allow.write_text("\n".join(ALLOWLIST) + "\n", encoding="utf-8")
    assert aa.main([str(path), "--allowlist", str(allow)]) == 0


def test_wheel_with_file_not_on_the_allowlist_fails(tmp_path: Path) -> None:
    path = _wheel(tmp_path, {"demo_pkg/tests_private_helper.py": "Y = 2\n"})
    rules = aa.load_rules(PUBLIC_RULES)
    assert aa.audit(path, ALLOWLIST, rules) == [
        f"{WHEEL}: file not expected (not on the allowlist): demo_pkg/tests_private_helper.py"
    ]
    allow = tmp_path / "allow.txt"
    allow.write_text("\n".join(ALLOWLIST) + "\n", encoding="utf-8")
    assert aa.main([str(path), "--allowlist", str(allow)]) == 1


# ---- public rules and normalization ----------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        NAME_,
        _fullwidth(NAME_),
        NAME_.upper(),
        NAME_ + ".runtime",
        _join("nvs", "_", "\u200b", "kernel"),
    ],
)
def test_internal_package_name_is_found_in_every_spelling(tmp_path: Path, variant: str) -> None:
    path = _wheel(tmp_path, {"demo_pkg/__init__.py": f"import {variant}\n"})
    problems = aa.audit(path, ALLOWLIST, aa.load_rules(PUBLIC_RULES))
    assert len(problems) == 1
    assert "rule name.internal-package (layer 1)" in problems[0]
    assert "demo_pkg/__init__.py:1" in problems[0]


@pytest.mark.parametrize(
    "text",
    [
        _join("nvs", "-", "kernel"),
        _join("NVS", "-", "Kernel"),
        _join("NVS", " ", "Kernel 0123456"),
        _join("nvs", "-", "kernel 0123456"),
    ],
)
def test_product_name_is_not_an_internal_package_name(tmp_path: Path, text: str) -> None:
    """The product name (any separator) is public; only the underscore form is internal."""
    path = _wheel(tmp_path, {"demo_pkg/__init__.py": f"# {text}\n"})
    assert aa.audit(path, ALLOWLIST, aa.load_rules(PUBLIC_RULES)) == []


@pytest.mark.parametrize("separator", ["-", " ", "_"])
def test_wording_that_points_at_a_private_repository_is_found(
    tmp_path: Path, separator: str
) -> None:
    text = _join("private ", "nvs", separator, "kernel repository")
    path = _wheel(tmp_path, {"demo_pkg/__init__.py": f"# {text}\n"})
    problems = aa.audit(path, ALLOWLIST, aa.load_rules(PUBLIC_RULES))
    assert any("rule wording.private-repository (layer 1)" in p for p in problems)


def test_public_word_in_a_file_name_is_found(tmp_path: Path) -> None:
    path = _wheel(tmp_path, {f"demo_pkg/{NAME_}.py": "Z = 1\n"})
    problems = aa.audit(path, ALLOWLIST, aa.load_rules(PUBLIC_RULES))
    assert any("name.internal-package" in p for p in problems)


def test_line_number_is_reported(tmp_path: Path) -> None:
    rules = aa.load_rules(PUBLIC_RULES)
    found = aa.scan_text("f", f"a\n\n  {NAME_}\n", rules, "content")
    assert [(f.rule.id, f.line) for f in found] == [("name.internal-package", 3)]


def test_public_rule_shows_the_matched_text(tmp_path: Path) -> None:
    rules = aa.load_rules(PUBLIC_RULES)
    (finding,) = aa.scan_text("f", NAME_, rules, "content")
    assert NAME_ in finding.describe()


# ---- dummy rule files: kana, aliases, zero-width, private rules ------------------------------


def test_hiragana_and_katakana_are_the_same(tmp_path: Path) -> None:
    rules = aa.load_rules(_rules_file(tmp_path, _dummy_rule("d.kana", "ないぶ")))
    assert len(aa.scan_text("f", "これはナイブです", rules, "content")) == 1
    assert len(aa.scan_text("f", "これはないぶです", rules, "content")) == 1


def test_aliases_are_matched(tmp_path: Path) -> None:
    body = _dummy_rule("d.alias", "dummy-internal-name", 'aliases = ["dmy-int", "ダミー内部名"]\n')
    rules = aa.load_rules(_rules_file(tmp_path, body))
    assert len(aa.scan_text("f", "see DMY_INT here", rules, "content")) == 1
    assert len(aa.scan_text("f", "see ﾀﾞﾐｰ内部名 here", rules, "content")) == 1
    assert len(aa.scan_text("f", "see Dummy Internal Name here", rules, "content")) == 1
    assert aa.scan_text("f", "nothing here", rules, "content") == []


def test_zero_width_and_separator_variants_are_found(tmp_path: Path) -> None:
    rules = aa.load_rules(_rules_file(tmp_path, _dummy_rule("d.zw", "dummy-internal-name")))
    variants = (
        "dum" + "\u200b" + "my-internal" + "\u200c" + "-name",
        "dummy___internal - name",
        "DUMMY-" + "\ufeff" + "Internal_Name",
    )
    for text in variants:
        assert len(aa.scan_text("f", text, rules, "content")) == 1, text


def test_rules_can_keep_or_fold_separators(tmp_path: Path) -> None:
    keep = _dummy_rule("d.keep", "dummy_name", extra="fold_separators = false\n")
    fold = _dummy_rule("d.fold", "dummy_other")
    rules = aa.load_rules(_rules_file(tmp_path, keep + fold))
    by_text = {
        text: sorted(f.rule.id for f in aa.scan_text("f", text, rules, "content"))
        for text in ("dummy_name", "dummy-name", "Dummy Name", "dummy_other", "dummy-other")
    }
    assert by_text == {
        "dummy_name": ["d.keep"],
        "dummy-name": [],
        "Dummy Name": [],
        "dummy_other": ["d.fold"],
        "dummy-other": ["d.fold"],
    }


def test_private_rule_does_not_print_the_word(tmp_path: Path) -> None:
    body = _dummy_rule("d.private", "dummy-internal-name")
    private = aa.load_rules(_rules_file(tmp_path, body, visibility="private"))
    path = _wheel(tmp_path, {"demo_pkg/__init__.py": "# dummy-internal-name\n"})
    problems = aa.audit(path, ALLOWLIST, private)
    assert len(problems) == 1
    assert "rule d.private (layer 2)" in problems[0]
    assert aa.PRIVATE_NOTE in problems[0]
    assert "dummy" not in problems[0].split("rule d.private")[1]
    public = aa.load_rules(_rules_file(tmp_path, body, visibility="public"))
    assert "dummy internal name" in aa.audit(path, ALLOWLIST, public)[0]


# ---- layer 3 ---------------------------------------------------------------------------------


def test_secret_shapes_are_found(tmp_path: Path) -> None:
    rules = aa.load_rules(PUBLIC_RULES)
    fake_key = _join("-----", "BEGIN RSA PRIVATE", " KEY-----")
    fake_token = _join("sk", "-", "Ab1Cd2Ef3Gh4Ij5Kl6Mn7")
    long_string = "a1" * 25
    for text, rule_id in (
        (fake_key, "secret.private-key-block"),
        (fake_token, "secret.known-token-shape"),
        (f"token = {long_string}", "secret.long-mixed-string"),
    ):
        found = aa.scan_text("f", text, rules, "content")
        assert [f.rule.id for f in found] == [rule_id], text


def test_hex_digests_and_record_hashes_are_not_secrets() -> None:
    rules = aa.load_rules(PUBLIC_RULES)
    assert aa.scan_text("f", "a" * 40 + " " + "0123456789abcdef" * 4, rules, "content") == []
    record = "pkg/x.py,sha256=" + "Ab1" * 14 + "Q,6\n"
    assert aa.scan_text("RECORD", record, rules, "content") == []


def test_binary_is_reported_and_scanned(tmp_path: Path) -> None:
    fake_token = _join("sk", "-", "Ab1Cd2Ef3Gh4Ij5Kl6Mn7").encode()
    path = _wheel(tmp_path, {"demo_pkg/blob.bin": b"\xff\xfe\x00" + fake_token})
    allow = [*ALLOWLIST, "src/demo_pkg/blob.bin"]
    problems = aa.audit(path, allow, aa.load_rules(PUBLIC_RULES))
    assert any("binary.not-decodable" in p for p in problems)
    assert any("secret.known-token-shape" in p for p in problems)


# ---- the rules files themselves --------------------------------------------------------------


def test_missing_rules_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(aa.RulesError):
        aa.load_rules(tmp_path / "does-not-exist.toml")
    path = _wheel(tmp_path)
    assert aa.main([str(path), "--rules", str(tmp_path / "does-not-exist.toml")]) == 2


@pytest.mark.parametrize(
    "body",
    [
        '[[rule]]\nid = "x"\nlayer = 2\nkind = "literal"\nscope = "both"\ndescription = "d"\n',
        _dummy_rule("x", "abc").replace('kind = "literal"', 'kind = "glob"'),
        _dummy_rule("x", "abc").replace('"literal"', '"regex"').replace('"abc"', '"(unclosed"'),
        _dummy_rule("x", "abc").replace("layer = 2", "layer = 9"),
        _dummy_rule("x", "abc").replace('scope = "both"', 'scope = "elsewhere"'),
        _dummy_rule("x", "abc") + _dummy_rule("x", "def"),
        _dummy_rule("x", "abc", extra='fold_separators = "no"\n'),
        _dummy_rule("x", "abc", extra="fold_separators = 1\n"),
        "not = [valid",
    ],
)
def test_malformed_rules_are_errors(tmp_path: Path, body: str) -> None:
    path = _rules_file(tmp_path, body)
    with pytest.raises(aa.RulesError):
        aa.load_rules(path)
    assert aa.main([str(_wheel(tmp_path)), "--rules", str(path)]) == 2


def test_public_rules_file_has_the_required_fields() -> None:
    data = tomllib.loads(PUBLIC_RULES.read_text(encoding="utf-8"))
    assert data["visibility"] == "public"
    assert data["rule"], "no rules"
    for rule in data["rule"]:
        for key in aa.REQUIRED:
            assert key in rule, (rule.get("id"), key)
        assert rule["layer"] in (1, 3)  # the public file holds layers 1 and 3 only
        assert rule["description"].strip()
        assert isinstance(rule.get("fold_separators", True), bool)
        assert isinstance(rule.get("normalize", True), bool)
    assert len(aa.load_rules(PUBLIC_RULES)) == len(data["rule"])
