"""(a) Package import boundary: stdlib / nvs_observation / optional numpy|nvs_edge only."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from _edge_refs import (
    REGISTRY_PATH,
    _is_import_module_call,
    _resolve_import_module_arg,
    iter_source_paths,
    load_registry,
    registry_modules,
)

ALLOWED_ROOTS = set(sys.stdlib_module_names) | {"__future__", "nvs_observation"}


def _root(name: str) -> str:
    return name.split(".", 1)[0]


def test_a_import_boundary_stdlib_or_nvs_observation_only() -> None:
    entries = load_registry()
    allowed_dynamic = {"numpy"} | registry_modules(entries)
    assert REGISTRY_PATH.is_file()

    def check(path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        loop_iters: dict[str, list[str]] = {}

        class Visitor(ast.NodeVisitor):
            def visit_For(self, node: ast.For) -> None:
                if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Tuple):
                    values: list[str] = []
                    ok = True
                    for elt in node.iter.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            values.append(elt.value)
                        else:
                            ok = False
                            break
                    if ok:
                        loop_iters[node.target.id] = values
                self.generic_visit(node)
                if isinstance(node.target, ast.Name):
                    loop_iters.pop(node.target.id, None)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    root = _root(alias.name)
                    assert root in ALLOWED_ROOTS, f"{path}: import {alias.name}"
                    assert "nvs_runtime" not in alias.name, f"{path}: {alias.name}"

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.level and not node.module:
                    return
                module = node.module or ""
                if node.level:
                    assert "nvs_runtime" not in module
                    return
                root = _root(module)
                assert root in ALLOWED_ROOTS, f"{path}: from {module}"
                assert "nvs_runtime" not in module, f"{path}: from {module}"

            def visit_Call(self, node: ast.Call) -> None:
                if _is_import_module_call(node) and node.args:
                    for module in _resolve_import_module_arg(node.args[0], loop_iters):
                        assert "nvs_runtime" not in module, f"{path}: import_module({module!r})"
                        assert module in allowed_dynamic, (
                            f"{path}: import_module({module!r}) not numpy or Registry module"
                        )
                self.generic_visit(node)

        Visitor().visit(tree)

    for source_path in iter_source_paths():
        check(source_path)
