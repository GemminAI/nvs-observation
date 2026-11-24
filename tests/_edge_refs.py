"""Static extraction of nvs_edge module/name refs from nvs_observation sources."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src" / "nvs_observation"
REGISTRY_PATH = PACKAGE_ROOT / "kernel-api-registry.json"

Ref = tuple[str, str | None]


def load_registry() -> list[dict[str, Any]]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def registry_ref_set(entries: list[dict[str, Any]]) -> set[Ref]:
    return {(entry["module"], entry["name"]) for entry in entries}


def registry_modules(entries: list[dict[str, Any]]) -> set[str]:
    return {entry["module"] for entry in entries}


def iter_source_paths() -> list[Path]:
    return sorted(path for path in SRC_ROOT.rglob("*.py") if path.is_file())


def _const_str(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _tuple_str_consts(node: ast.AST) -> list[str] | None:
    if not isinstance(node, ast.Tuple):
        return None
    values: list[str] = []
    for elt in node.elts:
        text = _const_str(elt)
        if text is None:
            return None
        values.append(text)
    return values


def _pair_tuple_strs(node: ast.AST) -> list[tuple[str, str]] | None:
    if not isinstance(node, ast.Tuple):
        return None
    pairs: list[tuple[str, str]] = []
    for elt in node.elts:
        if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
            return None
        left = _const_str(elt.elts[0])
        right = _const_str(elt.elts[1])
        if left is None or right is None:
            return None
        pairs.append((left, right))
    return pairs


def _is_import_module_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "import_module":
        return True
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id == "importlib"
    )


def _resolve_import_module_arg(
    arg: ast.AST,
    loop_iters: dict[str, list[str]],
) -> list[str]:
    direct = _const_str(arg)
    if direct is not None:
        return [direct]
    if isinstance(arg, ast.JoinedStr):
        # Only the form f"nvs_edge.{name}" with name bound to a known string tuple.
        if len(arg.values) != 2:
            raise ValueError(f"unsupported f-string import_module arg: {ast.dump(arg)}")
        prefix_node, formatted = arg.values
        prefix = _const_str(prefix_node)
        if prefix is None or not isinstance(formatted, ast.FormattedValue):
            raise ValueError(f"unsupported f-string import_module arg: {ast.dump(arg)}")
        if not isinstance(formatted.value, ast.Name):
            raise ValueError(f"unsupported f-string import_module name: {ast.dump(arg)}")
        values = loop_iters.get(formatted.value.id)
        if values is None:
            raise ValueError(
                f"import_module f-string uses unbound loop name {formatted.value.id!r}"
            )
        return [f"{prefix}{value}" for value in values]
    raise ValueError(f"unsupported import_module argument: {ast.dump(arg)}")


def extract_nvs_edge_refs(tree: ast.AST) -> set[Ref]:
    """Return {(module, name|None)} for nvs_edge imports and getattr/hasattr names.

    Supports the patterns used by capture_edge and edge_observe:
    - import_module(\"nvs_edge...\")
    - for name in (\"contract\", ...): import_module(f\"nvs_edge.{name}\")
    - wanted = ((\"contract\", \"ContractError\"), ...) with hasattr/getattr
    - name = \"Edge\"; getattr(module, name) after import_module(\"nvs_edge.edge\")
    """
    refs: set[Ref] = set()
    loop_iters: dict[str, list[str]] = {}
    str_consts: dict[str, str] = {}
    imported_into: dict[str, set[str]] = {}
    wanted_pairs: list[tuple[str, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_For(self, node: ast.For) -> None:
            if isinstance(node.target, ast.Name):
                values = _tuple_str_consts(node.iter)
                if values is not None:
                    loop_iters[node.target.id] = values
            self.generic_visit(node)
            if isinstance(node.target, ast.Name):
                loop_iters.pop(node.target.id, None)

        def visit_Assign(self, node: ast.Assign) -> None:
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target = node.targets[0].id
                const = _const_str(node.value)
                if const is not None:
                    str_consts[target] = const
                pairs = _pair_tuple_strs(node.value)
                if pairs is not None and target == "wanted":
                    wanted_pairs.extend(pairs)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if _is_import_module_call(node) and node.args:
                modules = _resolve_import_module_arg(node.args[0], loop_iters)
                for module in modules:
                    if module == "numpy":
                        continue
                    if not module.startswith("nvs_edge"):
                        raise ValueError(f"unexpected import_module target: {module!r}")
                    refs.add((module, None))
                    # Track assignment target if present: x = import_module(...)
                    parent_assign = getattr(node, "_nvs_assign_target", None)
                    if isinstance(parent_assign, str):
                        imported_into.setdefault(parent_assign, set()).add(module)
            if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "hasattr"}:
                if len(node.args) >= 2:
                    owner = node.args[0]
                    attr_node = node.args[1]
                    attr = _const_str(attr_node)
                    if attr is None and isinstance(attr_node, ast.Name):
                        attr = str_consts.get(attr_node.id)
                    if attr is None:
                        # modules[module], attribute — covered via wanted pairs
                        self.generic_visit(node)
                        return
                    modules: set[str] = set()
                    if isinstance(owner, ast.Name):
                        modules = set(imported_into.get(owner.id, ()))
                    if not modules:
                        # getattr(modules[module], attribute) handled by wanted
                        self.generic_visit(node)
                        return
                    for module in modules:
                        if module.startswith("nvs_edge"):
                            refs.add((module, attr))
            self.generic_visit(node)

    # Annotate simple Assign targets for import_module tracking.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                node.value._nvs_assign_target = target.id  # type: ignore[attr-defined]

    Visitor().visit(tree)

    for short_name, attribute in wanted_pairs:
        refs.add((f"nvs_edge.{short_name}", attribute))

    return refs


def extract_all_nvs_edge_refs() -> set[Ref]:
    refs: set[Ref] = set()
    for path in iter_source_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        refs |= extract_nvs_edge_refs(tree)
    return refs
