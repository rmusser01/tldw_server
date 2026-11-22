"""Independent check of a ChaChaNotes_DB move (TASK-13950.22, R22, R32). It shares no code with ``split.py``.

    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.diffpairs MANIFEST... --base REF

For each moved def it parses the base def (``git show REF:<source>``) and the head def (in the target), then
walks both trees in parallel. Only the header is normalized: the receiver parameter ``self`` is renamed to
``ctx`` (functions mode), a ``staticmethod`` decorator is dropped, and a ``classmethod`` decorator and its
``cls`` parameter are dropped. The body is never normalized.

The walk descends while node types, scalar fields and list lengths match. The first mismatch under a node is
recorded as ``(unparse(old), unparse(new))`` and not descended into. A recorded pair is allowed when the old
side is rooted at ``self``/``cls`` (an attribute, a call on one, or the bare name), or is a manifest rename.
Any other pair, a changed literal, operator or control flow, exits 1. Exits 0 when every pair is allowed.
In constants mode each constant's value is walked the same way, and any difference exits 1.
"""

from __future__ import annotations

import argparse
import ast
import subprocess  # nosec B404  # fixed argv, no shell
import sys
from collections import Counter
from pathlib import Path

if sys.version_info < (3, 12):  # R22: AST positions inside f-strings are 3.12+
    raise SystemExit("diffpairs.py needs Python 3.12+")

CLASS = "CharactersRAGDB"


def load_manifest(path: Path) -> dict:
    """Read the ``MANIFEST`` literal without importing the file (R23)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MANIFEST" for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError as exc:
                raise SystemExit(f"{path}: MANIFEST must be a literal: {exc}") from exc
    raise SystemExit(f"{path}: no MANIFEST")


def git_show(root: Path, ref: str, path: str) -> str:
    result = subprocess.run(  # nosec B603 B607  # fixed argv, no shell
        ["git", "-C", str(root), "show", f"{ref}:{path}"], capture_output=True, text=True, check=True
    )
    return result.stdout


def find_def(body: list[ast.stmt], name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def normalized_base(node: ast.FunctionDef | ast.AsyncFunctionDef, functions_mode: bool) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The base def with only its header changed."""
    if not functions_mode:  # a store keeps the def as it is: decorators and ``self`` stay
        return node
    decorators = [ast.unparse(d) for d in node.decorator_list]
    node.decorator_list = []
    args = node.args
    first = (args.posonlyargs or args.args or [None])[0]
    if "classmethod" in decorators and first is not None and first.arg == "cls":
        (args.posonlyargs or args.args).pop(0)
    elif first is not None and first.arg == "self" and "staticmethod" not in decorators:
        first.arg = "ctx"
    return node


def _scalars(node: ast.AST) -> list[tuple[str, object]]:
    return [(f, v) for f, v in ast.iter_fields(node) if not isinstance(v, (ast.AST, list))]


def walk(old: ast.AST, new: ast.AST, pairs: list[tuple[ast.AST, ast.AST]]) -> None:
    """Descend while the two nodes match in type, scalar fields and list lengths; else record the pair."""
    if type(old) is not type(new) or _scalars(old) != _scalars(new):
        pairs.append((old, new))
        return
    for field, old_value in ast.iter_fields(old):
        new_value = getattr(new, field)
        if isinstance(old_value, list):
            if len(old_value) != len(new_value):
                pairs.append((old, new))
                return
    for field, old_value in ast.iter_fields(old):
        new_value = getattr(new, field)
        if isinstance(old_value, list):
            for a, b in zip(old_value, new_value, strict=True):
                if isinstance(a, ast.AST):
                    walk(a, b, pairs)
        elif isinstance(old_value, ast.AST):
            walk(old_value, new_value, pairs)


def _root(text: str) -> str:
    """The leading name of an expression's text: ``self._a(x).b`` -> ``self``."""
    for i, ch in enumerate(text):
        if not (ch.isalnum() or ch == "_"):
            return text[:i]
    return text


def allowed(old: ast.AST, new: ast.AST, renames: dict[str, str]) -> bool:
    if _root(ast.unparse(old)) in {"self", "cls"}:
        return True
    return (
        isinstance(old, ast.Attribute)
        and isinstance(new, ast.Attribute)
        and renames.get(old.attr) == new.attr
        and ast.unparse(old.value) == ast.unparse(new.value)
    )


def constant_value(body: list[ast.stmt], name: str) -> ast.expr | None:
    for node in body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name:
            return node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return node.value
    return None


def check_constants(base_tree: ast.Module, base_class: ast.ClassDef, head_tree: ast.Module, manifest: dict, counts: Counter, bad: list[str]) -> int:
    """A moved constant keeps its value exactly: any differing node is disallowed (constants mode)."""
    checked = 0
    for name in manifest["constants"]:
        old = constant_value(base_class.body, name) or constant_value(base_tree.body, name)
        new = constant_value(head_tree.body, name)
        if old is None or new is None:
            bad.append(f"{name}: constant missing in {'base' if old is None else 'head'}")
            continue
        pairs: list[tuple[ast.AST, ast.AST]] = []
        walk(old, new, pairs)
        checked += 1
        for a, b in pairs:
            counts[(ast.unparse(a), ast.unparse(b))] += 1
            bad.append(f"{name}: {ast.unparse(a)[:80]!r} -> {ast.unparse(b)[:80]!r}")
    return checked


def check_manifest(root: Path, ref: str, manifest: dict, counts: Counter, bad: list[str]) -> int:
    functions_mode = manifest["mode"] == "functions"
    base_tree = ast.parse(git_show(root, ref, manifest["source"]))
    base_class = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    head_tree = ast.parse((root / manifest["target"]).read_text(encoding="utf-8"))
    head_body: list[ast.stmt] = head_tree.body
    if manifest.get("store_class"):
        head_body = next(n for n in head_tree.body if isinstance(n, ast.ClassDef) and n.name == manifest["store_class"]).body
    renames = manifest.get("renames", {})
    checked = check_constants(base_tree, base_class, head_tree, manifest, counts, bad) if manifest["mode"] == "constants" else 0
    for name in manifest["names"]:
        old = find_def(base_class.body, name)
        new = find_def(head_body, name)
        if old is None or new is None:
            bad.append(f"{name}: def missing in {'base' if old is None else 'head'}")
            continue
        pairs: list[tuple[ast.AST, ast.AST]] = []
        walk(normalized_base(old, functions_mode), new, pairs)
        checked += 1
        for a, b in pairs:
            key = (ast.unparse(a), ast.unparse(b))
            counts[key] += 1
            if not allowed(a, b, renames):
                bad.append(f"{name}: {key[0]!r} -> {key[1]!r}")
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    counts: Counter = Counter()
    bad: list[str] = []
    checked = sum(check_manifest(root, args.base, load_manifest(path), counts, bad) for path in args.manifests)
    for (old, new), count in sorted(counts.items()):
        print(f"{count}x  {old}  ->  {new}")
    for line in bad:
        print(f"DISALLOWED {line}")
    print(f"{checked} defs, {sum(counts.values())} pairs, {len(bad)} disallowed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
