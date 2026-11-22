"""Rewrite callers of delegated CharactersRAGDB names through their store (TASK-13950.22).

    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split callers --store ATTR --names N1,N2 [--check] [--tests]

``callers`` rewrites, under tldw_Server_API/app and tldw_Server_API/tests, ``<recv>.<name>`` to
``<recv>.<ATTR>.<name>`` and ``getattr/hasattr(<recv>, "<name>", ...)`` to a probe of ``<recv>.<ATTR>``.
``--check`` writes nothing and exits 1 while anything still reaches a name outside ``*.<ATTR>``.
Edits use AST positions (UTF-8 byte offsets), applied right to left, so no string value changes.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

if sys.version_info < (3, 12):  # R22: f-string tokenization and AST positions are 3.12+
    raise SystemExit("split.py needs Python 3.12+")

from tldw_Server_API.tests.lint import test_charactersragdb_delegated_callers_ratchet as receivers  # noqa: E402

CLASS = "CharactersRAGDB"
DB_MODULE = "tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py"
STORES = "tldw_Server_API/app/core/DB_Management/chacha/"
SCANNED = ("tldw_Server_API/app", "tldw_Server_API/tests")


class Hit(NamedTuple):
    kind: str  # load | probe | assign | classref | patch
    node: ast.AST
    name: str
    receiver: ast.expr | None
    rewritable: bool


def _is_store(node: ast.expr, attr: str) -> bool:
    """True for ``*.<attr>`` and for the ``getattr(x, "<attr>", ...)`` a probe rewrite leaves."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
        return len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value == attr
    return receivers.last_identifier(node) == attr


def _class_has(cls: ast.ClassDef, attr: str) -> bool:
    for node in ast.walk(cls):
        if isinstance(node, ast.Attribute) and node.attr == attr and isinstance(node.ctx, ast.Store):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == attr:
            return True
        if isinstance(node, ast.Name) and node.id == attr and isinstance(node.ctx, ast.Store):
            return True
    return False


def _defined(cls: ast.ClassDef, names: set[str]) -> set[str]:
    return {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names}


def scan(tree: ast.Module, names: set[str], attr: str, in_db_module: bool) -> tuple[list[Hit], list[str]]:
    """The references to ``names`` that bypass ``attr``, and notes on ``self.<name>`` in classes lacking it."""
    hits: list[Hit] = []
    notes: list[str] = []

    def walk(node: ast.AST, classes: tuple[ast.ClassDef, ...]) -> None:
        if isinstance(node, ast.ClassDef):
            classes += (node,)
        for child in ast.iter_child_nodes(node):
            walk(child, classes)
        if isinstance(node, ast.Attribute) and node.attr in names:
            recv = node.value
            if _is_store(recv, attr):
                return
            if isinstance(recv, ast.Name) and recv.id == "self":
                if in_db_module and classes and classes[-1].name == CLASS:
                    hits.append(Hit("load", node, node.attr, recv, isinstance(node.ctx, ast.Load)))
                elif classes and node.attr not in _defined(classes[-1], names):
                    notes.append(f"{node.lineno}: self.{node.attr} in {classes[-1].name}, which does not define it")
            elif isinstance(recv, ast.Name) and recv.id == CLASS or receivers.last_identifier(recv) == CLASS:
                hits.append(Hit("classref", node, node.attr, recv, False))
            elif isinstance(node.ctx, ast.Load):
                hits.append(Hit("load", node, node.attr, recv, True))
            else:
                hits.append(Hit("assign", node, node.attr, recv, False))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in names
            and not _is_store(node.args[0], attr)
        ):
            hits.append(Hit("probe", node, node.args[1].value, node.args[0], True))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for name in names:
                if node.value.endswith(f"{CLASS}.{name}"):
                    hits.append(Hit("patch", node, name, None, False))

    walk(tree, ())
    return hits, notes


def _py_files(root: Path) -> Iterator[tuple[str, Path]]:
    for base in SCANNED:
        for path in sorted((root / base).rglob("*.py")):
            yield path.relative_to(root).as_posix(), path


def _parse(path: Path) -> tuple[str, ast.Module]:
    text = path.read_text(encoding="utf-8")
    return text, ast.parse(text, filename=str(path))


def collisions(root: Path, names: set[str]) -> set[str]:
    """Names another app class also defines, outside ChaChaNotes_DB.py and chacha/."""
    found: set[str] = set()
    for key, path in _py_files(root):
        if not key.startswith(SCANNED[0]) or key == DB_MODULE or key.startswith(STORES):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                found |= _defined(node, names)
    return found


def _is_store_module(key: str, tree: ast.Module, names: set[str]) -> bool:
    return key.startswith(STORES) and any(
        isinstance(n, ast.ClassDef) and _defined(n, names) for n in ast.walk(tree)
    )


def _edits(hits: list[Hit], attr: str) -> list[tuple[int, int, str]]:
    """(line, col, text) insertions, with positions as AST gives them (col in UTF-8 bytes)."""
    out: list[tuple[int, int, str]] = []
    for hit in hits:
        if not hit.rewritable:
            continue
        if hit.kind == "load":
            recv = hit.node.value  # type: ignore[attr-defined]
            out.append((recv.end_lineno, recv.end_col_offset, f".{attr}"))
        else:
            recv = hit.node.args[0]  # type: ignore[attr-defined]
            out.append((recv.lineno, recv.col_offset, "getattr("))
            out.append((recv.end_lineno, recv.end_col_offset, f', "{attr}", None)'))
    return out


def apply(text: str, edits: list[tuple[int, int, str]]) -> str:
    raw = text.encode("utf-8")
    starts = [0]
    for line in raw.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    # Right to left; for one offset the later-listed edit goes first, so a pair opens and closes in order.
    ordered = sorted(enumerate(edits), key=lambda e: (starts[e[1][0] - 1] + e[1][1], e[0]), reverse=True)
    for _, (line, col, insert) in ordered:
        at = starts[line - 1] + col
        raw = raw[:at] + insert.encode("utf-8") + raw[at:]
    return raw.decode("utf-8")


def _spec_mock(tree: ast.Module) -> bool:
    return any(
        isinstance(n, ast.Call)
        and any(k.arg in {"spec", "spec_set"} and isinstance(k.value, ast.Name) and k.value.id == CLASS for k in n.keywords)
        for n in ast.walk(tree)
    )


def _fakes(tree: ast.Module, names: set[str], attr: str) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name != CLASS and _defined(node, names) and not _class_has(node, attr):
            yield f"{node.lineno}: class {node.name} defines {sorted(_defined(node, names))} but has no {attr}"


def run(root: Path, attr: str, wanted: set[str], check: bool, show_tests: bool) -> int:
    clash = collisions(root, wanted)
    names = wanted - clash
    for name in sorted(clash):
        print(f"collision {name}: another app class defines it; not rewritten")
    problems = rewritten = 0
    touched: list[str] = []
    for key, path in _py_files(root):
        text, tree = _parse(path)
        if _is_store_module(key, tree, names):
            continue
        hits, notes = scan(tree, names, attr, key == DB_MODULE)
        if not check:
            for note in notes:
                print(f"note {key}:{note}")
            if not any(h.rewritable for h in hits):
                continue
            for h in hits:
                if h.rewritable and not (isinstance(h.receiver, ast.Name) and h.receiver.id == "self") and not receivers.counts(h.name, h.receiver):
                    print(f"unusual receiver {key}:{h.node.lineno}: {ast.unparse(h.receiver)}.{h.name}")
            if _spec_mock(tree):
                print(f"spec mock {key}: needs {attr} = MagicMock()")
            path.write_text(apply(text, _edits(hits, attr)), encoding="utf-8")
            rewritten += sum(h.rewritable for h in hits)
            touched.append(key)
            continue
        found = [f"{h.node.lineno}: {h.kind} {h.name}" for h in hits]
        if key.startswith(SCANNED[1]):
            found += list(_fakes(tree, names, attr))
        for line in found:
            print(f"left {key}:{line}")
        problems += len(found)
    if check:
        print(f"{problems} left")
        return 1 if problems else 0
    print(f"rewrote {rewritten} sites in {len(touched)} files")
    if show_tests:
        print("\n".join(k for k in touched if k.startswith(SCANNED[1])))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    callers = sub.add_parser("callers", help="route delegated-name callers through the store attribute")
    callers.add_argument("--store", required=True)
    callers.add_argument("--names", required=True)
    callers.add_argument("--check", action="store_true")
    callers.add_argument("--tests", action="store_true")
    callers.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    names = {n for n in args.names.split(",") if n}
    return run(args.root.resolve(), args.store, names, args.check, args.tests)


if __name__ == "__main__":
    raise SystemExit(main())
