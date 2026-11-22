"""Rewrite callers of delegated CharactersRAGDB names through their store (TASK-13950.22).

    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split callers --store ATTR --names N1,N2 [--check] [--tests]
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split rewire --store <store module path>
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split patches --names N1,N2

``callers`` rewrites, under tldw_Server_API/app and tldw_Server_API/tests, ``<recv>.<name>`` to
``<recv>.<ATTR>.<name>`` and ``getattr/hasattr(<recv>, "<name>", ...)`` to a probe of ``<recv>.<ATTR>``.
``--check`` writes nothing and exits 1 while anything still reaches a name outside ``*.<ATTR>``,
string-argument patches included (``monkeypatch.setattr(db, "n", f)``, ``patch.object(db, "n")``, which are inert on a mock).
``rewire`` turns each ``self._db.<a>`` of a store module into a ``self._ctx`` read (R14, TASK-13950.22 PR C1).
``patches`` lists every patch of a name (R25), so a test cannot patch a name the code no longer runs through.
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


# ``rewire`` tables. A key is a DB member and a value is the ChaChaContext member that replaces it.
CTX_REWRITES = {
    "_normalize_nullable_text": "normalize_nullable_text",
    "_get_current_utc_timestamp_iso": "utc_now_iso",
    "_generate_uuid": "new_uuid",
    "_get_current_db_version": "current_db_version",
    "_map_table_for_backend": "map_table",
    "_selected_owner_filter": "owner_filter",
    "_prepare_backend_statement": "prepare_statement",
    "_CURRENT_SCHEMA_VERSION": "code_schema_version",
    "_SCHEMA_NAME": "schema_name",
}
# Public DB names the context exposes under the same name.
CTX_PUBLIC = frozenset(
    {
        "backend",
        "backend_type",
        "client_id",
        "owner_user_id",
        "db_path_str",
        "is_memory_db",
        "transaction",
        "get_connection",
        "execute_query",
        "execute_many",
    }
)
# Class constants that became module constants of the store (Task 3), read bare.
CTX_MODULE_CONSTANTS = frozenset(
    {
        "_ALLOWED_CONVERSATION_STATES",
        "_ALLOWED_CONVERSATION_CHARACTER_SCOPES",
        "_ALLOWED_CONVERSATION_SEARCH_ORDER",
        "_ALLOWED_CONVERSATION_ASSISTANT_KINDS",
        "_ALLOWED_PERSONA_MEMORY_MODES",
        "_ALLOWED_SCOPE_TYPES",
        "_DEFAULT_CONVERSATION_STATE",
    }
)
# A private DB member made public in the same PR: reached as ``ctx.db.<new>``.
CTX_DB_RENAMES = {"_ensure_message_metadata_table": "ensure_message_metadata_table"}


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


def references(tree: ast.Module, names: set[str]) -> bool:
    """Any mention of a name: an attribute, a string (probe or patch target), or a class def."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in names:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in names or any(node.value.endswith(f".{n}") for n in names):
                return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            return True
    return False


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


def _target_ok(node: ast.expr, ok_ids: tuple[str, ...]) -> bool:
    return receivers.last_identifier(node) in ok_ids


def _store_ids(tree: ast.Module, attr: str) -> tuple[str, ...]:
    """Names that stand for the store: ``attr``, its class, and a local bound from ``<x>.<attr>``."""
    ids = [attr, "".join(part.title() for part in attr.split("_"))]
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == attr
            and all(isinstance(t, ast.Name) for t in node.targets)
        ):
            ids += [t.id for t in node.targets]  # type: ignore[attr-defined]
    return tuple(ids)


def _is_patch_call(func: ast.expr, member: str) -> bool:
    return (
        isinstance(func, ast.Attribute)
        and func.attr == member
        and receivers.last_identifier(func.value) == "patch"
    )


def patch_hits(tree: ast.Module, names: set[str], ok_ids: tuple[str, ...]) -> list[tuple[int, str, str]]:
    """(line, form, name) for every R25 patch of ``names`` whose target is not named in ``ok_ids``."""
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func, args = node.func, node.args
            is_set = (isinstance(func, ast.Attribute) and func.attr in {"setattr", "delattr"}) or (
                isinstance(func, ast.Name) and func.id in {"setattr", "delattr"}
            )
            named = len(args) >= 2 and isinstance(args[1], ast.Constant) and args[1].value in names
            if is_set and named and not _target_ok(args[0], ok_ids):
                found.append((node.lineno, "setattr", args[1].value))
            elif _is_patch_call(func, "object") and named and not _target_ok(args[0], ok_ids):
                found.append((node.lineno, "patch.object", args[1].value))
            elif _is_patch_call(func, "multiple") and args and not _target_ok(args[0], ok_ids):
                found += [(node.lineno, "patch.multiple", k.arg) for k in node.keywords if k.arg in names]
            elif (
                (isinstance(func, ast.Attribute) and func.attr == "getsource")
                or (isinstance(func, ast.Name) and func.id == "getsource")
            ) and args and isinstance(args[0], ast.Attribute) and args[0].attr in names:
                if not _target_ok(args[0].value, ok_ids):
                    found.append((node.lineno, "getsource", args[0].attr))
            if not _is_patch_call(func, "multiple"):
                found += [(node.lineno, "kwarg", k.arg) for k in node.keywords if k.arg in names]
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and target.attr in names and not _target_ok(target.value, ok_ids):
                    found.append((node.lineno, "assign", target.attr))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found += [(node.lineno, "string", n) for n in names if node.value.endswith(f"{CLASS}.{n}")]
    return sorted(set(found))


def _is_test_file(key: str) -> bool:
    return key.startswith(SCANNED[1]) or "/tests/" in key


def run_patches(root: Path, names: set[str]) -> int:
    total = 0
    for key, path in _py_files(root):
        if not _is_test_file(key):
            continue
        for line, form, name in patch_hits(_parse(path)[1], names, ("ctx",)):
            print(f"{key}:{line}: {form} {name}")
            total += 1
    print(f"{total} patches")
    return 1 if total else 0


def _replacement(attr: str) -> str | None:
    """The text for ``self._db.<attr>``, or None when it is an unmapped private."""
    if attr in CTX_REWRITES:
        return f"self._ctx.{CTX_REWRITES[attr]}"
    if attr in CTX_PUBLIC:
        return f"self._ctx.{attr}"
    if attr in CTX_MODULE_CONSTANTS:
        return attr
    if attr in CTX_DB_RENAMES:
        return f"self._ctx.db.{CTX_DB_RENAMES[attr]}"
    return None if attr.startswith("_") else f"self._ctx.db.{attr}"


def _is_self_db(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_db"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def rewire(text: str, tree: ast.Module) -> tuple[list[tuple[int, int, int, int, str]], list[str]]:
    """Span replacements that route ``self._db`` through ``self._ctx``, and the unmapped privates."""
    edits: list[tuple[int, int, int, int, str]] = []
    unmapped: list[str] = []
    inside: set[int] = set()  # ids of ``self._db`` nodes already handled as a receiver
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_self_db(node.value):
            inside.add(id(node.value))
            new = _replacement(node.attr)
            if new is None:
                unmapped.append(f"{node.lineno}: self._db.{node.attr}")
            else:
                edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, new))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(_is_self_db(t) for t in node.targets):
            edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, "self._ctx = ctx"))
            inside.update(id(t) for t in node.targets)
        elif _is_self_db(node) and id(node) not in inside:
            edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, "self._ctx.db"))
    return edits, unmapped


def replace_spans(text: str, edits: list[tuple[int, int, int, int, str]]) -> str:
    raw = text.encode("utf-8")
    starts = [0]
    for line in raw.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    for l1, c1, l2, c2, new in sorted(edits, reverse=True):
        raw = raw[: starts[l1 - 1] + c1] + new.encode("utf-8") + raw[starts[l2 - 1] + c2 :]
    return raw.decode("utf-8")


def run_rewire(root: Path, store: str) -> int:
    path = root / store
    text, tree = _parse(path)
    edits, unmapped = rewire(text, tree)
    if unmapped:
        print("unmapped private DB members (resolve them in CTX_REWRITES, then re-run):")
        print("\n".join(f"  {store}:{u}" for u in unmapped))
        return 1
    path.write_text(replace_spans(text, edits), encoding="utf-8")
    print(f"rewired {len(edits)} sites in {store}")
    return 0


def run(root: Path, attr: str, wanted: set[str], check: bool, show_tests: bool) -> int:
    clash = collisions(root, wanted)
    names = wanted - clash
    for name in sorted(clash):
        print(f"collision {name}: another app class defines it; not rewritten")
    problems = rewritten = 0
    touched: list[str] = []
    referencing: list[str] = []
    for key, path in _py_files(root):
        text, tree = _parse(path)
        if _is_store_module(key, tree, names):
            continue
        hits, notes = scan(tree, names, attr, key == DB_MODULE)
        if not check and key.startswith(SCANNED[1]) and (hits or references(tree, names)):
            referencing.append(key)
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
        if _is_test_file(key):  # assign and string patches are already reported by scan
            found += [
                f"{line}: patch {form} {name}"
                for line, form, name in patch_hits(tree, names, _store_ids(tree, attr))
                if form not in {"assign", "string"}
            ]
        for line in found:
            print(f"left {key}:{line}")
        problems += len(found)
    if check:
        print(f"{problems} left")
        return 1 if problems else 0
    print(f"rewrote {rewritten} sites in {len(touched)} files")
    if show_tests:
        print("\n".join(referencing))  # every test file naming one, so a re-run lists the same files
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    callers = sub.add_parser("callers", help="route delegated-name callers through the store attribute")
    callers.add_argument("--store", required=True)
    callers.add_argument("--names", required=True)
    callers.add_argument("--check", action="store_true")
    callers.add_argument("--tests", action="store_true")
    rewire_cmd = sub.add_parser("rewire", help="route a store module's self._db reads through self._ctx")
    rewire_cmd.add_argument("--store", required=True, help="the store module path, relative to the root")
    patches_cmd = sub.add_parser("patches", help="list every patch of the given names")
    patches_cmd.add_argument("--names", required=True)
    for cmd in (callers, rewire_cmd, patches_cmd):
        cmd.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.cmd == "rewire":
        return run_rewire(root, args.store)
    names = {n for n in args.names.split(",") if n}
    if args.cmd == "patches":
        return run_patches(root, names)
    return run(root, args.store, names, args.check, args.tests)


if __name__ == "__main__":
    raise SystemExit(main())
