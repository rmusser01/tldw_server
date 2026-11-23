"""Rewrite callers of delegated CharactersRAGDB names through their store (TASK-13950.22).

    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split callers --store ATTR --names N1,N2 [--check] [--tests]
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split rewire --store <store module path>
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split patches (--names N1,N2 | MANIFEST...)
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split store MANIFEST...
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split functions MANIFEST...
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split select GROUP
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split verify MANIFEST... --base REF
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split check MANIFEST...
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split tests MANIFEST...
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split select schema-sql
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split constants MANIFEST...
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split registry-snapshot

``callers`` rewrites, under tldw_Server_API/app and tldw_Server_API/tests, ``<recv>.<name>`` to
``<recv>.<ATTR>.<name>`` and ``getattr/hasattr(<recv>, "<name>", ...)`` to a probe of ``<recv>.<ATTR>``.
``--check`` writes nothing and exits 1 while anything still reaches a name outside ``*.<ATTR>``,
string-argument patches included (``monkeypatch.setattr(db, "n", f)``, ``patch.object(db, "n")``, which are inert on a mock).
``rewire`` turns each ``self._db.<a>`` of a store module into a ``self._ctx`` read (R14, TASK-13950.22 PR C1).
``patches`` lists every patch of a name (R25), so a test cannot patch a name the code no longer runs through.
``store`` moves the manifest's defs from CharactersRAGDB into a store class, rewriting each ``self.<attr>`` (R8, Appendix A.3),
applies the manifest's ``renames`` and routes every caller through the store. ``verify`` proves the result against a base ref.
``check`` and ``tests`` list what a manifest still leaves behind. A manifest (``manifests/t<NN>_<name>.py``) is one literal, read, never imported.
``constants`` moves a manifest's class and module constants, values untouched, into a dependency-free module and re-imports them (R4, R20).
``functions`` moves a manifest's defs into module functions that take ``ctx`` (A.3, R24): ``self`` becomes ``ctx``, a call of a moved def
gets ``ctx`` inserted, a shim keeps a forwarder, and callers in app and tests go through the module.
``select GROUP`` prints the names a group moves (A.5): schema-sql, or the function groups of Tasks 6-9. Edits use AST positions (UTF-8 byte offsets), applied right to left, so no string value changes.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import difflib
import io
import re
import subprocess  # nosec B404  # fixed argv, no shell
import symtable
import sys
import tokenize
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
    "_runtime_schema_version": "runtime_schema_version",
    "_map_table_for_backend": "map_table",
    "_selected_owner_filter": "owner_filter",
    "_prepare_backend_statement": "prepare_statement",
    "_CURRENT_SCHEMA_VERSION": "code_schema_version",
    "_POSTGRES_SCHEMA_VERSION": "postgres_schema_version",
    "_run_sqlite_linear_migration_step": "run_sqlite_migration_step",
    "_SCHEMA_NAME": "schema_name",
    "_selected_keyword_link_filter": "keyword_link_filter",
    "_require_selected_owner_row": "require_owner_row",
    "_case_insensitive_order_expression": "case_insensitive_order_expression",
    "_case_insensitive_order_clause": "case_insensitive_order_clause",
    "_unique_text_match_sql": "unique_text_match_sql",
    "_is_unique_violation": "is_unique_violation",
    "_chunk_list": "chunk_list",
    "_canonical_json_text_v61": "canonical_json_text_v61",
    "_note_task_v60_hash": "note_task_v60_hash",
    "_deserialize_row_fields": "deserialize_row_fields",
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
CTX_MODULE_CONSTANTS = {
    "_ALLOWED_CONVERSATION_STATES": "conversation_store",
    "_ALLOWED_CONVERSATION_CHARACTER_SCOPES": "conversation_store",
    "_ALLOWED_CONVERSATION_SEARCH_ORDER": "conversation_store",
    "_ALLOWED_CONVERSATION_ASSISTANT_KINDS": "conversation_store",
    "_ALLOWED_PERSONA_MEMORY_MODES": "conversation_store",
    "_ALLOWED_SCOPE_TYPES": "conversation_store",
    "_DEFAULT_CONVERSATION_STATE": "conversation_store",
    "_SQLITE_PARAM_LIMIT": "note_store",
}  # constant -> the chacha/ module that defines it; pinned by test_every_split_rewrite_target_is_a_context_member
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
    camel = "".join(part.title() for part in attr.split("_"))  # the store class: ``WorkspaceStore.<name>``
    return receivers.last_identifier(node) in {attr, camel}


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


_PARSED: dict[tuple[str, int, int], tuple[str, ast.Module]] = {}


def _parse(path: Path) -> tuple[str, ast.Module]:
    """The text and tree of a file, cached by (path, mtime, size) so one run parses each file once."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _PARSED:
        text = path.read_text(encoding="utf-8")
        _PARSED[key] = (text, ast.parse(text, filename=str(path)))
    return _PARSED[key]


def collisions(root: Path, names: set[str]) -> set[str]:
    """Names another app class also defines, outside ChaChaNotes_DB.py and chacha/."""
    found: set[str] = set()
    for key, path in _py_files(root):
        if not key.startswith(SCANNED[0]) or key == DB_MODULE or key.startswith(STORES):
            continue
        for node in ast.walk(_parse(path)[1]):
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


FAKE_FACTORIES = frozenset({"SimpleNamespace", "Mock", "MagicMock"})


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
            if receivers.last_identifier(func) in FAKE_FACTORIES:  # a kwarg on any other call is just a kwarg
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


def _replacement(attr: str, renames: dict[str, str] | None = None) -> str | None:
    """The text for ``self._db.<attr>``, or None when it is an unmapped private."""
    if attr in CTX_REWRITES:
        return f"self._ctx.{CTX_REWRITES[attr]}"
    if attr in CTX_PUBLIC:
        return f"self._ctx.{attr}"
    if attr in CTX_MODULE_CONSTANTS:
        return attr
    renames = {**CTX_DB_RENAMES, **(renames or {})}
    if attr in renames:
        return f"self._ctx.db.{renames[attr]}"
    return None if attr.startswith("_") else f"self._ctx.db.{attr}"


def _is_self_db(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_db"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def rewire(text: str, tree: ast.Module, renames: dict[str, str] | None = None) -> tuple[list[tuple[int, int, int, int, str]], list[str]]:
    """Span replacements that route ``self._db`` through ``self._ctx``, and the unmapped privates."""
    edits: list[tuple[int, int, int, int, str]] = []
    unmapped: list[str] = []
    inside: set[int] = set()  # ids of ``self._db`` nodes already handled as a receiver
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _is_self_db(node.value):
            inside.add(id(node.value))
            new = _replacement(node.attr, renames)
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
    edits, unmapped = rewire(text, tree, all_renames(root))
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


# --- Manifests, ``store``, ``verify``, ``check``, ``tests`` (TASK-13950.22 PR C2) ------------------------------------

MANIFEST_DIR = "Helper_Scripts/chacha_split/manifests"
LIST_KEYS = ("names", "shims", "structural", "constants", "class_constants")
DICT_KEYS = ("rewrites", "renames")


def load_manifest(path: Path) -> dict:
    """The ``MANIFEST`` literal of a manifest file, read with ``ast.literal_eval`` and never imported (R23)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MANIFEST" for t in node.targets):
            try:
                manifest = ast.literal_eval(node.value)
            except ValueError as exc:
                raise SystemExit(f"{path}: MANIFEST must be a plain literal (it is read, never imported): {exc}") from exc
            for key in LIST_KEYS:
                manifest.setdefault(key, [])
            for key in DICT_KEYS:
                manifest.setdefault(key, {})
            manifest.setdefault("store_class", None)
            manifest.setdefault("store_attr", None)
            return manifest
    raise SystemExit(f"{path}: no MANIFEST assignment")


def committed_manifests(root: Path) -> list[dict]:
    return [load_manifest(p) for p in sorted((root / MANIFEST_DIR).glob("t*.py"))]


def all_renames(root: Path) -> dict[str, str]:
    """Every private DB name made public by a committed manifest, as old -> new."""
    renames: dict[str, str] = {}
    for manifest in committed_manifests(root):
        renames.update(manifest["renames"])
    return renames


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
    if len(found) != 1:
        raise SystemExit(f"expected one class {name}, found {len(found)}")
    return found[0]


def _defs(body: list[ast.stmt]) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {n.name: n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _span(lines: list[str], node: ast.AST) -> tuple[int, int]:
    """0-based [start, end) lines of a def: decorators, the body, and the comment lines directly above."""
    start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))]) - 1
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    return start, node.end_lineno  # type: ignore[attr-defined]


def _moved_index(root: Path, manifests: list[dict]) -> dict[str, str]:
    """Moved name -> store attribute, over every committed manifest and the ones being applied."""
    index: dict[str, str] = {}
    for manifest in [*committed_manifests(root), *manifests]:
        if manifest["store_attr"]:
            index.update(dict.fromkeys(manifest["names"], manifest["store_attr"]))
    return index


class Unmapped(Exception):
    """A private DB name no rule maps (A.3 rule 4: fail closed)."""


def store_replacement(attr: str, manifest: dict, index: dict[str, str], renames: dict[str, str]) -> str | None:
    """The text for ``self.<attr>`` inside a def moved into a store, or None to keep it as ``self.<attr>``."""
    if attr == "ctx":
        return "self._ctx"
    store_attr = manifest["store_attr"]
    if index.get(attr) == store_attr:
        return None
    if attr in index:
        return f"self._ctx.db.{index[attr]}.{attr}"
    if attr in manifest["class_constants"]:  # a constant only this group reads: a module constant of the store module
        return attr
    if attr in manifest["rewrites"]:
        value = manifest["rewrites"][attr]
        return "self._ctx" + value[3:] if value.startswith("ctx") else value
    if attr in CTX_REWRITES:
        return f"self._ctx.{CTX_REWRITES[attr]}"
    if attr in CTX_PUBLIC:
        return f"self._ctx.{attr}"
    if attr in CTX_MODULE_CONSTANTS:
        return attr
    if attr in renames:
        return f"self._ctx.db.{renames[attr]}"
    if not attr.startswith("_"):
        return f"self._ctx.db.{attr}"
    raise Unmapped(attr)


def _is_self(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "self"


STORE_DECORATORS = frozenset({"staticmethod", "classmethod", "overload"})


def class_receiver_replacement(recv: str, attr: str, manifest: dict, index: dict[str, str]) -> str | None:
    """The text for ``cls.<attr>`` or ``CharactersRAGDB.<attr>`` inside a def moved into a store.

    A name of the group is a member of the store class; a constant only the group reads is a module constant of the store
    module. Anything else has no class to reach and fails closed.
    """
    if index.get(attr) == manifest["store_attr"]:
        return f"{manifest['store_class']}.{attr}" if recv == CLASS else None
    if attr in manifest["class_constants"]:
        return attr
    raise Unmapped(attr)


def _store_edits(
    node: ast.AST, manifest: dict, index: dict[str, str], renames: dict[str, str]
) -> tuple[list[tuple[int, int, int, int, str]], list[str]]:
    edits: list[tuple[int, int, int, int, str]] = []
    problems: list[str] = []
    handled: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in {"cls", CLASS}:
            try:
                new = class_receiver_replacement(n.value.id, n.attr, manifest, index)
            except Unmapped:
                problems.append(f"{n.lineno}: {n.value.id}.{n.attr}")
                continue
            if new is not None:
                edits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, new))
        elif isinstance(n, ast.Attribute) and _is_self(n.value):
            handled.add(id(n.value))
            if not isinstance(n.ctx, ast.Load):
                problems.append(f"{n.lineno}: self.{n.attr} is assigned")
                continue
            try:
                new = store_replacement(n.attr, manifest, index, renames)
            except Unmapped:
                problems.append(f"{n.lineno}: self.{n.attr}")
                continue
            if new is not None:
                edits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, new))
        elif (isinstance(n, ast.Name) and n.id == "ctx" and not isinstance(n.ctx, ast.Load)) or (
            isinstance(n, ast.arg) and n.arg == "ctx"
        ):
            problems.append(f"{n.lineno}: the body binds a name ctx")
    for n in ast.walk(node):
        if _is_self(n) and isinstance(n.ctx, ast.Load) and id(n) not in handled:  # type: ignore[attr-defined]
            edits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, "self._ctx.db"))  # type: ignore[attr-defined]
    return edits, problems


def _lines(text: str) -> list[str]:
    return text.split("\n")


def _all_defs(body: list[ast.stmt]) -> dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every def of each name, in source order: a function with ``@overload`` stubs has more than one."""
    found: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for n in body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.setdefault(n.name, []).append(n)
    return found


def free_names(def_text: str, node: ast.AST) -> set[str]:
    """Module-level names a moved def reads: symtable globals, plus the names its annotations use."""
    table = symtable.symtable("from __future__ import annotations\nclass _W:\n" + def_text + "\n", "<moved>", "exec")
    found: set[str] = set()

    def visit(t: symtable.SymbolTable) -> None:
        found.update(s.get_name() for s in t.get_symbols() if s.is_global() and s.is_referenced())
        for child in t.get_children():
            visit(child)

    visit(table.get_children()[0])
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            notes = [n.returns, *(a.annotation for a in [*n.args.posonlyargs, *n.args.args, *n.args.kwonlyargs, n.args.vararg, n.args.kwarg] if a)]
            found.update(x.id for note in notes if note for x in ast.walk(note) if isinstance(x, ast.Name))
        elif isinstance(n, ast.AnnAssign):  # symtable skips the annotation of a local, but ruff F821 does not
            found.update(x.id for x in ast.walk(n.annotation) if isinstance(x, ast.Name))
    return {n for n in found if not hasattr(builtins, n)}


def module_names(tree: ast.Module) -> dict[str, ast.stmt]:
    """Top-level bound names of a module (imports, defs, classes, assignments), with the statement that binds each."""
    names: dict[str, ast.stmt] = {}

    def visit(statements: list[ast.stmt]) -> None:
        for st in statements:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                for alias in st.names:
                    names[(alias.asname or alias.name).split(".")[0]] = st
            elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names[st.name] = st
            elif isinstance(st, ast.Assign):
                names.update({t.id: st for t in st.targets if isinstance(t, ast.Name)})
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                names[st.target.id] = st
            elif isinstance(st, (ast.If, ast.Try)):
                for block in _blocks_of(st):
                    visit(block)

    visit(tree.body)
    return names


def _blocks_of(st: ast.stmt) -> list[list[ast.stmt]]:
    blocks = [getattr(st, f, []) for f in ("body", "orelse", "finalbody")]
    return [*blocks, *(h.body for h in getattr(st, "handlers", []))]


def _import_line(name: str, stmt: ast.stmt) -> str:
    """The import that binds ``name`` the way ChaChaNotes_DB binds it, or an import from ChaChaNotes_DB."""
    if isinstance(stmt, ast.ImportFrom):
        alias = next(a for a in stmt.names if (a.asname or a.name) == name)
        suffix = f" as {alias.asname}" if alias.asname else ""
        return f"from {'.' * stmt.level}{stmt.module or ''} import {alias.name}{suffix}"
    if isinstance(stmt, ast.Import):
        alias = next(a for a in stmt.names if (a.asname or a.name).split(".")[0] == name)
        return f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
    return f"from {DB_MODULE[:-3].replace('/', '.')} import {name}"


def _after_imports(tree: ast.Module) -> int:
    """The 1-based line after the last top-level import of a module."""
    last = max((n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))), default=0)
    return last  # type: ignore[return-value]


def _move_into_store(root: Path, manifest: dict, index: dict[str, str], renames: dict[str, str]) -> int:
    source, target = root / manifest["source"], root / manifest["target"]
    text, tree = _parse(source)
    cls = _class_node(tree, CLASS)
    defs = _all_defs(cls.body)
    missing = [n for n in manifest["names"] if n not in defs]
    if missing:
        print(f"not in {CLASS}: {missing}")
        return 1
    edits: list[tuple[int, int, int, int, str]] = []
    problems: list[str] = []
    for name in manifest["names"]:
        for node in defs[name]:
            if any(ast.unparse(d) not in STORE_DECORATORS for d in node.decorator_list):
                problems.append(f"{node.lineno}: {name} has a decorator other than staticmethod, classmethod or overload (store mode fails closed)")
            new_edits, new_problems = _store_edits(node, manifest, index, renames)
            edits += new_edits
            problems += [f"{name}: {p}" for p in new_problems]
    if problems:
        print("unmapped private DB members (resolve them in the manifest's rewrites or renames, then re-run):")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    rewritten = replace_spans(text, edits)
    new_tree = ast.parse(rewritten)
    new_defs = _all_defs(_class_node(new_tree, CLASS).body)
    new_lines = _lines(rewritten)
    pieces: dict[str, list[str]] = {}
    free: set[str] = set()
    for name in manifest["names"]:
        pieces[name] = []
        for node in new_defs[name]:  # every def of a name: the overloads of a function come with it
            start, end = _span(new_lines, node)
            pieces[name] += new_lines[start:end]
            free |= free_names("\n".join(new_lines[start:end]), node)
    # the source: delete each def (right to left), and one blank line where two would meet
    lines = _lines(text)
    for node in sorted((d for n in manifest["names"] for d in defs[n]), key=lambda d: d.lineno, reverse=True):
        start, end = _span(lines, node)
        del lines[start:end]
        if 0 < start < len(lines) and not lines[start - 1].strip() and not lines[start].strip():
            del lines[start]
    constants = _take_constants(manifest, tree, _lines(text), lines)
    db_names = module_names(tree)
    # the target: imports, constants, then the defs at the end of the store class
    t_text, t_tree = _parse(target)
    t_lines = _lines(t_text)
    t_names = module_names(t_tree)
    group = {n for n, attr in index.items() if attr == manifest["store_attr"]}
    t_lines = _retarget_self_db(t_lines, t_tree, group)
    t_text = "\n".join(t_lines)
    t_tree = ast.parse(t_text)
    store = _class_node(t_tree, manifest["store_class"])
    t_lines = _lines(t_text)
    add = [name for name in sorted(free) if name not in t_names and name not in {c[0] for c in constants}]
    unknown = [n for n in add if n not in db_names]
    if unknown:
        print(f"free names with no source in ChaChaNotes_DB: {unknown}")
        return 1
    body: list[str] = []
    for name in manifest["names"]:
        body += ["", *pieces[name]]
    t_lines[store.end_lineno : store.end_lineno] = body
    insert_at = _after_imports(t_tree)
    extra = [_import_line(n, db_names[n]) for n in add]
    for _, const_lines in constants:
        extra += ["", *const_lines]
    t_lines[insert_at:insert_at] = extra
    target.write_text("\n".join(t_lines), encoding="utf-8")
    source.write_text("\n".join(lines), encoding="utf-8")  # unused imports are left to ruff F401 (C2)
    print(f"moved {len(manifest['names'])} defs into {manifest['store_class']}; imports added: {add}")
    return 0


def _take_constants(manifest: dict, tree: ast.Module, original: list[str], lines: list[str]) -> list[tuple[str, list[str]]]:
    """Cut each module constant (``constants``) and class constant (``class_constants``) the manifest lists out of the source
    lines, and return its text at module level."""
    taken: list[tuple[str, list[str]]] = []
    class_body = _class_node(tree, CLASS).body
    sites = [(name, tree.body, False) for name in manifest["constants"]] + [(name, class_body, True) for name in manifest["class_constants"]]
    for name, body, in_class in sites:
        node = next((n for n in body if _const_name(n) == name), None)
        if node is None:
            raise SystemExit(f"constant {name} is not an assignment in {CLASS if in_class else 'the module of ' + DB_MODULE}")
        piece = original[node.lineno - 1 : node.end_lineno]
        at = next(i for i, line in enumerate(lines) if line == piece[0] and lines[i : i + len(piece)] == piece)
        del lines[at : at + len(piece)]
        if 0 < at < len(lines) and not lines[at - 1].strip() and not lines[at].strip():
            del lines[at]
        taken.append((name, _dedent(piece) if in_class else piece))
    return taken


def _retarget_self_db(lines: list[str], tree: ast.Module, group: set[str]) -> list[str]:
    """In a store module, ``self._db.<moved>`` and ``self._ctx.db.<moved>`` become ``self.<moved>``."""
    edits: list[tuple[int, int, int, int, str]] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in group and ast.unparse(n.value) in {"self._db", "self._ctx.db"}:
            edits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, f"self.{n.attr}"))
    return _lines(replace_spans("\n".join(lines), edits))


def _rename_edits(tree: ast.Module, renames: dict[str, str], in_class: bool) -> list[tuple[int, int, int, int, str]]:
    edits: list[tuple[int, int, int, int, str]] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in renames:
            edits.append((n.end_lineno, n.end_col_offset - len(n.attr), n.end_lineno, n.end_col_offset, renames[n.attr]))  # type: ignore[arg-type]
    if in_class:
        for n in _class_node(tree, CLASS).body:
            if isinstance(n, ast.FunctionDef) and n.name in renames:
                edits.append((n.lineno, n.col_offset + 4, n.lineno, n.col_offset + 4 + len(n.name), renames[n.name]))
    return edits


def _apply_renames(root: Path, renames: dict[str, str]) -> int:
    count = 0
    for key, path in _py_files(root):
        text, tree = _parse(path)
        edits = _rename_edits(tree, renames, key == DB_MODULE)
        if edits:
            path.write_text(replace_spans(text, edits), encoding="utf-8")
            count += len(edits)
    return count


def run_store(root: Path, manifest_paths: list[Path]) -> int:
    manifests = [load_manifest(p) for p in manifest_paths]
    for manifest in manifests:
        if manifest["mode"] != "store":
            print(f"{manifest['task']}: mode {manifest['mode']!r} is not implemented")
            return 1
    index = _moved_index(root, manifests)
    renames = {**all_renames(root), **{k: v for m in manifests for k, v in m["renames"].items()}}
    for manifest in manifests:
        if _move_into_store(root, manifest, index, renames):
            return 1
    for manifest in manifests:
        if run(root, manifest["store_attr"], set(manifest["names"]), False, False):
            return 1
    print(f"renamed {_apply_renames(root, renames)} references")
    return 0


class _Rewriter(ast.NodeTransformer):
    """Apply A.3 rules 3-5 to a base def, the way ``store`` does on text."""

    def __init__(self, manifest: dict, index: dict[str, str], renames: dict[str, str]) -> None:
        self.manifest, self.index, self.renames = manifest, index, renames
        self.unmapped: list[str] = []

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: N802
        if isinstance(node.value, ast.Name) and node.value.id in {"cls", CLASS}:
            try:
                new = class_receiver_replacement(node.value.id, node.attr, self.manifest, self.index)
            except Unmapped:
                self.unmapped.append(node.attr)
                return node
            return node if new is None else ast.parse(new, mode="eval").body
        if _is_self(node.value) and isinstance(node.ctx, ast.Load):
            try:
                new = store_replacement(node.attr, self.manifest, self.index, self.renames)
            except Unmapped:
                self.unmapped.append(node.attr)
                return node
            return node if new is None else ast.parse(new, mode="eval").body
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if node.id == "self" and isinstance(node.ctx, ast.Load):
            return ast.parse("self._ctx.db", mode="eval").body
        return node


class _CallSites(ast.NodeTransformer):
    """Remaining class defs: ``self.<moved>`` goes through the store, a renamed attribute takes its new name."""

    def __init__(self, moved: dict[str, str], renames: dict[str, str]) -> None:
        self.moved, self.renames = moved, renames

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        if _is_self(node.value) and node.attr in self.moved:
            return ast.Attribute(ast.Attribute(node.value, self.moved[node.attr], ast.Load()), node.attr, node.ctx)
        if node.attr in self.renames:
            node.attr = self.renames[node.attr]
        return node


def _diff(old: ast.AST, new: ast.AST | None) -> list[str]:
    a = ast.unparse(old).splitlines()
    b = ast.unparse(new).splitlines() if new is not None else []
    return [f"    {line}" for line in difflib.unified_diff(a, b, "base", "head", lineterm="", n=1)]


def run_verify(root: Path, manifest_paths: list[Path], base: str) -> int:
    manifests = [load_manifest(p) for p in manifest_paths]
    if any(m["mode"] == "functions" for m in manifests):
        if any(m["mode"] not in {"functions", "constants"} for m in manifests):
            raise SystemExit("verify takes functions (and constants) manifests, or neither")
        return run_verify_functions(root, manifests, base)
    index = _moved_index(root, manifests)
    renames = {k: v for m in manifests for k, v in m["renames"].items()}
    structural = {n for m in manifests for n in m["structural"]}
    source = manifests[0]["source"]
    base_tree = ast.parse(_git_show(root, base, source))
    head_tree = ast.parse((root / source).read_text(encoding="utf-8"))
    base_defs, head_defs = _defs(_class_node(base_tree, CLASS).body), _defs(_class_node(head_tree, CLASS).body)
    failures: list[str] = []
    moved = 0
    constants_ok = 0
    for manifest in manifests:
        if manifest["mode"] == "constants":
            constants_ok += _verify_constants(root, manifest, base_tree, head_tree, failures)
            continue
        t_tree = ast.parse((root / manifest["target"]).read_text(encoding="utf-8"))
        store_defs = _defs(_class_node(t_tree, manifest["store_class"]).body)
        for name in manifest["names"]:
            if name in head_defs:
                failures.append(f"{name}: still on {CLASS}")
            if name not in base_defs or name not in store_defs:
                failures.append(f"{name}: missing from the {'base class' if name not in base_defs else 'store'}")
                continue
            rewriter = _Rewriter(manifest, index, renames)
            expected = rewriter.visit(copy.deepcopy(base_defs[name]))
            if rewriter.unmapped or ast.dump(expected) != ast.dump(store_defs[name]):
                failures.append(f"{name}: the moved body differs from the base")
                failures += _diff(expected, store_defs[name])
            else:
                moved += 1
        for name in manifest["constants"]:
            old = _module_constant(base_tree, name)
            new = _module_constant(t_tree, name)
            if old is None or new is None or ast.dump(old) != ast.dump(new):
                failures.append(f"{name}: the moved constant differs from the base")
    identical, structural_seen = 0, []
    moved_names = set(index) & {n for m in manifests for n in m["names"]}
    remaining_head = dict(head_defs)
    for name, node in base_defs.items():
        if name in moved_names:
            continue
        new_name = renames.get(name, name)
        head = remaining_head.pop(new_name, None)
        if name in structural or new_name in structural:
            structural_seen.append(new_name)
            print(f"structural {new_name}:")
            print("\n".join(_diff(node, head)))
            continue
        if head is None:
            failures.append(f"{name}: gone from {CLASS} and not declared structural")
            continue
        expected = _CallSites({n: index[n] for n in moved_names}, renames).visit(copy.deepcopy(node))
        expected.name = new_name
        if ast.dump(expected) == ast.dump(head):
            identical += 1
        else:
            failures.append(f"{name}: the class def differs from the base")
            failures += _diff(expected, head)
    for name in remaining_head:
        if name in structural:
            structural_seen.append(name)
        else:
            failures.append(f"{name}: a new def on {CLASS} that is not declared structural")
    for line in failures:
        print(line)
    if constants_ok:
        print(f"constants {constants_ok} identical")
    print(f"moved {moved} identical; class defs {identical} identical; structural {sorted(set(structural_seen))}")
    return 1 if failures else 0


def _module_constant(tree: ast.Module, name: str) -> ast.expr | None:
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n.value
    return None


def _git_show(root: Path, ref: str, path: str) -> str:
    result = subprocess.run(  # nosec B603 B607  # fixed argv, no shell
        ["git", "-C", str(root), "show", f"{ref}:{path}"], capture_output=True, text=True, check=True
    )
    return result.stdout


def watched(manifest: dict) -> set[str]:
    """The DB names a manifest retires: the moved ones, the renamed ones and the ones it rewrites."""
    return {*manifest["names"], *manifest["renames"], *manifest["rewrites"]}


def moved_free_names(root: Path, manifest: dict) -> set[str]:
    """Names the moved bodies read from ChaChaNotes_DB's own namespace (R30): ``time``, a helper function."""
    target = root / manifest["target"]
    if not target.exists() or not (manifest["store_class"] or manifest["mode"] == "functions"):
        return set()
    t_tree = ast.parse(target.read_text(encoding="utf-8"))
    defs = _defs(t_tree.body if manifest["mode"] == "functions" else _class_node(t_tree, manifest["store_class"]).body)
    db_names = set(module_names(ast.parse((root / manifest["source"]).read_text(encoding="utf-8"))))
    db_names |= set(module_names(t_tree))  # a name the move left unused in ChaChaNotes_DB (ruff F401 dropped its import) is still patched there
    found: set[str] = set()
    for name in manifest["names"]:
        if name in defs:
            found |= free_names(textwrap_indent(ast.unparse(defs[name])), defs[name]) & db_names
    return found


def textwrap_indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


def module_patch_hits(tree: ast.Module, names: set[str]) -> list[tuple[int, str, str]]:
    """(line, form, name) for patches that target a name on the ChaChaNotes_DB module (R30)."""
    aliases = {"ChaChaNotes_DB"}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            aliases |= {a.asname or a.name for a in n.names if a.name == "ChaChaNotes_DB"}
        elif isinstance(n, ast.Import):
            aliases |= {a.asname for a in n.names if a.asname and a.name.endswith("ChaChaNotes_DB")}
    found: list[tuple[int, str, str]] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call) or not n.args:
            continue
        func = receivers.last_identifier(n.func)
        first = n.args[0]
        if func in {"setattr", "delattr", "patch"} and isinstance(first, ast.Constant) and isinstance(first.value, str):
            found += [(n.lineno, f"module {func}", x) for x in names if first.value.endswith(f"ChaChaNotes_DB.{x}")]
        elif (
            (func in {"setattr", "delattr"} or _is_patch_call(n.func, "object"))
            and len(n.args) >= 2
            and isinstance(n.args[1], ast.Constant)
            and n.args[1].value in names
            and receivers.last_identifier(first) in aliases
        ):
            found.append((n.lineno, "module setattr", n.args[1].value))
    return sorted(set(found))


def _schema_reads(root: Path, manifest: dict) -> set[str]:
    """Class constants the moved bodies read, as ``schema_sql.<name>`` (A.6: a class patch of one no longer reaches the reader)."""
    target = root / manifest["target"]
    if not target.exists():
        return set()
    return {
        n.attr
        for n in ast.walk(ast.parse(target.read_text(encoding="utf-8")))
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "schema_sql"
    }


def _db_reads(root: Path) -> set[str]:
    tree = ast.parse((root / DB_MODULE).read_text(encoding="utf-8"))
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def run_manifest_patches(root: Path, manifest_paths: list[Path], names: set[str]) -> int:
    manifests = [load_manifest(p) for p in manifest_paths]
    watch = set(names)
    free: set[str] = set()
    ok_ids: tuple[str, ...] = ("ctx",)
    for manifest in manifests:
        watch |= watched(manifest)
        free |= moved_free_names(root, manifest)
        if manifest["store_attr"]:
            ok_ids += (manifest["store_attr"], "".join(p.title() for p in manifest["store_attr"].split("_")))
        if manifest["mode"] == "functions":
            ok_ids += (manifest["alias"],)
            watch |= _schema_reads(root, manifest)
            if _schema_reads(root, manifest):
                ok_ids += ("schema_sql",)
    reads = _db_reads(root) if (root / DB_MODULE).exists() else set()
    total = 0
    for key, path in _py_files(root):
        if not _is_test_file(key):
            continue
        tree = _parse(path)[1]
        for line, form, name in patch_hits(tree, watch, ok_ids):
            print(f"{key}:{line}: {form} {name}")
            total += 1
        for line, form, name in module_patch_hits(tree, free):
            print(f"{key}:{line}: {form} {name}" + (" (both sides: patch the store module too)" if name in reads else ""))
            total += 1
    print(f"{total} patches")
    return 1 if total else 0


def _word_files(root: Path, names: set[str]) -> list[str]:
    pattern = re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, names))) + r")\b")
    return [key for key, path in _py_files(root) if _is_test_file(key) and pattern.search(path.read_text(encoding="utf-8"))]


def run_tests(root: Path, manifest_paths: list[Path]) -> int:
    names: set[str] = set()
    for path in manifest_paths:
        manifest = load_manifest(path)
        names |= {*manifest["names"], *manifest["renames"]}
    print("\n".join(_word_files(root, names)) if names else "")
    return 0


def run_check(root: Path, manifest_paths: list[Path]) -> int:
    """Exit 1 on any reference left to a moved or renamed name, or to a free name through ChaChaNotes_DB."""
    bad = 0
    reads = _db_reads(root)
    loaded = [load_manifest(p) for p in manifest_paths]
    if any(m["mode"] == "functions" for m in loaded):
        return run_check_functions(root, loaded)
    for path in manifest_paths:
        manifest = load_manifest(path)
        names = {*manifest["names"], *manifest["renames"]}
        bad += run(root, manifest["store_attr"] or "", names, True, False)
        free = moved_free_names(root, manifest)
        for key, file in _py_files(root):
            if not _is_test_file(key):
                continue
            for line, form, name in module_patch_hits(_parse(file)[1], free):
                if name not in reads:  # the DB module no longer reads it, so the patch is inert
                    print(f"left {key}:{line}: {form} {name}")
                    bad += 1
    return 1 if bad else 0


# --- ``constants`` and ``select schema-sql`` (TASK-13950.22 PR D, R4/R20) ------------------------------------------

SCHEMA_SQL_RX = re.compile(r"(_MIGRATION_SQL_|_FULL_SCHEMA_SQL|_CHAT_MACROS_SCHEMA_SQL|_SHARED_WORKSPACE_CHAT_V61_POSTGRES_VERIFY_SQL|_NOTE_GRAPH_SUGGESTION_RECEIPT_DELETE|_HISTORY_PROJECTIONS_SCHEMA_SQL|_RETIRED_SYNC_PLACEHOLDER_REPAIR_SQL|^_FTS_CONFIG$|^_POSTGRES_SEQUENCE_TABLES$|^_SQLITE_(MIGRATION_FIXTURE_DROP|SCHEMA_TABLE_INFO|SCHEMA_INDEX_LIST)_STATEMENTS$|^_NOTE_TASK_V60_(TABLES|RELATIONS)$|^_NOTES_MOODBOARD_STUDIO_V61_RELATIONS$|^_SQLITE_CORE_SCHEMA_TABLES$)")
SCHEMA_SQL_DOC = (
    "Dependency-free constants of CharactersRAGDB: schema and migration SQL, and constants that moved "
    "chacha modules share with the class (TASK-13950.22)."
)
INDENT = "    "


def _const_name(node: ast.stmt) -> str | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
        return node.target.id
    return None


def _const_sites(tree: ast.Module) -> dict[str, tuple[str, ast.stmt]]:
    """Constant name -> (``module`` or ``class``, statement), for the module body and the CharactersRAGDB class body."""
    sites: dict[str, tuple[str, ast.stmt]] = {}
    for level, body in (("module", tree.body), ("class", _class_node(tree, CLASS).body)):
        for st in body:
            name = _const_name(st)
            if name and name in sites and sites[name][0] == level:
                raise SystemExit(f"{name} is assigned twice at {level} level (lines {sites[name][1].lineno} and {st.lineno}); the tool will not guess")
            if name:
                sites[name] = (level, st)
    return sites


def _value_reads(node: ast.stmt) -> set[str]:
    parts = [node.value, _unwrap_classvar(node)]
    return {x.id for p in parts if p is not None for x in ast.walk(p) if isinstance(x, ast.Name) and not hasattr(builtins, x.id)}


def select_schema_sql(tree: ast.Module) -> list[str]:
    """Class constants matching SCHEMA_SQL_RX, plus every constant their values read (A.5), in source order."""
    sites = _const_sites(tree)
    chosen = {n for n, (level, _) in sites.items() if level == "class" and SCHEMA_SQL_RX.search(n)}
    todo = list(chosen)
    while todo:
        for read in _value_reads(sites[todo.pop()][1]):
            if read in sites and read not in chosen:
                chosen.add(read)
                todo.append(read)
    return sorted(chosen, key=lambda n: sites[n][1].lineno)


def _unwrap_classvar(node: ast.stmt) -> ast.expr | None:
    """The annotation of an AnnAssign with ``ClassVar[T]`` read as ``T``."""
    note = getattr(node, "annotation", None)
    if isinstance(note, ast.Subscript) and ast.unparse(note.value).split(".")[-1] == "ClassVar":
        return note.slice
    return note


def _continuation_rows(text: str) -> set[int]:
    """1-based rows that start inside a multi-line string or f-string token: their text is the string's value."""
    rows: set[int] = set()
    starts: list[int] = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
            rows.update(range(tok.start[0] + 1, tok.end[0] + 1))
        elif tok.type == tokenize.FSTRING_START:
            starts.append(tok.start[0])
        elif tok.type == tokenize.FSTRING_END:
            first = starts.pop()
            rows.update(range(first + 1, tok.end[0] + 1))
    return rows


def _dedent(piece: list[str]) -> list[str]:
    skip = _continuation_rows("\n".join(piece) + "\n")
    out: list[str] = []
    for row, line in enumerate(piece, 1):
        if row in skip or not line.strip():
            out.append(line)
        elif line.startswith(INDENT):
            out.append(line[len(INDENT) :])
        else:
            raise SystemExit(f"cannot dedent line {line!r}")
    return out


def _import_block(module: str, names: list[str], indent: str, noqa: str) -> list[str]:
    return [f"{indent}from {module} import (  # {noqa}", *(f"{indent}{INDENT}{n}," for n in names), f"{indent})"]


def _dotted(path: str) -> str:
    return path[:-3].replace("/", ".")


def _move_constants(root: Path, manifest: dict) -> int:
    source, target = root / manifest["source"], root / manifest["target"]
    text = source.read_text(encoding="utf-8")
    sites = _const_sites(ast.parse(text))
    names = manifest["constants"]
    missing = [n for n in names if n not in sites]
    if missing:
        print(f"not a constant of {CLASS} or of the module: {missing}")
        return 1
    moved = set(names)
    foreign = {n: sorted(_value_reads(sites[n][1]) - moved) for n in names}
    foreign = {n: reads for n, reads in foreign.items() if reads}
    if foreign:
        print("constants whose value reads a name that does not move (add it to the manifest, or resolve it):")
        print("\n".join(f"  {n}: {reads}" for n, reads in foreign.items()))
        return 1
    # ClassVar[T] -> T first (same line count), then re-locate each statement
    edits = []
    for n in names:
        st = sites[n][1]
        note = getattr(st, "annotation", None)
        if note is not None and _unwrap_classvar(st) is not note:
            edits.append((note.lineno, note.col_offset, note.end_lineno, note.end_col_offset, ast.get_source_segment(text, note.slice)))
    rewritten = replace_spans(text, edits)
    lines = _lines(rewritten)
    new_sites = _const_sites(ast.parse(rewritten))
    order = sorted(names, key=lambda n: new_sites[n][1].lineno)
    spans = {}
    for n in order:
        start, end = _span(lines, new_sites[n][1])
        spans[n] = (start, end, new_sites[n][0])
    pieces: list[str] = []
    previous_end = None
    for n in order:
        start, end, level = spans[n]
        piece = lines[start:end]
        if level == "class":
            piece = _dedent(piece)
        if previous_end is not None and start != previous_end:
            pieces.append("")
        pieces += piece
        previous_end = end
    # the source: the first constant of each level becomes the import, the rest are deleted, right to left
    module_path = _dotted(manifest["target"])
    first = {level: next(n for n in order if spans[n][2] == level) for level in ("class", "module") if any(spans[n][2] == level for n in order)}
    for n in sorted(order, key=lambda n: spans[n][0], reverse=True):
        start, end, level = spans[n]
        if first[level] == n:
            names_here = [m for m in order if spans[m][2] == level]
            block = _import_block(module_path, names_here, INDENT if level == "class" else "", "noqa: F401" if level == "class" else "noqa: E402,F401")
            lines[start:end] = block
            continue
        del lines[start:end]
        if 0 < start < len(lines) and not lines[start - 1].strip() and not lines[start].strip():
            del lines[start]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():  # a later PR adds constants it found it needs: append, in source order
        target.write_text(target.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + "\n".join(pieces) + "\n", encoding="utf-8")
    else:
        header = [f'"""{SCHEMA_SQL_DOC}"""', "", "from __future__ import annotations", ""]
        target.write_text("\n".join([*header, *pieces]) + "\n", encoding="utf-8")
    source.write_text("\n".join(lines), encoding="utf-8")
    print(f"moved {len(order)} constants into {manifest['target']}")
    return 0


def run_constants(root: Path, manifest_paths: list[Path]) -> int:
    for path in manifest_paths:
        manifest = load_manifest(path)
        if manifest["mode"] != "constants":
            print(f"{manifest['task']}: mode {manifest['mode']!r} is not the constants mode")
            return 1
        if _move_constants(root, manifest):
            return 1
    return 0


# --- ``select`` for the function groups (TASK-13950.22 PR E, A.5) -------------------------------------------------

SELECT_GROUPS = ("schema-sql", "sqlite-schema", "postgres-schema", "migration-support", "migrate-sqlite", "migrate-postgres", "init")
# Names the bucket regexes catch by accident (A.5).
SELECT_FALSE_POSITIVES = frozenset(
    {
        "_ensure_skill_registry_table", "_ensure_json_string", "_ensure_json_string_from_mixed", "ensure_chat_grammars_table",
        "_ensure_note_folder_path_locked", "_ensure_note_exists_for_folder_sync", "_validate_workspace_source_saved_view_schema_version",
        "_verify_workspace_migration_delete_eligibility", "_repair_flashcard_review_session_aggregates_if_unchanged",
        "reconcile_flashcard_asset_refs", "rebuild_full_text_indexes", "backfill_tool_calls_from_inline",
        "ensure_character_tables_ready", "_repair_conversation_assistant_identity",
    }
)  # fmt: skip
# The 13 shims and the bootstrap forwarder stay on the class (R3, R5); a hand-filled manifest lists the leaf shims.
SELECT_EXCLUDED = frozenset(
    {
        "_get_db_version", "_sqlite_linear_migration_steps", "_run_sqlite_linear_migration_step", "_sqlite_table_names",
        "_sqlite_column_names", "_sqlite_index_names", "_apply_postgres_migration_script", "_get_schema_version_postgres",
        "_set_schema_version_postgres", "_initialize_schema", "_initialize_schema_sqlite", "_initialize_schema_postgres",
        "_sqlite_schema_init_lock_for_path", "_ensure_bootstrap_for_backend", "repair_conversation_assistant_identity",
    }
)  # fmt: skip
SELECT_INIT = ("_initialize_schema_sqlite_legacy", "_postgres_schema_is_current")
VERSION_ATTRS = frozenset({"_CURRENT_SCHEMA_VERSION", "_POSTGRES_SCHEMA_VERSION", "_SCHEMA_NAME"})
_BUCKET_SCHEMA_RX = re.compile(r"^_?(ensure|verify|repair|rebuild|reconcile|bridge|configure|sync_postgres_seq)")
_BUCKET_BUILD_RX = re.compile(r"_(create|apply)_.*(schema|table|index|trigger)")


def _bucket(name: str) -> str:
    if name.startswith("_migrate"):
        return "migrate"
    low = name.lower()
    if re.search(r"rls|polic", low):
        return "rls"
    if "fts" in low:
        return "fts"
    if "backfill" in low:
        return "backfill"
    if _BUCKET_SCHEMA_RX.search(low) or "schema" in low or _BUCKET_BUILD_RX.search(low):
        return "schema"
    return "other"


def _all_class_defs(cls: ast.ClassDef) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Every def of the class, those nested in ``if``/``try`` included and the TYPE_CHECKING stubs not."""
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def walk(statements: list[ast.stmt]) -> None:
        for st in statements:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[st.name] = st
            elif isinstance(st, ast.If) and ast.unparse(st.test) == "TYPE_CHECKING":
                walk(st.orelse)
            elif isinstance(st, (ast.If, ast.Try)):
                for block in _blocks_of(st):
                    walk(block)

    walk(cls.body)
    return found


def _self_refs(defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]) -> dict[str, set[str]]:
    """Name -> the defs that read ``self.<name>``, ``cls.<name>`` or ``CharactersRAGDB.<name>``."""
    callers: dict[str, set[str]] = {}
    for caller, node in defs.items():
        for x in ast.walk(node):
            if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id in {"self", "cls", CLASS} and x.attr in defs and x.attr != caller:
                callers.setdefault(x.attr, set()).add(caller)
    return callers


def _outside_reads(root: Path) -> tuple[set[str], set[str]]:
    """Names the chacha stores reach as ``db.<name>``, and names app code outside DB_Management reads as ``.<name>``."""
    stores: set[str] = set()
    for path in sorted((root / STORES).rglob("*.py")):
        stores |= {m.group(1) for m in re.finditer(r"(?:_db|\bdb)\.(_?\w+)", path.read_text(encoding="utf-8"))}
    app: set[str] = set()
    for key, path in _py_files(root):
        if key.startswith(SCANNED[0]) and key != DB_MODULE and "DB_Management" not in key:
            app |= {m.group(1) for m in re.finditer(r"\.(_?[A-Za-z0-9_]+)\b", path.read_text(encoding="utf-8"))}
    return stores, app


def _reads_version_attr(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == "cls" and x.attr in VERSION_ATTRS for x in ast.walk(node))


# Domains (A.5): a class def in the "other" bucket whose first match in this ordered list is its domain.
DOMAINS = (
    ("quiz", re.compile(r"quiz|osce|attempt|question|study_assistant|_fill_blank|levenshtein|remediation")),
    ("flashcards", re.compile(r"flashcard|deck|review_session|study_pack|suggestion|srs|apkg")),
    ("workspace", re.compile(r"workspace|clone_target|source_review|saved_view|source_")),
)


def select_domain(root: Path, group: str) -> list[str]:
    """The names of the "other"-bucket class defs (the 14 false positives count as "other") in a domain, in source order. A name defined twice is listed once (see ``domain_def_count``)."""
    defs = _all_class_defs(_class_node(_parse(root / DB_MODULE)[1], CLASS))
    domain = {n for n in defs if (_bucket(n) == "other" or n in SELECT_FALSE_POSITIVES) and next((d for d, rx in DOMAINS if rx.search(n)), None) == group}
    return sorted(domain, key=lambda n: defs[n].lineno)


def domain_def_count(root: Path, group: str) -> int:
    """How many ``def`` statements the domain has: a name defined under both branches of an ``if`` counts twice."""
    cls = _class_node(_parse(root / DB_MODULE)[1], CLASS)
    names = set(select_domain(root, group))
    return sum(1 for x in ast.walk(cls) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name in names)


def select_functions(root: Path, group: str) -> list[str]:
    """The defs a function group moves, in source order: research B's ``t22_defs.py`` + ``t22_closure.py``."""
    cls = _class_node(_parse(root / DB_MODULE)[1], CLASS)
    defs = _all_class_defs(cls)
    callers = _self_refs(defs)
    stores, app = _outside_reads(root)
    chosen = {n for n in defs if _bucket(n) not in {"other"} and n not in SELECT_FALSE_POSITIVES}
    changed = True
    while changed:  # closure: a private helper whose callers all move, and nobody else reads
        changed = False
        for name in defs:
            if name in chosen or name in SELECT_FALSE_POSITIVES or _bucket(name) != "other" or not name.startswith("_"):
                continue
            who = callers.get(name, set())
            if who and who <= chosen and name not in stores and name not in app:
                chosen.add(name)
                changed = True
    moved = {
        n for n in chosen
        if n not in SELECT_EXCLUDED and n not in CTX_DB_RENAMES.values()  # a public DB method the stores call stays
        and not (any(ast.unparse(d) == "classmethod" for d in defs[n].decorator_list) and _reads_version_attr(defs[n]))
    }  # fmt: skip
    migrate = {n for n in moved if n.startswith("_migrate")}
    init = {n for n in moved if n in SELECT_INIT}

    def side(name: str) -> str:
        if "postgres" in name:
            return "pg"
        if "sqlite" in name or name in migrate and "postgres" not in name:
            return "sq"
        return "x"

    def kind(name: str) -> str:
        if "postgres" in name:
            return "pg"
        if "sqlite" in name:
            return "sq"
        sides = {side(c) for c in callers.get(name, set())}
        return "sq" if sides == {"sq"} else "pg" if sides == {"pg"} else "shared"

    helpers = moved - migrate - init
    picked = {
        "migrate-sqlite": {n for n in migrate if "postgres" not in n},
        "migrate-postgres": {n for n in migrate if "postgres" in n},
        "init": init,
        "sqlite-schema": {n for n in helpers if kind(n) == "sq"},
        "postgres-schema": {n for n in helpers if kind(n) == "pg"},
        "migration-support": {n for n in helpers if kind(n) == "shared"},
    }[group]
    return sorted(picked, key=lambda n: defs[n].lineno)


def run_select(root: Path, group: str) -> int:
    if group == "schema-sql":
        print(select_schema_sql(_parse(root / DB_MODULE)[1]))
    elif group in SELECT_GROUPS:
        print(select_functions(root, group))
    elif group in {d for d, _ in DOMAINS}:
        names = select_domain(root, group)
        print(names)
        print(f"{len(names)} names, {domain_def_count(root, group)} defs", file=sys.stderr)
    else:
        print(f"select {group!r} is not implemented yet")
        return 1
    return 0


# --- ``functions`` (TASK-13950.22 PR E, A.3, R24) ---------------------------------------------------------------------

_DBM = "tldw_Server_API/app/core/DB_Management/"
NON_CHACHA_DB = tuple(_DBM + d + "/" for d in ("media_db", "prompt_studio_db", "codegraph", "backends", "migrations"))
PKG = "tldw_Server_API.app.core.DB_Management.chacha"
CONTEXT_MODULE = STORES + "context.py"
KNOWN_MODULES = {"schema_sql": f"{PKG}.schema_sql", "conversation_store": f"{PKG}.conversation_store", "note_store": f"{PKG}.note_store"}


class FnMove(NamedTuple):
    alias: str
    target: str  # the target module path, relative to the root
    static: bool  # no ``ctx`` parameter: a staticmethod, or a classmethod made static (R24)
    shim: bool  # the class keeps a forwarder


class FnEnv(NamedTuple):
    manifest: dict
    index: dict[str, FnMove]
    store_index: dict[str, str]
    renames: dict[str, str]
    schema_consts: frozenset[str]
    ctx_static: frozenset[str]
    clash: frozenset[str]


class FnProblem(Exception):
    """A reference the tool will not rewrite on its own (A.3 rule 9: fail closed)."""


def _fn_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    decorators = tuple(ast.unparse(d) for d in node.decorator_list)
    return {(): "instance", ("staticmethod",): "static", ("classmethod",): "classmethod"}.get(decorators)


def _ctx_static_members(root: Path) -> frozenset[str]:
    path = root / CONTEXT_MODULE
    if not path.exists():
        return frozenset()
    cls = _class_node(ast.parse(path.read_text(encoding="utf-8")), "ChaChaContext")
    return frozenset(n.name for n in cls.body if isinstance(n, ast.FunctionDef) and any(ast.unparse(d) == "staticmethod" for d in n.decorator_list))


def _schema_constant_names(root: Path) -> tuple[frozenset[str], frozenset[str]]:
    """(class constants, module constants) that committed ``constants`` manifests moved into schema_sql."""
    path = root / STORES / "schema_sql.py"
    moved = {n for m in committed_manifests(root) if m["mode"] == "constants" for n in m["constants"]}
    if not path.exists():
        return frozenset(moved), frozenset()
    in_module = {n for st in ast.parse(path.read_text(encoding="utf-8")).body if (n := _const_name(st))}
    return frozenset(moved & in_module), frozenset(in_module)


def fn_index(root: Path, manifests: list[dict], defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]) -> dict[str, FnMove]:
    """Moved name -> where it went, over the committed ``functions`` manifests and the ones being applied."""
    applying = {m["target"] for m in manifests}
    index: dict[str, FnMove] = {}
    for manifest in [*(c for c in committed_manifests(root) if c["target"] not in applying), *manifests]:
        if manifest["mode"] != "functions":
            continue
        target = root / manifest["target"]
        t_defs = _defs(ast.parse(target.read_text(encoding="utf-8")).body) if target.exists() else {}
        for name in manifest["names"]:
            if name in defs:
                static = _fn_kind(defs[name]) != "instance"
            elif name in t_defs:
                params = t_defs[name].args.posonlyargs + t_defs[name].args.args
                static = not params or params[0].arg != "ctx"
            else:
                raise SystemExit(f"{name}: in neither {CLASS} nor {manifest['target']}")
            index[name] = FnMove(manifest["alias"], manifest["target"], static, name in manifest["shims"])
    return index


def fn_env(root: Path, manifests: list[dict], defs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef], clash: set[str]) -> FnEnv:
    return FnEnv(
        {},
        fn_index(root, manifests, defs),
        _moved_index(root, []),
        {**all_renames(root), **{k: v for m in manifests for k, v in m["renames"].items()}},
        _schema_constant_names(root)[0],
        _ctx_static_members(root),
        frozenset(clash),
    )


def fn_replacement(attr: str, recv: str, env: FnEnv) -> tuple[str, bool]:
    """The text for ``self.<attr>`` (``recv == "self"``) or ``cls.<attr>`` in a moved body, and whether a call to it takes ``ctx``."""
    if recv == "self" and attr == "ctx":
        return "ctx", False
    move = env.index.get(attr)
    if move is not None:
        if recv == "cls" and not move.static:
            raise Unmapped(attr)
        if move.target == env.manifest.get("target"):  # a module does not import itself
            return attr, not move.static
        return f"{move.alias}.{attr}", not move.static
    rewrite = env.manifest["rewrites"].get(attr) or (f"ctx.{CTX_REWRITES[attr]}" if attr in CTX_REWRITES else None)
    if recv == "cls":  # R24: only a static member of the context, or a schema_sql constant
        member = rewrite.split(".", 1)[1] if rewrite and rewrite.split(".", 1)[0] in {"ctx", "ChaChaContext"} else None
        if member in env.ctx_static:
            return f"chacha_context.ChaChaContext.{member}", False  # the module, not the class: the context imports ChaChaNotes_DB, so a top-level class import cycles
        if attr in env.schema_consts:
            return f"schema_sql.{attr}", False
        raise Unmapped(attr)
    if attr in env.store_index:
        return f"ctx.db.{env.store_index[attr]}.{attr}", False
    if rewrite:
        return rewrite, False
    if attr in env.schema_consts:
        return f"schema_sql.{attr}", False
    if attr in CTX_PUBLIC:
        return f"ctx.{attr}", False
    if attr in CTX_MODULE_CONSTANTS:
        return f"{CTX_MODULE_CONSTANTS[attr]}.{attr}", False
    if attr in env.renames:
        return f"ctx.db.{env.renames[attr]}", False
    if not attr.startswith("_"):
        return f"ctx.db.{attr}", False
    raise Unmapped(attr)


def fn_callsite(node: ast.Attribute, env: FnEnv, *, in_class: bool, is_call: bool, source: str = "", in_test: bool = False) -> tuple[str, str | None] | None:
    """A reference to a moved name outside the moved bodies: the text for the attribute and the ``ctx`` argument, or None to leave it.

    Raises FnProblem for a reference the rules do not cover. ``source`` is the receiver's text, for an external receiver.
    """
    move = env.index.get(node.attr)
    if move is None or not isinstance(node.ctx, ast.Load):
        return None
    recv, name = node.value, f"{move.alias}.{node.attr}"
    if isinstance(recv, ast.Name) and recv.id in {mv.alias for mv in env.index.values()}:
        return None  # already ``<module>.<name>``
    if isinstance(recv, ast.Name) and recv.id in {"self", "cls"}:
        if not in_class:  # another class's own method of the same name
            return None
        if move.static:
            return name, None
        if recv.id == "self" and is_call:
            return name, "self.ctx"
        if move.shim:  # ``bind(self._shim)``: the forwarder stays on the class, so the reference stays valid
            return None
        raise FnProblem(f"{node.lineno}: {recv.id}.{node.attr} is not a call to a method that takes ctx")
    if isinstance(recv, ast.Name) and recv.id == CLASS:
        return (name, None) if move.static else None  # an unbound instance method is listed by ``check`` (Appendix D)
    if move.shim or not is_call:
        return None
    if not receivers.RECEIVER.search(receivers.last_identifier(recv)):
        return None  # M1: outside the class only a ``db``-like receiver is ours (a free function's ``backend._x`` is not); ``check`` lists the rest
    if node.attr in env.clash and not in_test:
        return None  # another app class defines the name; a test call on a ``db``-like receiver is still ours
    return (name, None) if move.static else (name, f"{source}.ctx")


class _FnMoved(ast.NodeTransformer):
    """Apply A.3 rules 2-5 to a moved def, the way ``functions`` does on text."""

    def __init__(self, env: FnEnv, recv: str) -> None:
        self.env, self.recv = env, recv
        self.unmapped: list[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        func = node.func
        wants = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == self.recv
            and isinstance(func.ctx, ast.Load)
            and func.attr != "ctx"
            and func.attr in self.env.index
            and not self.env.index[func.attr].static
        )
        self.generic_visit(node)
        if wants:
            node.args.insert(0, ast.Name("ctx", ast.Load()))
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: N802
        if isinstance(node.value, ast.Name) and node.value.id == self.recv and not isinstance(node.ctx, ast.Load) and node.attr in self.env.manifest["rewrites"]:
            assigned = ast.parse(self.env.manifest["rewrites"][node.attr], mode="eval").body
            assigned.ctx = node.ctx  # type: ignore[attr-defined]
            return assigned
        if isinstance(node.value, ast.Name) and node.value.id == self.recv and isinstance(node.ctx, ast.Load):
            try:
                new, _ = fn_replacement(node.attr, self.recv, self.env)
            except Unmapped:
                self.unmapped.append(node.attr)
                return node
            return ast.parse(new, mode="eval").body
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        if isinstance(node.ctx, ast.Load) and node.id == self.recv:
            if self.recv == "cls":
                self.unmapped.append("cls")
                return node
            return ast.parse("ctx.db", mode="eval").body
        return node


def expected_moved(node: ast.FunctionDef | ast.AsyncFunctionDef, env: FnEnv) -> tuple[ast.AST, list[str]]:
    """The def a base method becomes in its target module, and the names no rule maps."""
    kind = _fn_kind(node)
    new = copy.deepcopy(node)
    recv = "cls" if kind == "classmethod" else "self"
    rewriter = _FnMoved(env, recv)
    new.body = [rewriter.visit(st) for st in new.body]
    new.decorator_list = []
    params = new.args.posonlyargs or new.args.args
    if kind == "instance" and params and params[0].arg == "self":
        params[0].arg = "ctx"
    elif kind == "classmethod" and params and params[0].arg == "cls":
        params.pop(0)
    return new, rewriter.unmapped


class _FnClass(ast.NodeTransformer):
    """A remaining class def: calls of moved names go through their module, a renamed attribute takes its new name."""

    def __init__(self, env: FnEnv) -> None:
        self.env = env
        self.problems: list[str] = []

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        hit = None
        if isinstance(node.func, ast.Attribute):
            try:
                hit = fn_callsite(node.func, self.env, in_class=True, is_call=True, source=ast.unparse(node.func.value))
            except FnProblem as exc:
                self.problems.append(str(exc))
        if hit is None:
            return self.generic_visit(node)
        node.args = [self.visit(a) for a in node.args]
        node.keywords = [self.visit(k) for k in node.keywords]
        if hit[1]:
            node.args.insert(0, ast.parse(hit[1], mode="eval").body)
        node.func = ast.parse(hit[0], mode="eval").body
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:  # noqa: N802
        try:
            hit = fn_callsite(node, self.env, in_class=True, is_call=False)
        except FnProblem as exc:
            self.problems.append(str(exc))
            hit = None
        if hit is not None:
            return ast.parse(hit[0], mode="eval").body
        self.generic_visit(node)
        if node.attr in self.env.renames:
            node.attr = self.env.renames[node.attr]
        return node


def expected_class_def(node: ast.FunctionDef | ast.AsyncFunctionDef, env: FnEnv, name: str, shim: FnMove | None) -> tuple[ast.AST, list[str]]:
    new = copy.deepcopy(node)
    if shim is not None:
        new.body = _shim_body(new, shim, name)
        new.name = env.renames.get(name, name)
        return new, []
    rewriter = _FnClass(env)
    new.body = [rewriter.visit(st) for st in new.body]
    new.name = env.renames.get(name, name)
    return new, rewriter.problems


def _forward_params(node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str) -> list[str]:
    args = node.args
    positional = [a.arg for a in [*args.posonlyargs, *args.args]]
    if kind in {"instance", "classmethod"}:
        positional = positional[1:]
    out = list(positional)
    if args.vararg:
        out.append(f"*{args.vararg.arg}")
    out += [f"{a.arg}={a.arg}" for a in args.kwonlyargs]
    if args.kwarg:
        out.append(f"**{args.kwarg.arg}")
    return out


def _shim_text(node: ast.FunctionDef | ast.AsyncFunctionDef, move: FnMove, name: str) -> str:
    kind = _fn_kind(node)
    params = _forward_params(node, kind or "instance")
    lead = [] if move.static else ["self.ctx"]
    return f"return {move.alias}.{name}({', '.join([*lead, *params])})"


def _shim_body(node: ast.FunctionDef | ast.AsyncFunctionDef, move: FnMove, name: str) -> list[ast.stmt]:
    first = node.body[0]
    keep = [first] if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str) else []
    return [*keep, ast.parse(_shim_text(node, move, name)).body[0]]


def _paren_after(raw: list[bytes], line: int, col: int) -> int | None:
    """The byte column of the ``(`` that follows (line, col) on the same line, or None."""
    rest = raw[line - 1][col:]
    stripped = rest.lstrip()
    return col + (len(rest) - len(stripped)) if stripped.startswith(b"(") else None


def _ctx_argument(text: str, raw: list[bytes], line: int, paren: int, call: ast.Call) -> str:
    """The text inserted after a call's ``(``: ``ctx`` alone, ``ctx, `` before arguments, ``ctx,`` when they start on the next line."""
    if not call.args and not call.keywords:
        return text
    return text + ("," if not raw[line - 1][paren + 1 :].strip() else ", ")


def _moved_edits(raw: list[bytes], node: ast.FunctionDef | ast.AsyncFunctionDef, env: FnEnv) -> tuple[list[tuple[int, int, int, int, str]], dict[str, int], list[str]]:
    edits: list[tuple[int, int, int, int, str]] = []
    unmapped: dict[str, int] = {}
    problems: list[str] = []
    kind = _fn_kind(node)
    if kind is None:
        return edits, unmapped, [f"{node.lineno}: decorators {[ast.unparse(d) for d in node.decorator_list]} (only staticmethod and classmethod move)"]
    recv = "cls" if kind == "classmethod" else "self"
    calls = {id(c.func): c for c in ast.walk(node) if isinstance(c, ast.Call)}
    problems += [f"{c.lineno}: super() cannot move out of the class" for c in calls.values() if isinstance(c.func, ast.Name) and c.func.id == "super"]
    handled: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == recv:
            handled.add(id(n.value))
            if not isinstance(n.ctx, ast.Load) and n.attr not in env.manifest["rewrites"]:  # an assigned attribute needs a declared rewrite (H: the runtime version)
                problems.append(f"{n.lineno}: {recv}.{n.attr} is assigned")
                continue
            try:
                new, wants = fn_replacement(n.attr, recv, env)
            except Unmapped:
                unmapped[n.attr] = unmapped.get(n.attr, 0) + 1
                continue
            edits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, new))  # type: ignore[arg-type]
            if wants:
                call = calls.get(id(n))
                paren = _paren_after(raw, n.end_lineno, n.end_col_offset) if call else None  # type: ignore[arg-type]
                if call is None or paren is None:
                    problems.append(f"{n.lineno}: self.{n.attr} is not a plain call (it would need ctx)")
                else:
                    edits.append((n.end_lineno, paren + 1, n.end_lineno, paren + 1, _ctx_argument("ctx", raw, n.end_lineno, paren, call)))  # type: ignore[arg-type]
        elif (isinstance(n, ast.Name) and n.id == "ctx" and not isinstance(n.ctx, ast.Load)) or (isinstance(n, ast.arg) and n.arg == "ctx"):
            problems.append(f"{n.lineno}: the body binds a name ctx")
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id == recv and isinstance(n.ctx, ast.Load) and id(n) not in handled:
            if recv == "cls":
                problems.append(f"{n.lineno}: cls is used other than as cls.<name> (R24)")
            else:
                edits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, "ctx.db"))  # type: ignore[arg-type]
    params = node.args.posonlyargs + node.args.args
    if kind == "instance":
        if not params or params[0].arg != "self":
            problems.append(f"{node.lineno}: the first parameter is not self")
        else:
            edits.append((params[0].lineno, params[0].col_offset, params[0].lineno, params[0].col_offset + 4, "ctx"))
    elif kind == "classmethod":
        if not params or params[0].arg != "cls":
            problems.append(f"{node.lineno}: the first parameter is not cls")
        else:
            rest = [*params[1:], *([node.args.vararg] if node.args.vararg else []), *node.args.kwonlyargs, *([node.args.kwarg] if node.args.kwarg else [])]
            end = min(rest, key=lambda a: (a.lineno, a.col_offset)) if rest else None
            first = params[0]
            if end is None:
                edits.append((first.lineno, first.col_offset, first.lineno, first.col_offset + 3, ""))
            else:
                edits.append((first.lineno, first.col_offset, end.lineno, end.col_offset, ""))
    return edits, unmapped, problems


def _callsite_edits(text: str, tree: ast.Module, env: FnEnv, in_db: bool, moved_defs: set[int], in_test: bool = False) -> tuple[list[tuple[int, int, int, int, str]], set[str], list[str]]:
    """Edits for every reference to a moved name outside the moved defs (A.3 rules 9 and 10)."""
    raw = text.encode("utf-8").split(b"\n")
    calls = {id(c.func): c for c in ast.walk(tree) if isinstance(c, ast.Call)}
    edits: list[tuple[int, int, int, int, str]] = []
    used: set[str] = set()
    problems: list[str] = []

    structural = set(env.manifest.get("structural", []))

    def walk(node: ast.AST, in_class: bool, hand_edited: bool = False) -> None:
        if id(node) in moved_defs:
            return
        if isinstance(node, ast.ClassDef):
            in_class = in_db and node.name == CLASS
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and in_class and node.name in structural:
            hand_edited = True  # a declared structural def is rewritten by hand, so its non-call references are not a problem
        if isinstance(node, ast.Attribute) and node.attr in env.index:
            call = calls.get(id(node))
            try:
                source = ast.get_source_segment(text, node.value) or ""
                hit = fn_callsite(node, env, in_class=in_class, is_call=call is not None, source=source, in_test=in_test)
            except FnProblem as exc:
                if not hand_edited:
                    problems.append(str(exc))
                hit = None
            if hit is not None:
                new, ctx_text = hit
                edits.append((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset, new))  # type: ignore[arg-type]
                used.add(env.index[node.attr].alias)
                if ctx_text:
                    paren = _paren_after(raw, node.end_lineno, node.end_col_offset) if call else None  # type: ignore[arg-type]
                    if call is None or paren is None:
                        problems.append(f"{node.lineno}: {node.attr} is not a plain call")
                    else:
                        edits.append((node.end_lineno, paren + 1, node.end_lineno, paren + 1, _ctx_argument(ctx_text, raw, node.end_lineno, paren, call)))  # type: ignore[arg-type]
        for child in ast.iter_child_nodes(node):
            walk(child, in_class, hand_edited)

    walk(tree, False)
    return edits, used, problems


def _shim_edit(text: str, node: ast.FunctionDef | ast.AsyncFunctionDef, move: FnMove, name: str) -> tuple[int, int, int, int, str]:
    body = node.body
    text_line = _shim_text(node, move, name)
    has_doc = isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)
    if has_doc and len(body) == 1:
        indent = " " * body[0].col_offset
        return (body[0].end_lineno, body[0].end_col_offset, body[0].end_lineno, body[0].end_col_offset, f"\n{indent}{text_line}")  # type: ignore[return-value]
    first = body[1] if has_doc else body[0]
    return (first.lineno, first.col_offset, node.end_lineno, node.end_col_offset, text_line)  # type: ignore[return-value]


def _module_import(alias: str, dotted: str) -> str:
    parent, leaf = dotted.rsplit(".", 1)
    return f"from {parent} import {leaf}" + (f" as {alias}" if leaf != alias else "")


def _add_import_lines(text: str, tree: ast.Module, lines: list[str]) -> str:
    """Insert import lines after the last top-level import (or the docstring) of a module."""
    at = _after_imports(tree)
    if not at and tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
        at = tree.body[0].end_lineno
    out = _lines(text)
    out[at:at] = lines
    return "\n".join(out)


def _fn_manifests(manifest_paths: list[Path]) -> list[dict]:
    manifests = [load_manifest(p) for p in manifest_paths]
    for manifest in manifests:
        if manifest["mode"] != "functions":
            raise SystemExit(f"{manifest['task']}: mode {manifest['mode']!r} is not the functions mode")
    return manifests


def _is_new_target(root: Path, manifest: dict) -> bool:
    return not (root / manifest["target"]).exists()


def run_functions(root: Path, manifest_paths: list[Path]) -> int:
    manifests = _fn_manifests(manifest_paths)
    source = root / manifests[0]["source"]
    text, tree = _parse(source)
    cls = _class_node(tree, CLASS)
    defs = _defs(cls.body)
    names = [n for m in manifests for n in m["names"]]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        print("\n".join(f"{n}: listed twice" for n in twice))
        return 1
    missing = [n for n in names if n not in defs]
    if missing:
        print(f"not in {CLASS}: {missing}")
        return 1
    for manifest in manifests:
        if (root / manifest["target"]).exists():
            print(f"{manifest['target']} exists already; the functions mode only creates a module")
            return 1
    env = fn_env(root, manifests, defs, collisions(root, set(names)))
    for name in sorted(env.clash):
        print(f"collision {name}: another app class defines it; rewritten only for a db-like receiver in tests")
    raw = text.encode("utf-8").split(b"\n")
    # 1. the moved bodies (rules 2-5), edited on a copy of the text
    edits: list[tuple[int, int, int, int, str]] = []
    unmapped: dict[str, dict[str, int]] = {}
    problems: list[str] = []
    for manifest in manifests:
        menv = env._replace(manifest=manifest)
        for name in manifest["names"]:
            new_edits, miss, probs = _moved_edits(raw, defs[name], menv)
            edits += new_edits
            for attr, count in miss.items():
                unmapped.setdefault(attr, {})[name] = count
            problems += [f"{name}: {p}" for p in probs]
    # 2. everything else in the source (rules 9-10), edited on the original text
    moved_ids = {id(defs[n]) for n in names}
    structural_all = {"structural": [n for m in manifests for n in m["structural"]]}
    outer, used, outer_problems = _callsite_edits(text, tree, env._replace(manifest={**env.manifest, **structural_all}), True, moved_ids)
    problems += [f"in {CLASS}: {p}" for p in outer_problems]
    shim_names = [n for m in manifests for n in m["shims"]]
    for name in shim_names:
        if name not in names:
            problems.append(f"{name}: a shim that is not in names")
    if unmapped or problems:
        if unmapped:
            print("unmapped private DB members (resolve each by A.3's order and record it in the manifest's rewrites or renames, then re-run):")
            for attr in sorted(unmapped):
                where = ", ".join(f"{n} x{c}" for n, c in sorted(unmapped[attr].items()))
                print(f"  self.{attr}: {sum(unmapped[attr].values())} uses: {where}")
        for problem in problems:
            print(f"  {problem}")
        return 1
    moved_text = replace_spans(text, edits)
    moved_tree = ast.parse(moved_text)
    moved_defs = _defs(_class_node(moved_tree, CLASS).body)
    moved_lines = _lines(moved_text)
    t_names_all = {n for m in manifests for n in m["names"]}
    db_names = module_names(tree)
    schema_module_names = _schema_constant_names(root)[1]
    imports_of: dict[str, set[str]] = {}
    pieces: dict[str, list[str]] = {}
    for manifest in manifests:
        free: set[str] = set()
        for name in manifest["names"]:
            start, end = _span(moved_lines, moved_defs[name])
            drop = {line for d in moved_defs[name].decorator_list for line in range(d.lineno, d.end_lineno + 1)}  # type: ignore[attr-defined]
            piece = [line for row, line in enumerate(moved_lines[start:end], start + 1) if row not in drop]
            free |= free_names("\n".join(piece), moved_defs[name])
            pieces[name] = _dedent(piece)
        lines_out: set[str] = set()
        for n in sorted(free):
            if n in manifest["names"]:
                continue
            if n in KNOWN_MODULES:
                lines_out.add(_module_import(n, KNOWN_MODULES[n]))
            elif n == "chacha_context":
                lines_out.add(f"from {PKG} import context as chacha_context")
            elif n in {m["alias"] for m in manifests} | {mv.alias for mv in env.index.values()}:
                move = next(mv for mv in env.index.values() if mv.alias == n)
                if move.target != manifest["target"]:
                    lines_out.add(_module_import(n, _dotted(move.target)))
            elif n in manifest["rewrites"].values():
                continue  # a bare module-level name the manifest rewrites to and defines by hand in its target
            elif n in schema_module_names:
                lines_out.add(f"from {PKG}.schema_sql import {n}")
            elif n in db_names:
                lines_out.add(_import_line(n, db_names[n]))
            else:
                print(f"free name {n} of {manifest['target']} has no source in {DB_MODULE}")
                return 1
        imports_of[manifest["target"]] = lines_out
    del t_names_all
    # the source: apply the outside edits, then delete the moved defs (the shims keep a forwarder)
    shim_edits = [_shim_edit(text, defs[n], env.index[n], n) for n in shim_names]
    new_text = replace_spans(text, [*outer, *shim_edits])
    new_tree = ast.parse(new_text)
    new_defs = _defs(_class_node(new_tree, CLASS).body)
    lines = _lines(new_text)
    for name in sorted((n for n in names if n not in shim_names), key=lambda n: new_defs[n].lineno, reverse=True):
        start, end = _span(lines, new_defs[name])
        del lines[start:end]
        if 0 < start < len(lines) and not lines[start - 1].strip() and not lines[start].strip():
            del lines[start]
    footer = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].startswith("# End of")), None)
    block = sorted(_module_import(a, _dotted(mv.target)) + "  # noqa: E402" for a, mv in {mv.alias: mv for mv in env.index.values() if mv.alias in used or mv.alias in {m["alias"] for m in manifests}}.items())  # import order, as ruff I wants it
    at = len(lines) if footer is None else footer - (1 if footer and lines[footer - 1] == "#" else 0)
    while at and not lines[at - 1].strip():  # two blank lines after the class, one before the footer
        at -= 1
        del lines[at]
    lines[at:at] = ["", "", *block, ""]
    source.write_text("\n".join(lines), encoding="utf-8")
    # the targets
    for manifest in manifests:
        target = root / manifest["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        init = target.parent / "__init__.py"
        if not init.exists():
            init.write_text(f'"""ChaCha package {target.parent.name} (TASK-13950.22)."""\n', encoding="utf-8")
        doc = manifest.get("doc") or f"Functions moved out of CharactersRAGDB into {manifest['alias']} (TASK-13950.22)."
        body: list[str] = []
        for name in manifest["names"]:
            body += ["", "", *pieces[name]]
        head = [f'"""{doc}"""', "", "from __future__ import annotations", ""]
        target.write_text("\n".join([*head, *sorted(imports_of[manifest["target"]]), *body]) + "\n", encoding="utf-8")
    # 3. outside callers in app and tests (rule 10), then the renames
    touched = 0
    targets = {m["target"] for m in manifests}
    for key, path in _py_files(root):
        if key == DB_MODULE or key in targets or key.startswith(NON_CHACHA_DB):
            continue  # M1: the other DB packages define same-named methods of their own
        f_text, f_tree = _parse(path)
        f_edits, f_used, f_problems = _callsite_edits(f_text, f_tree, env, False, set(), _is_test_file(key))
        for p in f_problems:
            print(f"left {key}:{p}")
        if not f_edits:
            continue
        have = module_names(f_tree)
        add = [_module_import(a, _dotted(mv.target)) for a, mv in sorted({mv.alias: mv for mv in env.index.values() if mv.alias in f_used}.items()) if a not in have]
        out = replace_spans(f_text, f_edits)
        if add:
            out = _add_import_lines(out, ast.parse(out), add)
        path.write_text(out, encoding="utf-8")
        touched += 1
    print(f"moved {len(names)} defs into {sorted(targets)}; rewrote callers in {touched} files")
    print(f"renamed {_apply_renames(root, env.renames)} references")
    return 0


def run_verify_functions(root: Path, manifests: list[dict], base: str) -> int:
    """``verify`` for ``functions`` manifests: each moved def and each remaining class def against the base."""
    source = manifests[0]["source"]
    base_tree = ast.parse(_git_show(root, base, source))
    head_tree = ast.parse((root / source).read_text(encoding="utf-8"))
    base_defs, head_defs = _defs(_class_node(base_tree, CLASS).body), _defs(_class_node(head_tree, CLASS).body)
    names = {n for m in manifests for n in m["names"]}
    env = fn_env(root, manifests, base_defs, collisions(root, names))
    structural = {n for m in manifests for n in m["structural"]}
    failures: list[str] = []
    moved = 0
    constants_ok = sum(_verify_constants(root, m, base_tree, head_tree, failures) for m in manifests if m["mode"] == "constants")
    for manifest in (m for m in manifests if m["mode"] == "functions"):
        t_defs = _defs(ast.parse((root / manifest["target"]).read_text(encoding="utf-8")).body)
        for name in manifest["names"]:
            if name not in base_defs or name not in t_defs:
                failures.append(f"{name}: missing from the {'base class' if name not in base_defs else 'target module'}")
                continue
            if name in head_defs and name not in manifest["shims"]:
                failures.append(f"{name}: still on {CLASS}")
            expected, unmapped = expected_moved(base_defs[name], env._replace(manifest=manifest))
            if unmapped or ast.dump(expected) != ast.dump(t_defs[name]):
                failures.append(f"{name}: the moved body differs from the base")
                failures += _diff(expected, t_defs[name])
            else:
                moved += 1
    shims = {n: env.index[n] for m in manifests for n in m["shims"]}
    identical, structural_seen = 0, []
    remaining_head = dict(head_defs)
    for name, node in base_defs.items():
        if name in names and name not in shims:
            continue
        new_name = env.renames.get(name, name)
        head = remaining_head.pop(new_name, None)
        if name in structural or new_name in structural:
            structural_seen.append(new_name)
            print(f"structural {new_name}:")
            print("\n".join(_diff(node, head)))
            continue
        if head is None:
            failures.append(f"{name}: gone from {CLASS} and not declared structural")
            continue
        expected, problems = expected_class_def(node, env, name, shims.get(name))
        if problems or ast.dump(expected) != ast.dump(head):
            failures.append(f"{name}: the class def differs from the base")
            failures += [f"    {p}" for p in problems]
            failures += _diff(expected, head)
        else:
            identical += 1
    for name in remaining_head:
        if name in structural:
            structural_seen.append(name)
        else:
            failures.append(f"{name}: a new def on {CLASS} that is not declared structural")
    for line in failures:
        print(line)
    if constants_ok:
        print(f"constants {constants_ok} identical")
    print(f"moved {moved} identical; class defs {identical} identical; structural {sorted(set(structural_seen))}")
    return 1 if failures else 0


def _left_references(tree: ast.Module, names: set[str], aliases: set[str], strings: bool = False) -> list[str]:
    """References to ``names`` that go through anything but a moved module: attributes, probes and patch strings.

    ``strings`` (M2, tests): also a string that is exactly a name (a ``method = "_x"`` variable), an f-string whose literal part is a name
    or the start of one, and a ``setattr``/``delattr`` probe, the forms ``patches`` cannot see.
    """
    findings: list[str] = []

    def walk(node: ast.AST, classes: tuple[ast.ClassDef, ...]) -> None:
        if isinstance(node, ast.ClassDef):
            classes += (node,)
        if (
            strings and isinstance(node, ast.Call) and len(node.args) >= 3 and isinstance(node.args[0], ast.Name) and node.args[0].id in aliases
            and isinstance(node.func, (ast.Name, ast.Attribute)) and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr) == "setattr"
        ):
            for child in node.args[2:]:  # ``setattr(<moved module>, "<name>", fake)`` is the right patch, not a left reference
                walk(child, classes)
            return
        if isinstance(node, ast.Attribute) and node.attr in names and isinstance(node.ctx, ast.Load):
            recv = node.value
            own = isinstance(recv, ast.Name) and recv.id in {"self", "cls"} and classes and classes[-1].name != CLASS and bool(_defined(classes[-1], {node.attr}))
            if not (isinstance(recv, ast.Name) and recv.id in aliases) and not own:
                findings.append(f"{node.lineno}: {ast.unparse(recv)}.{node.attr}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ({"getattr", "hasattr", "setattr", "delattr"} if strings else {"getattr", "hasattr"}) and len(node.args) >= 2:
            if isinstance(node.args[1], ast.Constant) and node.args[1].value in names:
                findings.append(f"{node.lineno}: {node.func.id} probe of {node.args[1].value}")
        elif strings and isinstance(node, ast.JoinedStr):
            for piece in node.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    text = piece.value.strip()
                    if any(n in text or (len(text) >= 12 and text.startswith("_") and n.startswith(text)) for n in names):
                        findings.append(f"{node.lineno}: f-string {text}")
            for piece in node.values:  # its literal parts are judged above; only the ``{...}`` parts are walked on
                if isinstance(piece, ast.FormattedValue):
                    walk(piece, classes)
            return
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            findings.extend(f"{node.lineno}: string {node.value}" for n in names if node.value.endswith(f"{CLASS}.{n}"))
            if strings and node.value in names:
                findings.append(f"{node.lineno}: string {node.value}")
        for child in ast.iter_child_nodes(node):
            walk(child, classes)

    walk(tree, ())
    return findings


def _db_like(finding: str) -> bool:
    """A ``<recv>.<name>`` finding whose receiver names a ChaCha DB (the receiver rule of ``callers``)."""
    recv = finding.split(": ", 1)[1].rsplit(".", 1)[0]
    return bool(receivers.RECEIVER.search(recv.split(".")[-1].split("(")[0])) and "probe" not in finding and "string" not in finding


def run_check_functions(root: Path, manifests: list[dict]) -> int:
    """Exit 1 on any reference left to a moved name that the rewrite rules did not handle (Appendix D lists the rules)."""
    names = {n for m in manifests for n in m["names"]} - {n for m in manifests for n in m["shims"]}
    names |= {n for m in manifests for n in m["renames"]}
    aliases = {m["alias"] for m in manifests}
    targets = {m["target"] for m in manifests}
    clash = collisions(root, names)
    bad = 0
    for key, path in _py_files(root):
        if key in targets:
            continue
        text, tree = _parse(path)
        if key.startswith(NON_CHACHA_DB):
            continue  # M1: their same-named methods are their own
        findings = _left_references(tree, names - clash, aliases, _is_test_file(key) and "tests/Helper_Scripts/" not in key)  # the tool's own self-tests hold seed text
        if _is_test_file(key):
            findings += [f for f in _left_references(tree, names & clash, aliases, True) if _db_like(f)]
            findings += [f"{line}: {m}" for line, m in _fakes_of(tree, names)]
        for finding in findings:
            print(f"left {key}:{finding}")
        bad += len(findings)
    reads = _db_reads(root)
    for manifest in manifests:
        free = moved_free_names(root, manifest)
        for key, file in _py_files(root):
            if _is_test_file(key):
                for line, form, name in module_patch_hits(_parse(file)[1], free):
                    if name not in reads:
                        print(f"left {key}:{line}: {form} {name}")
                        bad += 1
    print(f"{bad} left")
    return 1 if bad else 0


def _fakes_of(tree: ast.Module, names: set[str]) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name != CLASS and _defined(node, names):
            yield node.lineno, f"class {node.name} defines {sorted(_defined(node, names))}, so a rewritten call may reach the module with a fake"


def _verify_constants(root: Path, manifest: dict, base_tree: ast.Module, head_tree: ast.Module, failures: list[str]) -> int:
    """Each constant equals its base value and annotation (ClassVar read as its argument), and the class or module imports it."""
    t_tree = ast.parse((root / manifest["target"]).read_text(encoding="utf-8"))
    base_sites, head_sites = _const_sites(base_tree), _const_sites(head_tree)
    target = {n: st for st in t_tree.body if (n := _const_name(st))}
    module = _dotted(manifest["target"])
    imported: dict[str, set[str]] = {"module": set(), "class": set()}
    for level, body in (("module", head_tree.body), ("class", _class_node(head_tree, CLASS).body)):
        for st in body:
            if isinstance(st, ast.ImportFrom) and st.module == module:
                imported[level] |= {a.name for a in st.names}

    def shape(st: ast.stmt) -> tuple[str, str]:
        note = _unwrap_classvar(st)
        return (ast.dump(note) if note is not None else "", ast.dump(st.value))  # type: ignore[attr-defined]

    identical = 0
    for name in manifest["constants"]:
        if name not in base_sites or name not in target:
            failures.append(f"{name}: missing from the {'base' if name not in base_sites else 'target'}")
            continue
        level, old = base_sites[name]
        if shape(old) != shape(target[name]):
            failures.append(f"{name}: the moved constant differs from the base")
        elif name in head_sites:
            failures.append(f"{name}: still assigned in the source")
        elif name not in imported[level]:
            failures.append(f"{name}: not imported by the {level} from {module}")
        else:
            identical += 1
    return identical


# --- ``registry-snapshot`` (TASK-13950.22 PR G, R3, R12) --------------------------------------------------------------

SQLITE_STEPS_DEF = "_sqlite_linear_migration_steps"
POSTGRES_INIT_DEF = "_initialize_schema_postgres"
_POSTGRES_STEP_TEST = re.compile(r"target_version >= (\d+) and current_version < (\d+)")


def _self_attr(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def sqlite_snapshot(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[int, str]]:
    """Source version -> step name: keys from the dict literal, the loop tuple and the ``steps[N] = bind_migration(f, ...)`` assignment."""
    found: dict[int, str] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, int) and (name := _self_attr(value)):
                    found[key.value] = name
        elif isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            for pair in node.iter.elts:
                if isinstance(pair, ast.Tuple) and len(pair.elts) == 2 and all(isinstance(e, ast.Constant) for e in pair.elts):
                    found[pair.elts[0].value] = pair.elts[1].value  # type: ignore[attr-defined]
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            target, value = node.targets[0], node.value
            if isinstance(target.value, ast.Name) and target.value.id == "steps" and isinstance(target.slice, ast.Constant) and isinstance(value, ast.Call) and value.args and isinstance(value.args[0], ast.Name):
                found[target.slice.value] = value.args[0].id
    return sorted(found.items())


def postgres_snapshot(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """The 73..91 if-chains of ``_initialize_schema_postgres``, fast path then full path, as source version -> step name."""
    chains: list[list[tuple[int, str]]] = []
    for node in sorted((n for n in ast.walk(fn) if isinstance(n, ast.If)), key=lambda n: n.lineno):
        if not (m := _POSTGRES_STEP_TEST.fullmatch(ast.unparse(node.test))) or m.group(1) != m.group(2) or int(m.group(1)) < 73:
            continue  # the v71/v72 stamps are not steps
        calls = [_self_attr(c.func) for c in ast.walk(ast.Module(node.body, [])) if isinstance(c, ast.Call) and (_self_attr(c.func) or "").startswith("_migrate")]
        if len(calls) != 1:
            raise SystemExit(f"v{m.group(1)} chain: expected one _migrate call, found {calls}")
        version = int(m.group(1)) - 1
        if version == 72:
            chains.append([])
        chains[-1].append((version, calls[0]))  # type: ignore[arg-type]
    if len(chains) != 2:
        raise SystemExit(f"expected a fast-path and a full-path chain, found {len(chains)}")
    return chains[0], chains[1]


def run_registry_snapshot(root: Path) -> int:
    cls = _class_node(_parse(root / DB_MODULE)[1], CLASS)
    defs = _all_class_defs(cls)
    sqlite = sqlite_snapshot(defs[SQLITE_STEPS_DEF])
    fast, full = postgres_snapshot(defs[POSTGRES_INIT_DEF])
    if fast != full:
        print("the fast-path and full-path chains differ")
        return 1
    for label, rows in (("FROZEN_SQLITE", sqlite), ("FROZEN_POSTGRES", full)):
        print(f"{label} = [")
        for version, name in rows:
            print(f"    ({version}, {name!r}),")
        print("]")
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
    patches_cmd = sub.add_parser("patches", help="list every patch of the given names, or of a manifest's names")
    patches_cmd.add_argument("--names", default="")
    verify_cmd = sub.add_parser("verify", help="prove a manifest's move against a base ref")
    verify_cmd.add_argument("--base", required=True)
    manifest_cmds = [patches_cmd, verify_cmd]
    select_cmd = sub.add_parser("select", help="print the names a group moves (schema-sql, sqlite-schema, ...)")
    select_cmd.add_argument("group")
    snapshot_cmd = sub.add_parser("registry-snapshot", help="print the frozen SQLite and PostgreSQL migration registries of a tree, by AST")
    for label, text in (("constants", "move a manifest's constants into its module"), ("store", "move a manifest's defs into its store"), ("functions", "move a manifest's defs into module functions"), ("check", "list references a manifest leaves"), ("tests", "list test files naming a manifest's names")):
        manifest_cmds.append(sub.add_parser(label, help=text))
    for cmd in manifest_cmds:
        cmd.add_argument("manifests", nargs="*" if cmd is patches_cmd else "+", type=Path)
    for cmd in (callers, rewire_cmd, select_cmd, snapshot_cmd, *manifest_cmds):
        cmd.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.cmd == "rewire":
        return run_rewire(root, args.store)
    if args.cmd == "patches":
        names = {n for n in args.names.split(",") if n}
        if not args.manifests and not names:
            parser.error("patches needs --names or a manifest")
        return run_manifest_patches(root, args.manifests, names) if args.manifests else run_patches(root, names)
    if args.cmd == "registry-snapshot":
        return run_registry_snapshot(root)
    if args.cmd == "select":
        return run_select(root, args.group)
    if args.cmd == "constants":
        return run_constants(root, args.manifests)
    if args.cmd == "store":
        return run_store(root, args.manifests)
    if args.cmd == "functions":
        return run_functions(root, args.manifests)
    if args.cmd == "verify":
        return run_verify(root, args.manifests, args.base)
    if args.cmd == "check":
        return run_check(root, args.manifests)
    if args.cmd == "tests":
        return run_tests(root, args.manifests)
    return run(root, args.store, {n for n in args.names.split(",") if n}, args.check, args.tests)


if __name__ == "__main__":
    raise SystemExit(main())
