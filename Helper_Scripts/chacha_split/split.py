"""Rewrite callers of delegated CharactersRAGDB names through their store (TASK-13950.22).

    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split callers --store ATTR --names N1,N2 [--check] [--tests]
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split rewire --store <store module path>
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split patches (--names N1,N2 | MANIFEST...)
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split store MANIFEST...
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split verify MANIFEST... --base REF
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split check MANIFEST...
    PYTHONPATH=. $PY -m Helper_Scripts.chacha_split.split tests MANIFEST...

``callers`` rewrites, under tldw_Server_API/app and tldw_Server_API/tests, ``<recv>.<name>`` to
``<recv>.<ATTR>.<name>`` and ``getattr/hasattr(<recv>, "<name>", ...)`` to a probe of ``<recv>.<ATTR>``.
``--check`` writes nothing and exits 1 while anything still reaches a name outside ``*.<ATTR>``,
string-argument patches included (``monkeypatch.setattr(db, "n", f)``, ``patch.object(db, "n")``, which are inert on a mock).
``rewire`` turns each ``self._db.<a>`` of a store module into a ``self._ctx`` read (R14, TASK-13950.22 PR C1).
``patches`` lists every patch of a name (R25), so a test cannot patch a name the code no longer runs through.
``store`` moves the manifest's defs from CharactersRAGDB into a store class, rewriting each ``self.<attr>`` (R8, Appendix A.3),
applies the manifest's ``renames`` and routes every caller through the store. ``verify`` proves the result against a base ref.
``check`` and ``tests`` list what a manifest still leaves behind. A manifest (``manifests/t<NN>_<name>.py``) is one literal, read, never imported.
Edits use AST positions (UTF-8 byte offsets), applied right to left, so no string value changes.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import difflib
import re
import subprocess  # nosec B404  # fixed argv, no shell
import symtable
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
    "_runtime_schema_version": "runtime_schema_version",
    "_map_table_for_backend": "map_table",
    "_selected_owner_filter": "owner_filter",
    "_prepare_backend_statement": "prepare_statement",
    "_CURRENT_SCHEMA_VERSION": "code_schema_version",
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
LIST_KEYS = ("names", "shims", "structural", "constants")
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


def _store_edits(
    node: ast.AST, manifest: dict, index: dict[str, str], renames: dict[str, str]
) -> tuple[list[tuple[int, int, int, int, str]], list[str]]:
    edits: list[tuple[int, int, int, int, str]] = []
    problems: list[str] = []
    handled: set[int] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and _is_self(n.value):
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
    defs = _defs(cls.body)
    missing = [n for n in manifest["names"] if n not in defs]
    if missing:
        print(f"not in {CLASS}: {missing}")
        return 1
    edits: list[tuple[int, int, int, int, str]] = []
    problems: list[str] = []
    for name in manifest["names"]:
        node = defs[name]
        if node.decorator_list:
            problems.append(f"{node.lineno}: {name} has a decorator (store mode fails closed)")
        new_edits, new_problems = _store_edits(node, manifest, index, renames)
        edits += new_edits
        problems += [f"{name}: {p}" for p in new_problems]
    if problems:
        print("unmapped private DB members (resolve them in the manifest's rewrites or renames, then re-run):")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    rewritten = replace_spans(text, edits)
    new_tree = ast.parse(rewritten)
    new_defs = _defs(_class_node(new_tree, CLASS).body)
    new_lines = _lines(rewritten)
    pieces: dict[str, list[str]] = {}
    free: set[str] = set()
    for name in manifest["names"]:
        start, end = _span(new_lines, new_defs[name])
        pieces[name] = new_lines[start:end]
        free |= free_names("\n".join(pieces[name]), new_defs[name])
    # the source: delete each def (right to left), and one blank line where two would meet
    lines = _lines(text)
    for name in sorted(manifest["names"], key=lambda n: defs[n].lineno, reverse=True):
        start, end = _span(lines, defs[name])
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
    source.write_text(_drop_unused_imports("\n".join(lines)), encoding="utf-8")
    print(f"moved {len(manifest['names'])} defs into {manifest['store_class']}; imports added: {add}")
    return 0


def _drop_unused_imports(text: str) -> str:
    return text  # ponytail: ruff F401 is the check (C2); the tool never edits the DB module's imports


def _take_constants(manifest: dict, tree: ast.Module, original: list[str], lines: list[str]) -> list[tuple[str, list[str]]]:
    """Cut each module constant the manifest lists out of the source lines, and return its text."""
    taken: list[tuple[str, list[str]]] = []
    for name in manifest["constants"]:
        node = next(
            (n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)),
            None,
        )
        if node is None:
            raise SystemExit(f"constant {name} is not a module-level assignment in {DB_MODULE}")
        piece = original[node.lineno - 1 : node.end_lineno]
        at = next(i for i, line in enumerate(lines) if line == piece[0] and lines[i : i + len(piece)] == piece)
        del lines[at : at + len(piece)]
        if 0 < at < len(lines) and not lines[at - 1].strip() and not lines[at].strip():
            del lines[at]
        taken.append((name, piece))
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
    index = _moved_index(root, manifests)
    renames = {k: v for m in manifests for k, v in m["renames"].items()}
    structural = {n for m in manifests for n in m["structural"]}
    source = manifests[0]["source"]
    base_tree = ast.parse(_git_show(root, base, source))
    head_tree = ast.parse((root / source).read_text(encoding="utf-8"))
    base_defs, head_defs = _defs(_class_node(base_tree, CLASS).body), _defs(_class_node(head_tree, CLASS).body)
    failures: list[str] = []
    moved = 0
    for manifest in manifests:
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
    if not manifest["store_class"] or not target.exists():
        return set()
    t_tree = ast.parse(target.read_text(encoding="utf-8"))
    defs = _defs(_class_node(t_tree, manifest["store_class"]).body)
    db_names = set(module_names(ast.parse((root / manifest["source"]).read_text(encoding="utf-8"))))
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
    print("\n".join(_word_files(root, names)))
    return 0


def run_check(root: Path, manifest_paths: list[Path]) -> int:
    """Exit 1 on any reference left to a moved or renamed name, or to a free name through ChaChaNotes_DB."""
    bad = 0
    reads = _db_reads(root)
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
    for label, text in (("store", "move a manifest's defs into its store"), ("check", "list references a manifest leaves"), ("tests", "list test files naming a manifest's names")):
        manifest_cmds.append(sub.add_parser(label, help=text))
    for cmd in manifest_cmds:
        cmd.add_argument("manifests", nargs="*" if cmd is patches_cmd else "+", type=Path)
    for cmd in (callers, rewire_cmd, *manifest_cmds):
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
    if args.cmd == "store":
        return run_store(root, args.manifests)
    if args.cmd == "verify":
        return run_verify(root, args.manifests, args.base)
    if args.cmd == "check":
        return run_check(root, args.manifests)
    if args.cmd == "tests":
        return run_tests(root, args.manifests)
    return run(root, args.store, {n for n in args.names.split(",") if n}, args.check, args.tests)


if __name__ == "__main__":
    raise SystemExit(main())
