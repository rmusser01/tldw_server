"""Ratchet: the CharactersRAGDB surface does not grow (TASK-13950.8).

CharactersRAGDB (core/DB_Management/ChaChaNotes_DB.py) is a ~46,000-line class that the
chacha/ store extraction is meant to shrink. Three counts are frozen in
charactersragdb_surface_baseline.json, each a ceiling ``{"count": int, "reason": ...}``:

  * methods: ``def``s anywhere in the class body (including inside ``if``/``try``
    blocks, excluding the typing-only stubs under ``if TYPE_CHECKING:``), plus every
    other binding onto the class from module code in ChaChaNotes_DB.py
    (``CharactersRAGDB.x = f`` or ``setattr(CharactersRAGDB, ...)`` with a value that is
    not a delegation).
  * delegated_methods: bindings onto the class whose value is a ``_delegate_store_method``
    call, directly or through a module-level wrapper that returns one (for example
    ``_delegate_conversation_store_method``). A binding inside a ``for`` loop over a
    literal tuple or list counts once per element.
  * private_member_reaches: in every module under core/DB_Management/chacha/, an
    attribute ``_x`` (single leading underscore, not a dunder) read, written or deleted
    on the db object, where the db object is ``self._db`` or a name ``db``; plus
    ``getattr``/``setattr``/``hasattr``/``delattr`` on the db object with a literal ``"_x"``;
    plus, in a store class whose ``__getattr__`` forwards to ``self._db``, every
    ``self._x`` the class does not define itself (a ``def``, a class attribute or a
    ``self._x = ...`` assignment), since that lookup lands on the db.

The counter fails closed: a ``_delegate_store_method`` call it cannot attribute to a
binding, or a binding in a loop it cannot size, raises instead of being skipped.

Growth fails: move the method into its chacha/ store and call the store, or use the
db's public API instead of a private member. If growth is unavoidable, raise the count
with a "reason" (reviewed in the PR; see _ceilings.py). A count below its ceiling passes;
lock the gain in with
``python -m tldw_Server_API.tests.lint.test_charactersragdb_surface_ratchet --tighten``.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeGuard

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import REPO_ROOT, parse_tree

pytestmark = pytest.mark.unit

CLASS = "CharactersRAGDB"
DELEGATE = "_delegate_store_method"
DB_MODULE = Path("core/DB_Management/ChaChaNotes_DB.py")
STORE_DIR = Path("core/DB_Management/chacha")
TYPE_CHECKING = frozenset({"TYPE_CHECKING", "typing.TYPE_CHECKING"})

SURFACE = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("charactersragdb_surface_baseline.json"),
    command="python -m tldw_Server_API.tests.lint.test_charactersragdb_surface_ratchet",
    fix=(
        "Put new persistence code in its chacha/ store and call the store; reach the db "
        "through its public API, not a private member."
    ),
    stale="The baseline lists a count this ratchet no longer measures.",
)


# --- The counter --------------------------------------------------------------------------------


def _blocks(statement: ast.stmt) -> Iterator[list[ast.stmt]]:
    """The statement lists nested directly in a compound statement."""
    for field in ("body", "orelse", "finalbody"):
        yield getattr(statement, field, [])
    for part in [*getattr(statement, "handlers", []), *getattr(statement, "cases", [])]:
        yield part.body


def _is_call_to(node: ast.AST | None, names: Iterable[str]) -> TypeGuard[ast.Call]:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in names


def _is_class(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == CLASS


def _the_class(tree: ast.Module) -> ast.ClassDef:
    found = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == CLASS]
    if len(found) != 1:
        raise ValueError(f"expected one class {CLASS} in {DB_MODULE}, found {len(found)}")
    return found[0]


def _class_body_defs(statements: list[ast.stmt]) -> int:
    count = 0
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            count += 1
        elif isinstance(statement, ast.If) and ast.unparse(statement.test) in TYPE_CHECKING:
            count += _class_body_defs(statement.orelse)  # the body is typing-only stubs
        elif not isinstance(statement, ast.ClassDef):
            count += sum(_class_body_defs(block) for block in _blocks(statement))
    return count


def _bound_value(statement: ast.stmt) -> ast.expr | None:
    """The value ``statement`` binds onto the class: ``C.x = v``, ``C.x: T = v`` or ``setattr(C, name, v)``."""
    if isinstance(statement, ast.Assign) and any(
        isinstance(target, ast.Attribute) and _is_class(target.value) for target in statement.targets
    ):
        return statement.value
    if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Attribute):
        return statement.value if _is_class(statement.target.value) else None
    call = statement.value if isinstance(statement, ast.Expr) else None
    if _is_call_to(call, {"setattr"}) and len(call.args) == 3 and _is_class(call.args[0]):
        return call.args[2]
    return None


def _bindings(tree: ast.Module) -> tuple[int, int]:
    """(delegated, other) bindings onto the class from module code. Fails closed on shapes it cannot count."""
    wrappers = {
        fn.name
        for fn in tree.body
        if isinstance(fn, ast.FunctionDef)
        and any(isinstance(s, ast.Return) and _is_call_to(s.value, {DELEGATE}) for s in fn.body)
    }
    delegates = {DELEGATE, *wrappers}
    counted: set[int] = set()
    totals = {"delegated": 0, "other": 0}

    def visit(statements: list[ast.stmt], times: int | None) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.ClassDef)) and statement.name in {*delegates, CLASS}:
                continue  # definitions, not bindings; the class body is counted as methods
            value = _bound_value(statement)
            if value is not None:
                if times is None:
                    raise ValueError(f"{DB_MODULE}:{statement.lineno}: cannot size the loop around this binding")
                totals["delegated" if _is_call_to(value, delegates) else "other"] += times
                counted.add(id(value))
            if isinstance(statement, (ast.For, ast.AsyncFor, ast.While)):
                items = getattr(statement, "iter", None)
                size = None
                if isinstance(items, (ast.Tuple, ast.List)) and all(isinstance(e, ast.Constant) for e in items.elts):
                    size = len(items.elts)
                visit(statement.body, None if size is None or times is None else times * size)
                visit(statement.orelse, times)
            else:
                for block in _blocks(statement):
                    visit(block, times)

    visit(tree.body, 1)
    defining = [fn for fn in tree.body if isinstance(fn, ast.FunctionDef) and fn.name in delegates]
    definitions = {id(node) for fn in defining for node in ast.walk(fn)}
    for node in ast.walk(tree):
        if _is_call_to(node, delegates) and id(node) not in counted | definitions:
            raise ValueError(f"{DB_MODULE}:{node.lineno}: cannot attribute this {DELEGATE} call to a {CLASS} binding")
    return totals["delegated"], totals["other"]


def _is_db(node: ast.AST) -> bool:
    """``db`` or ``self._db``: how chacha/ code names the CharactersRAGDB it wraps."""
    if isinstance(node, ast.Name):
        return node.id == "db"
    return isinstance(node, ast.Attribute) and node.attr == "_db" and _is_self(node.value)


def _is_self(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "self"


def _private(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _forwarded_reaches(cls: ast.ClassDef) -> int:
    """``self._x`` the class does not define, in a class whose ``__getattr__`` forwards to the db."""
    forwards = any(
        isinstance(fn, ast.FunctionDef)
        and fn.name == "__getattr__"
        and any(_is_call_to(node, {"getattr"}) and node.args and _is_db(node.args[0]) for node in ast.walk(fn))
        for fn in cls.body
    )
    if not forwards:
        return 0
    if cls.bases:
        raise ValueError(f"{cls.name} forwards to the db but has bases this counter cannot resolve")
    on_self = [n for n in ast.walk(cls) if isinstance(n, ast.Attribute) and _is_self(n.value)]
    defined = {n.attr for n in on_self if isinstance(n.ctx, ast.Store)}
    defined |= {s.name for s in cls.body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for s in cls.body:
        targets = s.targets if isinstance(s, ast.Assign) else [s.target] if isinstance(s, ast.AnnAssign) else []
        defined |= {t.id for t in targets if isinstance(t, ast.Name)}
    return sum(1 for n in on_self if _private(n.attr) and n.attr not in defined)


def _reaches_privately(node: ast.AST) -> bool:
    """``<db>._x``, or ``getattr(<db>, "_x", ...)`` and its set/has/del siblings."""
    if isinstance(node, ast.Attribute):
        return _private(node.attr) and _is_db(node.value)
    if not _is_call_to(node, {"getattr", "setattr", "hasattr", "delattr"}) or len(node.args) < 2:
        return False
    target, name = node.args[:2]
    return _is_db(target) and isinstance(name, ast.Constant) and isinstance(name.value, str) and _private(name.value)


def _private_reaches(tree: ast.Module) -> int:
    return sum(
        _forwarded_reaches(node) if isinstance(node, ast.ClassDef) else _reaches_privately(node)
        for node in ast.walk(tree)
    )


def measure(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, int]]:
    app = (repo_root / "tldw_Server_API" / "app").resolve()
    trees = parse_tree(app)
    db_tree = trees[app / DB_MODULE]
    delegated, other = _bindings(db_tree)
    reaches = sum(_private_reaches(tree) for path, tree in trees.items() if path.is_relative_to(app / STORE_DIR))
    return {
        "methods": {"count": _class_body_defs(_the_class(db_tree).body) + other},
        "delegated_methods": {"count": delegated},
        "private_member_reaches": {"count": reaches},
    }


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------

_DB = '''
from typing import TYPE_CHECKING


class CharactersRAGDB:
    _shared = 1
    if TYPE_CHECKING:
        def add_note(self): ...
    def __init__(self): ...
    def get(self): ...
    @property
    def size(self): ...
    @size.setter
    def size(self, value): ...


def _delegate_store_method(store_attr, method_name):
    def _delegated(self, *args, **kwargs):
        return getattr(getattr(self, store_attr), method_name)(*args, **kwargs)
    return _delegated


def _delegate_note_store_method(method_name):
    return _delegate_store_method("note_store", method_name)


for _name in ("add_note", "get_note"):
    setattr(CharactersRAGDB, _name, _delegate_note_store_method(_name))

CharactersRAGDB.count_notes = _delegate_store_method("note_store", "count_notes")
CharactersRAGDB.get = _patched_get
'''

_STORE = '''
class NoteStore:
    def __init__(self, db):
        self._db = db
        self._cache = {}

    def read(self):
        self._db._execute()
        self._db.public()
        self._db.__class__
        self._cache.clear()
        getattr(self._db, "_flag", 0)
        db = self._db
        return db._backend


class ForwardingStore:
    def __init__(self, db):
        self._db = db

    def __getattr__(self, name):
        return getattr(self._db, name)

    @staticmethod
    def _own(): ...

    def write(self):
        self._own()
        self._prepare()
        return self.backend_type
'''

_SEEDED = {"methods": {"count": 5}, "delegated_methods": {"count": 3}, "private_member_reaches": {"count": 4}}


def _measure_seeded(root: Path, db: str = _DB, store: str = _STORE) -> dict[str, dict[str, int]]:
    app = root / "tldw_Server_API" / "app"
    for relative, source in ((DB_MODULE, db), (STORE_DIR / "note_store.py", store)):
        (app / relative).parent.mkdir(parents=True, exist_ok=True)
        (app / relative).write_text(source, encoding="utf-8")
    return measure(root)


def _baseline(counts: dict[str, dict[str, int]]) -> dict:
    return {key: {**value, "reason": None} for key, value in counts.items()}


def test_the_counter_counts_each_seeded_shape(tmp_path: Path) -> None:
    assert _measure_seeded(tmp_path) == _SEEDED


def test_a_new_delegate_registration_fails(tmp_path: Path) -> None:
    grown = _DB.replace('("add_note", "get_note")', '("add_note", "get_note", "list_notes")')

    (problem,) = _ceilings.violations(SURFACE, _measure_seeded(tmp_path, db=grown), _baseline(_SEEDED))

    assert problem.startswith("delegated_methods: count 4 > ceiling 3"), problem
    assert '"delegated_methods": {"count": 4, "reason": "<why' in problem


@pytest.mark.parametrize(
    "registration",
    [
        'setattr(CharactersRAGDB, "list_notes", _delegate_store_method("note_store", "list_notes"))',
        'CharactersRAGDB.list_notes = _delegate_note_store_method("list_notes")',
        'def _install():\n    setattr(CharactersRAGDB, "list_notes", _delegate_note_store_method("list_notes"))',
    ],
)
def test_every_registration_shape_counts(tmp_path: Path, registration: str) -> None:
    assert _measure_seeded(tmp_path, db=_DB + registration + "\n")["delegated_methods"] == {"count": 4}


@pytest.mark.parametrize(
    "unsized",
    [
        'held = _delegate_store_method("note_store", "list_notes")',
        'for _name in NAMES:\n    setattr(CharactersRAGDB, _name, _delegate_note_store_method(_name))',
    ],
)
def test_a_registration_the_counter_cannot_size_fails_closed(tmp_path: Path, unsized: str) -> None:
    with pytest.raises(ValueError, match="cannot"):
        _measure_seeded(tmp_path, db=_DB + unsized + "\n")


def test_a_new_method_fails_but_a_typing_stub_does_not(tmp_path: Path) -> None:
    stub = _DB.replace("        def add_note(self): ...", "        def add_note(self): ...\n        def x(self): ...")
    method = _DB.replace("    def get(self): ...", "    def get(self): ...\n    def new(self): ...")

    assert _measure_seeded(tmp_path / "stub", db=stub) == _SEEDED
    (problem,) = _ceilings.violations(SURFACE, _measure_seeded(tmp_path / "method", db=method), _baseline(_SEEDED))
    assert problem.startswith("methods: count 6 > ceiling 5"), problem


@pytest.mark.parametrize(
    "reach",
    ["self._db._new()", "db._new", 'hasattr(self._db, "_new")', "self._db._new = 1"],
)
def test_a_new_private_reach_fails(tmp_path: Path, reach: str) -> None:
    store = _STORE.replace("        return db._backend", f"        {reach}\n        return db._backend")

    (problem,) = _ceilings.violations(SURFACE, _measure_seeded(tmp_path, store=store), _baseline(_SEEDED))

    assert problem.startswith("private_member_reaches: count 5 > ceiling 4"), problem


def test_a_reach_through_a_forwarding_getattr_fails(tmp_path: Path) -> None:
    store = _STORE.replace("        self._prepare()", "        self._prepare()\n        self._commit()")

    assert _measure_seeded(tmp_path, store=store)["private_member_reaches"] == {"count": 5}


def test_shrinking_passes_without_a_baseline_edit(tmp_path: Path) -> None:
    shrunk = _DB.replace("CharactersRAGDB.get = _patched_get\n", "")

    assert _ceilings.violations(SURFACE, _measure_seeded(tmp_path, db=shrunk), _baseline(_SEEDED)) == []


# --- The ratchet --------------------------------------------------------------------------------


def test_charactersragdb_surface_does_not_grow() -> None:
    counts = measure()
    baseline = _ceilings.load(SURFACE.baseline)

    problems = _ceilings.violations(SURFACE, counts, baseline)

    assert not problems, "\n".join(problems)
    _ceilings.warn_if_tightenable(SURFACE, counts, baseline)


def test_raised_surface_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(SURFACE)
    assert not problems, "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(_ceilings.main(SURFACE, measure))
