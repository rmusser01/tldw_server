"""Ratchet: app code calls the delegated CharactersRAGDB names, not the stores (TASK-13950.22).

``CharactersRAGDB`` re-attaches store methods to itself (``setattr`` loops at the end of
core/DB_Management/ChaChaNotes_DB.py), so ``db.add_note(...)`` works and hides which store owns
the method. This freezes, per app module, how many call sites still use such a delegated name,
in charactersragdb_delegated_callers_baseline.json, each a ceiling ``{"calls": int, "reason": ...}``.
New code calls ``db.<store>.<method>``; shrink the ceilings with ``--tighten`` as callers move.

Counted, in every ``*.py`` under the app root outside core/DB_Management/: an attribute load whose
attr is a delegated name, and a ``getattr``/``hasattr`` whose second argument is such a name literal.
The receiver (after ``last_identifier`` strips calls and subscripts) must match RECEIVER, must not
end in ``store`` and must not be ``asyncio``. Names that other classes share (COLLISIONS) count only
on LITERAL_RECEIVERS. The names are read from the class, so a new delegation is covered at once.

Known limit: a db held under a name RECEIVER does not match is not counted.

    python -m tldw_Server_API.tests.lint.test_charactersragdb_delegated_callers_ratchet --write-baseline
    python -m tldw_Server_API.tests.lint.test_charactersragdb_delegated_callers_ratchet --tighten
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint import test_charactersragdb_surface_ratchet as surface
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree

pytestmark = pytest.mark.unit

RECEIVER = re.compile(r"(^|_)db(\d+|_\w+)?$|database$|^cdb$|^chacha|chachanotes", re.I)
# Names other classes (jobs, keyword stores) also define; only an unmistakable ChaCha db counts.
COLLISIONS = frozenset({"get_task", "list_tasks", "create_task", "add_keyword", "soft_delete_keyword"})
LITERAL_RECEIVERS = frozenset({"chacha_db", "notes_db", "character_db"})
_DB_MODULE = "core/DB_Management/ChaChaNotes_DB.py"
_IGNORED = "core/DB_Management/"

RATCHET = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("charactersragdb_delegated_callers_baseline.json"),
    command="python -m tldw_Server_API.tests.lint.test_charactersragdb_delegated_callers_ratchet",
    fix="Call the store, `db.<store>.<method>`, not the delegated `db.<method>` (TASK-13950.22).",
    stale="The module no longer calls a delegated CharactersRAGDB name.",
)


def last_identifier(node: ast.expr) -> str:
    """The name a receiver ends in: calls and subscripts stripped, then an attr or a name."""
    while isinstance(node, (ast.Call, ast.Subscript)):
        node = node.func if isinstance(node, ast.Call) else node.value
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def counts(name: str, receiver: ast.expr) -> bool:
    ident = last_identifier(receiver)
    if not RECEIVER.search(ident) or ident.lower().endswith("store") or ident == "asyncio":
        return False
    return name not in COLLISIONS or ident in LITERAL_RECEIVERS


def _is_probe(node: ast.AST, names: dict[str, str]) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"getattr", "hasattr"}
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in names
        and counts(node.args[1].value, node.args[0])
    )


def measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    root = root.resolve()
    trees = {p.relative_to(root).as_posix(): t for p, t in parse_tree(root).items()}
    names = surface.delegated_names(trees[_DB_MODULE])
    out: dict[str, dict[str, int]] = {}
    for key, tree in sorted(trees.items()):
        if key.startswith(_IGNORED):
            continue
        calls = sum(
            (isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and n.attr in names and counts(n.attr, n.value))
            or _is_probe(n, names)
            for n in ast.walk(tree)
        )
        if calls:
            out[key] = {"calls": calls}
    return out


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------

API = "api/x.py"


def _seed(root: Path, source: str, key: str = API, db: str = surface._DB) -> Path:
    for relative, text in {_DB_MODULE: db, key: source}.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return root


def test_each_receiver_shape_counts(tmp_path: Path) -> None:
    src = """
    def f(db, chacha_db, db_instance):
        db.add_note()
        self.db.add_note()
        self._db.add_note()
        chacha_db.add_note()
        db_instance.add_note()
        get_chacha_db().add_note()
    """
    assert measure(_seed(tmp_path / "app", src)) == {API: {"calls": 6}}


def test_store_and_asyncio_receivers_do_not_count(tmp_path: Path) -> None:
    src = "db.note_store.add_note()\nasyncio.add_note()\nloop.add_note()\n"
    assert measure(_seed(tmp_path / "app", src)) == {}


def test_collision_names_count_only_on_literal_receivers(tmp_path: Path) -> None:
    db = surface._DB.replace('("add_note", "get_note")', '("add_note", "create_task")')
    assert measure(_seed(tmp_path / "a", "db.create_task()\n", db=db)) == {}
    src = "chacha_db.create_task()\nself.notes_db.create_task()\n"
    assert measure(_seed(tmp_path / "b", src, db=db)) == {API: {"calls": 2}}


def test_getattr_and_hasattr_probes_count(tmp_path: Path) -> None:
    src = 'getattr(db, "add_note", None)\nhasattr(chacha_db, "get_note")\ngetattr(db, name)\n'
    assert measure(_seed(tmp_path / "app", src)) == {API: {"calls": 2}}


def test_db_management_is_ignored(tmp_path: Path) -> None:
    assert measure(_seed(tmp_path / "app", "db.add_note()\n", key="core/DB_Management/chacha/s.py")) == {}


def test_names_are_read_from_the_class(tmp_path: Path) -> None:
    assert measure(_seed(tmp_path / "a", "db.list_notes()\n")) == {}
    db = surface._DB.replace('("add_note", "get_note")', '("add_note", "get_note", "list_notes")')
    assert measure(_seed(tmp_path / "b", "db.list_notes()\n", db=db)) == {API: {"calls": 1}}


def test_growth_and_stale_entries_fail(tmp_path: Path) -> None:
    measured = measure(_seed(tmp_path / "app", "db.add_note()\ndb.add_note()\ndb.get_note()\n"))
    baseline = {API: {"calls": 2, "reason": None}, "api/gone.py": {"calls": 1, "reason": None}}
    problems = "\n".join(_ceilings.violations(RATCHET, measured, baseline))
    assert "calls 3 > ceiling 2" in problems
    assert "stale entry" in problems


def test_raised_ceiling_without_reason_fails() -> None:
    base = {"m.py": {"calls": 2, "reason": None}}
    bumped = {"m.py": {"calls": 3, "reason": None}}
    problems = _ceilings.unreasoned_raises(RATCHET, bumped, base)
    assert len(problems) == 1 and "without a new reason" in problems[0]


# --- The ratchet, against the real tree ---------------------------------------------------------


def test_delegated_callers_do_not_grow() -> None:
    measured, baseline = measure(), _ceilings.load(RATCHET.baseline)

    problems = _ceilings.violations(RATCHET, measured, baseline)

    assert problems == [], "\n".join(problems)
    _ceilings.warn_if_tightenable(RATCHET, measured, baseline)


def test_raised_delegated_caller_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(RATCHET, measure())
    assert problems == [], "\n".join(problems)


def test_delegated_names_match_the_surface_count() -> None:
    names = surface.delegated_names(parse_tree(APP_ROOT)[APP_ROOT / _DB_MODULE])
    assert len(names) == surface.measure()["delegated_methods"]["count"]


if __name__ == "__main__":
    raise SystemExit(_ceilings.main(RATCHET, measure))
