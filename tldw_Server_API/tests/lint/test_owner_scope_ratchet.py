"""Ratchet: ChaCha owner-scope helpers that drop their tenant predicate on None (TASK-13950.23).

A helper such as ``_selected_owner_filter(owner_client_id)`` that returns ``""`` when the owner is
None quietly turns a tenant-scoped query into an unscoped one. This freezes, per module, how many
call sites such helpers have and how many calls opt out explicitly with ``OwnerScope.NOT_APPLICABLE``
in owner_scope_baseline.json, each a ceiling ``{"calls": int, "opt_outs": int, "reason": ...}``.
A later task fixes the helpers and shrinks the ceilings with ``--tighten``.

A function in SCOPED_MODULES is a drop-on-None helper when it has a parameter in OWNER_PARAMS and an
``if`` testing ``<param> is None`` (directly or inside a BoolOp) whose body is exactly one ``return``,
bare or returning a tuple that starts with ``""``. A ``raise`` body or ``return None`` does not count.

Counted per module: ``calls`` (every call whose function name is a helper's) and ``opt_outs`` (call
arguments that are exactly ``OwnerScope.NOT_APPLICABLE``).

    python -m tldw_Server_API.tests.lint.test_owner_scope_ratchet --write-baseline
    python -m tldw_Server_API.tests.lint.test_owner_scope_ratchet --tighten
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Mapping
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree

pytestmark = pytest.mark.unit

OWNER_PARAMS = frozenset({"owner_client_id", "client_id", "owner_user_id", "user_id", "owner_id", "tenant_id"})
_CHACHA_DIR = "core/DB_Management/chacha/"
_CHACHA_DB = "core/DB_Management/ChaChaNotes_DB.py"

RATCHET = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("owner_scope_baseline.json"),
    command="python -m tldw_Server_API.tests.lint.test_owner_scope_ratchet",
    fix=(
        "Pass the owner and make the helper raise on None for PostgreSQL, or opt out with "
        "OwnerScope.NOT_APPLICABLE where the table has no tenant (TASK-13950.23)."
    ),
    stale="The module no longer calls a helper that drops the owner predicate on None.",
)


def _is_scoped(key: str) -> bool:
    return key == _CHACHA_DB or (key.startswith(_CHACHA_DIR) and key.endswith(".py") and "/" not in key[len(_CHACHA_DIR):])


def _trees(root: Path) -> dict[str, ast.Module]:
    root = root.resolve()
    keyed = {p.relative_to(root).as_posix(): t for p, t in parse_tree(root).items()}
    return {k: t for k, t in keyed.items() if _is_scoped(k)}


def _tests_none(test: ast.expr, param: str) -> bool:
    if isinstance(test, ast.BoolOp):
        return any(_tests_none(v, param) for v in test.values)
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == param
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def _drops_predicate(body: list[ast.stmt]) -> bool:
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return False
    value = body[0].value
    if value is None:
        return True
    return (
        isinstance(value, ast.Tuple)
        and bool(value.elts)
        and isinstance(value.elts[0], ast.Constant)
        and value.elts[0].value == ""
    )


def drop_on_none_helpers(trees: Mapping[str, ast.Module]) -> set[str]:
    names: set[str] = set()
    for key, tree in trees.items():
        if not _is_scoped(key):
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = fn.args
            params = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)} & OWNER_PARAMS
            if any(
                isinstance(n, ast.If) and _drops_predicate(n.body) and any(_tests_none(n.test, p) for p in params)
                for n in ast.walk(fn)
            ):
                names.add(fn.name)
    return names


def _is_opt_out(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "NOT_APPLICABLE"
        and isinstance(node.value, ast.Name)
        and node.value.id == "OwnerScope"
    )


def _func_name(call: ast.Call) -> str | None:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    trees = _trees(root)
    helpers = drop_on_none_helpers(trees)
    out: dict[str, dict[str, int]] = {}
    for key, tree in sorted(trees.items()):
        calls = opt_outs = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            calls += _func_name(node) in helpers
            opt_outs += sum(_is_opt_out(a) for a in (*node.args, *(k.value for k in node.keywords)))
        if calls or opt_outs:
            out[key] = {"calls": calls, "opt_outs": opt_outs}
    return out


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


CHACHA = "core/DB_Management/chacha/s.py"


def _calls(helper: str, n: int) -> str:
    return "".join(f"    def c{i}(self):\n        return self.{helper}(1)\n" for i in range(n))


def test_early_return_tuple_helper_and_its_calls_count(tmp_path: Path) -> None:
    src = (
        "class S:\n    def _f(self, owner_client_id):\n"
        '        if owner_client_id is None or self.x != 1:\n            return "", ()\n        return "a", (1,)\n'
        + _calls("_f", 2)
    )
    assert measure(_seed(tmp_path / "app", {CHACHA: src})) == {CHACHA: {"calls": 2, "opt_outs": 0}}


def test_bare_return_and_owner_user_id_count(tmp_path: Path) -> None:
    src = (
        "class S:\n    def _f(self, *, owner_user_id):\n"
        "        if self.x and owner_user_id is None:\n            return\n        self.y = 1\n"
        + _calls("_f", 1)
    )
    assert measure(_seed(tmp_path / "app", {CHACHA: src})) == {CHACHA: {"calls": 1, "opt_outs": 0}}


@pytest.mark.parametrize(
    "body",
    [
        'if owner_client_id is None:\n            raise ValueError("x")',
        "if owner_client_id is None:\n            return None",
    ],
)
def test_raise_and_return_none_are_not_helpers(tmp_path: Path, body: str) -> None:
    src = f"class S:\n    def _f(self, owner_client_id):\n        {body}\n" + _calls("_f", 1)
    assert measure(_seed(tmp_path / "app", {CHACHA: src})) == {}


def test_non_owner_param_is_not_a_helper(tmp_path: Path) -> None:
    src = 'class S:\n    def _f(self, other):\n        if other is None:\n            return "", ()\n' + _calls("_f", 1)
    assert measure(_seed(tmp_path / "app", {CHACHA: src})) == {}


def test_module_outside_chacha_is_ignored(tmp_path: Path) -> None:
    src = 'class S:\n    def _f(self, owner_client_id):\n        if owner_client_id is None:\n            return "", ()\n' + _calls("_f", 1)
    assert measure(_seed(tmp_path / "app", {"core/DB_Management/other_store.py": src})) == {}


def test_opt_outs_are_counted(tmp_path: Path) -> None:
    src = (
        "class S:\n    def _f(self, owner_client_id):\n"
        '        if owner_client_id is None:\n            return "", ()\n'
        "    def a(self):\n        self._f(owner_client_id=OwnerScope.NOT_APPLICABLE)\n"
        "    def b(self):\n        self._f(OwnerScope.NOT_APPLICABLE)\n"
        "    def c(self, x):\n        return x is OwnerScope.NOT_APPLICABLE\n"
    )
    assert measure(_seed(tmp_path / "app", {CHACHA: src})) == {CHACHA: {"calls": 2, "opt_outs": 2}}


def test_growth_and_stale_entries_fail(tmp_path: Path) -> None:
    src = 'class S:\n    def _f(self, owner_client_id):\n        if owner_client_id is None:\n            return "", ()\n' + _calls("_f", 3)
    measured = measure(_seed(tmp_path / "app", {CHACHA: src}))
    baseline = {CHACHA: {"calls": 2, "opt_outs": 0, "reason": None},
                "core/DB_Management/chacha/gone.py": {"calls": 1, "opt_outs": 0, "reason": None}}
    problems = _ceilings.violations(RATCHET, measured, baseline)
    assert "calls 3 > ceiling 2" in "\n".join(problems)
    assert "stale entry" in "\n".join(problems)


def test_raised_ceiling_without_reason_fails() -> None:
    base = {"m.py": {"calls": 2, "opt_outs": 0, "reason": None}}
    bumped = {"m.py": {"calls": 3, "opt_outs": 0, "reason": None}}
    problems = _ceilings.unreasoned_raises(RATCHET, bumped, base)
    assert len(problems) == 1 and "without a new reason" in problems[0]


# --- The ratchet, against the real tree ---------------------------------------------------------


def test_owner_scope_does_not_grow() -> None:
    measured, baseline = measure(), _ceilings.load(RATCHET.baseline)

    problems = _ceilings.violations(RATCHET, measured, baseline)

    assert problems == [], "\n".join(problems)
    _ceilings.warn_if_tightenable(RATCHET, measured, baseline)


def test_raised_owner_scope_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(RATCHET, measure())
    assert problems == [], "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(_ceilings.main(RATCHET, measure))
