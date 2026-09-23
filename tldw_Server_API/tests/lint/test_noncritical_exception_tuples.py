"""Ratchet for TASK-13295: a "noncritical" tuple may not contain BaseException.

The `*_NONCRITICAL_EXCEPTIONS` tuples across the codebase exist to replace bare
`except Exception:` and satisfy ruff BLE001. Several were written with
`asyncio.CancelledError` as a member -- which `except Exception:` never caught, because

    issubclass(asyncio.CancelledError, Exception) is False
    asyncio.CancelledError.__mro__ == (CancelledError, BaseException, object)

so the remediation *widened* what is swallowed and introduced a defect the lint rule it
was written for cannot see. Cancellation is never noncritical: suppressing it means an
awaiting coroutine keeps running for a client that is gone, and `Task.cancel()` never
completes, so graceful shutdown blocks on it.

The allowance is zero and must stay zero. Code that genuinely needs to act on
cancellation should name `asyncio.CancelledError` in its own `except` clause and
re-raise, not fold it into a tuple whose name promises the opposite.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# BaseException subclasses that are NOT Exception subclasses, by unqualified name. A
# tuple named "noncritical" must not contain any of them.
_BASE_ONLY_EXCEPTIONS = frozenset(
    {
        "BaseException",
        "KeyboardInterrupt",
        "SystemExit",
        "GeneratorExit",
    }
)

# `CancelledError` cannot be judged on its unqualified name, and an earlier version of
# this rule got it wrong -- caught by review. There are two distinct classes:
#
#   asyncio.exceptions.CancelledError            BaseException only  -> banned
#   concurrent.futures._base.CancelledError      Exception subclass  -> allowed
#
# They are NOT aliases of one another on any version this repo supports (verified on
# 3.12: `asyncio.CancelledError is concurrent.futures.CancelledError` is False, and the
# futures one has Exception in its MRO). Banning the bare name therefore rejected
# correct code, so the asyncio one is identified by its qualification or its import.
_ASYNCIO_CANCELLED = "asyncio.CancelledError"
_CANCELLED_NAME = "CancelledError"

_TUPLE_NAME_SUFFIX = "NONCRITICAL_EXCEPTIONS"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _app_root() -> Path:
    return _repo_root() / "tldw_Server_API" / "app"


def _dotted_name(node: ast.expr) -> str | None:
    """`ValueError` -> "ValueError"; `asyncio.CancelledError` -> "asyncio.CancelledError".

    Qualification is preserved deliberately: two different classes share the
    unqualified name `CancelledError`, and only one of them is BaseException-only.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _bare_cancelled_is_asyncio(tree: ast.Module) -> bool:
    """Resolve an unqualified `CancelledError` to its module via the file's imports.

    A bare name with no matching import would be a NameError at runtime, so absence of
    an import means the name came from somewhere this scan cannot see; that is treated
    as the banned case so the rule fails loudly rather than silently allowing it.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not any(alias.name == _CANCELLED_NAME for alias in node.names):
            continue
        module = node.module or ""
        return not (module.startswith("concurrent.futures") or module == "concurrent")
    return True


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    if isinstance(node, ast.AnnAssign):
        return [node.target.id] if isinstance(node.target, ast.Name) else []
    return [t.id for t in node.targets if isinstance(t, ast.Name)]


def _scan() -> tuple[int, list[str]]:
    """Return (tuples inspected, violations) across app/."""
    app_root = _app_root()
    inspected = 0
    violations: list[str] = []

    for path in sorted(app_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:  # pragma: no cover - a broken file fails its own tests
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            names = [n for n in _assigned_names(node) if n.endswith(_TUPLE_NAME_SUFFIX)]
            if not names or not isinstance(node.value, ast.Tuple):
                continue
            inspected += 1
            for element in node.value.elts:
                member = _dotted_name(element)
                if member is None:
                    continue
                banned = member.rsplit(".", 1)[-1] in _BASE_ONLY_EXCEPTIONS
                if member == _ASYNCIO_CANCELLED:
                    banned = True
                elif member == _CANCELLED_NAME:
                    banned = _bare_cancelled_is_asyncio(tree)
                if banned:
                    rel = path.relative_to(_repo_root())
                    violations.append(f"{rel}:{node.lineno} {names[0]} contains {member}")

    return inspected, violations


def test_the_premise_holds_on_this_interpreter() -> None:
    """If this ever fails, the rule below is arguing against the language."""
    import asyncio

    assert not issubclass(asyncio.CancelledError, Exception)
    assert issubclass(asyncio.CancelledError, BaseException)


def test_the_two_cancellederrors_are_different_classes() -> None:
    """The distinction the first version of this rule got wrong.

    Banning the unqualified name `CancelledError` rejected
    `concurrent.futures.CancelledError`, which is an ordinary Exception subclass and is
    legitimate in a noncritical tuple. If these ever do become aliases, this test fails
    and the rule can be simplified back to a name check.
    """
    import asyncio
    import concurrent.futures

    assert asyncio.CancelledError is not concurrent.futures.CancelledError
    assert issubclass(concurrent.futures.CancelledError, Exception)
    assert not issubclass(asyncio.CancelledError, Exception)


def test_the_scanner_distinguishes_the_two_by_qualification() -> None:
    """Exercise the resolution logic directly, both spellings and both import forms."""
    banned_qualified = ast.parse("X_NONCRITICAL_EXCEPTIONS = (asyncio.CancelledError,)")
    banned_bare = ast.parse(
        "from asyncio import CancelledError\nX_NONCRITICAL_EXCEPTIONS = (CancelledError,)"
    )
    allowed_qualified = ast.parse(
        "X_NONCRITICAL_EXCEPTIONS = (concurrent.futures.CancelledError,)"
    )
    allowed_bare = ast.parse(
        "from concurrent.futures import CancelledError\n"
        "X_NONCRITICAL_EXCEPTIONS = (CancelledError,)"
    )

    def _members(tree: ast.Module) -> list[str]:
        assign = next(n for n in tree.body if isinstance(n, ast.Assign))
        return [_dotted_name(e) or "" for e in assign.value.elts]

    assert _members(banned_qualified) == ["asyncio.CancelledError"]
    assert _members(allowed_qualified) == ["concurrent.futures.CancelledError"]
    assert _bare_cancelled_is_asyncio(banned_bare) is True
    assert _bare_cancelled_is_asyncio(allowed_bare) is False


def test_noncritical_tuples_contain_no_baseexception_members() -> None:
    """Scan every *_NONCRITICAL_EXCEPTIONS tuple in app/ and reject BaseException-only members."""
    inspected, violations = _scan()

    assert inspected > 0, "scanner matched nothing -- the tuple naming convention moved"
    assert not violations, (
        "A *_NONCRITICAL_EXCEPTIONS tuple contains a BaseException-derived member. "
        "The bare `except Exception:` these tuples replaced did not catch it, so adding "
        "it changed behaviour: cancellation is swallowed, the coroutine runs on for a "
        "client that is gone, and Task.cancel() never completes.\n\n"
        "If a site genuinely needs to act on cancellation, give it its own "
        "`except asyncio.CancelledError:` clause that re-raises.\n\n"
        + "\n".join(violations)
    )
