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

# BaseException subclasses that are NOT Exception subclasses. A tuple named
# "noncritical" must not contain any of them.
_BASE_ONLY_EXCEPTIONS = frozenset(
    {
        "BaseException",
        "KeyboardInterrupt",
        "SystemExit",
        "GeneratorExit",
        # asyncio.CancelledError and concurrent.futures.CancelledError are the same
        # class since 3.8, and it derives from BaseException.
        "CancelledError",
    }
)

_TUPLE_NAME_SUFFIX = "NONCRITICAL_EXCEPTIONS"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _app_root() -> Path:
    return _repo_root() / "tldw_Server_API" / "app"


def _member_name(node: ast.expr) -> str | None:
    """`ValueError` -> "ValueError"; `asyncio.CancelledError` -> "CancelledError"."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


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
                member = _member_name(element)
                if member in _BASE_ONLY_EXCEPTIONS:
                    rel = path.relative_to(_repo_root())
                    violations.append(f"{rel}:{node.lineno} {names[0]} contains {member}")

    return inspected, violations


def test_the_premise_holds_on_this_interpreter() -> None:
    """If this ever fails, the rule below is arguing against the language."""
    import asyncio

    assert not issubclass(asyncio.CancelledError, Exception)
    assert issubclass(asyncio.CancelledError, BaseException)


def test_noncritical_tuples_contain_no_baseexception_members() -> None:
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
