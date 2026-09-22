"""Guard: a *_NONCRITICAL_EXCEPTIONS tuple must not swallow BaseException-derived types.

BLE001 remediation replaced `except Exception:` with explicit exception tuples. That is
safe for Exception subclasses, but `asyncio.CancelledError` derives from BaseException
(Python 3.8+), so listing it in such a tuple *widens* the catch relative to the
`except Exception:` it replaced and breaks cooperative cancellation. The lint rule that
motivated the change cannot see this.

This test is an AST ratchet over the declared tuples, in the style of
tests/lint/test_endpoint_auth_deps_import_boundary.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"

# Names that are BaseException-derived and therefore must never appear in a
# "noncritical"/swallowable exception tuple.
BANNED_MEMBERS = {
    "CancelledError",          # asyncio.CancelledError
    "KeyboardInterrupt",
    "SystemExit",
    "GeneratorExit",
    "BaseException",
}

TUPLE_NAME_SUFFIXES = ("_NONCRITICAL_EXCEPTIONS", "_NONCRITICAL_EXCS")


def _member_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _offenders() -> list[str]:
    offenders: list[str] = []
    for path in APP_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not any(t.endswith(TUPLE_NAME_SUFFIXES) for t in targets):
                continue
            if not isinstance(node.value, ast.Tuple):
                continue
            for element in node.value.elts:
                name = _member_name(element)
                if name in BANNED_MEMBERS:
                    rel = path.relative_to(REPO_ROOT)
                    offenders.append(f"{rel}:{element.lineno}: {targets[0]} includes {name}")
    return sorted(offenders)


def test_noncritical_exception_tuples_exclude_base_exceptions() -> None:
    offenders = _offenders()
    assert not offenders, (
        "BaseException-derived members in swallowable exception tuples "
        "(these widen the catch beyond the `except Exception:` they replaced):\n"
        + "\n".join(offenders)
    )
