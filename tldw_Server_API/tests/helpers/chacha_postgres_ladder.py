"""Read the ChaCha PostgreSQL migration ladder by AST (TASK-13950.22, R26, R33).

Source-text tests of ``_initialize_schema_postgres`` break when a call site is rewritten
(``self._x(conn)`` becomes ``postgres_schema._x(self.ctx, conn)``). These helpers read the
callee names and keyword arguments instead, so the same tests survive a move. Positional
arguments are not kept, so the extra ``ctx`` argument of a moved call changes nothing. A loop over
``POSTGRES_STEPS`` stands for the registry's steps in key order, so a test sees the v72 to v91 steps.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable, Iterable
from typing import NamedTuple


class LadderCall(NamedTuple):
    """One call in the ladder: the callee's name and its keyword arguments."""

    name: str
    kwargs: dict[str, object]


def postgres_init_function() -> Callable:
    """Return the function that holds the PostgreSQL ladder: the bootstrap module's, not the class's forwarder."""
    from tldw_Server_API.app.core.DB_Management.chacha import schema_bootstrap

    return schema_bootstrap._initialize_schema_postgres


def _tree(fn: Callable) -> ast.FunctionDef:
    module = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = module.body[0]
    assert isinstance(node, ast.FunctionDef), "expected a plain function definition"
    return node


def _keyword_value(node: ast.expr) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return ast.unparse(node)


def _is_registry_loop(node: ast.AST) -> bool:
    """A ``for ... in postgres_steps.POSTGRES_STEPS.items()`` loop, the shape that replaced the 73..91 if-chains."""
    return isinstance(node, ast.For) and ast.unparse(node.iter).endswith("POSTGRES_STEPS.items()")


def _registry_names() -> list[str]:
    from tldw_Server_API.app.core.DB_Management.chacha.migrations import postgres_steps

    return list(postgres_steps.POSTGRES_STEPS.values())


def _calls_in(nodes: Iterable[ast.AST]) -> list[LadderCall]:
    """Calls in source order. A registry loop counts as one call per registry step, in key order, at the loop's position."""
    roots = list(nodes)
    found: list[tuple[int, int, LadderCall]] = []
    skipped: set[int] = set()
    for root in roots:
        for node in ast.walk(root):
            if _is_registry_loop(node):
                found.extend((node.lineno, node.col_offset, LadderCall(name, {})) for name in _registry_names())  # type: ignore[attr-defined]
                skipped.update(id(call) for call in ast.walk(node) if isinstance(call, ast.Call))
    for root in roots:
        for node in ast.walk(root):
            if not isinstance(node, ast.Call) or id(node) in skipped:
                continue
            func = node.func
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            else:
                continue
            kwargs = {kw.arg: _keyword_value(kw.value) for kw in node.keywords if kw.arg is not None}
            found.append((node.lineno, node.col_offset, LadderCall(name, kwargs)))
    return [call for _, _, call in sorted(found, key=lambda item: item[:2])]


def ladder_calls(fn: Callable) -> list[LadderCall]:
    """Every call in ``fn``'s body, in source order."""
    return _calls_in(_tree(fn).body)


def branch_calls(fn: Callable, version: int) -> list[LadderCall]:
    """The calls inside each ``if current_version < <version>:`` of ``fn``, in source order."""
    wanted = f"current_version < {version}"
    branches = [n for n in ast.walk(_tree(fn)) if isinstance(n, ast.If) and ast.unparse(n.test) == wanted]
    return _calls_in(stmt for branch in branches for stmt in branch.body)


def names(calls: Iterable[LadderCall]) -> list[str]:
    """The callee names alone, for order checks."""
    return [call.name for call in calls]


def assigned_names(fn: Callable, target: str) -> set[str]:
    """Every attribute and bare name read by the right-hand side of each ``<target> = ...`` in ``fn`` (empty when never assigned)."""
    found: set[str] = set()
    for node in ast.walk(_tree(fn)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in node.targets):
            for part in ast.walk(node.value):
                if isinstance(part, ast.Attribute):
                    found.add(part.attr)
                elif isinstance(part, ast.Name):
                    found.add(part.id)
    return found
