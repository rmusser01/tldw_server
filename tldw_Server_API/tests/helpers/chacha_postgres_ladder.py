"""Read the ChaCha PostgreSQL migration ladder by AST (TASK-13950.22, R26, R33).

Source-text tests of ``_initialize_schema_postgres`` break when a call site is rewritten
(``self._x(conn)`` becomes ``postgres_schema._x(self.ctx, conn)``). These helpers read the
callee names and keyword arguments instead, so the same tests survive a move. Positional
arguments are not kept, so the extra ``ctx`` argument of a moved call changes nothing.
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
    """Return the function that holds the PostgreSQL ladder (Task 9 repoints this one line)."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    return CharactersRAGDB._initialize_schema_postgres


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


def _calls_in(nodes: Iterable[ast.AST]) -> list[LadderCall]:
    found: list[tuple[int, int, LadderCall]] = []
    for root in nodes:
        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
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


def loaded_names(fn: Callable) -> set[str]:
    """Every attribute name and bare name ``fn`` reads, for 'the function reads X' checks."""
    out: set[str] = set()
    for node in ast.walk(_tree(fn)):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Name):
            out.add(node.id)
    return out
