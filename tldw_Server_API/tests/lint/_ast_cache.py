"""One cached AST parse of the backend tree, shared by the lint ratchets (TASK-13950.8).

Every AST ratchet needs the whole of tldw_Server_API/app, and parsing it takes seconds.
``parse_tree`` parses a root once per process, so all the lint test files one pytest
run collects share a single parse.

API (reused by TASK-13950.9 and .11):

    parse_tree(root: Path) -> Mapping[Path, ast.Module]
        Every ``*.py`` file under ``root``, recursively, keyed by its absolute path, in
        sorted path order. Cached per resolved root and never invalidated: a root's files
        are parsed once, so a self-test that rewrites seeded files needs a fresh root.
        A root inside APP_ROOT is served from APP_ROOT's parse, so it never holds a second
        copy. The mapping is read-only; do not mutate the trees. A file that does not
        parse raises SyntaxError naming it.

    REPO_ROOT, APP_ROOT
        The repository root and tldw_Server_API/app. Scan ``parse_tree(APP_ROOT)`` so every
        ratchet hits the same cache entry.

Import it by its full name, ``from tldw_Server_API.tests.lint._ast_cache import ...``, so
there is one module, and one cache, per process.

Ceiling: the parsed app tree, about 2.5 GB, is held for the rest of the pytest session.
Under pytest-xdist each worker would hold its own copy, and a runner with 7 GB or less
would be tight; there, switch to per-file derived facts (parse, extract, drop) instead.
``gc.freeze()`` after the parse moves the whole pytest heap at that moment, not only the
trees, out of the cyclic GC for the rest of the session.
"""

from __future__ import annotations

import ast
import gc
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API" / "app"


def parse_tree(root: Path) -> Mapping[Path, ast.Module]:
    """Parse every ``*.py`` under ``root`` once per process; see the module docstring."""
    root = root.resolve()
    if root != APP_ROOT and root.is_relative_to(APP_ROOT):
        app = _parse_tree(APP_ROOT)
        return MappingProxyType({path: tree for path, tree in app.items() if path.is_relative_to(root)})
    return _parse_tree(root)


@cache
def _parse_tree(root: Path) -> Mapping[Path, ast.Module]:
    # The app tree is ~8M AST nodes. Left on, the cyclic GC rescans them while they are
    # built (9.6 s instead of 3.4 s) and on every later full collection (~5 s each).
    # AST nodes hold no reference cycles, so freezing them out of the GC loses nothing.
    enabled = gc.isenabled()
    gc.disable()
    try:
        return MappingProxyType(
            {path: ast.parse(path.read_bytes(), filename=str(path)) for path in sorted(root.rglob("*.py"))}
        )
    finally:
        gc.freeze()
        if enabled:
            gc.enable()
