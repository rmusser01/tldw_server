"""Guard against blocking calls reappearing inside ``async def`` bodies.

A blocking call awaited on the API event loop stalls every other request in the
worker for its whole duration. This is not a proportional slowdown: one
occurrence degrades the entire process.

The failure this guards against was real. Twelve workflow adapters called
``subprocess.run(cmd, timeout=300)`` from async functions, and ``WorkflowEngine``
awaits those adapters on the request loop. Serving ``/health`` while one 1.5 s
media step ran gave a 1565 ms worst-case latency for unrelated requests, against
9 ms once the call was awaited properly.

Scope note: ``open()`` and other filesystem calls are deliberately *not* checked.
They are pervasive in the current tree and mostly touch small local files, so
enforcing them would mean a large exclusion list that quickly stops meaning
anything. The four categories below are all at zero today and are the ones that
block for unbounded time -- network round trips, sleeps, process waits, and
database locks.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP_ROOT = Path("tldw_Server_API/app")

# Dotted call -> why it blocks. Matched on the full dotted name and on the last
# two segments, so both `subprocess.run(...)` and `mod.subprocess.run(...)` hit.
BLOCKING_CALLS: dict[str, str] = {
    # Unbounded network waits
    "requests.get": "network",
    "requests.post": "network",
    "requests.put": "network",
    "requests.patch": "network",
    "requests.delete": "network",
    "requests.head": "network",
    "requests.request": "network",
    "requests.Session": "network",
    "httpx.get": "network",
    "httpx.post": "network",
    "httpx.put": "network",
    "httpx.delete": "network",
    "httpx.request": "network",
    "httpx.Client": "network",
    "urllib.request.urlopen": "network",
    "urlopen": "network",
    # Sleeps
    "time.sleep": "sleep",
    # Process waits
    "subprocess.run": "subprocess",
    "subprocess.call": "subprocess",
    "subprocess.check_call": "subprocess",
    "subprocess.check_output": "subprocess",
    "subprocess.Popen": "subprocess",
    "os.system": "subprocess",
    # Database locks
    "sqlite3.connect": "database",
}

REMEDIATION = {
    "network": "await the shared http_client helpers (afetch / create_async_client)",
    "sleep": "use `await asyncio.sleep(...)`",
    "subprocess": (
        "use Workflows.subprocess_utils.run_checked_async, or "
        "`await asyncio.create_subprocess_exec(...)`"
    ),
    "database": (
        "open the connection through sqlite_policy and run the work in "
        "`asyncio.to_thread`, or use an async driver"
    ),
}


def _dotted_name(node: ast.AST) -> str | None:
    """Render `a.b.c` from an attribute/name chain, else None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class _AsyncBodyScanner(ast.NodeVisitor):
    """Collect blocking calls in one async body, ignoring nested functions.

    A nested `def` is not awaited on the loop by virtue of its position -- it is
    typically handed to `asyncio.to_thread` or `run_in_executor`, which is the
    fix rather than the defect. Nested `async def` bodies are visited separately
    by the module-level walk.
    """

    def __init__(self) -> None:
        self.hits: list[tuple[int, str, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted_name(node.func)
        if dotted is not None:
            category = BLOCKING_CALLS.get(dotted)
            name = dotted
            if category is None:
                tail = ".".join(dotted.split(".")[-2:])
                category = BLOCKING_CALLS.get(tail)
                name = tail
            if category is not None:
                self.hits.append((node.lineno, name, category))
        self.generic_visit(node)


def _iter_source_files() -> list[Path]:
    return sorted(
        path
        for path in APP_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts and "tests" not in path.parts
    )


def _find_blocking_calls() -> list[str]:
    findings: list[str] = []
    for path in _iter_source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncFunctionDef):
                continue
            scanner = _AsyncBodyScanner()
            for statement in node.body:
                scanner.visit(statement)
            for lineno, name, category in scanner.hits:
                findings.append(
                    f"{path}:{lineno} {name}() in async def {node.name}() "
                    f"[{category}] -- {REMEDIATION[category]}"
                )
    return findings


@pytest.mark.unit
def test_no_blocking_calls_inside_async_functions() -> None:
    """No unbounded blocking call may sit directly in an ``async def`` body."""
    findings = _find_blocking_calls()
    assert not findings, (
        f"{len(findings)} blocking call(s) found inside async functions. "
        "Each one stalls every concurrent request in the worker:\n  "
        + "\n  ".join(findings)
    )


@pytest.mark.unit
def test_guard_detects_a_planted_blocking_call() -> None:
    """The scanner must actually catch what it claims to, not vacuously pass."""
    source = (
        "import subprocess\n"
        "async def handler():\n"
        "    subprocess.run(['ffmpeg'], timeout=300)\n"
    )
    scanner = _AsyncBodyScanner()
    tree = ast.parse(source)
    async_def = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef))
    for statement in async_def.body:
        scanner.visit(statement)
    assert [(name, category) for _, name, category in scanner.hits] == [
        ("subprocess.run", "subprocess")
    ]


@pytest.mark.unit
def test_guard_ignores_calls_dispatched_to_a_thread() -> None:
    """Work moved into a nested sync function is the fix, not a finding."""
    source = (
        "import asyncio, subprocess\n"
        "async def handler():\n"
        "    def _work():\n"
        "        subprocess.run(['ffmpeg'])\n"
        "    await asyncio.to_thread(_work)\n"
    )
    scanner = _AsyncBodyScanner()
    tree = ast.parse(source)
    async_def = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef))
    for statement in async_def.body:
        scanner.visit(statement)
    assert scanner.hits == []
