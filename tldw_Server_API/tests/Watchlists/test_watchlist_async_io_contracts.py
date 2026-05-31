"""Async I/O contract tests for Watchlists endpoints."""

from __future__ import annotations

import ast
import inspect

from tldw_Server_API.app.api.v1.endpoints import watchlists


def _threadpool_targets(function: object) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name) or call.func.id != "run_in_threadpool":
            continue
        if not call.args:
            continue
        target = call.args[0]
        if isinstance(target, ast.Name):
            targets.add(target.id)
    return targets


def test_watchlist_run_stream_reads_log_files_via_threadpool() -> None:
    targets = _threadpool_targets(watchlists.stream_run)

    assert "_read_log_tail" in targets
    assert "_read_log_chunk" in targets
