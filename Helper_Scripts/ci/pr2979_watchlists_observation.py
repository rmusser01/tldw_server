"""Scoped content-free Watchlists timing; instrumented results are diagnostic only."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

TARGET = "test_watchlists_runs_scale_endpoints_within_budget"


def unavailable() -> None:
    """Disclose missing observation without replacing the original outcome."""
    try:
        print("UAT552_DIAGNOSTIC_UNAVAILABLE")
    except OSError:
        pass


def bindings(item: Any, database: Any) -> dict[str, Any]:
    """Record loaded source identity and versions without emitting machine paths."""
    root = Path.cwd().resolve()
    test_path = getattr(getattr(item, "module", None), "__file__", None)
    db_path = getattr(sys.modules.get(database.__module__), "__file__", None)
    return {
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_origin_matches": bool(test_path)
        and Path(test_path).resolve() == root / "tldw_Server_API/tests/Watchlists/test_watchlists_scale_load_api.py",
        "database_origin_matches": bool(db_path)
        and Path(db_path).resolve() == root / "tldw_Server_API/app/core/DB_Management/Watchlists_DB.py",
        "target_source_sha256": hashlib.sha256(Path(test_path).read_bytes()).hexdigest() if test_path else None,
        "runtime_versions": {
            name: getattr(sys.modules.get(name), "__version__", None)
            for name in ("fastapi", "pydantic", "pydantic_core", "starlette", "pytest")
        },
        "python_version": list(sys.version_info[:3]),
    }


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: Any) -> Generator[None, None, None]:
    """Delegate the selected case once and restore every patched function."""
    if item.name != TARGET:
        yield
        return
    prefix = os.environ.get("UAT552_PROBE_OUTPUT")
    if not prefix:
        try:
            yield
        finally:
            unavailable()
        return
    import fastapi.routing as routing
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase

    try:
        targets = [
            *[
                (routing._IncludedRouter, method, "route." + method)
                for method in ("effective_candidates", "_build_effective_context")
            ],
            *[
                (WatchlistsDatabase, method, "db." + method)
                for method in ("__init__", "ensure_schema_once", "list_runs", "list_runs_for_job", "get_run")
            ],
            *[
                (routing, method, "fastapi." + method)
                for method in ("serialize_response", "run_endpoint_function", "solve_dependencies")
            ],
        ]
        originals = [getattr(owner, method) for owner, method, _ in targets]
        if not all(callable(original) for original in originals):
            raise AttributeError
    except AttributeError:
        try:
            yield
        finally:
            unavailable()
        return

    records: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None

    def timed(name: str, original: Any) -> Any:
        def record(started: int, cpu: int) -> None:
            if active is not None:
                metric = active["sections"].setdefault(name, {"calls": 0, "wall_ns": 0, "thread_cpu_ns": 0})
                metric["calls"] += 1
                metric["wall_ns"] += time.perf_counter_ns() - started
                metric["thread_cpu_ns"] += time.thread_time_ns() - cpu

        @functools.wraps(original)
        def sync(*args: Any, **kwargs: Any) -> Any:
            started, cpu = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return original(*args, **kwargs)
            finally:
                record(started, cpu)

        @functools.wraps(original)
        async def asynchronous(*args: Any, **kwargs: Any) -> Any:
            started, cpu = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return await original(*args, **kwargs)
            finally:
                record(started, cpu)

        return asynchronous if inspect.iscoroutinefunction(original) else sync

    original_get, original_call = TestClient.get, FastAPI.__call__

    @functools.wraps(original_get)
    def measured_get(*args: Any, **kwargs: Any) -> Any:
        nonlocal active
        active = {"ordinal": len(records) + 1, "sections": {}}
        started, cpu = time.perf_counter_ns(), time.process_time_ns()
        try:
            return original_get(*args, **kwargs)
        finally:
            active["wall_ns"] = time.perf_counter_ns() - started
            active["process_cpu_ns"] = time.process_time_ns() - cpu
            records.append(active)
            active = None

    @functools.wraps(original_call)
    async def measured_call(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter_ns()
        try:
            return await original_call(*args, **kwargs)
        finally:
            if active is not None:
                active["asgi_wall_ns"] = time.perf_counter_ns() - started

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(TestClient, "get", measured_get)
            patch.setattr(FastAPI, "__call__", measured_call)
            for (owner, method, label), original in zip(targets, originals, strict=True):
                patch.setattr(owner, method, timed(label, original))
            yield
    finally:
        try:
            document = {
                "schema_version": 1,
                "sections_are_inclusive": True,
                "bindings": bindings(item, WatchlistsDatabase),
                "records": records,
            }
            Path(prefix + ".json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        except (OSError, MemoryError, AttributeError, TypeError):
            unavailable()
