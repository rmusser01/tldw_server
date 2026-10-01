"""Synthetic delegation/privacy controls; no application or hosted acceptance."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest

from Helper_Scripts.ci import pr2979_watchlists_observation as observer

PRIVATE = "PRIVATE-UAT552-SENTINEL"


def runtime(monkeypatch, output, failure=None):
    response = SimpleNamespace(status_code=200, content=PRIVATE.encode())

    class Router:
        def effective_candidates(self):
            return self._build_effective_context(None)

        def _build_effective_context(self, route):
            return response

    class Database:
        def __init__(self):
            pass

        def ensure_schema_once(self):
            pass

        def list_runs(self):
            return response

        list_runs_for_job = list_runs
        get_run = list_runs

    async def delegated(*args, **kwargs):
        return response

    routing = SimpleNamespace(
        _IncludedRouter=Router,
        serialize_response=delegated,
        run_endpoint_function=delegated,
        solve_dependencies=delegated,
    )

    class App:
        async def __call__(self, scope, receive, send):
            assert Router().effective_candidates() is response
            db = Database()
            db.ensure_schema_once()
            assert db.list_runs() is response
            assert await routing.solve_dependencies() is response
            assert await routing.run_endpoint_function() is response
            assert await routing.serialize_response() is response
            if failure is not None:
                raise failure
            return response

    class Client:
        def get(self, url, *args, **kwargs):
            return asyncio.run(App()({"type": "http", "path": url}, None, None))

    monkeypatch.setitem(sys.modules, "fastapi", SimpleNamespace(FastAPI=App, routing=routing))
    monkeypatch.setitem(sys.modules, "fastapi.routing", routing)
    monkeypatch.setitem(sys.modules, "starlette.testclient", SimpleNamespace(TestClient=Client))
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.DB_Management.Watchlists_DB",
        SimpleNamespace(WatchlistsDatabase=Database),
    )
    monkeypatch.setenv("UAT552_PROBE_OUTPUT", str(output))
    item = SimpleNamespace(name=observer.TARGET)
    originals = (
        Client.get,
        App.__call__,
        Router.effective_candidates,
        Router._build_effective_context,
        Database.__init__,
        Database.list_runs,
        routing.solve_dependencies,
    )
    return item, Client, response, originals, App, Router, Database, routing


def test_content_free_clocks_and_original_return(monkeypatch, tmp_path, capsys):
    output = tmp_path / "observation"
    item, client, response, originals, app, router, db, routing = runtime(monkeypatch, output)
    hook = observer.pytest_runtest_call(item)
    next(hook)
    assert client().get("/" + PRIVATE, secret=PRIVATE) is response
    with pytest.raises(StopIteration):
        next(hook)
    raw = output.with_suffix(".json").read_text()
    assert PRIVATE not in raw + capsys.readouterr().out
    record = json.loads(raw)["records"][0]
    assert record["ordinal"] == 1
    assert record["wall_ns"] >= record["asgi_wall_ns"] >= 0
    assert record["process_cpu_ns"] >= 0
    assert record["sections"]["db.list_runs"]["calls"] == 1
    assert all(section["thread_cpu_ns"] >= 0 for section in record["sections"].values())
    assert originals == (
        client.get,
        app.__call__,
        router.effective_candidates,
        router._build_effective_context,
        db.__init__,
        db.list_runs,
        routing.solve_dependencies,
    )


def test_original_exception_and_privacy(monkeypatch, tmp_path, capsys):
    output = tmp_path / "exception"
    failure = RuntimeError(PRIVATE)
    item, client, _, originals, app, router, db, routing = runtime(monkeypatch, output, failure)
    hook = observer.pytest_runtest_call(item)
    next(hook)
    with pytest.raises(RuntimeError) as caught:
        client().get("/" + PRIVATE)
    assert caught.value is failure
    with pytest.raises(StopIteration):
        next(hook)
    assert PRIVATE not in output.with_suffix(".json").read_text() + capsys.readouterr().out
    assert originals == (
        client.get,
        app.__call__,
        router.effective_candidates,
        router._build_effective_context,
        db.__init__,
        db.list_runs,
        routing.solve_dependencies,
    )


@pytest.mark.parametrize("failure", [None, RuntimeError(PRIVATE), SystemExit(11)])
def test_unavailable_output_preserves_outcome_and_restoration(monkeypatch, tmp_path, capsys, failure):
    output = tmp_path / "absent" / "observation"
    item, client, response, originals, app, router, db, routing = runtime(monkeypatch, output, failure)
    hook = observer.pytest_runtest_call(item)
    next(hook)
    if failure is None:
        assert client().get("/" + PRIVATE) is response
    else:
        with pytest.raises(type(failure)) as caught:
            client().get("/" + PRIVATE)
        assert caught.value is failure
    with pytest.raises(StopIteration):
        next(hook)
    assert capsys.readouterr().out.strip() == "UAT552_DIAGNOSTIC_UNAVAILABLE"
    assert originals == (
        client.get,
        app.__call__,
        router.effective_candidates,
        router._build_effective_context,
        db.__init__,
        db.list_runs,
        routing.solve_dependencies,
    )


def test_other_cases_are_not_instrumented(monkeypatch, tmp_path):
    output = tmp_path / "unused"
    monkeypatch.setenv("UAT552_PROBE_OUTPUT", str(output))
    hook = observer.pytest_runtest_call(SimpleNamespace(name="unrelated_case"))
    next(hook)
    with pytest.raises(StopIteration):
        next(hook)
    assert not output.with_suffix(".json").exists()


def test_missing_framework_hook_leaves_original_case_available(monkeypatch, tmp_path, capsys):
    output = tmp_path / "missing-hook"
    item, client, response, _, _, _, db, _ = runtime(monkeypatch, output)
    monkeypatch.delattr(db, "get_run")
    original_get = client.get
    hook = observer.pytest_runtest_call(item)
    next(hook)
    assert client.get is original_get
    assert client().get("/" + PRIVATE) is response
    with pytest.raises(StopIteration):
        next(hook)
    assert capsys.readouterr().out.strip() == "UAT552_DIAGNOSTIC_UNAVAILABLE"
