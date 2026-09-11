"""Behavioral checks for the release measurement harness (not release capacity)."""

import asyncio
import importlib
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import httpx
import pytest


def profile():
    """Return a short, explicit test-only operating envelope."""
    limits = {"queue_depth": 3, "db_pool_in_use": 2, "storage_bytes": 1000}
    return {
        "name": "in-process-fixture",
        "base_url": "http://fixture.local",
        "artifact_sha256": "a" * 64,
        "source_revision": "b" * 40,
        "telemetry_path": "/observations",
        "timeout_seconds": 0.1,
        "sample_interval_seconds": 0.01,
        "max_sample_age_seconds": 1,
        "max_storage_growth_bytes": 100,
        "max_recovery_seconds": 0.08,
        "phases": [
            {
                "name": name,
                "duration_seconds": 0.08,
                "concurrency": count,
                "pause_seconds": 0.005,
                "min_successes": 1,
                "max_error_ratio": 0,
                "max_rejection_ratio": reject,
                "p95_seconds": 0.1,
                "metric_maxima": dict(limits),
            }
            for name, count, reject in [("steady", 1, 0), ("overload", 2, 1), ("recovery", 1, 0)]
        ],
    }


def dataset():
    """Use distinct auth and workflow requests so starvation is detectable."""
    return [
        {
            "name": "authenticated-read",
            "category": "authentication",
            "method": "GET",
            "path": "/auth",
            "success_statuses": [200],
        },
        {
            "name": "terminal-workflow",
            "category": "workflow",
            "method": "POST",
            "path": "/workflow",
            "json": {"fixture": "synthetic"},
            "success_statuses": [200],
            "response_equals": {"state": "complete"},
        },
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_sha256", "latest"),
        ("source_revision", "dev"),
        ("timeout_seconds", float("nan")),
        ("sample_interval_seconds", 0),
        ("max_storage_growth_bytes", -1),
        ("base_url", "http://user:secret@example.test"),
        ("base_url", "file:///etc/passwd"),
    ],
)
def test_invalid_profile_fails_before_requests(field, value):
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    config = profile()
    config[field] = value
    with pytest.raises(ValueError):
        runner.validate(config, dataset())


@pytest.mark.parametrize("path", ["//evil.test/read", "https://evil.test", "/\\evil", "/ok#secret"])
def test_dataset_cannot_escape_origin(path):
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    rows = dataset()
    rows[0]["path"] = path
    with pytest.raises(ValueError):
        runner.validate(profile(), rows)


def test_all_required_metrics_and_categories_must_be_measured():
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    config = profile()
    del config["phases"][1]["metric_maxima"]["db_pool_in_use"]
    with pytest.raises(ValueError):
        runner.validate(config, dataset())
    with pytest.raises(ValueError):
        runner.validate(profile(), dataset()[:1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault,expected",
    [
        (None, True),
        ("missing-metric", False),
        ("stale", False),
        ("drift", False),
        ("server-error", False),
        ("wrong-terminal-state", False),
        ("unrecovered", False),
        ("storage-growth", False),
        ("redirect", False),
    ],
)
async def test_http_run_measures_workloads_and_rejects_incomplete_evidence(fault, expected):
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    calls = []
    samples = 0
    active = 0
    workload_count = 0

    async def application(scope, receive, send):
        nonlocal samples, active, workload_count
        calls.append(scope["path"])
        code = 200
        if scope["path"] == "/observations":
            samples += 1
            body = {
                "artifact_sha256": "a" * 64,
                "source_revision": "b" * 40,
                "sampled_at": time.time(),
                "queue_depth": 0,
                "db_pool_in_use": 1,
                "storage_bytes": 200,
            }
            if fault == "missing-metric":
                del body["db_pool_in_use"]
            if fault == "stale":
                body["sampled_at"] = 1
            if fault == "drift" and samples > 2:
                body["artifact_sha256"] = "c" * 64
            if fault == "unrecovered":
                body["queue_depth"] = 20
            if fault == "storage-growth":
                body["storage_bytes"] += samples * 30
        else:
            active += 1
            workload_count += 1
            rejected = active > 1 and workload_count % 3 == 0
            await asyncio.sleep(0.002)
            active -= 1
            code = 429 if rejected else 200
            body = {"state": "complete", "unretained_content": "NEVER_RETAIN_RESPONSE"}
            if fault == "server-error":
                code = 500
            if fault == "wrong-terminal-state":
                body["state"] = "queued"
            if fault == "redirect":
                code = 302
        await send({"type": "http.response.start", "status": code, "headers": []})
        await send({"type": "http.response.body", "body": json.dumps(body).encode()})

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://fixture.local"
    ) as client:
        evidence = await runner.run(profile(), dataset(), client)
    assert evidence["passed"] is expected
    assert "NEVER_RETAIN_RESPONSE" not in json.dumps(evidence)
    if expected:
        assert all(phase["workloads"]["terminal-workflow"]["successes"] > 0 for phase in evidence["phases"])
        assert evidence["recovery_seconds"] is not None
        assert evidence["phases"][1]["concurrency"] == 2
    else:
        assert evidence["failures"]
    if fault in ("missing-metric", "stale"):
        assert calls == ["/observations"]
    else:
        assert "/auth" in calls and "/workflow" in calls


@pytest.mark.asyncio
async def test_timeout_is_counted_and_does_not_hang_phase():
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    import asyncio

    async def slow(request):
        if request.url.path == "/observations":
            return httpx.Response(
                200,
                json={
                    "artifact_sha256": "a" * 64,
                    "source_revision": "b" * 40,
                    "sampled_at": time.time(),
                    "queue_depth": 0,
                    "db_pool_in_use": 0,
                    "storage_bytes": 0,
                },
            )
        await asyncio.sleep(10)
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow), base_url="http://fixture.local") as client:
        evidence = await asyncio.wait_for(runner.run(profile(), dataset(), client), 2)
    assert not evidence["passed"]
    assert evidence["phases"][0]["workloads"]["authenticated-read"]["errors"] > 0


def test_unknown_fields_do_not_hide_misspelled_thresholds_or_leak_secrets():
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    config = profile()
    config["phases"][0]["unrecognized_field"] = "DO_NOT_ECHO"
    with pytest.raises(ValueError):
        runner.validate(config, dataset())


@pytest.mark.asyncio
async def test_higher_concurrency_without_observed_overload_cannot_pass():
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")

    async def healthy(request):
        if request.url.path == "/observations":
            return httpx.Response(
                200,
                json={
                    "artifact_sha256": "a" * 64,
                    "source_revision": "b" * 40,
                    "sampled_at": time.time(),
                    "queue_depth": 0,
                    "db_pool_in_use": 0,
                    "storage_bytes": 0,
                },
            )
        return httpx.Response(200, json={"state": "complete"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(healthy), base_url="http://fixture.local") as client:
        evidence = await runner.run(profile(), dataset(), client)
    assert not evidence["passed"]
    assert "overload: no rejection observed" in evidence["failures"]


def test_cli_rejects_existing_output_without_overwriting(tmp_path):
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    inputs = tmp_path / "profile.json"
    rows = tmp_path / "dataset.json"
    output = tmp_path / "evidence.json"
    inputs.write_text(json.dumps(profile()))
    rows.write_text(json.dumps(dataset()))
    output.write_text("earlier evidence")
    assert runner.main(["--profile", str(inputs), "--dataset", str(rows), "--output", str(output)]) == 2
    assert output.read_text() == "earlier evidence"


@pytest.mark.asyncio
async def test_response_limit_and_non_json_overload_handling():
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")

    async def responder(request):
        if request.url.path == "/oversized":
            return httpx.Response(200, content=b"x" * (1024 * 1024 + 1))
        return httpx.Response(429, text="capacity exceeded")

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder), base_url="http://fixture.local") as client:
        with pytest.raises(ValueError):
            await runner.request(client, {"method": "GET", "path": "/oversized"}, 1)
        code, body = await runner.request(client, {"method": "GET", "path": "/reject"}, 1, json_body=True)
    assert (code, body) == (429, None)


def test_extremely_large_numbers_are_invalid_not_unhandled_overflow():
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    config = profile()
    config["timeout_seconds"] = 10**1000
    with pytest.raises(ValueError):
        runner.validate(config, dataset())


def test_cli_writes_failed_evidence_without_credentials(tmp_path, monkeypatch):
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    seen_headers = []

    class Collector(BaseHTTPRequestHandler):
        def do_GET(self):
            seen_headers.append(self.headers.get("X-API-KEY"))
            body = json.dumps({"artifact_sha256": "wrong-target"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Collector)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("SOAK_TEST_TOKEN", "DO_NOT_WRITE_THIS_CREDENTIAL")
        config = profile()
        config["base_url"] = f"http://127.0.0.1:{server.server_port}"
        inputs = tmp_path / "profile.json"
        rows = tmp_path / "dataset.json"
        output = tmp_path / "evidence.json"
        inputs.write_text(json.dumps(config))
        rows.write_text(json.dumps(dataset()))
        result = runner.main(
            [
                "--profile",
                str(inputs),
                "--dataset",
                str(rows),
                "--output",
                str(output),
                "--api-key-env",
                "SOAK_TEST_TOKEN",
            ]
        )
        assert result == 1
        assert seen_headers == ["DO_NOT_WRITE_THIS_CREDENTIAL"]
        assert "DO_NOT_WRITE_THIS_CREDENTIAL" not in output.read_text()
        report = json.loads(output.read_text())
        assert report["phases"] == []
        assert not report["passed"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_pacing_sleep_does_not_extend_phase_duration():
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    config = profile()
    for phase in config["phases"]:
        phase["pause_seconds"] = 1

    async def responder(request):
        if request.url.path == "/observations":
            return httpx.Response(
                200,
                json={
                    "artifact_sha256": "a" * 64,
                    "source_revision": "b" * 40,
                    "sampled_at": time.time(),
                    "queue_depth": 0,
                    "db_pool_in_use": 0,
                    "storage_bytes": 0,
                },
            )
        return httpx.Response(200, json={"state": "complete"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder), base_url="http://fixture.local") as client:
        report = await asyncio.wait_for(runner.run(config, dataset(), client), 0.8)
    assert report["elapsed_seconds"] < 0.8


@pytest.mark.asyncio
async def test_final_observation_cannot_regress_after_sustained_recovery():
    runner = importlib.import_module("Helper_Scripts.load_tests.release_soak")
    config = profile()
    config["sample_interval_seconds"] = 0.04
    samples = 0
    requests = 0

    async def responder(request):
        nonlocal samples, requests
        if request.url.path == "/observations":
            samples += 1
            return httpx.Response(
                200,
                json={
                    "artifact_sha256": "a" * 64,
                    "source_revision": "b" * 40,
                    "sampled_at": time.time(),
                    "queue_depth": 99 if samples == 8 else 0,
                    "db_pool_in_use": 0,
                    "storage_bytes": 0,
                },
            )
        requests += 1
        status = 429 if samples in (4, 5) and requests % 3 == 0 else 200
        return httpx.Response(status, json={"state": "complete"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder), base_url="http://fixture.local") as client:
        report = await runner.run(config, dataset(), client)
    assert samples == 8
    assert all(phase["telemetry"]["maxima"]["queue_depth"] == 0 for phase in report["phases"])
    assert not report["passed"]
    assert "final observation exceeds recovery resource ceilings" in report["failures"]
