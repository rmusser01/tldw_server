"""Release evidence must retain every upload and stay on disposable targets."""

import json
import os
import runpy
import subprocess  # nosec B404
import sys
from pathlib import Path
from secrets import token_urlsafe

import httpx
import pytest

PROBES = Path(__file__).resolve().parents[3] / "Docs/Operations/probes"
FLAGS = {
    "EMAIL_NATIVE_PERSIST_ENABLED": "true",
    "EMAIL_OPERATOR_SEARCH_ENABLED": "true",
    "EMAIL_MEDIA_SEARCH_DELEGATION_MODE": "opt_in",
    "EMAIL_GMAIL_CONNECTOR_ENABLED": "false",
    "CONNECTORS_WORKER_ENABLED": "false",
}
EXPECTED_IDS = {101, 102, 103}


@pytest.fixture
def release_settings(monkeypatch):
    """Use real flag parsing and a local metrics registry with bounded samples."""
    from tldw_Server_API.app.core import config
    from tldw_Server_API.app.core.Metrics import metrics_manager

    for key, value in FLAGS.items():
        monkeypatch.setenv(key, value)
    loaded = config.load_settings()
    for key in FLAGS:
        if key in loaded:
            monkeypatch.setitem(config.settings, key, loaded[key])
    baseline = {key: config.settings.get(key) for key in FLAGS}
    baseline['CONNECTORS_WORKER_ENABLED'] = False
    registry = metrics_manager.MetricsRegistry()
    registry.increment("email_ingestion_parse_total", labels={"format": "mbox", "outcome": "parsed"})
    registry.increment("email_ingestion_dedupe_total", labels={"backend": "sqlite"})
    monkeypatch.setattr(metrics_manager, "get_metrics_registry", lambda: registry)
    yield config.settings, baseline


def release_client(settings, *, baseline_ids=EXPECTED_IDS, rollback_ids=EXPECTED_IDS):
    """Simulate HTTP responses while the helper changes the effective real flags."""
    def respond(request):
        enabled = settings["EMAIL_OPERATOR_SEARCH_ENABLED"]
        if request.url.path == "/api/v1/media/search":
            body = json.loads(request.content)
            if not enabled and body.get("email_query_mode") == "operators":
                return httpx.Response(422, json={"detail": "Operator search is disabled"})
            ids = baseline_ids if enabled else rollback_ids
            if body["query"].startswith('"'):
                ids = {101}
            page = int(request.url.params.get("page", "1"))
            ordered = sorted(ids)
            return httpx.Response(200, json={
                "items": [{"id": ident} for ident in ordered[(page - 1) * 100:page * 100]],
                "pagination": {"total_pages": max(1, (len(ordered) + 99) // 100)},
            })
        if request.url.path == "/api/v1/email/search":
            return httpx.Response(200 if enabled else 404, json={"items": []})
        if request.url.path == "/api/v1/email/messages/101":
            return httpx.Response(200 if enabled else 404, json={"subject": "ArchiveThroughput 0-000"})
        raise AssertionError(f"Unexpected release HTTP request: {request.url.path}")

    return httpx.AsyncClient(base_url="http://127.0.0.1", transport=httpx.MockTransport(respond))


@pytest.mark.asyncio
@pytest.mark.parametrize("rollback_ids", [{101}, {101, 102, 104}])
async def test_release_rejects_partial_or_replaced_legacy_rows_after_rollback(release_settings, rollback_ids):
    """A nonempty result or unchanged count cannot establish ID preservation."""
    settings, baseline = release_settings
    validate = runpy.run_path(str(PROBES / "email_local_release_checks_13376.py"))["validate_local_release"]
    async with release_client(settings, rollback_ids=rollback_ids) as client:
        with pytest.raises(AssertionError):
            await validate(client, read_headers={}, sample_media_id=101, expected_media_ids=EXPECTED_IDS)
    assert {key: settings.get(key) for key in FLAGS if key != 'CONNECTORS_WORKER_ENABLED'} == {
        key: value for key, value in baseline.items() if key != 'CONNECTORS_WORKER_ENABLED'
    }
    assert {key: os.environ[key] for key in FLAGS} == FLAGS


@pytest.mark.asyncio
async def test_release_supports_env_only_worker_flag_and_restores_flags(release_settings):
    """The worker env flag is not a key returned by the application settings loader."""
    settings, baseline = release_settings
    validate = runpy.run_path(str(PROBES / "email_local_release_checks_13376.py"))["validate_local_release"]
    async with release_client(settings) as client:
        evidence = await validate(client, read_headers={}, sample_media_id=101, expected_media_ids=EXPECTED_IDS)
    assert evidence["rollback"]["legacy_search_count"] == 3
    assert evidence["restored_flags"] == baseline
    assert {key: os.environ[key] for key in FLAGS} == FLAGS


@pytest.mark.asyncio
async def test_release_rejects_incomplete_baseline_even_when_native_and_legacy_agree(release_settings):
    """Agreement on a subset of the uploads cannot certify complete parity."""
    settings, _baseline = release_settings
    validate = runpy.run_path(str(PROBES / "email_local_release_checks_13376.py"))["validate_local_release"]
    async with release_client(settings, baseline_ids={101, 102}) as client:
        with pytest.raises(AssertionError):
            await validate(client, read_headers={}, sample_media_id=101, expected_media_ids=EXPECTED_IDS)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled_flag", ["EMAIL_GMAIL_CONNECTOR_ENABLED", "CONNECTORS_WORKER_ENABLED"])
async def test_release_rejects_effective_connector_or_worker_enablement(release_settings, monkeypatch, enabled_flag):
    """Env-only worker enablement must fail before release HTTP checks proceed."""
    from tldw_Server_API.app.core import config

    settings, _baseline = release_settings
    monkeypatch.setenv(enabled_flag, "true")
    if enabled_flag != "CONNECTORS_WORKER_ENABLED":
        monkeypatch.setitem(settings, enabled_flag, config.load_settings()[enabled_flag])
    validate = runpy.run_path(str(PROBES / "email_local_release_checks_13376.py"))["validate_local_release"]
    async with release_client(settings) as client:
        with pytest.raises(AssertionError):
            await validate(client, read_headers={}, sample_media_id=101, expected_media_ids=EXPECTED_IDS)


def guard_preamble(script, tmp_path, *, manifest=None):
    """Execute guard setup in a process that stops before application imports.

    No database, model, network or server module can import. The temporary-root
    factory records attempts instead of creating a retained probe directory.
    """
    code = '''
import json
import os
import runpy
import sys
import tempfile

class ApplicationImportStopped(RuntimeError):
    pass

class StopApplicationImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fastapi":
            raise ApplicationImportStopped()

roots_created = []
def root_factory(*args, **kwargs):
    roots_created.append(True)
    return sys.argv[2]

tempfile.mkdtemp = root_factory
sys.meta_path.insert(0, StopApplicationImports())
sys.path.insert(0, os.path.dirname(sys.argv[1]))
try:
    runpy.run_path(sys.argv[1], run_name="probe_guard_test")
except ApplicationImportStopped:
    status = "configured"
except ValueError:
    status = "rejected"
except Exception as exc:
    status = type(exc).__name__
else:
    status = "unexpected_completion"
print(json.dumps({
    "status": status,
    "roots_created": len(roots_created),
    "content_mode": os.environ.get("CONTENT_DB_MODE"),
    "backend": os.environ.get("TLDW_CONTENT_DB_BACKEND"),
    "has_pg_dsn": bool(os.environ.get("TLDW_CONTENT_PG_DSN")),
}))
'''
    environment = dict(os.environ)
    environment.update({
        "CONTENT_DB_MODE": "postgresql",
        "TLDW_CONTENT_DB_BACKEND": "postgresql",
        "TLDW_CONTENT_PG_DSN": "postgresql://synthetic.invalid/unused",
    })
    if manifest is not None:
        environment["EMAIL_PROBE_PG_MANIFEST"] = str(manifest)
    # Fixed Python code and repository probe paths; no shell or application imports.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", code, str(PROBES / script), str(tmp_path)],
        env=environment, capture_output=True, text=True, timeout=15, check=True,
    )
    return json.loads(result.stdout)


def test_sqlite_guard_overrides_inherited_postgres_target_before_application_imports(tmp_path):
    observed = guard_preamble("email_archive_throughput_sqlite_2026_09_25.py", tmp_path)
    assert observed == {
        "status": "configured", "roots_created": 1,
        "content_mode": "sqlite", "backend": "sqlite", "has_pg_dsn": False,
    }


@pytest.mark.parametrize("unsafe", ["unrelated", "public", "symlink"])
def test_postgres_guard_rejects_unsafe_manifest_before_creating_root(tmp_path, unsafe):
    manifest = tmp_path / "private.json"
    data = {
        "host": "127.0.0.1", "port": 5434, "role": "email_probe_012345abcd",
        "password": token_urlsafe(24), "auth_db": "email_auth_012345abcd",
        "content_db": "email_content_012345abcd",
    }
    if unsafe == "unrelated":
        data["content_db"] = "unrelated_local_database"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    manifest.chmod(0o644 if unsafe == "public" else 0o600)
    if unsafe == "symlink":
        target = manifest.with_name("target.json")
        manifest.rename(target)
        manifest.symlink_to(target)
    observed = guard_preamble("email_archive_throughput_postgres_2026_09_25.py", tmp_path, manifest=manifest)
    assert observed["status"] == "rejected"
    assert observed["roots_created"] == 0


def test_postgres_guard_accepts_private_generated_target_before_application_imports(tmp_path):
    manifest = tmp_path / "private.json"
    manifest.write_text(json.dumps({
        "host": "127.0.0.1", "port": 5434, "role": "email_probe_012345abcd",
        "password": token_urlsafe(24), "auth_db": "email_auth_012345abcd",
        "content_db": "email_content_012345abcd",
    }), encoding="utf-8")
    manifest.chmod(0o600)
    observed = guard_preamble("email_archive_throughput_postgres_2026_09_25.py", tmp_path, manifest=manifest)
    assert observed == {
        "status": "configured", "roots_created": 1,
        "content_mode": "postgresql", "backend": "postgresql", "has_pg_dsn": True,
    }
