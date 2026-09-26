"""PostgreSQL timing reuse must retain provenance and recheck persisted security."""

import copy
import hashlib
import json
import runpy
from pathlib import Path
from secrets import token_urlsafe

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "Docs/Operations/probes/email_million_search_13376.py"
RESOURCES = {
    "role": "email_probe_012345abcd", "auth_db": "email_auth_012345abcd", "content_db": "email_content_012345abcd",
}
SETUP = {
    "planner_statistics": "analyze_after_load", "statistics_seconds": 0.5,
    "loader": "bulk_synthetic", "messages": 120, "attachments": 24, "attachment_ratio": 0.2,
    "batches": 1, "batch_size": 2000, "sender_pool": 200, "recipient_pool": 500,
    "seed": 42, "duration_seconds": 1.25,
}
SECURITY = {
    "backend": "postgresql", "native_rows": 120, "owner_media_rows": 120, "linked_legacy_rows": 120,
    "legacy_indexed_body_rows": 120, "matching_body_version_rows": 120, "synthetic_identity_rows": 120,
    "distinct_senders": 120, "distinct_recipients": 120, "superuser": False, "bypass_rls": False,
    "rls_enabled": True, "rls_forced": True, "other_media_rows": 0, "owner_scope_restored": True,
}


@pytest.fixture
def runner():
    return runpy.run_path(str(SCRIPT), run_name="email_postgres_reuse_test")


@pytest.fixture
def provenance(runner, tmp_path):
    """A valid timing report may miss latency targets but must pass fixture checks."""
    manifest = dict(RESOURCES, host="127.0.0.1", port=5434, password=token_urlsafe(24))
    identity = runner["_source_identity"]()
    report = {
        "report_version": 1,
        "benchmark": {"backend": "postgresql", "scope_user_id": 42, "client_id": "42", "tenant_id": "email-benchmark:42"},
        "dataset_profile": {"total_messages": 120, "total_attachments": 24, "distinct_labels": 23, "fixture_setup": SETUP},
        "fixture_validation": {"messages": 120, "attachments": 24, "cross_tenant_matches": 0,
                               "validated_scope": {"user_id": 42, "is_admin": False}},
        "fixture_security": SECURITY,
        "validation_guards": {"synthetic_only": True, "gmail_connector_enabled": False, "outbound_attempts": 0,
                              "model_attempts": 0, "background_tasks_blocked": True, "non_loopback_socket_and_dns_blocked": True},
        "reproduction": {"runner": "Docs/Operations/probes/email_million_search_13376.py", "messages": 120,
                         "configured_shape": {"attachment_ratio": 0.2, "label_cardinality": 20, "sender_pool": 200,
                                              "recipient_pool": 500, "seed": 42}, "setup_is_ingestion_throughput": False},
        "cleanup": {"postgres_resources": RESOURCES},
        "source_identity": identity, "source_identity_after_probe": identity, "source_unchanged_during_probe": True,
        "targets": {"nfr_performance_gate_met": False},
    }
    path = tmp_path / "original.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, manifest, copy.deepcopy(report)


def test_postgres_reuse_accepts_matching_failed_latency_report_and_preserves_setup(runner, provenance):
    path, manifest, report = provenance
    loaded = runner["_load_postgres_provenance"](path, manifest, 120)
    assert loaded["fixture_setup"] == SETUP
    assert loaded["source_identity"] == report["source_identity"]
    assert loaded["source_report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_measured_source_identity_detects_sql_rewrite_changes(runner, tmp_path, monkeypatch):
    identity = runner['_source_identity']
    original_root = runner['REPOSITORY']
    utility = 'tldw_Server_API/app/core/DB_Management/backends/query_utils.py'
    for name in set(identity()['sha256']) | {utility}:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((original_root / name).read_bytes())
    monkeypatch.setitem(identity.__globals__, 'REPOSITORY', tmp_path)
    monkeypatch.setattr(runner['shutil'], 'which', lambda _name: None)
    before = identity()['sha256']
    with (tmp_path / utility).open('a') as output:
        output.write('\n# synthetic rewrite change\n')
    after = identity()['sha256']
    assert before != after


@pytest.mark.parametrize("invalid", ["resource", "scope", "shape", "guard", "privileged", "index", "setup", "source"])
def test_postgres_reuse_rejects_inconsistent_provenance(runner, provenance, invalid):
    path, manifest, report = provenance
    if invalid == "resource":
        report["cleanup"]["postgres_resources"]["content_db"] = "email_content_deadbeef01"
    elif invalid == "scope":
        report["fixture_validation"]["validated_scope"]["is_admin"] = True
    elif invalid == "shape":
        report["dataset_profile"]["total_messages"] = 119
    elif invalid == "guard":
        report["validation_guards"]["outbound_attempts"] = 1
    elif invalid == "privileged":
        report["fixture_security"]["superuser"] = True
    elif invalid == "index":
        report["fixture_security"]["legacy_indexed_body_rows"] = 119
    elif invalid == "setup":
        report["dataset_profile"]["fixture_setup"]["statistics_seconds"] = -1
    else:
        report["source_unchanged_during_probe"] = False
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        runner["_load_postgres_provenance"](path, manifest, 120)


def test_postgres_reuse_rejects_symlink_report(runner, provenance):
    path, manifest, _report = provenance
    link = path.with_name("link.json")
    link.symlink_to(path)
    with pytest.raises(ValueError):
        runner["_load_postgres_provenance"](link, manifest, 120)


def test_postgres_reuse_accepts_regular_report_through_parent_alias(runner, provenance):
    """macOS /tmp is an alias; the report itself must remain a regular file."""
    path, manifest, _report = provenance
    alias = path.parent / "alias"
    alias.symlink_to(path.parent, target_is_directory=True)
    assert runner["_load_postgres_provenance"](alias / path.name, manifest, 120)["fixture_setup"] == SETUP


def test_postgres_reuse_retains_original_load_source_and_previous_maintenance(runner, provenance, maintenance):
    path, manifest, report = provenance
    _maintenance_path, record = maintenance
    origin = copy.deepcopy(report["source_identity"])
    origin["sha256"]["Docs/Operations/probes/email_million_search_13376.py"] = "0" * 64
    report["fixture_provenance"] = {"origin_source_identity": origin}
    report["fixture_maintenance"] = [record]
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = runner["_load_postgres_provenance"](path, manifest, 120)
    assert loaded["origin_source_identity"] == origin
    assert loaded["fixture_maintenance"] == [record]


@pytest.fixture
def maintenance(tmp_path):
    record = {
        "report_version": 1, "backend": "postgresql", "postgres_resources": RESOURCES,
        "generated_at": "2026-09-26T18:30:00+00:00", "source_revision": "a" * 40,
        "actions": [{"name": "vacuum_analyze", "duration_seconds": 0.75}],
    }
    path = tmp_path / "maintenance.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, copy.deepcopy(record)


def test_maintenance_accepts_bound_generated_resource_record(runner, maintenance):
    path, record = maintenance
    assert runner["_load_postgres_maintenance"](path, RESOURCES) == record


@pytest.mark.parametrize("invalid", ["resource", "duration", "nonfinite", "name", "sql", "credential"])
def test_maintenance_rejects_unbound_or_unsafe_record(runner, maintenance, invalid):
    path, record = maintenance
    if invalid == "resource":
        record["postgres_resources"]["role"] = "email_probe_deadbeef01"
    elif invalid == "duration":
        record["actions"][0]["duration_seconds"] = -1
    elif invalid == "nonfinite":
        record["actions"][0]["duration_seconds"] = float("inf")
    elif invalid == "name":
        record["actions"][0]["name"] = "VACUUM; SELECT secret"
    else:
        record["sql" if invalid == "sql" else "password"] = "blocked"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        runner["_load_postgres_maintenance"](path, RESOURCES)


@pytest.mark.parametrize("live_failure", [None, "index", "privileged"])
def test_postgres_reuse_validates_before_timing_skips_seed_and_preserves_maintenance(runner, provenance, maintenance, monkeypatch, tmp_path, live_failure):
    """The runner must not call the seed loader or time an unchecked existing DB."""
    from Helper_Scripts.benchmarks import email_search_bench as bench
    from loguru import logger

    path, manifest, original = provenance
    maintenance_path, maintenance_record = maintenance
    private = tmp_path / "private.json"
    private.write_text(json.dumps(manifest), encoding="utf-8")
    private.chmod(0o600)
    monkeypatch.setenv("EMAIL_PROBE_PG_MANIFEST", str(private))
    args = runner["_parser"]().parse_args([
        "--backend", "postgresql", "--messages", "120", "--postgres-existing-report", str(path),
        "--postgres-maintenance-report", str(maintenance_path), "--out", str(tmp_path / "rerun.json"),
    ])
    namespace = runner["_run"].__globals__
    events = []
    identity = runner["_source_identity"]()
    monkeypatch.setitem(namespace, "_source_identity", lambda: identity)
    monkeypatch.setitem(namespace, "_hardware_profile", lambda: {})
    monkeypatch.setattr(logger, "remove", lambda: None)
    monkeypatch.setattr(logger, "add", lambda *args, **kwargs: None)
    monkeypatch.setattr("runpy.run_path", lambda *args, **kwargs: {"ROOT": tmp_path, "outbound_attempts": [], "model_attempts": []})

    def inspect(*args, **kwargs):
        events.append("inspect")
        security = copy.deepcopy(original["fixture_security"])
        if live_failure == "index":
            security["legacy_indexed_body_rows"] = 119
        elif live_failure == "privileged":
            security["superuser"] = True
        runner["_validate_fixture_security"](security, 120, postgres=True)
        return original["fixture_validation"], original["fixture_security"]

    def time_existing():
        import sys

        events.append("timing")
        assert "--ensure-fixture" not in sys.argv
        timed = {"dataset_profile": {"total_messages": 120}, "environment": {},
                 "warm_pass": {"summary": {}}, "targets": {"nfr_performance_gate_met": False}}
        args.out.write_text(json.dumps(timed), encoding="utf-8")
        return 0

    monkeypatch.setitem(namespace, "_inspect_fixture", inspect)
    monkeypatch.setattr(bench, "main", time_existing)
    if live_failure:
        with pytest.raises(ValueError):
            runner["_run"](args)
        assert events == ["inspect"]
        return
    assert runner["_run"](args) == 0
    report = json.loads(args.out.read_text(encoding="utf-8"))
    assert events == ["inspect", "timing", "inspect"]
    assert report["dataset_profile"]["fixture_setup"] == SETUP
    assert report["fixture_maintenance"] == [maintenance_record]
    assert report["fixture_provenance"]["source_identity"] == original["source_identity"]


@pytest.mark.parametrize("corrupt", [False, True])
def test_fixture_inspection_rechecks_real_stored_bodies_and_index_counts(runner, tmp_path, monkeypatch, corrupt):
    """Prior JSON cannot conceal a body change made after its measurement."""
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_benchmark_fixture import (
        seed_email_benchmark_fixture,
    )

    monkeypatch.setenv("CONTENT_DB_MODE", "sqlite")
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "sqlite")
    path = tmp_path / "fixture.sqlite"
    database = MediaDatabase(db_path=str(path), client_id="42")
    try:
        seed_email_benchmark_fixture(database, tenant_id="email-benchmark:42", message_target=120)
        if corrupt:
            with database.transaction() as connection:
                database._execute_with_connection(connection, "UPDATE email_messages SET body_text=? WHERE id=?", ("changed", 1))
    finally:
        database.close_connection()
    if corrupt:
        with pytest.raises(ValueError, match="expected synthetic email"):
            runner["_inspect_fixture"](path, postgres=False, messages=120)
    else:
        validation, security = runner["_inspect_fixture"](path, postgres=False, messages=120)
        assert validation["messages"] == 120
        assert security["legacy_indexed_body_rows"] == 120


@pytest.mark.parametrize("invalid", ["sqlite_report", "unbound_maintenance"])
def test_reuse_cli_rejects_mixed_backend_or_unbound_maintenance_before_guards(runner, monkeypatch, tmp_path, invalid):
    if invalid == "sqlite_report":
        options = ["--postgres-existing-report", str(tmp_path / "original.json")]
    else:
        options = ["--backend", "postgresql", "--postgres-maintenance-report", str(tmp_path / "maintenance.json")]
    monkeypatch.setattr("sys.argv", ["probe", *options, "--out", str(tmp_path / "output.json")])
    with pytest.raises(SystemExit) as error:
        runner["main"]()
    assert error.value.code == 2


def test_reuse_cli_rejects_overwriting_original_report_before_guards(runner, provenance, monkeypatch):
    path, _manifest, _report = provenance
    before = path.read_bytes()
    monkeypatch.setattr("sys.argv", [
        "probe", "--backend", "postgresql", "--postgres-existing-report", str(path), "--out", str(path),
    ])
    with pytest.raises(SystemExit) as error:
        runner["main"]()
    assert error.value.code == 2
    assert path.read_bytes() == before
