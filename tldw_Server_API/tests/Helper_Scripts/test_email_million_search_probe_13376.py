"""A scale probe must reject unrelated files and leave imports inert."""

import json
import os
import runpy
import socket
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "Docs/Operations/probes/email_million_search_13376.py"


def test_runner_import_does_not_install_guards_or_change_environment():
    """Only an explicit main invocation may change process-wide networking."""
    before = dict(os.environ)
    before_connect = socket.socket.connect
    before_dns = socket.getaddrinfo
    assert SCRIPT.is_file(), "The reproducible scale runner is missing"
    runpy.run_path(str(SCRIPT), run_name="email_scale_import_test")
    assert dict(os.environ) == before
    assert socket.socket.connect is before_connect
    assert socket.getaddrinfo is before_dns


@pytest.fixture
def runner():
    return runpy.run_path(str(SCRIPT), run_name="email_scale_test")


@pytest.fixture
def reusable_fixture():
    with tempfile.TemporaryDirectory(prefix="tldw-email-live-sqlite-") as temporary:
        root = Path(temporary).resolve()
        database = root / "million.sqlite"
        database.touch()
        marker = {
            "fixture_kind": "email_bulk_synthetic_13376",
            "tenant_id": "email-benchmark:42",
            "database_file": "million.sqlite",
            "messages": 120,
            "seed": 42,
        }
        (root / "email_million_fixture_13376.json").write_text(json.dumps(marker), encoding="utf-8")
        yield database, marker


def test_reuse_accepts_generated_temporary_fixture_with_matching_provenance(runner, reusable_fixture):
    database, marker = reusable_fixture
    assert runner["_load_sqlite_provenance"](database, "email-benchmark:42", 120) == marker


@pytest.mark.parametrize("change", ["tenant", "size", "kind", "file", "missing"])
def test_reuse_rejects_mismatched_or_missing_provenance(runner, reusable_fixture, change):
    database, marker = reusable_fixture
    marker_path = database.parent / "email_million_fixture_13376.json"
    if change == "missing":
        marker_path.unlink()
    else:
        field, value = {
            "tenant": ("tenant_id", "user:42"),
            "size": ("messages", 121),
            "kind": ("fixture_kind", "personal"),
            "file": ("database_file", "personal.sqlite"),
        }[change]
        marker[field] = value
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        runner["_load_sqlite_provenance"](database, "email-benchmark:42", 120)


def test_reuse_rejects_database_outside_generated_temporary_root(runner, tmp_path):
    database = tmp_path / "million.sqlite"
    database.touch()
    with pytest.raises(ValueError, match="generated temporary"):
        runner["_load_sqlite_provenance"](database, "email-benchmark:42", 120)


def test_reuse_rejects_symlink_database(runner, reusable_fixture):
    database, _marker = reusable_fixture
    target = database.with_name("real.sqlite")
    database.rename(target)
    database.symlink_to(target)
    with pytest.raises(ValueError, match="regular"):
        runner["_load_sqlite_provenance"](database, "email-benchmark:42", 120)


def test_postgres_manifest_validation_rejects_unrelated_and_public_credentials(runner, tmp_path):
    manifest = tmp_path / "private.json"
    data = {
        "host": "127.0.0.1",
        "port": 5434,
        "role": "email_probe_012345abcd",
        "password": "private",
        "auth_db": "email_auth_012345abcd",
        "content_db": "email_content_012345abcd",
    }
    manifest.write_text(json.dumps(data), encoding="utf-8")
    manifest.chmod(0o600)
    assert runner["_validate_postgres_manifest"](manifest)["content_db"] == data["content_db"]
    data["host"] = "192.0.2.1"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="generated local"):
        runner["_validate_postgres_manifest"](manifest)
    data["host"] = "127.0.0.1"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    manifest.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        runner["_validate_postgres_manifest"](manifest)


def test_persisted_fixture_validation_checks_real_native_identity_and_isolation(runner, tmp_path):
    from Helper_Scripts.benchmarks.email_search_bench import _fetch_fixture_profile

    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_benchmark_fixture import (
        seed_email_benchmark_fixture,
    )

    database = MediaDatabase(db_path=str(tmp_path / "fixture.sqlite"), client_id="42")
    try:
        seed_email_benchmark_fixture(database, tenant_id="email-benchmark:42", message_target=120)
        profile = _fetch_fixture_profile(database, "email-benchmark:42")
        evidence = runner["_validate_fixture"](database, profile, "email-benchmark:42", 120)
        assert evidence == {
            "native_samples": 3,
            "cross_tenant_matches": 0,
            "messages": 120,
            "attachments": 24,
            "negation": {"positive_matches": 20, "negated_matches": 19, "removed_matches": 1},
        }
        with pytest.raises(ValueError, match="shape"):
            runner["_validate_fixture"](database, profile, "email-benchmark:42", 121)
    finally:
        database.close_connection()


def test_source_hashes_cover_runner_and_search_implementation(runner):
    identity = runner["_source_identity"]()
    assert identity["sha256"]["Docs/Operations/probes/email_million_search_13376.py"]
    assert identity["sha256"]["tldw_Server_API/app/core/DB_Management/media_db/runtime/email_query_ops.py"]


def test_main_error_output_omits_exception_payload(runner, monkeypatch, capsys, tmp_path):
    namespace = runner["main"].__globals__

    def fail(_args):
        raise ValueError("synthetic-private-credential-sentinel")

    monkeypatch.setitem(namespace, "_run", fail)
    monkeypatch.setattr("sys.argv", ["probe", "--out", str(tmp_path / "report.json")])
    assert runner["main"]() == 2
    assert capsys.readouterr().err == "Synthetic search probe aborted (error_type=ValueError).\n"


def test_database_fixture_inspection_reports_actual_links_versions_and_address_pools(tmp_path):
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_benchmark_fixture import (
        describe_fixture_security,
        seed_email_benchmark_fixture,
    )

    database = MediaDatabase(db_path=str(tmp_path / "inspection.sqlite"), client_id="42")
    try:
        seed_email_benchmark_fixture(database, tenant_id="email-benchmark:42", message_target=120)
        evidence = describe_fixture_security(database)
        assert evidence == {
            "backend": "sqlite",
            "legacy_indexed_body_rows": 120,
            "native_rows": 120,
            "owner_media_rows": 120,
            "linked_legacy_rows": 120,
            "matching_body_version_rows": 120,
            "synthetic_identity_rows": 120,
            "distinct_senders": 120,
            "distinct_recipients": 120,
        }
        with database.transaction() as connection:
            database._execute_with_connection(
                connection, "UPDATE email_messages SET body_text=? WHERE id=?", ("altered synthetic native body", 1)
            )
        assert describe_fixture_security(database)["matching_body_version_rows"] == 119
    finally:
        database.close_connection()
