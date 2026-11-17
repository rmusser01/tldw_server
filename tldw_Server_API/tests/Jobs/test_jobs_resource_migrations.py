"""Resource metadata survives upgrades and rejects partial or mutable bindings."""

import base64
import json
import sqlite3

import pytest

from tldw_Server_API.app.core.claims_cluster_rebuild_contract import validate_cluster_rebuild_payload
from tldw_Server_API.app.core.Jobs.migrations import JOBS_SQLITE_DDL, ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg
from tldw_Server_API.app.core.Security.crypto import decrypt_json_blob_with_key, encrypt_json_blob_with_key

AUTHORITY = "58978735-69ad-47d1-a324-04157989d9a8"
JOB_UUID = "a9dfb098-7959-419a-b017-7884f87eb782"
FINGERPRINT = "a" * 64
FIELDS = (
    "resource",
    "resource_authority_uuid",
    "resource_generation",
    "resource_settings_json",
    "resource_settings_fingerprint",
)
BINDING = ("cluster-rebuild", AUTHORITY, 1, '{"version":1}', FINGERPRINT)


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def database(request, tmp_path):
    if request.param == "sqlite":
        path = tmp_path / "jobs.db"
        # Exercise the pre-resource schema as well as fresh/repeated migrations.
        with sqlite3.connect(path) as legacy:
            legacy.executescript(JOBS_SQLITE_DDL)
            legacy.execute(
                "INSERT INTO jobs(domain,queue,job_type,status) VALUES ('other','default','legacy','queued')"
            )
        ensure_jobs_tables(path)
        ensure_jobs_tables(path)
        with sqlite3.connect(path, isolation_level=None) as conn:
            yield conn, "?", sqlite3.IntegrityError
    else:
        import psycopg

        dsn = request.getfixturevalue("jobs_pg_dsn")
        ensure_jobs_tables_pg(dsn)
        with psycopg.connect(dsn, autocommit=True) as conn:
            yield conn, "%s", psycopg.IntegrityError


def _insert_job(conn, marker, table, binding, *, job_uuid=JOB_UUID, payload=None, owner="u1"):
    columns = "domain,owner_user_id,queue,job_type,status,uuid,payload," + ",".join(FIELDS)
    values = ("claims", owner, "default", "claims_rebuild_clusters", "completed", job_uuid, payload, *binding)
    return conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({','.join([marker] * len(values))})", values)


def test_fresh_sqlite_resource_schema(tmp_path):
    path = ensure_jobs_tables(tmp_path / "fresh.db")
    with sqlite3.connect(path) as conn:
        for table in ("jobs", "jobs_archive"):
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            assert set(FIELDS) <= columns
        conn.execute(
            "SELECT domain,owner_user_id,resource,authority_uuid,generation,active_job_uuid FROM job_resource_heads"
        )
        conn.execute(
            "SELECT domain,owner_user_id,resource,request_key_digest,settings_json,settings_fingerprint,job_uuid,expires_at FROM job_resource_request_receipts"
        )


@pytest.mark.parametrize("table", ["jobs", "jobs_archive"])
def test_complete_binding_roundtrips_on_live_and_archive(database, table):
    conn, marker, _ = database
    _insert_job(conn, marker, table, BINDING)
    assert (
        tuple(conn.execute(f"SELECT {','.join(FIELDS)} FROM {table} WHERE uuid={marker}", (JOB_UUID,)).fetchone())
        == BINDING
    )
    conn.execute(f"UPDATE {table} SET queue='another' WHERE uuid={marker}", (JOB_UUID,))
    assert (
        conn.execute(f"SELECT resource FROM {table} WHERE uuid={marker}", (JOB_UUID,)).fetchone()[0]
        == "cluster-rebuild"
    )


@pytest.mark.parametrize("table", ["jobs", "jobs_archive"])
def test_all_null_binding_remains_opted_out(database, table):
    conn, marker, _ = database
    _insert_job(conn, marker, table, (None,) * 5)
    conn.execute(f"UPDATE {table} SET owner_user_id='u2',payload='{{}}' WHERE uuid={marker}", (JOB_UUID,))
    assert conn.execute(f"SELECT resource FROM {table} WHERE uuid={marker}", (JOB_UUID,)).fetchone()[0] is None


@pytest.mark.parametrize("table", ["jobs", "jobs_archive"])
@pytest.mark.parametrize("missing", range(5))
def test_partial_binding_is_rejected(database, table, missing):
    conn, marker, error = database
    binding = list(BINDING)
    binding[missing] = None
    with pytest.raises(error):
        _insert_job(conn, marker, table, binding)


@pytest.mark.parametrize("table", ["jobs", "jobs_archive"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("resource", "other"),
        ("resource_authority_uuid", "42d32b39-dbed-4adc-b278-2f59496e3163"),
        ("resource_generation", 2),
        ("resource_settings_json", '{"version":2}'),
        ("resource_settings_fingerprint", "b" * 64),
        ("owner_user_id", "u2"),
        ("domain", "other"),
        ("uuid", "42d32b39-dbed-4adc-b278-2f59496e3163"),
        ("job_type", "claims_rebuild_clusters.v2"),
    ],
)
def test_admitted_binding_is_immutable(database, table, field, value):
    conn, marker, error = database
    _insert_job(conn, marker, table, BINDING)
    with pytest.raises(error):
        conn.execute(f"UPDATE {table} SET {field}={marker} WHERE uuid={marker}", (value, JOB_UUID))


@pytest.mark.parametrize("table", ["jobs", "jobs_archive"])
@pytest.mark.parametrize("representation", ["serialization", "encrypted_rotation"])
def test_bound_payload_representation_can_be_rewritten(database, table, representation):
    """Schema must not block Task 3's later validated/decrypting replacement boundary."""
    conn, marker, _ = database
    payload = {
        "version": 1,
        "owner_user_id": "1",
        "method": "exact",
        "min_size": 2,
        "similarity_threshold": None,
        "algorithm_version": 1,
    }
    if representation == "serialization":
        original = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        replacement = json.dumps(dict(reversed(list(payload.items()))), indent=2)
        decoded = json.loads(replacement)
    else:
        first_key = base64.b64encode(b"1" * 32).decode("ascii")
        rotated_key = base64.b64encode(b"2" * 32).decode("ascii")
        first_envelope = encrypt_json_blob_with_key(payload, first_key)
        rotated_envelope = encrypt_json_blob_with_key(payload, rotated_key)
        assert first_envelope is not None and rotated_envelope is not None
        assert decrypt_json_blob_with_key(first_envelope, first_key) == payload
        decoded = decrypt_json_blob_with_key(rotated_envelope, rotated_key)
        original = json.dumps({"_encrypted": first_envelope})
        replacement = json.dumps({"_encrypted": rotated_envelope})
    assert original != replacement
    assert validate_cluster_rebuild_payload(decoded) == payload
    _insert_job(conn, marker, table, BINDING, payload=original, owner="1")
    updated = conn.execute(f"UPDATE {table} SET payload={marker} WHERE uuid={marker}", (replacement, JOB_UUID))
    assert updated.rowcount == 1
    row = conn.execute(f"SELECT payload,{','.join(FIELDS)} FROM {table} WHERE uuid={marker}", (JOB_UUID,)).fetchone()
    stored = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    assert stored == json.loads(replacement)
    assert tuple(row[1:]) == BINDING


@pytest.mark.parametrize("table", ["jobs", "jobs_archive"])
def test_sqlite_repeated_migration_refreshes_installed_payload_freezing_trigger(tmp_path, table):
    path = ensure_jobs_tables(tmp_path / "previous-resource-trigger.db")
    with sqlite3.connect(path) as conn:
        _insert_job(conn, "?", table, BINDING, payload='{"version":1}')
        conn.execute(f"DROP TRIGGER trg_{table}_resource_immutable")
        # Model the exact prior guard, including its representation-sensitive payload comparison.
        changed = " OR ".join(f"OLD.{field} IS NOT NEW.{field}" for field in FIELDS)
        identity_changed = " OR ".join(
            f"OLD.{field} IS NOT NEW.{field}" for field in ("domain", "owner_user_id", "uuid", "job_type", "payload")
        )
        conn.execute(
            f"CREATE TRIGGER trg_{table}_resource_immutable BEFORE UPDATE ON {table} "
            f"WHEN ({changed}) OR (OLD.resource IS NOT NULL AND ({identity_changed})) "
            "BEGIN SELECT RAISE(ABORT,'resource binding is immutable'); END"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE {table} SET payload='{{ \"version\": 1 }}' WHERE uuid=?", (JOB_UUID,))
    ensure_jobs_tables(path)
    ensure_jobs_tables(path)
    with sqlite3.connect(path) as conn:
        updated = conn.execute(f"UPDATE {table} SET payload='{{ \"version\": 1 }}' WHERE uuid=?", (JOB_UUID,))
        assert updated.rowcount == 1
        assert (
            conn.execute(f"SELECT payload FROM {table} WHERE uuid=?", (JOB_UUID,)).fetchone()[0] == '{ "version": 1 }'
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE {table} SET resource_generation=2 WHERE uuid=?", (JOB_UUID,))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE {table} SET owner_user_id='u2' WHERE uuid=?", (JOB_UUID,))


def test_resource_head_is_owner_scoped_and_generation_is_nonnegative(database):
    conn, marker, error = database
    query = f"INSERT INTO job_resource_heads(domain,owner_user_id,resource,authority_uuid,generation) VALUES ({','.join([marker] * 5)})"
    conn.execute(query, ("claims", "u1", "cluster-rebuild", AUTHORITY, 0))
    conn.execute(query, ("claims", "u2", "cluster-rebuild", AUTHORITY, 0))
    with pytest.raises(error):
        conn.execute(query, ("claims", "u1", "cluster-rebuild", AUTHORITY, 1))
    with pytest.raises(error):
        conn.execute(query, ("claims", "u3", "cluster-rebuild", AUTHORITY, -1))


def test_request_receipt_key_uses_full_scope_without_lifecycle_copy(database):
    conn, marker, error = database
    columns = "domain,owner_user_id,resource,request_key_digest,settings_json,settings_fingerprint,job_uuid,expires_at"
    query = f"INSERT INTO job_resource_request_receipts({columns}) VALUES ({','.join([marker] * 8)})"
    values = (
        "claims",
        "u1",
        "cluster-rebuild",
        "c" * 64,
        '{"version":1}',
        FINGERPRINT,
        JOB_UUID,
        "2026-11-02 00:00:00+00:00",
    )
    conn.execute(query, values)
    conn.execute(query, ("claims", "u2", *values[2:]))
    with pytest.raises(error):
        conn.execute(query, values)
    cursor = conn.execute("SELECT * FROM job_resource_request_receipts")
    assert {column[0] for column in cursor.description} == set(columns.split(","))


@pytest.mark.pg_jobs
def test_postgres_upgrade_preserves_legacy_live_and_archived_rows(jobs_pg_dsn):
    import psycopg

    with psycopg.connect(jobs_pg_dsn, autocommit=True) as conn:
        for table in ("jobs", "jobs_archive"):
            conn.execute(f"DROP TRIGGER trg_{table}_resource_immutable ON {table}")
            conn.execute(f"ALTER TABLE {table} DROP CONSTRAINT {table}_resource_binding_valid")
            for field in FIELDS:
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {field}")
            conn.execute(
                f"INSERT INTO {table}(domain,queue,job_type,status,uuid) VALUES ('other','default','legacy','completed','legacy')"
            )
    ensure_jobs_tables_pg(jobs_pg_dsn)
    ensure_jobs_tables_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn) as conn:
        for table in ("jobs", "jobs_archive"):
            assert conn.execute(f"SELECT {','.join(FIELDS)} FROM {table} WHERE uuid='legacy'").fetchone() == (None,) * 5


@pytest.mark.pg_jobs
@pytest.mark.parametrize("table", ["job_resource_heads", "job_resource_request_receipts"])
def test_postgres_resource_tables_enforce_owner_and_domain_rls(jobs_pg_dsn, monkeypatch, table):
    import psycopg

    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg
    from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

    _, restricted = _dsn_or_skip(monkeypatch)
    ensure_jobs_rls_policies_pg(jobs_pg_dsn)
    with psycopg.connect(jobs_pg_dsn, autocommit=True) as admin:
        for domain, owner in (("claims", "u1"), ("claims", "u2"), ("other", "u1")):
            admin.execute(
                "INSERT INTO job_resource_heads VALUES (%s,%s,'cluster-rebuild',%s,0,NULL)", (domain, owner, AUTHORITY)
            )
            admin.execute(
                "INSERT INTO job_resource_request_receipts VALUES (%s,%s,'cluster-rebuild',%s,'{}',%s,%s,NOW())",
                (domain, owner, "c" * 64, FINGERPRINT, JOB_UUID),
            )
    with psycopg.connect(restricted, autocommit=True) as conn:
        assert conn.execute(f"SELECT * FROM {table}").fetchall() == []
        if table == "job_resource_heads":
            insert_sql = "INSERT INTO job_resource_heads VALUES (%s,%s,%s,%s,0,NULL)"
            extra_values = (AUTHORITY,)
            update_sql = "UPDATE job_resource_heads SET generation=1"
            changed_field = "generation"
            expected = 1
        else:
            insert_sql = "INSERT INTO job_resource_request_receipts VALUES (%s,%s,%s,%s,'{}',%s,%s,NOW())"
            extra_values = ("c" * 64, FINGERPRINT, JOB_UUID)
            update_sql = "UPDATE job_resource_request_receipts SET settings_json='{\"version\":1}'"
            changed_field = "settings_json"
            expected = '{"version":1}'
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(insert_sql, ("claims", "u1", "missing-context", *extra_values))
        conn.execute(
            "SELECT set_config('app.owner_user_id','u1',false), set_config('app.domain_allowlist','claims',false)"
        )
        assert conn.execute(f"SELECT domain,owner_user_id FROM {table}").fetchall() == [("claims", "u1")]
        assert conn.execute(insert_sql, ("claims", "u1", "same-scope", *extra_values)).rowcount == 1
        for domain, owner in (("claims", "u2"), ("other", "u1")):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(insert_sql, (domain, owner, "foreign-scope", *extra_values))
        assert conn.execute(update_sql + " WHERE resource='same-scope'").rowcount == 1
        assert (
            conn.execute(f"SELECT {changed_field} FROM {table} WHERE resource='same-scope'").fetchone()[0] == expected
        )
        for field, value in (("owner_user_id", "u2"), ("domain", "other")):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(f"UPDATE {table} SET {field}=%s WHERE resource='same-scope'", (value,))
        assert conn.execute(update_sql + " WHERE owner_user_id='u2'").rowcount == 0
        assert conn.execute(update_sql + " WHERE domain='other'").rowcount == 0
        assert conn.execute(f"DELETE FROM {table} WHERE owner_user_id='u2'").rowcount == 0
        assert conn.execute(f"DELETE FROM {table} WHERE domain='other'").rowcount == 0
        assert conn.execute(f"DELETE FROM {table} WHERE resource='same-scope'").rowcount == 1
        assert conn.execute(f"SELECT domain,owner_user_id FROM {table}").fetchall() == [("claims", "u1")]
    with psycopg.connect(jobs_pg_dsn) as admin:
        assert admin.execute(f"SELECT domain,owner_user_id FROM {table} ORDER BY domain,owner_user_id").fetchall() == [
            ("claims", "u1"),
            ("claims", "u2"),
            ("other", "u1"),
        ]
