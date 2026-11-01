"""Additive SQLite schema for Jobs-owned exclusive resource authority."""

import sqlite3

from ...resource_contracts import RESOURCE_METADATA_FIELDS


def _uuid_sql(column: str) -> str:
    """Validate canonical UUID text using SQLite's built-in scalar operations."""
    return (
        f"length({column})=36 AND substr({column},9,1)='-' "
        f"AND substr({column},14,1)='-' AND substr({column},19,1)='-' "
        f"AND substr({column},24,1)='-' AND length(replace({column},'-',''))=32 "
        f"AND replace({column},'-','') NOT GLOB '*[^0-9a-f]*'"
    )


def _digest_sql(column: str) -> str:
    """Validate lowercase SHA-256 encodings without SQLite extensions."""
    return f"length({column})=64 AND {column} NOT GLOB '*[^0-9a-f]*'"


def ensure_resource_schema(conn: sqlite3.Connection) -> None:
    """Install resource tables and immutable nullable bindings in the caller txn."""
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS job_resource_heads (
          domain TEXT NOT NULL CHECK (length(trim(domain))>0),
          owner_user_id TEXT NOT NULL CHECK (length(trim(owner_user_id))>0),
          resource TEXT NOT NULL CHECK (length(trim(resource))>0),
          authority_uuid TEXT NOT NULL CHECK ({_uuid_sql('authority_uuid')}),
          generation INTEGER NOT NULL CHECK (typeof(generation)='integer' AND generation>=0),
          active_job_uuid TEXT CHECK (active_job_uuid IS NULL OR ({_uuid_sql('active_job_uuid')})),
          PRIMARY KEY (domain, owner_user_id, resource)
        )
        """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS job_resource_request_receipts (
          domain TEXT NOT NULL CHECK (length(trim(domain))>0),
          owner_user_id TEXT NOT NULL CHECK (length(trim(owner_user_id))>0),
          resource TEXT NOT NULL CHECK (length(trim(resource))>0),
          request_key_digest TEXT NOT NULL CHECK ({_digest_sql('request_key_digest')}),
          settings_json TEXT NOT NULL CHECK (json_valid(settings_json) AND json_type(settings_json)='object'),
          settings_fingerprint TEXT NOT NULL CHECK ({_digest_sql('settings_fingerprint')}),
          job_uuid TEXT NOT NULL CHECK ({_uuid_sql('job_uuid')}),
          expires_at TEXT NOT NULL,
          PRIMARY KEY (domain, owner_user_id, resource, request_key_digest)
        )
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_resource_request_receipts_uuid ON job_resource_request_receipts(job_uuid)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_resource_request_receipts_expiry ON job_resource_request_receipts(expires_at)"
    )
    for table in ("jobs", "jobs_archive"):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name in RESOURCE_METADATA_FIELDS:
            if name not in columns:
                kind = "INTEGER" if name == "resource_generation" else "TEXT"
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
        all_null = " AND ".join(f"NEW.{name} IS NULL" for name in RESOURCE_METADATA_FIELDS)
        all_present = " AND ".join(f"NEW.{name} IS NOT NULL" for name in RESOURCE_METADATA_FIELDS)
        valid = (
            f"({all_null}) OR ({all_present} "
            "AND length(trim(NEW.resource))>0 AND length(trim(NEW.domain))>0 "
            "AND NEW.owner_user_id IS NOT NULL AND length(trim(NEW.owner_user_id))>0 "
            "AND NEW.uuid IS NOT NULL "
            f"AND ({_uuid_sql('NEW.uuid')}) AND ({_uuid_sql('NEW.resource_authority_uuid')}) "
            "AND typeof(NEW.resource_generation)='integer' AND NEW.resource_generation>0 "
            "AND json_valid(NEW.resource_settings_json) AND json_type(NEW.resource_settings_json)='object' "
            f"AND ({_digest_sql('NEW.resource_settings_fingerprint')}))"
        )
        for event in ("INSERT", "UPDATE"):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS trg_{table}_resource_valid_{event.lower()} "
                f"BEFORE {event} ON {table} WHEN NOT COALESCE(({valid}),0) "
                "BEGIN SELECT RAISE(ABORT,'invalid resource binding'); END"
            )
        changed = " OR ".join(f"OLD.{name} IS NOT NEW.{name}" for name in RESOURCE_METADATA_FIELDS)
        identity_changed = " OR ".join(
            f"OLD.{name} IS NOT NEW.{name}" for name in ("domain", "owner_user_id", "uuid", "job_type", "payload")
        )
        conn.execute(
            f"CREATE TRIGGER IF NOT EXISTS trg_{table}_resource_immutable BEFORE UPDATE ON {table} "
            f"WHEN ({changed}) OR (OLD.resource IS NOT NULL AND ({identity_changed})) "
            "BEGIN SELECT RAISE(ABORT,'resource binding is immutable'); END"
        )
