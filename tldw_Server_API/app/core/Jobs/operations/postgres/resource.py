"""Additive PostgreSQL schema for Jobs-owned exclusive resource authority."""

from typing import Any
from uuid import uuid4

from ...resource_contracts import RESOURCE_METADATA_FIELDS, ResourceScope
from ..resource import ResourceAdmissionStore

_UUID_PATTERN = "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class PostgresResourceAdmissionStore(ResourceAdmissionStore):
    """Create an absent head safely, then lock it before exact Job rows."""

    def __init__(self, executor: Any) -> None:
        super().__init__(executor, postgres=True)

    def lock_head(self, scope: ResourceScope) -> dict[str, Any]:
        """ON CONFLICT waits for the creator; the next READ COMMITTED read locks it."""
        self.execute(
            "INSERT INTO job_resource_heads (domain,owner_user_id,resource,authority_uuid,generation) "
            "VALUES (%s,%s,%s,%s,0) ON CONFLICT(domain,owner_user_id,resource) DO NOTHING",
            (*self.scope_values(scope), str(uuid4())),
        )
        row = self.execute(
            "SELECT * FROM job_resource_heads WHERE domain=%s AND owner_user_id=%s AND resource=%s FOR UPDATE",
            self.scope_values(scope),
        ).fetchone()
        if row is None:
            raise RuntimeError("resource head unavailable")
        return dict(row)


def ensure_resource_schema(cur: Any) -> None:
    """Install required resource constraints in the caller's migration transaction."""
    # Legacy archives gained this scope field in an optional migration phase.
    cur.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS owner_user_id TEXT")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS job_resource_heads (
          domain TEXT NOT NULL CHECK (length(btrim(domain))>0),
          owner_user_id TEXT NOT NULL CHECK (length(btrim(owner_user_id))>0),
          resource TEXT NOT NULL CHECK (length(btrim(resource))>0),
          authority_uuid TEXT NOT NULL CHECK (authority_uuid ~ '{_UUID_PATTERN}'),
          generation BIGINT NOT NULL CHECK (generation>=0),
          active_job_uuid TEXT CHECK (active_job_uuid IS NULL OR active_job_uuid ~ '{_UUID_PATTERN}'),
          PRIMARY KEY (domain, owner_user_id, resource)
        )
        """)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS job_resource_request_receipts (
          domain TEXT NOT NULL CHECK (length(btrim(domain))>0),
          owner_user_id TEXT NOT NULL CHECK (length(btrim(owner_user_id))>0),
          resource TEXT NOT NULL CHECK (length(btrim(resource))>0),
          request_key_digest TEXT NOT NULL CHECK (request_key_digest ~ '^[0-9a-f]{{64}}$'),
          settings_json TEXT NOT NULL CHECK (jsonb_typeof(settings_json::jsonb)='object'),
          settings_fingerprint TEXT NOT NULL CHECK (settings_fingerprint ~ '^[0-9a-f]{{64}}$'),
          job_uuid TEXT NOT NULL CHECK (job_uuid ~ '{_UUID_PATTERN}'),
          expires_at TIMESTAMPTZ NOT NULL,
          PRIMARY KEY (domain, owner_user_id, resource, request_key_digest)
        )
        """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_resource_request_receipts_uuid ON job_resource_request_receipts(job_uuid)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_resource_request_receipts_expiry ON job_resource_request_receipts(expires_at)"
    )
    # Payload semantic equality belongs to the validated/decrypting replacement boundary.
    cur.execute("""
        CREATE OR REPLACE FUNCTION jobs_resource_binding_immutable() RETURNS trigger AS $$
        BEGIN
          IF ROW(OLD.resource, OLD.resource_authority_uuid, OLD.resource_generation,
                 OLD.resource_settings_json, OLD.resource_settings_fingerprint)
             IS DISTINCT FROM
             ROW(NEW.resource, NEW.resource_authority_uuid, NEW.resource_generation,
                 NEW.resource_settings_json, NEW.resource_settings_fingerprint)
             OR (OLD.resource IS NOT NULL AND
                 ROW(OLD.domain, OLD.owner_user_id, OLD.uuid, OLD.job_type)
                 IS DISTINCT FROM
                 ROW(NEW.domain, NEW.owner_user_id, NEW.uuid, NEW.job_type)) THEN
            RAISE EXCEPTION 'resource binding is immutable' USING ERRCODE='23514';
          END IF;
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """)
    for table in ("jobs", "jobs_archive"):
        for name in RESOURCE_METADATA_FIELDS:
            kind = "BIGINT" if name == "resource_generation" else "TEXT"
            cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {kind}")
        all_null = " AND ".join(f"{name} IS NULL" for name in RESOURCE_METADATA_FIELDS)
        all_present = " AND ".join(f"{name} IS NOT NULL" for name in RESOURCE_METADATA_FIELDS)
        valid = (
            f"({all_null}) OR ({all_present} "
            "AND length(btrim(resource))>0 AND length(btrim(domain))>0 "
            "AND owner_user_id IS NOT NULL AND length(btrim(owner_user_id))>0 "
            f"AND uuid IS NOT NULL AND uuid ~ '{_UUID_PATTERN}' "
            f"AND resource_authority_uuid ~ '{_UUID_PATTERN}' AND resource_generation>0 "
            "AND jsonb_typeof(resource_settings_json::jsonb)='object' "
            "AND resource_settings_fingerprint ~ '^[0-9a-f]{64}$')"
        )
        # Table/metadata names and predicates come only from closed module constants.
        cur.execute(f"""
            DO $$ BEGIN
              IF NOT EXISTS (SELECT 1 FROM pg_constraint
                WHERE conrelid='{table}'::regclass AND conname='{table}_resource_binding_valid') THEN
                ALTER TABLE {table} ADD CONSTRAINT {table}_resource_binding_valid CHECK ({valid}) NOT VALID;
              END IF;
            END $$
            """)  # nosec B608
        cur.execute(f"ALTER TABLE {table} VALIDATE CONSTRAINT {table}_resource_binding_valid")
        cur.execute(f"DROP TRIGGER IF EXISTS trg_{table}_resource_immutable ON {table}")
        cur.execute(
            f"CREATE TRIGGER trg_{table}_resource_immutable BEFORE UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION jobs_resource_binding_immutable()"
        )


def ensure_resource_rls(cur: Any, *, admin_expr: str, domain_filter: str, owner_filter: str) -> None:
    """Apply existing Jobs owner/domain policies with explicit write checks."""
    predicate = f"({admin_expr}) OR (({domain_filter}) AND ({owner_filter}))"
    for table in ("job_resource_heads", "job_resource_request_receipts"):
        cur.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        cur.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        cur.execute(f"DROP POLICY IF EXISTS {table}_owner_domain ON {table}")
        cur.execute(
            f"CREATE POLICY {table}_owner_domain ON {table} FOR ALL " f"USING ({predicate}) WITH CHECK ({predicate})"
        )
