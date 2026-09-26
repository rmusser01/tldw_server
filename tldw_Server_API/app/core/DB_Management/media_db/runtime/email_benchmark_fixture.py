"""Bounded, disposable synthetic email search fixtures, never production ingestion.

Both persisted representations and their canonical full-text indexes are populated. This
setup path requires a synthetic tenant and empty database and refuses retries;
use a new disposable database after an interrupted load.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.runtime.fts_ops import _update_fts_media
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context


def seed_email_benchmark_fixture(
    db: Any,
    *,
    tenant_id: str,
    message_target: int,
    batch_size: int = 2000,
    attachment_ratio: float = 0.2,
    label_cardinality: int = 20,
    sender_pool: int = 200,
    recipient_pool: int = 500,
    seed: int = 42,
    source_key: str = "benchmark-mailbox",
) -> dict[str, Any]:
    """Seed equivalent native/legacy records in bounded atomic batches.

    The target must be disposable and empty. PostgreSQL additionally requires
    the generated probe database naming contract, an unused Media sequence and
    a scoped, non-admin user. All values are parameterized.
    """
    suffix = tenant_id.removeprefix("email-benchmark:")
    if not tenant_id.startswith("email-benchmark:") or not suffix.isdigit() or int(suffix) <= 0:
        raise ValueError("bulk fixture requires a synthetic tenant email-benchmark:<positive user ID>")
    owner = int(suffix)
    if message_target <= 0 or not 1 <= batch_size <= 10000:
        raise ValueError("positive message target and batch size between 1 and 10000 required")
    if sender_pool <= 0 or recipient_pool <= 0 or label_cardinality <= 0 or not 0 <= attachment_ratio <= 1:
        raise ValueError("invalid fixture shape")
    postgres = db.backend_type == BackendType.POSTGRESQL
    with db.transaction() as conn:
        if postgres:
            scope = get_scope()
            if scope is None or scope.user_id != owner or scope.is_admin:
                raise ValueError("bulk PostgreSQL fixture requires matching non-admin user scope")
            name = db._fetchone_with_connection(conn, "SELECT current_database() AS name")["name"]
            if re.fullmatch(r"email_content_[0-9a-f]{10}", name) is None:
                raise ValueError("bulk PostgreSQL fixture requires a generated disposable probe database")
            sequence = db._fetchone_with_connection(conn, "SELECT last_value, is_called FROM media_id_seq")
            if sequence["is_called"] or sequence["last_value"] != 1:
                raise ValueError("bulk fixture requires an empty database with unused Media sequence")
        existing = db._fetchone_with_connection(conn, "SELECT COUNT(*) AS n FROM Media")["n"]
        native = db._fetchone_with_connection(conn, "SELECT COUNT(*) AS n FROM email_messages")["n"]
        if existing or native:
            raise ValueError("bulk fixture requires an empty database")
        db._execute_with_connection(
            conn,
            "INSERT INTO email_sources(id,tenant_id,provider,source_key) VALUES (1,?,?,?)",
            (tenant_id, "upload", source_key),
        )
        senders = [f"sender{i}@bench.example" for i in range(sender_pool)]
        recipients = [f"recipient{i}@bench.example" for i in range(recipient_pool)]
        addresses = senders + recipients
        db._executemany_with_connection(
            conn,
            "INSERT INTO email_participants(id,tenant_id,email_normalized) VALUES (?,?,?)",
            [(i + 1, tenant_id, address) for i, address in enumerate(addresses)],
        )
        labels = ["Inbox"] + [f"Label-{i + 1:02d}" for i in range(label_cardinality)] + ["Finance", "Alerts"]
        db._executemany_with_connection(
            conn,
            "INSERT INTO email_labels(id,tenant_id,label_key,label_name) VALUES (?,?,?,?)",
            [(i + 1, tenant_id, label.lower(), label) for i, label in enumerate(labels)],
        )
    started = time.perf_counter()
    end_date = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=1)
    start_date = end_date - timedelta(days=365)
    topics = ["Quarterly Report", "Budget Update", "Incident Alert", "Invoice Notice", "Team Sync", "Release Planning"]
    attachment_count = 0
    batches = 0
    for start in range(0, message_target, batch_size):
        media, versions, messages, participants, mappings, attachments = [], [], [], [], [], []
        for i in range(start, min(message_target, start + batch_size)):
            ident = i + 1
            sender_idx, recipient_idx = i % sender_pool, (i * 3) % recipient_pool
            sender, recipient = senders[sender_idx], recipients[recipient_idx]
            topic = topics[i % len(topics)]
            subject = f"{topic} #{i}"
            body = f"Benchmark email body {i}. Topic: {topic}. Sender: {sender}. Recipient: {recipient}. Synthetic operator benchmark."
            stamp = (start_date + (end_date - start_date) * i / max(1, message_target - 1)).isoformat()
            message_id = f"<bench-{seed}-{i}@bench.example>"
            # Evenly spread attachments without a random-memory stream or future dates.
            has_attachment = int((i + 1) * attachment_ratio) > int(i * attachment_ratio)
            names = [1, 2 + i % label_cardinality]
            if i % 5 == 0:
                names.append(label_cardinality + 2)
            if i % 11 == 0:
                names.append(label_cardinality + 3)
            message_labels = [labels[n - 1] for n in names]
            descriptors = (
                [{"name": f"file-{i}.pdf", "content_type": "application/pdf", "size": 1024 + i % 4096}]
                if has_attachment
                else []
            )
            metadata = json.dumps(
                {
                    "title": subject,
                    "email": {
                        "subject": subject,
                        "from": sender,
                        "to": recipient,
                        "date": stamp,
                        "message_id": message_id,
                        "labels": message_labels,
                        "attachments": descriptors,
                    },
                }
            )
            content_hash = hashlib.sha256(body.encode()).hexdigest()
            media.append(
                (
                    ident,
                    f"email://bench/{tenant_id}/{i}",
                    subject,
                    "email",
                    body,
                    sender,
                    stamp,
                    content_hash,
                    str(uuid5(NAMESPACE_URL, f"email-fixture-{seed}-{i}")),
                    stamp,
                    owner,
                    str(owner),
                )
            )
            versions.append(
                (
                    ident,
                    ident,
                    1,
                    body,
                    metadata,
                    stamp,
                    str(uuid5(NAMESPACE_URL, f"email-fixture-version-{seed}-{i}")),
                    stamp,
                    str(owner),
                )
            )
            messages.append(
                (
                    ident,
                    tenant_id,
                    ident,
                    1,
                    str(i),
                    message_id,
                    subject,
                    body,
                    stamp,
                    sender,
                    recipient,
                    "",
                    "",
                    " ".join(message_labels),
                    has_attachment,
                    metadata,
                )
            )
            participants.extend([(ident, sender_idx + 1, "from"), (ident, sender_pool + recipient_idx + 1, "to")])
            mappings.extend((ident, n) for n in names)
            if has_attachment:
                attachment_count += 1
                attachments.append((ident, f"file-{i}.pdf", "application/pdf", 1024 + i % 4096, "attachment"))
        with db.transaction() as conn:
            db._executemany_with_connection(
                conn,
                "INSERT INTO Media(id,url,title,type,content,author,ingestion_date,content_hash,uuid,last_modified,owner_user_id,client_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                media,
            )
            db._executemany_with_connection(
                conn,
                "INSERT INTO DocumentVersions(id,media_id,version_number,content,safe_metadata,created_at,uuid,last_modified,client_id) VALUES (?,?,?,?,?,?,?,?,?)",
                versions,
            )
            if not postgres:
                # SQLite Media FTS is maintained explicitly by the normal API,
                # unlike PostgreSQL's Media vector trigger. Preserve that path.
                for row in media:
                    _update_fts_media(db, conn, row[0], row[2], row[4])
            db._executemany_with_connection(
                conn,
                "INSERT INTO email_messages(id,tenant_id,media_id,source_id,source_message_id,message_id,subject,body_text,internal_date,from_text,to_text,cc_text,bcc_text,label_text,has_attachments,raw_metadata_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                messages,
            )
            db._executemany_with_connection(
                conn,
                "INSERT INTO email_message_participants(email_message_id,participant_id,role) VALUES (?,?,?)",
                participants,
            )
            db._executemany_with_connection(
                conn, "INSERT INTO email_message_labels(email_message_id,label_id) VALUES (?,?)", mappings
            )
            if attachments:
                db._executemany_with_connection(
                    conn,
                    "INSERT INTO email_attachments(email_message_id,filename,content_type,size_bytes,disposition) VALUES (?,?,?,?,?)",
                    attachments,
                )
        batches += 1
        if (start + len(media)) % 50000 == 0:
            logger.info(
                "Synthetic search fixture progress messages={} elapsed_s={:.2f}",
                start + len(media),
                time.perf_counter() - started,
            )
    if postgres:
        # Fixed table names only; keep later writes valid despite explicit fixture IDs.
        with db.transaction() as conn:
            for statement in (
                "SELECT setval(pg_get_serial_sequence('media','id'), (SELECT MAX(id) FROM Media), true)",
                "SELECT setval(pg_get_serial_sequence('documentversions','id'), (SELECT MAX(id) FROM DocumentVersions), true)",
                "SELECT setval(pg_get_serial_sequence('email_messages','id'), (SELECT MAX(id) FROM email_messages), true)",
                "SELECT setval(pg_get_serial_sequence('email_sources','id'), (SELECT MAX(id) FROM email_sources), true)",
                "SELECT setval(pg_get_serial_sequence('email_participants','id'), (SELECT MAX(id) FROM email_participants), true)",
                "SELECT setval(pg_get_serial_sequence('email_labels','id'), (SELECT MAX(id) FROM email_labels), true)",
            ):
                db._execute_with_connection(conn, statement)
    statistics_started = time.perf_counter()
    with db.transaction() as conn:
        db._execute_with_connection(conn, "ANALYZE")
    statistics_seconds = time.perf_counter() - statistics_started
    return {
        "planner_statistics": "analyze_after_load",
        "statistics_seconds": statistics_seconds,
        "loader": "bulk_synthetic",
        "messages": message_target,
        "attachments": attachment_count,
        "attachment_ratio": attachment_count / message_target,
        "batches": batches,
        "batch_size": batch_size,
        "sender_pool": sender_pool,
        "recipient_pool": recipient_pool,
        "seed": seed,
        "duration_seconds": time.perf_counter() - started,
    }


def describe_fixture_security(db: Any, *, tenant_id: str = "email-benchmark:42") -> dict[str, Any]:
    """Inspect synthetic persisted parity and the real PostgreSQL authorization boundary.

    Call outside any open transaction, with the matching non-admin PostgreSQL
    scope active. The other-user check uses a separate transaction and restores
    the caller's scope. Returned values contain only counts and booleans, never
    connection strings, credentials, subjects, bodies or resource names.
    """
    suffix = tenant_id.removeprefix("email-benchmark:")
    if not tenant_id.startswith("email-benchmark:") or not suffix.isdigit() or int(suffix) <= 0:
        raise ValueError("fixture inspection requires a synthetic tenant")
    owner = int(suffix)
    postgres = db.backend_type == BackendType.POSTGRESQL
    scope = get_scope()
    if postgres and (scope is None or scope.user_id != owner or scope.is_admin):
        raise ValueError("fixture inspection requires matching non-admin PostgreSQL scope")
    with db.transaction() as conn:
        native = db._fetchone_with_connection(
            conn,
            "SELECT COUNT(*) AS n FROM email_messages WHERE tenant_id=?",
            (tenant_id,),
        )
        media = db._fetchone_with_connection(conn, "SELECT COUNT(*) AS n FROM Media")
        parity = db._fetchone_with_connection(
            conn,
            "SELECT COUNT(*) AS linked, "
            "SUM(CASE WHEN em.body_text=m.content AND em.body_text=dv.content THEN 1 ELSE 0 END) AS matching, "
            "SUM(CASE WHEN em.message_id LIKE ? AND m.url LIKE ? THEN 1 ELSE 0 END) AS synthetic "
            "FROM email_messages em JOIN Media m ON m.id=em.media_id "
            "JOIN DocumentVersions dv ON dv.media_id=m.id AND dv.version_number=1 WHERE em.tenant_id=?",
            ("<bench-42-%@bench.example>", f"email://bench/{tenant_id}/%", tenant_id),
        )
        addresses = db._fetchone_with_connection(
            conn,
            "SELECT COUNT(DISTINCT CASE WHEN emp.role='from' THEN ep.email_normalized END) AS senders, "
            "COUNT(DISTINCT CASE WHEN emp.role='to' THEN ep.email_normalized END) AS recipients "
            "FROM email_message_participants emp JOIN email_participants ep ON ep.id=emp.participant_id "
            "JOIN email_messages em ON em.id=emp.email_message_id WHERE em.tenant_id=? AND ep.tenant_id=?",
            (tenant_id, tenant_id),
        )
        legacy_indexed = db._fetchone_with_connection(
            conn,
            "SELECT COUNT(*) AS n FROM Media WHERE media_fts_tsv @@ to_tsquery('english', ?)"
            if postgres else "SELECT COUNT(*) AS n FROM media_fts WHERE media_fts MATCH ?",
            ("benchmark",),
        )
        report = {
            "legacy_indexed_body_rows": int(legacy_indexed["n"]),
            "backend": "postgresql" if postgres else "sqlite",
            "native_rows": int(native["n"]),
            "owner_media_rows": int(media["n"]),
            "linked_legacy_rows": int(parity["linked"]),
            "matching_body_version_rows": int(parity["matching"] or 0),
            "synthetic_identity_rows": int(parity["synthetic"] or 0),
            "distinct_senders": int(addresses["senders"]),
            "distinct_recipients": int(addresses["recipients"]),
        }
        if postgres:
            role = db._fetchone_with_connection(
                conn,
                "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user",
            )
            rls = db._fetchone_with_connection(
                conn,
                "SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE oid='media'::regclass",
            )
            report.update(
                {
                    "superuser": bool(role["rolsuper"]),
                    "bypass_rls": bool(role["rolbypassrls"]),
                    "rls_enabled": bool(rls["relrowsecurity"]),
                    "rls_forced": bool(rls["relforcerowsecurity"]),
                }
            )
    if postgres:
        with scoped_context(user_id=owner + 1, is_admin=False):
            with db.transaction() as conn:
                other = db._fetchone_with_connection(conn, "SELECT COUNT(*) AS n FROM Media")
                report["other_media_rows"] = int(other["n"])
        with db.transaction() as conn:
            restored = db._fetchone_with_connection(conn, "SELECT COUNT(*) AS n FROM Media")
        report["owner_scope_restored"] = get_scope() == scope and int(restored["n"]) == report["owner_media_rows"]
    return report
