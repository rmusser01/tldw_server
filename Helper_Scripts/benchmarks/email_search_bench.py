#!/usr/bin/env python3
"""
email_search_bench.py

Purpose
- Benchmark Stage-1 email operator search performance.
- Optionally build a deterministic synthetic fixture in Media DB v2.
- Produce a JSON report with cold and warm latency summaries (p50/p95).
- Optionally seed query mix from workload traces and capture SQLite query plans.

Examples
  # 1) Build a fixture (20k messages) and benchmark
  python Helper_Scripts/benchmarks/email_search_bench.py \
    --db-path .benchmarks/email_search_bench.sqlite \
    --ensure-fixture \
    --fixture-messages 20000 \
    --runs 30 \
    --warmup-runs 5 \
    --out .benchmarks/email_search_report.json

  # 2) Benchmark an existing tenant only (no writes)
  python Helper_Scripts/benchmarks/email_search_bench.py \
    --db-path /path/to/media.db \
    --tenant-id user:1 \
    --runs 20 \
    --warmup-runs 3

  # 3) Use custom query mix
  python Helper_Scripts/benchmarks/email_search_bench.py \
    --db-path .benchmarks/email_search_bench.sqlite \
    --query-mix-file Helper_Scripts/benchmarks/email_search_query_mix.sample.jsonc

  # 4) Use workload trace file and capture query plans (SQLite)
  python Helper_Scripts/benchmarks/email_search_bench.py \
    --db-path .benchmarks/email_search_bench.sqlite \
    --workload-trace-file /tmp/synthetic_email_workload_trace.json \
    --capture-query-plans

  # 5) PostgreSQL (configure an isolated TLDW_CONTENT_PG_* target first)
  TLDW_CONTENT_DB_BACKEND=postgresql \
    python Helper_Scripts/benchmarks/email_search_bench.py \
    --backend postgresql --scope-user-id 42 \
    --ensure-fixture --fixture-messages 10000 \
    --out /tmp/email_search_postgresql.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shlex
import sqlite3
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

_HELPERS_ROOT = Path(__file__).resolve()
for _parent in [_HELPERS_ROOT, *_HELPERS_ROOT.parents]:
    if _parent.name == "Helper_Scripts":
        _parent_str = str(_parent)
        if _parent_str not in sys.path:
            sys.path.insert(0, _parent_str)
        break

from common.repo_utils import ensure_repo_root

ensure_repo_root()

from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

try:
    from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
except ImportError as exc:
    logger.error(
        "tldw_Server_API import failed (run from repo root or set PYTHONPATH): {}",
        exc,
    )
    raise SystemExit(1) from exc


MediaDbLike = Any


def _open_media_db(*, db_path: Path, client_id: str) -> MediaDbLike:
    """Open the media DB handle used by this benchmark for the given tenant path."""
    return create_media_database(client_id, db_path=db_path)


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((max(0.0, min(100.0, p)) / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    avg_ms = float(sum(values) / len(values))
    return {
        "count": int(len(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "avg_ms": avg_ms,
        "p50_ms": _pct(values, 50.0),
        "p95_ms": _pct(values, 95.0),
    }


def _first_word(text: str | None, fallback: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback
    parts = [p for p in raw.replace("/", " ").replace("-", " ").split() if p]
    return parts[0] if parts else fallback


@dataclass
class QueryCase:
    name: str
    query: str
    notes: str = ""


def _load_query_mix_from_file(path: Path) -> list[QueryCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("query mix file must be a JSON array")  # noqa: TRY003
    cases: list[QueryCase] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"query mix entry {idx} must be an object")  # noqa: TRY003
        name = str(item.get("name") or f"query_{idx}").strip()
        query = str(item.get("query") or "").strip()
        if not query:
            raise ValueError(f"query mix entry {idx} missing query")  # noqa: TRY003
        notes = str(item.get("notes") or "").strip()
        cases.append(QueryCase(name=name, query=query, notes=notes))
    if not cases:
        raise ValueError("query mix file contained zero queries")  # noqa: TRY003
    return cases


def _load_query_mix_from_workload_trace(
    *,
    path: Path,
    top_n: int,
    min_count: int,
) -> list[QueryCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        query_rows = payload.get("queries")
    elif isinstance(payload, list):
        query_rows = payload
    else:
        raise ValueError("workload trace must be a JSON object or array")  # noqa: TRY003

    if not isinstance(query_rows, list):
        raise ValueError("workload trace missing 'queries' array")  # noqa: TRY003

    normalized: dict[str, dict[str, Any]] = {}
    min_count_int = max(1, int(min_count))
    for idx, row in enumerate(query_rows):
        if not isinstance(row, dict):
            continue
        query = str(row.get("query") or "").strip()
        if not query:
            continue
        try:
            count = int(row.get("count") or 1)
        except (TypeError, ValueError):
            count = 1
        if count < min_count_int:
            continue
        name = str(row.get("name") or "").strip() or f"trace_query_{idx + 1}"
        notes = str(row.get("notes") or "").strip()
        existing = normalized.get(query)
        if existing is None or int(existing["count"]) < count:
            normalized[query] = {
                "query": query,
                "count": count,
                "name": name,
                "notes": notes,
            }

    ranked = sorted(
        normalized.values(),
        key=lambda item: (-int(item["count"]), str(item["query"]).lower()),
    )
    selected = ranked[: max(1, int(top_n))]
    if not selected:
        raise ValueError("workload trace produced zero benchmark queries")  # noqa: TRY003

    cases: list[QueryCase] = []
    for idx, row in enumerate(selected):
        notes = str(row.get("notes") or "").strip()
        count_note = f"trace_count={int(row['count'])}"
        notes = f"{notes}; {count_note}".strip("; ").strip()
        cases.append(
            QueryCase(
                name=str(row.get("name") or f"trace_{idx + 1:02d}"),
                query=str(row["query"]),
                notes=notes,
            )
        )
    return cases


def _fetch_fixture_profile(db: MediaDbLike, tenant_id: str) -> dict[str, Any]:
    with db.transaction() as conn:
        total_row = db._fetchone_with_connection(  # noqa: SLF001 - benchmark helper
            conn,
            "SELECT COUNT(*) AS total FROM email_messages WHERE tenant_id = ?",
            (tenant_id,),
        )
        total_messages = int((total_row or {}).get("total", 0) or 0)

        attachment_row = db._fetchone_with_connection(  # noqa: SLF001
            conn,
            (
                "SELECT COUNT(*) AS total "
                "FROM email_attachments ea "
                "JOIN email_messages em ON em.id = ea.email_message_id "
                "WHERE em.tenant_id = ?"
            ),
            (tenant_id,),
        )
        total_attachments = int((attachment_row or {}).get("total", 0) or 0)

        labels_row = db._fetchone_with_connection(  # noqa: SLF001
            conn,
            (
                "SELECT COUNT(DISTINCT eml.label_id) AS total "
                "FROM email_message_labels eml "
                "JOIN email_messages em ON em.id = eml.email_message_id "
                "WHERE em.tenant_id = ?"
            ),
            (tenant_id,),
        )
        distinct_labels = int((labels_row or {}).get("total", 0) or 0)

        span_row = db._fetchone_with_connection(  # noqa: SLF001
            conn,
            "SELECT MIN(internal_date) AS min_date, MAX(internal_date) AS max_date "
            "FROM email_messages WHERE tenant_id = ?",
            (tenant_id,),
        )
        min_date = span_row.get("min_date") if span_row else None
        max_date = span_row.get("max_date") if span_row else None
        if isinstance(min_date, datetime):
            min_date = min_date.isoformat()
        if isinstance(max_date, datetime):
            max_date = max_date.isoformat()

        sender_row = db._fetchone_with_connection(  # noqa: SLF001
            conn,
            (
                "SELECT ep.email_normalized AS email "
                "FROM email_message_participants emp "
                "JOIN email_participants ep ON ep.id = emp.participant_id "
                "JOIN email_messages em ON em.id = emp.email_message_id "
                "WHERE em.tenant_id = ? AND emp.role = 'from' "
                "GROUP BY ep.email_normalized "
                "ORDER BY COUNT(*) DESC, ep.email_normalized ASC "
                "LIMIT 1"
            ),
            (tenant_id,),
        )
        top_sender = str((sender_row or {}).get("email") or "")

        recipient_row = db._fetchone_with_connection(  # noqa: SLF001
            conn,
            (
                "SELECT ep.email_normalized AS email "
                "FROM email_message_participants emp "
                "JOIN email_participants ep ON ep.id = emp.participant_id "
                "JOIN email_messages em ON em.id = emp.email_message_id "
                "WHERE em.tenant_id = ? AND emp.role = 'to' "
                "GROUP BY ep.email_normalized "
                "ORDER BY COUNT(*) DESC, ep.email_normalized ASC "
                "LIMIT 1"
            ),
            (tenant_id,),
        )
        top_recipient = str((recipient_row or {}).get("email") or "")

        label_row = db._fetchone_with_connection(  # noqa: SLF001
            conn,
            (
                "SELECT el.label_name AS label_name "
                "FROM email_message_labels eml "
                "JOIN email_labels el ON el.id = eml.label_id "
                "JOIN email_messages em ON em.id = eml.email_message_id "
                "WHERE em.tenant_id = ? "
                "GROUP BY el.label_name "
                "ORDER BY COUNT(*) DESC, el.label_name ASC "
                "LIMIT 1"
            ),
            (tenant_id,),
        )
        top_label = str((label_row or {}).get("label_name") or "")

        subject_row = db._fetchone_with_connection(  # noqa: SLF001
            conn,
            (
                "SELECT subject, from_text FROM email_messages "
                "WHERE tenant_id = ? AND subject IS NOT NULL AND subject <> '' "
                "ORDER BY id DESC LIMIT 1"
            ),
            (tenant_id,),
        )
        sample_subject = str((subject_row or {}).get("subject") or "")

    return {
        "total_messages": total_messages,
        "total_attachments": total_attachments,
        "distinct_labels": distinct_labels,
        "min_internal_date": min_date,
        "max_internal_date": max_date,
        "top_sender": top_sender,
        "top_recipient": top_recipient,
        "top_label": top_label,
        "sample_subject": sample_subject,
        "sample_sender": str((subject_row or {}).get("from_text") or top_sender),
    }


def _build_default_query_mix(profile: dict[str, Any]) -> list[QueryCase]:
    top_sender = str(profile.get("top_sender") or "sender0@bench.example")
    top_recipient = str(profile.get("top_recipient") or "recipient0@bench.example")
    top_label = str(profile.get("top_label") or "Inbox")
    subject_token = _first_word(str(profile.get("sample_subject") or ""), "benchmark")

    min_dt_raw = str(profile.get("min_internal_date") or "")
    max_dt_raw = str(profile.get("max_internal_date") or "")
    after_date = ""
    before_date = ""
    try:
        min_dt = datetime.fromisoformat(min_dt_raw.replace("Z", "+00:00"))
        max_dt = datetime.fromisoformat(max_dt_raw.replace("Z", "+00:00"))
        midpoint = min_dt + (max_dt - min_dt) / 2
        after_date = midpoint.date().isoformat()
        before_date = midpoint.date().isoformat()
    except (TypeError, ValueError, OverflowError):
        # Fallback to static date windows if profile lacks date span.
        after_date = "2025-01-01"
        before_date = "2030-01-01"

    return [
        QueryCase(name="from_filter", query=f"from:{top_sender}", notes="Participant sender filter"),
        QueryCase(name="to_filter", query=f"to:{top_recipient}", notes="Participant recipient filter"),
        QueryCase(name="subject_filter", query=f"subject:{subject_token}", notes="Subject field filter"),
        QueryCase(name="label_filter", query=f"label:{top_label}", notes="Label filter"),
        QueryCase(name="has_attachment", query="has:attachment", notes="Attachment existence filter"),
        QueryCase(name="after_date", query=f"after:{after_date}", notes="Date lower bound"),
        QueryCase(name="before_date", query=f"before:{before_date}", notes="Date upper bound"),
        QueryCase(
            name="mixed_text_and_negation",
            query=f"{subject_token} -from:{profile.get('sample_sender') or top_sender}",
            notes="Free text + unary negation",
        ),
        QueryCase(
            name="explicit_or",
            query=f"from:{top_sender} OR subject:{subject_token}",
            notes="Explicit OR semantics",
        ),
        QueryCase(
            name="relative_window",
            query="newer_than:90d",
            notes="Relative time operator",
        ),
    ]


def _build_fixture(
    *,
    db: MediaDbLike,
    tenant_id: str,
    source_key: str,
    message_target: int,
    attachment_ratio: float,
    label_cardinality: int,
    sender_pool: int,
    recipient_pool: int,
    seed: int,
) -> dict[str, Any]:
    profile_before = _fetch_fixture_profile(db, tenant_id)
    existing = int(profile_before.get("total_messages") or 0)
    if existing >= message_target:
        logger.info(
            "Fixture already meets target; tenant_id={} existing={} target={}",
            tenant_id,
            existing,
            message_target,
        )
        return profile_before

    rnd = random.Random(seed)  # nosec B311 - deterministic benchmark fixture generator, not cryptographic
    label_names = [f"Label-{idx + 1:02d}" for idx in range(max(1, label_cardinality))]
    subject_topics = [
        "Quarterly Report",
        "Budget Update",
        "Incident Alert",
        "Invoice Notice",
        "Team Sync",
        "Release Planning",
    ]
    base_date = datetime.now(timezone.utc) - timedelta(days=120)

    to_create = message_target - existing
    logger.info(
        "Building synthetic email fixture; tenant_id={} existing={} target={} creating={}",
        tenant_id,
        existing,
        message_target,
        to_create,
    )
    started = time.perf_counter()
    for i in range(existing, message_target):
        sender_idx = i % max(1, sender_pool)
        recipient_idx = (i * 3) % max(1, recipient_pool)
        sender = f"sender{sender_idx}@bench.example"
        recipient = f"recipient{recipient_idx}@bench.example"
        cc_participant = f"cc{(i * 7) % max(1, recipient_pool)}@bench.example"

        topic = subject_topics[i % len(subject_topics)]
        subject = f"{topic} #{i}"
        internal_dt = base_date + timedelta(minutes=i)
        date_header = internal_dt.strftime("%a, %d %b %Y %H:%M:%S +0000")

        labels = ["Inbox", label_names[i % len(label_names)]]
        if i % 5 == 0:
            labels.append("Finance")
        if i % 11 == 0:
            labels.append("Alerts")

        has_attachment = rnd.random() < max(0.0, min(1.0, attachment_ratio))
        attachments: list[dict[str, Any]] = []
        if has_attachment:
            attachments.append(
                {
                    "name": f"file-{i}.pdf",
                    "content_type": "application/pdf",
                    "size": 1024 + (i % 4096),
                    "content_id": f"<cid-{i}>",
                    "disposition": "attachment",
                }
            )

        body = (
            f"Benchmark email body {i}. Topic: {topic}. "
            f"Sender: {sender}. Recipient: {recipient}. "
            "This text exists to drive deterministic email operator benchmarks."
        )

        media_id, _uuid, _msg = db.add_media_with_keywords(
            url=f"email://bench/{tenant_id}/{i}",
            title=subject,
            media_type="email",
            content=body,
            keywords=["benchmark", "email"],
            author=sender,
        )
        if media_id is None:
            continue

        metadata = {
            "title": subject,
            "email": {
                "from": sender,
                "to": recipient,
                "cc": cc_participant if (i % 4 == 0) else "",
                "bcc": "",
                "subject": subject,
                "date": date_header,
                "message_id": f"<bench-{tenant_id}-{i}@bench.example>",
                "attachments": attachments,
                "labels": labels,
            },
        }
        db.upsert_email_message_graph(
            media_id=int(media_id),
            metadata=metadata,
            body_text=body,
            tenant_id=tenant_id,
            provider="upload",
            source_key=source_key,
            source_message_id=f"{tenant_id}-src-{i}",
            labels=labels,
        )

        if (i + 1) % 1000 == 0:
            logger.info(
                "Fixture progress tenant_id={} created={} elapsed_s={:.2f}",
                tenant_id,
                i + 1 - existing,
                time.perf_counter() - started,
            )

    elapsed = time.perf_counter() - started
    profile_after = _fetch_fixture_profile(db, tenant_id)
    logger.info(
        "Fixture build complete tenant_id={} created={} elapsed_s={:.2f} total_messages={}",
        tenant_id,
        to_create,
        elapsed,
        profile_after.get("total_messages"),
    )
    return profile_after


def _validate_negation_workload(
    *,
    db_path: Path,
    client_id: str,
    tenant_id: str,
    queries: list[QueryCase],
) -> dict[str, Any]:
    """Compare the negation case with its positive baseline outside timed passes."""
    cases = [case for case in queries if case.name == "mixed_text_and_negation"]
    validation: dict[str, Any] = {
        "name": "mixed_text_and_negation",
        "query": cases[0].query if len(cases) == 1 else None,
        "positive_query": None,
        "positive_total_matches": None,
        "negated_total_matches": None,
        "meaningful": False,
        "reason": None,
    }
    if len(cases) != 1:
        validation["reason"] = "missing_case" if not cases else "ambiguous_case"
        return validation

    case = cases[0]
    db = _open_media_db(db_path=db_path, client_id=client_id)
    try:
        groups = db._parse_email_operator_query(case.query)  # noqa: SLF001 - use search grammar
        terms = [term for group in groups for term in group]
        if not any(term["negated"] for term in terms):
            validation["reason"] = "missing_negated_term"
            return validation
        if not any(term["kind"] == "text" and not term["negated"] for term in terms):
            validation["reason"] = "missing_positive_free_text"
            return validation

        positive_groups: list[list[str]] = [[]]
        for token in shlex.split(case.query):
            token = token.strip()
            if token.upper() == "OR":
                positive_groups.append([])
            elif token and not token.startswith("-"):
                positive_groups[-1].append(token)
        # A branch consisting only of negated terms becomes true after removal;
        # OR with that branch makes the baseline the unrestricted mailbox.
        positive_query = (
            " OR ".join(shlex.join(group) for group in positive_groups)
            if all(positive_groups) else ""
        )
        validation["positive_query"] = positive_query
        _, positive_total = db.search_email_messages(query=positive_query, tenant_id=tenant_id, limit=1)
        _, negated_total = db.search_email_messages(query=case.query, tenant_id=tenant_id, limit=1)
        validation["positive_total_matches"] = int(positive_total)
        validation["negated_total_matches"] = int(negated_total)
        if negated_total <= 0:
            validation["reason"] = "no_retained_matches"
        elif negated_total >= positive_total:
            validation["reason"] = "no_match_reduction"
        else:
            validation["meaningful"] = True
        return validation
    finally:
        db.close_connection()


def _run_query_once(
    *,
    db_path: Path,
    client_id: str,
    tenant_id: str,
    query: str,
    limit: int,
    offset: int,
) -> tuple[float, int]:
    db = _open_media_db(db_path=db_path, client_id=client_id)
    try:
        t0 = time.perf_counter()
        _rows, total = db.search_email_messages(
            query=query,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms, int(total)
    finally:
        db.close_connection()


def _run_cold_pass(
    *,
    db_path: Path,
    client_id: str,
    tenant_id: str,
    queries: list[QueryCase],
    limit: int,
    offset: int,
) -> dict[str, Any]:
    query_rows: list[dict[str, Any]] = []
    all_latencies: list[float] = []
    for case in queries:
        latency_ms, total = _run_query_once(
            db_path=db_path,
            client_id=client_id,
            tenant_id=tenant_id,
            query=case.query,
            limit=limit,
            offset=offset,
        )
        all_latencies.append(latency_ms)
        query_rows.append(
            {
                "name": case.name,
                "query": case.query,
                "latency_ms": latency_ms,
                "total_matches": total,
            }
        )
    return {"queries": query_rows, "summary": _summary(all_latencies)}


def _is_sqlite_backend(db: MediaDbLike) -> bool:
    backend_type = getattr(db, "backend_type", None)
    backend_name = str(getattr(backend_type, "name", backend_type) or "").strip().lower()
    return "sqlite" in backend_name


def _capture_sqlite_query_plan(
    *,
    db: MediaDbLike,
    tenant_id: str,
    query: str,
    limit: int,
    offset: int,
    max_statements: int,
) -> dict[str, Any] | None:
    if not _is_sqlite_backend(db):
        return None

    captured_sql: list[str] = []
    with db.transaction() as conn:
        if not isinstance(conn, sqlite3.Connection):
            return None

        def _trace_callback(statement: str) -> None:
            text = str(statement or "").strip().rstrip(";")
            if not text:
                return
            upper = text.upper()
            if upper.startswith(("BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT", "RELEASE", "PRAGMA")):
                return
            if not upper.startswith("SELECT"):
                return
            lower = text.lower()
            # Restrict capture to normalized email search statements.
            if "email_" not in lower and " media " not in f" {lower} ":
                return
            captured_sql.append(text)

        conn.set_trace_callback(_trace_callback)
        try:
            db.search_email_messages(
                query=query,
                tenant_id=tenant_id,
                limit=limit,
                offset=offset,
            )
        finally:
            conn.set_trace_callback(None)

        unique_sql: list[str] = []
        seen: set[str] = set()
        for statement in captured_sql:
            if statement in seen:
                continue
            seen.add(statement)
            unique_sql.append(statement)

        plan_statements: list[dict[str, Any]] = []
        index_hits: list[str] = []
        explained_statements = 0
        for statement in unique_sql[: max(1, int(max_statements))]:
            details: list[str] = []
            index_names: list[str] = []
            uses_index = False
            try:
                rows = conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
                explained_statements += 1
                for row in rows:
                    detail = str(row[3]) if len(row) > 3 else str(row)
                    details.append(detail)
                    detail_upper = detail.upper()
                    if (
                        "USING INDEX" in detail_upper
                        or "USING COVERING INDEX" in detail_upper
                        or "USING AUTOMATIC INDEX" in detail_upper
                    ):
                        index_hits.append(detail)
                        uses_index = True
                        detail_parts = detail.replace(",", " ").split()
                        for idx, token in enumerate(detail_parts):
                            if token.upper() == "INDEX" and idx + 1 < len(detail_parts):
                                index_names.append(detail_parts[idx + 1])
            except Exception as exc:  # noqa: BLE001
                details.append(f"EXPLAIN failed: {type(exc).__name__}: {exc}")

            deduped_index_names: list[str] = []
            seen_index_names: set[str] = set()
            for name in index_names:
                if name in seen_index_names:
                    continue
                seen_index_names.add(name)
                deduped_index_names.append(name)
            plan_statements.append(
                {
                    "statement_sql": statement,
                    "plan_rows": details,
                    "uses_index": uses_index,
                    "index_names": deduped_index_names,
                    # Backward-compatible aliases.
                    "sql": statement,
                    "details": details,
                }
            )

        deduped_index_hits: list[str] = []
        seen_hits: set[str] = set()
        for hit in index_hits:
            if hit in seen_hits:
                continue
            seen_hits.add(hit)
            deduped_index_hits.append(hit)

        return {
            "captured_statement_count": int(len(unique_sql)),
            "explained_statement_count": int(explained_statements),
            "index_hits": deduped_index_hits,
            "statements": plan_statements,
        }


def _run_warm_pass(
    *,
    db_path: Path,
    client_id: str,
    tenant_id: str,
    queries: list[QueryCase],
    warmup_runs: int,
    runs: int,
    limit: int,
    offset: int,
    capture_query_plans: bool,
    query_plan_statements_max: int,
) -> dict[str, Any]:
    db = _open_media_db(db_path=db_path, client_id=client_id)
    try:
        query_rows: list[dict[str, Any]] = []
        all_latencies: list[float] = []
        total_plans_with_index_hits = 0
        total_plans_captured = 0
        total_explained_statements = 0
        total_index_hit_rows = 0
        for case in queries:
            plan_payload: dict[str, Any] | None = None
            if capture_query_plans:
                plan_payload = _capture_sqlite_query_plan(
                    db=db,
                    tenant_id=tenant_id,
                    query=case.query,
                    limit=limit,
                    offset=offset,
                    max_statements=query_plan_statements_max,
                )
                if plan_payload is not None:
                    total_plans_captured += 1
                    plan_hits = plan_payload.get("index_hits") or []
                    total_index_hit_rows += int(len(plan_hits))
                    total_explained_statements += int(plan_payload.get("explained_statement_count") or 0)
                    if plan_hits:
                        total_plans_with_index_hits += 1

            for _ in range(max(0, warmup_runs)):
                db.search_email_messages(
                    query=case.query,
                    tenant_id=tenant_id,
                    limit=limit,
                    offset=offset,
                )

            latencies: list[float] = []
            total_matches = 0
            for _ in range(max(1, runs)):
                t0 = time.perf_counter()
                _rows, total = db.search_email_messages(
                    query=case.query,
                    tenant_id=tenant_id,
                    limit=limit,
                    offset=offset,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                all_latencies.append(elapsed_ms)
                total_matches = int(total)

            query_rows.append(
                {
                    "name": case.name,
                    "query": case.query,
                    "notes": case.notes,
                    "total_matches": total_matches,
                    "latency": _summary(latencies),
                    "sqlite_query_plan": plan_payload,
                }
            )

        return {
            "warmup_runs_per_query": int(max(0, warmup_runs)),
            "measured_runs_per_query": int(max(1, runs)),
            "queries": query_rows,
            "summary": _summary(all_latencies),
            "query_plan_summary": {
                "enabled": bool(capture_query_plans),
                "captured_queries": int(total_plans_captured),
                "queries_with_index_hits": int(total_plans_with_index_hits),
                "total_explained_statements": int(total_explained_statements),
                "total_index_hit_rows": int(total_index_hit_rows),
            },
        }
    finally:
        db.close_connection()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark email operator search performance.")
    parser.add_argument(
        "--backend",
        choices=("sqlite", "postgresql"),
        default="sqlite",
        help="Expected content backend. PostgreSQL connection comes from TLDW_CONTENT_PG_* settings.",
    )
    parser.add_argument(
        "--scope-user-id",
        type=int,
        default=None,
        help="Positive user ID required for PostgreSQL RLS scope.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(".benchmarks/email_search_bench.sqlite"),
        help="Path to Media DB (SQLite file); ignored for PostgreSQL.",
    )
    parser.add_argument("--client-id", type=str, default="email-bench", help="Media DB client_id context.")
    parser.add_argument("--tenant-id", type=str, default="bench-tenant", help="Email tenant scope to benchmark.")
    parser.add_argument(
        "--ensure-fixture",
        action="store_true",
        help="Populate deterministic synthetic fixture up to --fixture-messages for --tenant-id.",
    )
    parser.add_argument("--fixture-loader", choices=("ingestion", "bulk"), default="ingestion", help="Bulk requires an empty disposable synthetic target; never measures production ingestion.")
    parser.add_argument("--fixture-messages", type=int, default=10000, help="Target synthetic fixture size.")
    parser.add_argument("--attachment-ratio", type=float, default=0.25, help="Attachment ratio for fixture creation.")
    parser.add_argument("--label-cardinality", type=int, default=20, help="Distinct label count in fixture.")
    parser.add_argument("--sender-pool", type=int, default=200, help="Distinct sender addresses in fixture.")
    parser.add_argument("--recipient-pool", type=int, default=500, help="Distinct recipient addresses in fixture.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for fixture generation.")
    parser.add_argument("--source-key", type=str, default="benchmark-mailbox", help="Source key used for fixture data.")
    parser.add_argument(
        "--query-mix-file",
        type=Path,
        default=None,
        help="Optional JSON file describing query mix array: [{name,query,notes?}, ...].",
    )
    parser.add_argument(
        "--workload-trace-file",
        type=Path,
        default=None,
        help=(
            "Optional workload trace JSON (array or {queries:[...]}) with entries "
            "like {query,count,name?,notes?}. Used when --query-mix-file is not set."
        ),
    )
    parser.add_argument(
        "--workload-top-n",
        type=int,
        default=20,
        help="When using --workload-trace-file, keep top N queries by count.",
    )
    parser.add_argument(
        "--workload-min-count",
        type=int,
        default=1,
        help="Minimum count threshold for workload trace queries.",
    )
    parser.add_argument("--warmup-runs", type=int, default=3, help="Warmup calls per query before measured runs.")
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per query in warm pass.")
    parser.add_argument("--limit", type=int, default=50, help="Search limit parameter.")
    parser.add_argument("--offset", type=int, default=0, help="Search offset parameter.")
    parser.add_argument(
        "--capture-query-plans",
        action="store_true",
        help="Capture SQLite EXPLAIN QUERY PLAN details for each warm-pass query.",
    )
    parser.add_argument(
        "--query-plan-statements-max",
        type=int,
        default=2,
        help="Maximum traced SELECT statements to explain per query when capturing plans.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".benchmarks/email_search_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print full JSON report to stdout in addition to writing --out.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.backend == "postgresql":
        if args.scope_user_id is None or args.scope_user_id <= 0:
            parser.error("--backend postgresql requires a positive --scope-user-id")
        if args.client_id not in {"email-bench", str(args.scope_user_id)}:
            parser.error("PostgreSQL --client-id must match --scope-user-id")
        args.client_id = str(args.scope_user_id)
        if args.tenant_id == "bench-tenant":
            args.tenant_id = f"user:{args.scope_user_id}"
    elif args.scope_user_id is not None:
        parser.error("--scope-user-id is only used with --backend postgresql")

    context = scoped_context(user_id=args.scope_user_id) if args.backend == "postgresql" else nullcontext()
    with context:
        return _run_benchmark(args)


def _run_benchmark(args: argparse.Namespace) -> int:
    """Run the fixture and query passes under the selected backend scope."""
    db_path: Path = args.db_path.expanduser().resolve()
    if args.backend == "sqlite":
        db_path.parent.mkdir(parents=True, exist_ok=True)

    report_started = time.perf_counter()
    fixture_profile: dict[str, Any]

    db = _open_media_db(db_path=db_path, client_id=args.client_id)
    try:
        actual_backend = str(getattr(db.backend_type, "name", "")).lower()
        if actual_backend != args.backend:
            logger.error(
                "Email benchmark requested backend={} but opened backend={}",
                args.backend,
                actual_backend,
            )
            return 2
        if args.ensure_fixture:
            fixture_setup = None
            if args.fixture_loader == "bulk":
                from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_benchmark_fixture import (
                    seed_email_benchmark_fixture,
                )
                fixture_setup = seed_email_benchmark_fixture(
                    db, tenant_id=args.tenant_id, source_key=args.source_key, message_target=args.fixture_messages,
                    attachment_ratio=args.attachment_ratio, label_cardinality=args.label_cardinality,
                    sender_pool=args.sender_pool, recipient_pool=args.recipient_pool, seed=args.seed,
                )
                fixture_profile = _fetch_fixture_profile(db, args.tenant_id)
                fixture_profile["fixture_setup"] = fixture_setup
            else:
                fixture_profile = _build_fixture(
                db=db,
                tenant_id=args.tenant_id,
                source_key=args.source_key,
                message_target=max(1, int(args.fixture_messages)),
                attachment_ratio=float(args.attachment_ratio),
                label_cardinality=max(1, int(args.label_cardinality)),
                sender_pool=max(1, int(args.sender_pool)),
                recipient_pool=max(1, int(args.recipient_pool)),
                seed=int(args.seed),
            )
        else:
            fixture_profile = _fetch_fixture_profile(db, args.tenant_id)
    finally:
        db.close_connection()

    total_messages = int(fixture_profile.get("total_messages") or 0)
    if total_messages <= 0:
        logger.error(
            "No messages available for tenant_id={} at db_path={}. Use --ensure-fixture or point to populated DB.",
            args.tenant_id,
            db_path,
        )
        return 2

    query_mix_source = "default"
    if args.query_mix_file is not None:
        query_mix = _load_query_mix_from_file(args.query_mix_file.expanduser().resolve())
        query_mix_source = "query_mix_file"
    elif args.workload_trace_file is not None:
        query_mix = _load_query_mix_from_workload_trace(
            path=args.workload_trace_file.expanduser().resolve(),
            top_n=max(1, int(args.workload_top_n)),
            min_count=max(1, int(args.workload_min_count)),
        )
        query_mix_source = "workload_trace_file"
    else:
        query_mix = _build_default_query_mix(fixture_profile)

    cold = _run_cold_pass(
        db_path=db_path,
        client_id=args.client_id,
        tenant_id=args.tenant_id,
        queries=query_mix,
        limit=max(1, int(args.limit)),
        offset=max(0, int(args.offset)),
    )
    warm = _run_warm_pass(
        db_path=db_path,
        client_id=args.client_id,
        tenant_id=args.tenant_id,
        queries=query_mix,
        warmup_runs=max(0, int(args.warmup_runs)),
        runs=max(1, int(args.runs)),
        limit=max(1, int(args.limit)),
        offset=max(0, int(args.offset)),
        capture_query_plans=bool(args.capture_query_plans),
        query_plan_statements_max=max(1, int(args.query_plan_statements_max)),
    )

    negation_validation = _validate_negation_workload(
        db_path=db_path,
        client_id=args.client_id,
        tenant_id=args.tenant_id,
        queries=query_mix,
    )

    required_operator_names = (
        "from_filter", "to_filter", "subject_filter", "label_filter", "has_attachment",
        "after_date", "before_date", "relative_window", "mixed_text_and_negation", "explicit_or",
    )
    warm_cases = {row["name"]: row for row in warm["queries"]}
    operator_coverage_met = all(
        name in warm_cases and int(warm_cases[name]["total_matches"]) > 0 for name in required_operator_names
    )
    operator_latency_met = operator_coverage_met and all(
        warm_cases[name]["latency"]["p50_ms"] <= 250.0 and warm_cases[name]["latency"]["p95_ms"] <= 900.0
        for name in required_operator_names
    )
    mailbox_size_met = total_messages >= 1_000_000

    report = {
        "report_version": 1,
        "generated_at": _iso_utc_now(),
        "duration_seconds": round(time.perf_counter() - report_started, 4),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "benchmark": {
            "backend": actual_backend,
            "db_path": str(db_path) if actual_backend == "sqlite" else None,
            "scope_user_id": args.scope_user_id,
            "client_id": str(args.client_id),
            "tenant_id": str(args.tenant_id),
            "limit": int(max(1, args.limit)),
            "offset": int(max(0, args.offset)),
            "query_mix_source": query_mix_source,
            "query_mix_file": str(args.query_mix_file) if args.query_mix_file else None,
            "workload_trace_file": str(args.workload_trace_file) if args.workload_trace_file else None,
            "workload_top_n": int(max(1, int(args.workload_top_n))),
            "workload_min_count": int(max(1, int(args.workload_min_count))),
        },
        "dataset_profile": fixture_profile,
        "query_mix": [asdict(case) for case in query_mix],
        "cold_pass": cold,
        "warm_pass": warm,
        "negation_validation": negation_validation,
        "targets": {
            "nfr_p50_ms": 250.0,
            "nfr_p95_ms": 900.0,
            "warm_pass_met_p50": warm["summary"]["p50_ms"] <= 250.0,
            "warm_pass_met_p95": warm["summary"]["p95_ms"] <= 900.0,
            "required_mailbox_messages": 1_000_000,
            "mailbox_size_met": mailbox_size_met,
            "operator_coverage_met": operator_coverage_met,
            "operator_latency_met": operator_latency_met,
            "meaningful_negation_met": negation_validation["meaningful"],
            "nfr_performance_gate_met": (
                mailbox_size_met and operator_coverage_met and negation_validation["meaningful"]
                and warm["summary"]["p50_ms"] <= 250.0
                and warm["summary"]["p95_ms"] <= 900.0
            ),
            "latency_gate_method": "aggregate_warm_pass_per_documented_protocol",
            "per_operator_latency_is_diagnostic": True,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info(
        "Email benchmark complete. warm_p50_ms={:.2f} warm_p95_ms={:.2f} out={}",
        warm["summary"]["p50_ms"],
        warm["summary"]["p95_ms"],
        args.out,
    )
    if args.print_json:
        print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
