"""Guarded synthetic million-message search evidence (TASK-13376.4).

Run from the repository root with the project virtual environment active and
PYTHONPATH=$PWD. Fresh SQLite:

  python Docs/Operations/probes/email_million_search_13376.py \
    --backend sqlite --out /tmp/email_million_sqlite.json

PostgreSQL requires EMAIL_PROBE_PG_MANIFEST from the adjacent disposable database
provisioner; only its generated local targets are accepted. SQLite reuse:

  python Docs/Operations/probes/email_million_search_13376.py \
    --sqlite-existing /generated/temp/root/million.sqlite --out /tmp/recheck.json

PostgreSQL reuse requires its original guarded report and the matching private
manifest. Optional maintenance records describe completed tuning, never execute SQL:

  python Docs/Operations/probes/email_million_search_13376.py --backend postgresql \
    --postgres-existing-report /tmp/original_postgres.json \
    --postgres-maintenance-report /tmp/postgres_maintenance.json --out /tmp/recheck_pg.json

SQLite reuse requires the provenance marker written by a successful fresh run. Fresh
fixtures and guard roots are retained for inspection; the report lists exact
cleanup targets. PostgreSQL databases/role are cleaned by the adjacent provisioner.
Bulk fixture speed is setup evidence, never production ingestion throughput.
The cold pass reopens connections; it does not flush the OS filesystem cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import runpy
import shutil
import stat
import subprocess  # nosec B404
import sys
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

PROBES = Path(__file__).resolve().parent
REPOSITORY = PROBES.parents[2]
MARKER = "email_million_fixture_13376.json"
FIXTURE_KIND = "email_bulk_synthetic_13376"
TENANT = "email-benchmark:42"


def _load_sqlite_provenance(path: Path, tenant_id: str, messages: int) -> dict[str, Any]:
    """Accept only marked, regular synthetic files inside direct generated temp roots."""
    if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
        raise ValueError("Existing fixture must be a regular SQLite file")
    resolved = path.resolve()
    root = resolved.parent
    allowed_parent = Path(tempfile.gettempdir()).resolve()
    if root.parent != allowed_parent or re.fullmatch(r"tldw-email-live-sqlite-[a-z0-9_]+", root.name) is None:
        raise ValueError("Existing fixture must be inside a generated temporary SQLite probe root")
    marker_path = root / MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("Existing fixture requires synthetic provenance")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "fixture_kind": FIXTURE_KIND,
        "tenant_id": tenant_id,
        "database_file": resolved.name,
        "messages": messages,
        "seed": 42,
    }
    if not isinstance(marker, dict) or any(marker.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing fixture provenance does not match requested synthetic dataset")
    return marker


def _validate_postgres_manifest(path: Path) -> dict[str, Any]:
    """Validate the private local generated target before the guard probe imports it."""
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("PostgreSQL probe manifest must be a private regular file")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # Match the adjacent provisioner's contract without importing its setup CLI,
    # which requires environment configuration even when only reading a manifest.
    match = re.fullmatch(r"email_probe_([0-9a-f]{10})", str(manifest.get("role", "")))
    suffix = match.group(1) if match else ""
    if (
        match is None
        or manifest.get("host") != "127.0.0.1"
        or manifest.get("port") != 5434
        or manifest.get("auth_db") != f"email_auth_{suffix}"
        or manifest.get("content_db") != f"email_content_{suffix}"
    ):
        raise ValueError("PostgreSQL manifest must target generated local probe resources")
    if not isinstance(manifest.get("password"), str) or not manifest["password"]:
        raise ValueError("PostgreSQL private manifest has no credential")
    return manifest


def _read_json_report(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read one regular report once, rejecting links and non-object JSON."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Provenance requires a regular report file")
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report, dict):
        raise ValueError("Provenance report must be an object")
    return report, raw


def _validate_postgres_maintenance(record: dict[str, Any], resources: dict[str, Any]) -> None:
    """Accept only bounded, credential-free descriptions tied to these resources."""
    keys = {"report_version", "backend", "postgres_resources", "generated_at", "actions", "source_revision"}
    if (
        not isinstance(record, dict) or set(record) != keys
        or record["report_version"] != 1 or record["backend"] != "postgresql"
        or record["postgres_resources"] != resources
        or not isinstance(record["actions"], list) or not record["actions"]
    ):
        raise ValueError("Maintenance provenance does not match generated PostgreSQL resources")
    stamp = datetime.fromisoformat(str(record["generated_at"]))
    if stamp.tzinfo is None or re.fullmatch(r"[0-9a-f]{40}", str(record["source_revision"])) is None:
        raise ValueError("Maintenance provenance requires a dated source revision")
    for action in record["actions"]:
        if (
            not isinstance(action, dict) or set(action) != {"name", "duration_seconds"}
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", str(action["name"])) is None
            or type(action["duration_seconds"]) not in (int, float)
            or not math.isfinite(action["duration_seconds"]) or action["duration_seconds"] < 0
        ):
            raise ValueError("Maintenance actions require bounded names and finite nonnegative durations")


def _load_postgres_maintenance(path: Path, resources: dict[str, Any]) -> dict[str, Any]:
    """Load validated maintenance evidence without reading credentials or running SQL."""
    report, _raw = _read_json_report(path)
    _validate_postgres_maintenance(report, resources)
    return report


def _validate_fixture_security(security: dict[str, Any], messages: int, *, postgres: bool) -> None:
    """Require real persisted body/index parity, pools and scoped authorization."""
    counts = ("native_rows", "owner_media_rows", "linked_legacy_rows", "legacy_indexed_body_rows",
              "matching_body_version_rows", "synthetic_identity_rows")
    if (
        security.get("backend") != ("postgresql" if postgres else "sqlite")
        or any(security.get(key) != messages for key in counts)
        or security.get("distinct_senders") != min(messages, 200)
        or security.get("distinct_recipients") != min(messages, 500)
    ):
        raise ValueError("Synthetic fixture legacy/native/version parity failed")
    if postgres and (
        security.get("superuser") is not False or security.get("bypass_rls") is not False
        or security.get("rls_enabled") is not True or security.get("rls_forced") is not True
        or security.get("other_media_rows") != 0 or security.get("owner_scope_restored") is not True
    ):
        raise ValueError("Synthetic PostgreSQL fixture authorization boundary failed")


def _load_postgres_provenance(path: Path, manifest: dict[str, Any], messages: int) -> dict[str, Any]:
    """Bind reuse to a guarded synthetic report, preserving original load/statistics."""
    report, raw = _read_json_report(path)
    resources = {key: manifest[key] for key in ("role", "auth_db", "content_db")}
    sections = ("benchmark", "dataset_profile", "fixture_validation", "fixture_security", "validation_guards",
                "reproduction", "cleanup", "source_identity", "source_identity_after_probe")
    if report.get("report_version") != 1 or any(not isinstance(report.get(key), dict) for key in sections):
        raise ValueError("PostgreSQL reuse requires a complete guarded source report")
    benchmark, profile, validation = (report[key] for key in sections[:3])
    reproduction = report["reproduction"]
    if (
        report["cleanup"].get("postgres_resources") != resources
        or any(benchmark.get(key) != value for key, value in {
            "backend": "postgresql", "scope_user_id": 42, "client_id": "42", "tenant_id": TENANT,
        }.items())
        or validation.get("validated_scope") != {"user_id": 42, "is_admin": False}
        or validation.get("messages") != messages or validation.get("attachments") != int(messages * 0.2)
        or validation.get("cross_tenant_matches") != 0
        or profile.get("total_messages") != messages or profile.get("total_attachments") != int(messages * 0.2)
        or profile.get("distinct_labels") != 23 or reproduction.get("messages") != messages
        or reproduction.get("runner") != "Docs/Operations/probes/email_million_search_13376.py"
        or reproduction.get("setup_is_ingestion_throughput") is not False
        or reproduction.get("configured_shape") != {
            "attachment_ratio": 0.2, "label_cardinality": 20, "sender_pool": 200, "recipient_pool": 500, "seed": 42,
        }
    ):
        raise ValueError("PostgreSQL report provenance does not match the requested synthetic resources and shape")
    _validate_fixture_security(report["fixture_security"], messages, postgres=True)
    guards = report["validation_guards"]
    if any(guards.get(key) != value for key, value in {
        "synthetic_only": True, "gmail_connector_enabled": False, "outbound_attempts": 0, "model_attempts": 0,
        "background_tasks_blocked": True, "non_loopback_socket_and_dns_blocked": True,
    }.items()):
        raise ValueError("PostgreSQL source report did not pass synthetic guards")
    identity = report["source_identity"]
    if (
        report.get("source_unchanged_during_probe") is not True or identity != report["source_identity_after_probe"]
    ):
        raise ValueError("PostgreSQL source report requires stable, measured source identity")
    previous = report.get("fixture_provenance", {})
    if not isinstance(previous, dict):
        raise ValueError("PostgreSQL original fixture provenance must be an object")
    origin = previous.get("origin_source_identity", identity)
    measured_files = set(_source_identity()["sha256"])
    # Original reports predate these later RLS/native-persistence changes.
    required_files = measured_files - {
        "tldw_Server_API/app/core/DB_Management/media_db/schema/features/postgres_rls.py",
        "tldw_Server_API/app/core/DB_Management/media_db/runtime/email_graph_persistence_ops.py",
    }
    for source in (identity, origin):
        hashes = source.get("sha256") if isinstance(source, dict) else None
        if (
            not isinstance(source, dict) or set(source) != {"revision", "sha256"} or not isinstance(hashes, dict)
            or not required_files.issubset(hashes) or not set(hashes).issubset(measured_files)
            or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in hashes.values())
            or re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision"))) is None
        ):
            raise ValueError("PostgreSQL source report requires complete original measured source identity")
    setup = profile.get("fixture_setup")
    setup_expected = {
        "planner_statistics": "analyze_after_load", "loader": "bulk_synthetic", "messages": messages,
        "attachments": int(messages * 0.2), "attachment_ratio": 0.2, "batch_size": 2000,
        "batches": math.ceil(messages / 2000), "sender_pool": 200, "recipient_pool": 500, "seed": 42,
    }
    if (
        not isinstance(setup, dict) or set(setup) != set(setup_expected) | {"duration_seconds", "statistics_seconds"}
        or any(setup.get(key) != value for key, value in setup_expected.items())
        or any(type(setup[key]) not in (int, float) or not math.isfinite(setup[key]) or setup[key] < 0
               for key in ("duration_seconds", "statistics_seconds"))
    ):
        raise ValueError("PostgreSQL source report requires original bounded fixture load/statistics evidence")
    maintenance = report.get("fixture_maintenance", [])
    if not isinstance(maintenance, list):
        raise ValueError("PostgreSQL prior maintenance must be a list")
    for record in maintenance:
        _validate_postgres_maintenance(record, resources)
    return {
        "fixture_setup": setup, "source_identity": identity, "origin_source_identity": origin, "source_report": str(path.resolve()),
        "source_report_sha256": hashlib.sha256(raw).hexdigest(), "fixture_maintenance": maintenance,
    }


def _validate_fixture(db: Any, profile: dict[str, Any], tenant_id: str, messages: int) -> dict[str, Any]:
    """Verify persisted synthetic identities, shape, and other-tenant invisibility."""
    expected_attachments = int(messages * 0.2)
    if (
        profile.get("total_messages") != messages
        or profile.get("total_attachments") != expected_attachments
        or profile.get("distinct_labels") != 23
    ):
        raise ValueError("Persisted synthetic fixture shape does not match the requested benchmark")
    for ident in (1, messages // 2, messages):
        detail = db.get_email_message_detail(email_message_id=ident, tenant_id=tenant_id)
        index = ident - 1
        if (
            detail is None
            or detail.get("message_id") != f"<bench-42-{index}@bench.example>"
            or not str(detail.get("body_text", "")).startswith(f"Benchmark email body {index}. Topic: ")
            or detail.get("media", {}).get("url") != f"email://bench/{tenant_id}/{index}"
            or detail.get("source", {}).get("source_key") != "benchmark-mailbox"
            or not str(detail.get("search_text", {}).get("from", "")).endswith("@bench.example")
            or not str(detail.get("search_text", {}).get("to", "")).endswith("@bench.example")
        ):
            raise ValueError("Persisted fixture sample is not the expected synthetic email")
    _rows, other_total = db.search_email_messages(query="", tenant_id="email-benchmark:43", limit=1)
    if other_total:
        raise ValueError("Synthetic benchmark cross-tenant search exposed messages")

    from Helper_Scripts.benchmarks.email_search_bench import _build_default_query_mix

    mixed = next(case.query for case in _build_default_query_mix(profile) if case.name == "mixed_text_and_negation")
    positive_query = mixed.split(" -from:", 1)[0]
    _rows, positive_total = db.search_email_messages(query=positive_query, tenant_id=tenant_id, limit=1)
    _rows, negated_total = db.search_email_messages(query=mixed, tenant_id=tenant_id, limit=1)
    if not 0 < negated_total < positive_total:
        raise ValueError("Synthetic benchmark negation must remove matching messages and retain nonempty results")
    return {
        "native_samples": 3,
        "cross_tenant_matches": 0,
        "messages": messages,
        "attachments": expected_attachments,
        "negation": {
            "positive_matches": positive_total,
            "negated_matches": negated_total,
            "removed_matches": positive_total - negated_total,
        },
    }


def _source_identity() -> dict[str, Any]:
    """Hash the measured search code, fixture loader and guards, including local edits."""
    files = (
        "Docs/Operations/probes/email_million_search_13376.py",
        "Docs/Operations/probes/email_archive_throughput_sqlite_2026_09_25.py",
        "Docs/Operations/probes/email_archive_throughput_postgres_2026_09_25.py",
        "Docs/Operations/probes/email_local_release_checks_13376.py",
        "Helper_Scripts/benchmarks/email_search_bench.py",
        "tldw_Server_API/app/core/DB_Management/backends/postgresql_backend.py",
        "tldw_Server_API/app/core/DB_Management/backends/sqlite_backend.py",
        "tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py",
        "tldw_Server_API/app/core/DB_Management/media_db/runtime/execution_ops.py",
        "tldw_Server_API/app/core/DB_Management/media_db/runtime/email_query_ops.py",
        "tldw_Server_API/app/core/DB_Management/media_db/runtime/email_benchmark_fixture.py",
        "tldw_Server_API/app/core/DB_Management/media_db/runtime/email_graph_persistence_ops.py",
        "tldw_Server_API/app/core/DB_Management/media_db/schema/email_schema_structures.py",
        "tldw_Server_API/app/core/DB_Management/media_db/schema/features/postgres_rls.py",
        "tldw_Server_API/app/core/DB_Management/media_db/repositories/media_repository.py",
        "tldw_Server_API/app/core/DB_Management/media_db/repositories/media_search_repository.py",
        "tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py",
        "tldw_Server_API/app/core/Ingestion_Media_Processing/Email/Email_Processing_Lib.py",
        "tldw_Server_API/app/core/Ingestion_Media_Processing/Email/email_ingestion_metrics.py",
        "tldw_Server_API/app/core/Ingestion_Media_Processing/Email/attachment_policy.py",
        "tldw_Server_API/app/core/Metrics/metrics_manager.py",
    )
    hashes = {name: hashlib.sha256((REPOSITORY / name).read_bytes()).hexdigest() for name in files}
    git = shutil.which("git")
    revision = None
    if git:
        # Fixed git introspection, with an absolute executable and no shell.
        result = subprocess.run(  # nosec B603
            [git, "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
    return {"revision": revision, "sha256": hashes}


def _hardware_profile() -> dict[str, Any]:
    """Record portable hardware fields and macOS CPU/RAM without reading environment secrets."""
    profile = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    sysctl = shutil.which("sysctl")
    if sys.platform == "darwin" and sysctl:
        for key, field in (("machdep.cpu.brand_string", "cpu_model"), ("hw.memsize", "memory_bytes")):
            # Fixed hardware keys, with an absolute executable and no shell.
            result = subprocess.run(  # nosec B603
                [sysctl, "-n", key],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                value = result.stdout.strip()
                profile[field] = int(value) if field == "memory_bytes" else value
    return profile


def _parser() -> argparse.ArgumentParser:
    """Expose the fixed reference workload and bounded small-run diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("sqlite", "postgresql"), default="sqlite")
    parser.add_argument("--sqlite-existing", type=Path)
    parser.add_argument("--postgres-existing-report", type=Path)
    parser.add_argument("--postgres-maintenance-report", type=Path)
    parser.add_argument("--messages", type=int, default=1_000_000)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _inspect_fixture(
    db_path: Path, *, postgres: bool, messages: int, profile: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recheck stored data with the owner scope before or after measured queries."""
    from Helper_Scripts.benchmarks import email_search_bench as bench

    from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_benchmark_fixture import (
        describe_fixture_security,
    )
    from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context

    with scoped_context(user_id=42) if postgres else nullcontext():
        scope = get_scope()
        if postgres and (scope is None or scope.user_id != 42 or scope.is_admin):
            raise ValueError("Synthetic PostgreSQL validation requires the matching non-admin user scope")
        db = bench._open_media_db(db_path=db_path, client_id="42")
        try:
            if profile is None:
                profile = bench._fetch_fixture_profile(db, TENANT)
            validation = _validate_fixture(db, profile, TENANT, messages)
            security = describe_fixture_security(db, tenant_id=TENANT)
            _validate_fixture_security(security, messages, postgres=postgres)
            if postgres:
                validation["validated_scope"] = {"user_id": scope.user_id, "is_admin": scope.is_admin}
            return validation, security
        finally:
            db.close_connection()


def _run(args: argparse.Namespace) -> int:
    """Install guards, build or validate synthetic data, and append reproducible evidence."""
    provenance = None
    postgres = None
    maintenance = []
    if args.sqlite_existing is not None:
        provenance = _load_sqlite_provenance(args.sqlite_existing, TENANT, args.messages)
    if args.backend == "postgresql":
        manifest_path = Path(os.environ["EMAIL_PROBE_PG_MANIFEST"])
        postgres = _validate_postgres_manifest(manifest_path)
        if args.postgres_existing_report is not None:
            provenance = _load_postgres_provenance(args.postgres_existing_report, postgres, args.messages)
            maintenance = list(provenance["fixture_maintenance"])
        if args.postgres_maintenance_report is not None:
            resources = {key: postgres[key] for key in ("role", "auth_db", "content_db")}
            record = _load_postgres_maintenance(args.postgres_maintenance_report, resources)
            if record not in maintenance:
                maintenance.append(record)
    probe_name = f"email_archive_throughput_{'postgres' if postgres else 'sqlite'}_2026_09_25.py"
    guard = runpy.run_path(str(PROBES / probe_name), run_name="email_scale_guarded_setup")
    from loguru import logger

    # Keep per-row DEBUG maintenance logs out of the measured bulk setup.
    logger.remove()
    logger.add(sys.stderr, level="INFO", backtrace=False, diagnose=False)
    guard_root = Path(guard["ROOT"])
    db_path = args.sqlite_existing.resolve() if args.sqlite_existing is not None else guard_root / "million.sqlite"
    source_identity = _source_identity()

    from Helper_Scripts.benchmarks import email_search_bench as bench

    pre_validation = pre_security = None
    if provenance:
        pre_validation, pre_security = _inspect_fixture(db_path, postgres=bool(postgres), messages=args.messages)
        if guard["outbound_attempts"] or guard["model_attempts"]:
            raise RuntimeError("Synthetic fixture validation encountered blocked network/model work")
    benchmark_args = [
        "email_search_bench.py",
        "--backend",
        args.backend,
        "--db-path",
        str(db_path),
        "--client-id",
        "42",
        "--tenant-id",
        TENANT,
        "--runs",
        str(args.runs),
        "--warmup-runs",
        str(args.warmup_runs),
        "--limit",
        "50",
        "--out",
        str(args.out),
    ]
    if postgres:
        benchmark_args.extend(["--scope-user-id", "42"])
    if not provenance:
        benchmark_args.extend(
            [
                "--ensure-fixture",
                "--fixture-loader",
                "bulk",
                "--fixture-messages",
                str(args.messages),
                "--attachment-ratio",
                "0.2",
                "--label-cardinality",
                "20",
                "--sender-pool",
                "200",
                "--recipient-pool",
                "500",
                "--seed",
                "42",
            ]
        )
    original_argv = sys.argv
    try:
        sys.argv = benchmark_args
        result = bench.main()
    finally:
        sys.argv = original_argv
    if result:
        return result
    report = json.loads(args.out.read_text(encoding="utf-8"))
    validation, security = _inspect_fixture(
        db_path, postgres=bool(postgres), messages=args.messages, profile=report["dataset_profile"],
    )
    report["validation_guards"] = {
        "synthetic_only": True,
        "gmail_connector_enabled": False,
        "outbound_attempts": len(guard["outbound_attempts"]),
        "model_attempts": len(guard["model_attempts"]),
        "background_tasks_blocked": True,
        "non_loopback_socket_and_dns_blocked": True,
    }
    report["fixture_validation"] = validation
    report["fixture_security"] = security
    if provenance:
        report["dataset_profile"]["fixture_setup"] = provenance.get("fixture_setup")
    if postgres and provenance:
        report["fixture_provenance"] = {
            key: provenance[key] for key in ("source_report", "source_report_sha256", "source_identity", "origin_source_identity")
        }
        report["fixture_provenance"]["pre_measurement_fixture_validation"] = pre_validation
        report["fixture_provenance"]["pre_measurement_fixture_security"] = pre_security
        report["fixture_maintenance"] = maintenance
    report["environment"].update(_hardware_profile())
    report["source_identity"] = source_identity
    report["source_identity_after_probe"] = _source_identity()
    report["source_unchanged_during_probe"] = source_identity == report["source_identity_after_probe"]
    report["cold_connection_cache_scope"] = "Fresh MediaDatabase handle per query; backend pools may reuse physical connections. OS filesystem and database server caches are unchanged."
    report["reproduction"] = {
        "runner": str(Path(__file__).resolve().relative_to(REPOSITORY)),
        "messages": args.messages,
        "runs": args.runs,
        "warmup_runs": args.warmup_runs,
        "fixture_reused": bool(provenance),
        "configured_shape": {
            "attachment_ratio": 0.2,
            "label_cardinality": 20,
            "sender_pool": 200,
            "recipient_pool": 500,
            "seed": 42,
        },
        "setup_is_ingestion_throughput": False,
    }
    report["cleanup"] = {
        "retained_temporary_roots": sorted({str(guard_root), str(db_path.parent)}),
        "instructions": "After handles/processes close, remove only these generated temporary roots.",
    }
    if postgres:
        report["cleanup"]["postgres_resources"] = {key: postgres[key] for key in ("role", "auth_db", "content_db")}
        report["cleanup"]["postgres_instructions"] = (
            "Keep the private EMAIL_PROBE_PG_MANIFEST and run email_archive_probe_databases_2026_09_25.py cleanup."
        )
    elif not provenance:
        marker = {
            "fixture_kind": FIXTURE_KIND,
            "tenant_id": TENANT,
            "database_file": db_path.name,
            "messages": args.messages,
            "seed": 42,
            "fixture_setup": report["dataset_profile"]["fixture_setup"],
        }
        (db_path.parent / MARKER).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if guard["outbound_attempts"] or guard["model_attempts"]:
        raise RuntimeError("Synthetic probe encountered blocked network/model work")
    if not report["source_unchanged_during_probe"]:
        raise RuntimeError("Synthetic probe source files changed during the benchmark; rerun with stable source")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "backend": args.backend,
                "guards": report["validation_guards"],
                "warm_summary": report["warm_pass"]["summary"],
                "nfr_performance_gate_met": report["targets"]["nfr_performance_gate_met"],
            }
        )
    )
    return 0


def main() -> int:
    """Run in a dedicated process; guard installation intentionally lasts until exit."""
    parser = _parser()
    args = parser.parse_args()
    if (
        args.messages < 120
        or args.messages > 10_000_000
        or not 1 <= args.runs <= 1000
        or not 0 <= args.warmup_runs <= 100
    ):
        parser.error("Require 120..10000000 messages, 1..1000 runs, and 0..100 warmups")
    if args.backend == "postgresql" and args.sqlite_existing is not None:
        parser.error("--sqlite-existing is supported only with --backend sqlite")
    if args.postgres_existing_report is not None and args.backend != "postgresql":
        parser.error("--postgres-existing-report requires --backend postgresql")
    if args.postgres_maintenance_report is not None and args.postgres_existing_report is None:
        parser.error("--postgres-maintenance-report requires --postgres-existing-report")
    protected = (args.postgres_existing_report, args.postgres_maintenance_report)
    if args.backend == "postgresql":
        manifest = os.environ.get("EMAIL_PROBE_PG_MANIFEST")
        protected += (Path(manifest) if manifest else None,)
    if any(path is not None and path.resolve() == args.out.resolve() for path in protected):
        parser.error("--out must not overwrite source provenance or the private manifest")
    try:
        return _run(args)
    except Exception as exc:  # noqa: BLE001 - top-level privacy boundary; do not render credentials or traceback locals
        print(f"Synthetic search probe aborted (error_type={type(exc).__name__}).", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
