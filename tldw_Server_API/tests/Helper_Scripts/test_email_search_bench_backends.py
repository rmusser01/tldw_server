"""Backend selection and RLS scope contracts for the email search benchmark."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from Helper_Scripts.benchmarks import email_search_bench as bench

from tldw_Server_API.app.core.DB_Management.scope_context import get_scope


def test_postgres_fixture_profile_serializes_date_bounds() -> None:
    """PostgreSQL datetime results must fit the benchmark JSON report."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 2, 1, tzinfo=timezone.utc)
    rows = iter(
        [
            {"total": 1},
            {"total": 0},
            {"total": 0},
            {"min_date": start, "max_date": end},
            {"email": "sender@example.test"},
            {"email": "to@example.test"},
            {"label_name": "Inbox"},
            {"subject": "Synthetic"},
        ]
    )
    db = SimpleNamespace(
        transaction=nullcontext,
        _fetchone_with_connection=lambda *_args: next(rows),
    )

    profile = bench._fetch_fixture_profile(db, "user:42")

    assert profile["min_internal_date"] == start.isoformat()
    assert profile["max_internal_date"] == end.isoformat()
    json.dumps(profile)


def test_postgres_benchmark_requires_numeric_user_scope(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A PostgreSQL run cannot silently measure an unscoped RLS session."""
    monkeypatch.setattr(sys, "argv", ["email_search_bench.py", "--backend", "postgresql"])
    monkeypatch.setattr(bench, "_open_media_db", lambda **_kwargs: pytest.fail("database opened"))

    with pytest.raises(SystemExit, match="2"):
        bench.main()
    assert "requires a positive --scope-user-id" in capsys.readouterr().err


def test_postgres_benchmark_rejects_sqlite_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit backend selection must match the database actually opened."""
    fake_db = SimpleNamespace(
        backend_type=SimpleNamespace(name="SQLITE"),
        close_connection=lambda: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "email_search_bench.py",
            "--backend",
            "postgresql",
            "--scope-user-id",
            "42",
            "--db-path",
            str(tmp_path / "unused.sqlite"),
        ],
    )
    monkeypatch.setattr(bench, "_open_media_db", lambda **_kwargs: fake_db)

    assert bench.main() == 2


def test_postgres_benchmark_scopes_fixture_and_search_and_reports_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fixture and query passes retain one user scope and emit no DSN."""
    seen: list[str] = []

    def require_scope(stage: str) -> None:
        scope = get_scope()
        assert scope is not None and scope.user_id == 42
        seen.append(stage)

    fake_db = SimpleNamespace(
        backend_type=SimpleNamespace(name="POSTGRESQL"),
        initialize_db=lambda: pytest.fail("Media factory already initializes the handle"),
        close_connection=lambda: None,
    )

    def open_db(**kwargs: Any) -> SimpleNamespace:
        require_scope("open")
        assert kwargs["client_id"] == "42"
        return fake_db

    def fixture(**kwargs: Any) -> dict[str, Any]:
        require_scope("fixture")
        assert kwargs["tenant_id"] == "user:42"
        return {"total_messages": 1}

    def cold(**_kwargs: Any) -> dict[str, Any]:
        require_scope("cold")
        return {"queries": [], "summary": {"p50_ms": 1.0, "p95_ms": 1.0}}

    def warm(**_kwargs: Any) -> dict[str, Any]:
        require_scope("warm")
        return {
            "queries": [
                {"name": name, "total_matches": 1, "latency": {"p50_ms": 1.0, "p95_ms": 1.0}}
                for name in (
                    "from_filter",
                    "to_filter",
                    "relative_window",
                    "mixed_text_and_negation",
                    "explicit_or",
                    "subject_filter",
                    "label_filter",
                    "has_attachment",
                    "after_date",
                    "before_date",
                )
            ],
            "summary": {"p50_ms": 1.0, "p95_ms": 1.0},
        }

    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "email_search_bench.py",
            "--backend",
            "postgresql",
            "--scope-user-id",
            "42",
            "--ensure-fixture",
            "--db-path",
            str(tmp_path / "unused.sqlite"),
            "--out",
            str(report_path),
        ],
    )
    monkeypatch.setattr(bench, "_open_media_db", open_db)
    monkeypatch.setattr(bench, "_build_fixture", fixture)
    monkeypatch.setattr(bench, "_build_default_query_mix", lambda _profile: [])
    monkeypatch.setattr(bench, "_run_cold_pass", cold)
    monkeypatch.setattr(bench, "_run_warm_pass", warm)

    assert bench.main() == 0
    report = json.loads(report_path.read_text())
    assert report["benchmark"]["backend"] == "postgresql"
    assert report["benchmark"]["scope_user_id"] == 42
    assert report["benchmark"]["db_path"] is None
    assert report["targets"]["mailbox_size_met"] is False
    assert report["targets"]["operator_latency_met"] is True
    assert report["targets"]["nfr_performance_gate_met"] is False
    assert seen == ["open", "fixture", "cold", "warm"]


@pytest.fixture
def synthetic_benchmark_db(tmp_path: Path) -> Iterator[Any]:
    """Use a real persisted mailbox while keeping release-gate tests small."""
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_benchmark_fixture import (
        seed_email_benchmark_fixture,
    )

    db = MediaDatabase(db_path=str(tmp_path / "gate.sqlite"), client_id="42")
    seed_email_benchmark_fixture(
        db, tenant_id="email-benchmark:42", message_target=120, sender_pool=10, recipient_pool=20,
    )
    try:
        yield db
    finally:
        db.close_connection()


@pytest.mark.parametrize("query,meaningful", [
    ('"Budget Update" -from:no-such-sender', False),
    ('"Budget Update" -from:sender1@bench.example', True),
])
def test_custom_negation_gate_requires_real_match_reduction(
    synthetic_benchmark_db: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, query: str, meaningful: bool,
) -> None:
    """Populated custom cases cannot certify a no-op negation predicate."""
    db = synthetic_benchmark_db
    profile = bench._fetch_fixture_profile(db, "email-benchmark:42")
    cases = bench._build_default_query_mix(profile)
    next(case for case in cases if case.name == "mixed_text_and_negation").query = query
    mix_path = tmp_path / "custom.json"
    mix_path.write_text(json.dumps([vars(case) for case in cases]))
    # Control only the scale/latency inputs; validation reads the actual small mailbox.
    profile["total_messages"] = 1_000_000
    warm_rows = [
        {"name": case.name, "query": case.query,
         "total_matches": db.search_email_messages(query=case.query, tenant_id="email-benchmark:42")[1],
         "latency": {"p50_ms": 1.0, "p95_ms": 1.0}}
        for case in cases
    ]
    stages: list[str] = []

    def timed_pass(stage: str, **_kwargs: Any) -> dict[str, Any]:
        stages.append(stage)
        return {"queries": warm_rows, "summary": {"p50_ms": 1.0, "p95_ms": 1.0}}

    monkeypatch.setattr(bench, "_fetch_fixture_profile", lambda *_args: profile)
    monkeypatch.setattr(bench, "_run_cold_pass", lambda **kwargs: timed_pass("cold", **kwargs))
    monkeypatch.setattr(bench, "_run_warm_pass", lambda **kwargs: timed_pass("warm", **kwargs))
    report_path = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", [
        "email_search_bench.py", "--db-path", db.db_path_str, "--client-id", "42",
        "--tenant-id", "email-benchmark:42", "--query-mix-file", str(mix_path), "--out", str(report_path),
    ])

    assert bench.main() == 0
    report = json.loads(report_path.read_text())
    assert report["targets"]["operator_coverage_met"] is True
    assert report["targets"]["nfr_performance_gate_met"] is meaningful
    assert report["targets"]["meaningful_negation_met"] is meaningful
    validation = report["negation_validation"]
    assert validation["query"] == query
    assert validation["positive_total_matches"] == 20
    assert validation["negated_total_matches"] == (16 if meaningful else 20)
    assert validation["reason"] == (None if meaningful else "no_match_reduction")
    assert next(case for case in report["query_mix"] if case["name"] == "mixed_text_and_negation")["query"] == query
    assert stages == ["cold", "warm"]


@pytest.mark.parametrize("query,positive_count,negated_count", [
    ('"Budget Update" -from:sender1@bench.example OR "Team Sync" -from:sender0@bench.example', 40, 32),
    ('"Budget Update" OR -from:sender1@bench.example', 120, 112),
    ('"Budget Update" -"sender1@bench.example"', 20, 16),
])
def test_negation_validation_preserves_quoted_text_and_or_branches(
    synthetic_benchmark_db: Any, query: str, positive_count: int, negated_count: int,
) -> None:
    """Removing negation must preserve each OR branch and quoted text semantics."""
    validation = bench._validate_negation_workload(
        db_path=Path(synthetic_benchmark_db.db_path_str), client_id="42", tenant_id="email-benchmark:42",
        queries=[bench.QueryCase(name="mixed_text_and_negation", query=query)],
    )
    assert validation["meaningful"] is True
    assert validation["positive_total_matches"] == positive_count
    assert validation["negated_total_matches"] == negated_count


@pytest.mark.parametrize("query,reason", [
    ('"Budget Update"', "missing_negated_term"),
    ('from:sender1@bench.example -label:Inbox', "missing_positive_free_text"),
    ('"Budget Update" -label:Inbox', "no_retained_matches"),
])
def test_negation_validation_flags_invalid_custom_workload(synthetic_benchmark_db: Any, query: str, reason: str) -> None:
    """An invalid named case is explained in the report rather than certified."""
    validation = bench._validate_negation_workload(
        db_path=Path(synthetic_benchmark_db.db_path_str), client_id="42", tenant_id="email-benchmark:42",
        queries=[bench.QueryCase(name="mixed_text_and_negation", query=query)],
    )
    assert validation["meaningful"] is False
    assert validation["reason"] == reason


@pytest.mark.parametrize("case_count,reason", [(0, "missing_case"), (2, "ambiguous_case")])
def test_negation_validation_rejects_missing_or_duplicate_cases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case_count: int, reason: str,
) -> None:
    """Missing or duplicate named cases fail closed without opening a database."""
    monkeypatch.setattr(bench, "_open_media_db", lambda **_kwargs: pytest.fail("database opened"))
    validation = bench._validate_negation_workload(
        db_path=tmp_path / "unused.sqlite", client_id="42", tenant_id="email-benchmark:42",
        queries=[bench.QueryCase(name="mixed_text_and_negation", query="Budget -from:sender")
                 for _ in range(case_count)],
    )
    assert validation["meaningful"] is False
    assert validation["reason"] == reason
