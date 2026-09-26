"""Backend selection and RLS scope contracts for the email search benchmark."""

from __future__ import annotations

import json
import sys
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
