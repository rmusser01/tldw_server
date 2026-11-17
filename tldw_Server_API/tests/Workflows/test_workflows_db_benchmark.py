"""Behavior checks for the standalone, temporary-database cache benchmark."""

import json
import subprocess  # nosec B404: exercise the fixed local CLI with the active interpreter.
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Helper_Scripts/benchmarks/workflows_sqlite_cache.py"


def test_workflows_cache_benchmark_reports_real_reads(tmp_path):
    assert SCRIPT.is_file(), "The reproducible workflow cache benchmark is missing"
    output = tmp_path / "report.json"
    subprocess.run(  # nosec B603: fixed interpreter/script and test-generated temporary output.
        [sys.executable, str(SCRIPT), "--iterations", "4", "--trials", "1", "--threads", "2", "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    report = json.loads(output.read_text())
    assert report["runtime"]["sqlite"]
    assert len(report["cases"]) == 8
    for case in report["cases"]:
        assert {trial["cache_size"] for trial in case["trials"]} == {0, 128}
        for trial in case["trials"]:
            assert trial["operations"] == 4 * case["workers"]
            assert trial["median_us"] > 0
            assert trial["p95_us"] >= trial["median_us"]
            assert trial["ops_per_s"] > 0
            if trial["cache_size"] == 0:
                assert trial["mismatches"] == 0
                assert trial["errors"] == {}


@pytest.mark.parametrize("option", ["--iterations", "--trials", "--threads"])
def test_workflows_cache_benchmark_rejects_empty_workloads(option):
    assert SCRIPT.is_file(), "The reproducible workflow cache benchmark is missing"
    result = subprocess.run(  # nosec B603: fixed local CLI and parametrized literal flags.
        [sys.executable, str(SCRIPT), option, "0"], cwd=ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 2
    assert "positive" in result.stderr


def test_workflow_pool_recovers_all_connections_after_exhaustion(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

    monkeypatch.setenv("WORKFLOWS_SQLITE_POOL_SIZE", "4")
    db = WorkflowsDatabase(str(tmp_path / "pool.db"))
    borrowed = [db._acquire_sqlite() for _ in range(3)]
    try:
        for connection in borrowed:
            db._release_sqlite(connection)
        assert {id(connection) for connection in db._sqlite_pool} == {id(connection) for connection in borrowed}
    finally:
        db.close()
        for connection in borrowed:
            connection.close()


def test_workflows_cache_benchmark_counts_corrupt_row_errors(tmp_path, monkeypatch):
    from Helper_Scripts.benchmarks import workflows_sqlite_cache as benchmark

    original = benchmark.workflows_db.WorkflowsDatabase.get_run
    reads = 0

    def corrupt_after_warmup(db, run_id):
        nonlocal reads
        reads += 1
        if reads > 100:
            raise IndexError("corrupt cached row")
        return original(db, run_id)

    monkeypatch.setattr(benchmark.workflows_db.WorkflowsDatabase, "get_run", corrupt_after_warmup)
    sample = benchmark._trial(tmp_path / "corrupt.db", 128, 0, 1, "run", 4)
    assert sample["errors"] == {"IndexError": 4}
    assert sample["valid"] is False
    assert sample["valid_ops_per_s"] == 0
