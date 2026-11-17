#!/usr/bin/env python3
"""Compare workflow read caching in disposable databases; never change production policy.

Run from a project environment: python Helper_Scripts/benchmarks/workflows_sqlite_cache.py
--output /tmp/workflows-cache.json. Cached controls are deliberately unsafe under
contention. A comparison is valid only when every measured read and pool return
is correct. Timings exclude setup/warmup and include no HTTP/WebSocket transport.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sqlite3
import statistics
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any
from unittest.mock import patch

from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tldw_Server_API.app.core.DB_Management import Workflows_DB as workflows_db  # noqa: E402

RUNS = 32
PAYLOAD_BYTES = 256


def _positive_int(value: str) -> int:
    """Require a nonempty workload rather than emit meaningless timings."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _trial(path: Path, cache_size: int, pool_size: int, workers: int, workload: str, iterations: int) -> dict[str, Any]:
    """Measure real adapter reads with only construction-time cache settings changed."""
    connect = sqlite3.connect

    def controlled_connect(*args, **kwargs):
        kwargs["cached_statements"] = cache_size
        return connect(*args, **kwargs)

    # Restore sqlite3.connect before any read threads start. No runtime policy switch.
    with patch.dict(os.environ, {"WORKFLOWS_SQLITE_POOL_SIZE": str(pool_size)}):
        with patch.object(workflows_db.sqlite3, "connect", side_effect=controlled_connect):
            db = workflows_db.WorkflowsDatabase(str(path))
    try:
        for index in range(RUNS):
            db.create_run(
                run_id=f"run-{index}",
                tenant_id="benchmark",
                user_id=f"owner-{index}",
                inputs={},
                workflow_id=None,
                definition_version=1,
                definition_snapshot={"name": "cache-benchmark", "version": 1, "steps": []},
            )
            db.append_event(
                "benchmark", f"run-{index}", "benchmark", {"owner": f"owner-{index}", "body": "x" * PAYLOAD_BYTES}
            )
        for index in range(100):
            db.get_run(f"run-{index % RUNS}")
            if workload == "poll":
                db.get_events(f"run-{index % RUNS}", since=0)
        pool_before = len(db._sqlite_pool)
        ready = Barrier(workers + 1, timeout=30) if workers > 1 else None

        def read(index: int) -> tuple[list[int], int, Counter[str]]:
            latencies: list[int] = []
            mismatches = 0
            errors: Counter[str] = Counter()
            if ready is not None:
                ready.wait()
            for offset in range(iterations):
                target = (index + offset * workers) % RUNS
                run_id, owner = f"run-{target}", f"owner-{target}"
                started = time.perf_counter_ns()
                try:
                    run = db.get_run(run_id)
                    events = db.get_events(run_id, since=0) if workload == "poll" else None
                    elapsed = time.perf_counter_ns() - started
                    valid = run is not None and (run.run_id, str(run.user_id)) == (run_id, owner)
                    if events is not None:
                        valid = (
                            valid
                            and len(events) == 1
                            and events[0]["run_id"] == run_id
                            and events[0]["payload_json"]["owner"] == owner
                        )
                    mismatches += not valid
                except (sqlite3.Error, TypeError, ValueError, KeyError, AttributeError, IndexError, SystemError) as exc:
                    elapsed = time.perf_counter_ns() - started
                    errors[type(exc).__name__] += 1
                latencies.append(elapsed)
            return latencies, mismatches, errors

        if workers == 1:
            started = time.perf_counter()
            results = [read(0)]
            elapsed_s = time.perf_counter() - started
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(read, index) for index in range(workers)]
                started = time.perf_counter()
                ready.wait()
                results = [future.result() for future in futures]
                elapsed_s = time.perf_counter() - started
        latencies = sorted(latency for result in results for latency in result[0])
        mismatches = sum(result[1] for result in results)
        errors = sum((result[2] for result in results), Counter())
        operations = iterations * workers
        pool_after = len(db._sqlite_pool)
        return {
            "cache_size": cache_size,
            "operations": operations,
            "elapsed_s": elapsed_s,
            "median_us": statistics.median(latencies) / 1000,
            "p95_us": latencies[math.ceil(len(latencies) * 0.95) - 1] / 1000,
            "ops_per_s": operations / elapsed_s,
            "valid_ops_per_s": (operations - mismatches - sum(errors.values())) / elapsed_s,
            "mismatches": mismatches,
            "errors": dict(errors),
            "pool_available_before": pool_before,
            "pool_available_after": pool_after,
            "valid": mismatches == 0 and not errors and pool_before == pool_after,
        }
    finally:
        db.close()


def benchmark(iterations: int, trials: int, threads: int) -> dict[str, Any]:
    """Run alternating paired controls and retain every trial for inspection."""
    cases = []
    with tempfile.TemporaryDirectory(prefix="workflows-cache-") as directory:
        for pool_size in (0, 4):
            for workers in sorted({1, threads}):
                for workload in ("run", "poll"):
                    samples = []
                    for trial in range(trials):
                        for cache_size in (128, 0) if trial % 2 == 0 else (0, 128):
                            path = Path(directory) / f"{pool_size}-{workers}-{workload}-{trial}-{cache_size}.db"
                            sample = _trial(path, cache_size, pool_size, workers, workload, iterations)
                            sample["trial"] = trial
                            samples.append(sample)
                    cached = sorted((s for s in samples if s["cache_size"] == 128), key=lambda s: s["trial"])
                    uncached = sorted((s for s in samples if s["cache_size"] == 0), key=lambda s: s["trial"])
                    valid = all(sample["valid"] for sample in samples)
                    cases.append(
                        {
                            "pool_size": pool_size,
                            "workers": workers,
                            "workload": workload,
                            "trials": samples,
                            "comparison_valid": valid,
                            "uncached_latency_change_pct": (
                                statistics.median(
                                    (u["median_us"] / c["median_us"] - 1) * 100 for c, u in zip(cached, uncached)
                                )
                                if valid
                                else None
                            ),
                            "uncached_throughput_change_pct": (
                                statistics.median(
                                    (u["ops_per_s"] / c["ops_per_s"] - 1) * 100 for c, u in zip(cached, uncached)
                                )
                                if valid
                                else None
                            ),
                        }
                    )
    return {
        "schema_version": 1,
        "source_sha256": {
            "benchmark": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "workflows_db": hashlib.sha256(Path(workflows_db.__file__).read_bytes()).hexdigest(),
        },
        "runtime": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "sqlite_threadsafety": sqlite3.threadsafety,
        },
        "method": {
            "runs": RUNS,
            "payload_bytes": PAYLOAD_BYTES,
            "iterations_per_worker": iterations,
            "trials": trials,
            "parallel_workers": threads,
            "warmup_reads": 100,
            "order": "alternating paired cache sizes",
            "latency": "adapter calls; validation excluded",
            "throughput": "includes validation and start barrier; attempted and valid operations reported",
            "poll": "get_run plus get_events(since=0), one event",
        },
        "cases": cases,
    }


def main() -> int:
    """Write a complete JSON report; fail if any uncached control is incorrect."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=_positive_int, default=1000)
    parser.add_argument("--trials", type=_positive_int, default=7)
    parser.add_argument("--threads", type=_positive_int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logger.disable("tldw_Server_API")
    report = benchmark(args.iterations, args.trials, args.threads)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return int(
        any(not sample["valid"] for case in report["cases"] for sample in case["trials"] if sample["cache_size"] == 0)
    )


if __name__ == "__main__":
    raise SystemExit(main())
