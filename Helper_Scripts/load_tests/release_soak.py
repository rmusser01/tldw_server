#!/usr/bin/env python3
"""Measure an operator-declared HTTP envelope; never infer release certification.

See Docs/Development/Release_Capacity_Soak.md for profile and collector contracts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

METRICS = ("queue_depth", "db_pool_in_use", "storage_bytes")
MAX_RESPONSE_BYTES = 1024 * 1024


def number(value: Any, minimum: float, maximum: float) -> bool:
    """Reject booleans, nonfinite measurements and out-of-range values."""
    return type(value) in (int, float) and minimum <= value <= maximum


def path_is_local(value: Any) -> bool:
    """Allow only origin-relative HTTP paths without redirect-like syntax."""
    return (
        isinstance(value, str)
        and value.startswith("/")
        and not value.startswith("//")
        and "\\" not in value
        and not urlsplit(value).fragment
        and not any(ord(char) < 32 for char in value)
    )


def validate(profile: dict, dataset: list) -> None:
    """Validate all run inputs before issuing requests; no defaults hide omissions."""
    try:
        if not isinstance(profile, dict) or not isinstance(dataset, list):
            raise ValueError("profile must be an object and dataset an array")
        if set(profile) != {
            "name",
            "base_url",
            "artifact_sha256",
            "source_revision",
            "telemetry_path",
            "timeout_seconds",
            "sample_interval_seconds",
            "max_sample_age_seconds",
            "max_storage_growth_bytes",
            "max_recovery_seconds",
            "phases",
        }:
            raise ValueError("unknown or missing profile field")
        target = urlsplit(profile["base_url"])
        if (
            target.scheme not in ("http", "https")
            or not target.hostname
            or target.username is not None
            or target.password is not None
            or target.path not in ("", "/")
            or target.query
            or target.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) origin without credentials")
        for key, pattern in (
            ("name", r"[a-zA-Z0-9_.-]{1,80}"),
            ("artifact_sha256", r"[0-9a-f]{64}"),
            ("source_revision", r"[0-9a-f]{40}"),
        ):
            if not isinstance(profile[key], str) or not re.fullmatch(pattern, profile[key]):
                raise ValueError(f"invalid {key}")
        if not path_is_local(profile["telemetry_path"]):
            raise ValueError("telemetry_path must be origin-relative")
        for key, low, high in (
            ("timeout_seconds", 0.01, 120),
            ("sample_interval_seconds", 0.005, 60),
            ("max_sample_age_seconds", 0.01, 120),
            ("max_storage_growth_bytes", 0, 10**18),
            ("max_recovery_seconds", 0.01, 86400),
        ):
            if not number(profile[key], low, high):
                raise ValueError(f"invalid {key}")
        phases = profile["phases"]
        if [phase["name"] for phase in phases] != ["steady", "overload", "recovery"]:
            raise ValueError("phases must be steady, overload, recovery in order")
        for phase in phases:
            if set(phase) != {
                "name",
                "duration_seconds",
                "concurrency",
                "pause_seconds",
                "min_successes",
                "max_error_ratio",
                "max_rejection_ratio",
                "p95_seconds",
                "metric_maxima",
            }:
                raise ValueError("unknown or missing phase field")
            if type(phase["concurrency"]) is not int or not 1 <= phase["concurrency"] <= 128:
                raise ValueError("concurrency must be an integer from 1 to 128")
            if type(phase["min_successes"]) is not int or phase["min_successes"] < 1:
                raise ValueError("min_successes must be a positive integer per workload")
            for key, low, high in (
                ("duration_seconds", 0.01, 86400),
                ("pause_seconds", 0, 60),
                ("max_error_ratio", 0, 1),
                ("max_rejection_ratio", 0, 1),
                ("p95_seconds", 0.001, 120),
            ):
                if not number(phase[key], low, high):
                    raise ValueError(f"invalid phase {key}")
            if phase["duration_seconds"] < 2 * profile["sample_interval_seconds"]:
                raise ValueError("each phase must allow at least two telemetry intervals")
            if set(phase["metric_maxima"]) != set(METRICS) or not all(
                number(value, 0, 10**18) for value in phase["metric_maxima"].values()
            ):
                raise ValueError("all three metric maxima must be finite and nonnegative")
        if phases[1]["concurrency"] <= phases[0]["concurrency"]:
            raise ValueError("overload concurrency must exceed steady concurrency")
        if phases[2]["concurrency"] > phases[0]["concurrency"]:
            raise ValueError("recovery concurrency must not exceed steady concurrency")
        if profile["max_recovery_seconds"] > phases[2]["duration_seconds"]:
            raise ValueError("recovery deadline must fit within the recovery phase")
        if not 2 <= len(dataset) <= 32:
            raise ValueError("dataset must contain 2 to 32 workloads")
        names = set()
        for row in dataset:
            if set(row) - {"name", "category", "method", "path", "json", "success_statuses", "response_equals"}:
                raise ValueError("unknown workload field")
            if (
                not isinstance(row["name"], str)
                or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", row["name"])
                or row["name"] in names
            ):
                raise ValueError("workload names must be unique identifiers")
            names.add(row["name"])
            if row["category"] not in ("authentication", "workflow"):
                raise ValueError("category must be authentication or workflow")
            if row["method"] not in ("GET", "POST") or not path_is_local(row["path"]):
                raise ValueError("workloads require GET/POST and origin-relative paths")
            statuses = row["success_statuses"]
            if (
                not isinstance(statuses, list)
                or not statuses
                or not all(type(code) is int and 200 <= code < 300 for code in statuses)
            ):
                raise ValueError("success_statuses must contain explicit 2xx statuses")
            if "response_equals" in row and not isinstance(row["response_equals"], dict):
                raise ValueError("response_equals must be an object of top-level JSON fields")
        if {row["category"] for row in dataset} != {"authentication", "workflow"}:
            raise ValueError("authentication and workflow observations are required")
        json.dumps([profile, dataset], allow_nan=False)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("missing or malformed profile/dataset field") from exc


async def request(client: httpx.AsyncClient, row: dict, timeout: float, *, json_body: bool = False) -> tuple[int, Any]:
    """Bound response memory and entire request time, including a slow body."""

    async def fetch() -> tuple[int, Any]:
        async with client.stream(
            row["method"],
            row["path"],
            json=row.get("json"),
            timeout=timeout,
            follow_redirects=False,
        ) as response:
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise ValueError("response size limit exceeded")
                body.extend(chunk)
            return response.status_code, json.loads(body) if json_body and 200 <= response.status_code < 300 else None

    return await asyncio.wait_for(fetch(), timeout=timeout)


def digest(value: Any) -> str:
    """Hash normalized JSON so equivalent profile formatting compares equally."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


async def run(profile: dict, dataset: list, client: httpx.AsyncClient) -> dict:
    """Run closed-loop phases and return evidence with explicit threshold failures."""
    validate(profile, dataset)
    if str(client.base_url).rstrip("/") != profile["base_url"].rstrip("/"):
        raise ValueError("HTTP client origin differs from the profile")
    started = time.monotonic()
    evidence = {
        "schema_version": 1,
        "profile_name": profile["name"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": profile["artifact_sha256"],
        "source_revision": profile["source_revision"],
        "identity_basis": "operator declaration matched against collector observations; not provenance verification",
        "profile_sha256": digest(profile),
        "dataset_sha256": digest(dataset),
        "measurement_scope": "closed-loop HTTP request completion; no inferred background-job completion",
        "phases": [],
        "failures": [],
        "recovery_seconds": None,
    }
    failures: set[str] = set()
    storage_first = None
    storage_last = None
    previous_sample = None
    timeout = profile["timeout_seconds"]

    async def observe() -> dict:
        nonlocal previous_sample, storage_first, storage_last
        status, sample = await request(
            client, {"method": "GET", "path": profile["telemetry_path"]}, timeout, json_body=True
        )
        if status != 200 or not isinstance(sample, dict):
            raise ValueError("telemetry response invalid")
        if any(sample.get(key) != profile[key] for key in ("artifact_sha256", "source_revision")):
            raise ValueError("target identity mismatch")
        timestamp = sample.get("sampled_at")
        now = time.time()
        if not number(timestamp, now - profile["max_sample_age_seconds"], now + 1) or (
            previous_sample is not None and timestamp <= previous_sample
        ):
            raise ValueError("telemetry timestamp stale or repeated")
        if not all(number(sample.get(key), 0, 10**18) for key in METRICS):
            raise ValueError("required telemetry metric missing or invalid")
        previous_sample = timestamp
        if storage_first is None:
            storage_first = sample["storage_bytes"]
        storage_last = sample["storage_bytes"]
        return sample

    # Verify the target before generating potentially mutating workload traffic.
    try:
        await observe()
    except (httpx.HTTPError, TimeoutError, ValueError):
        failures.add("initial target identity/telemetry verification failed")
    else:

        async def run_phase(phase: dict) -> dict:
            phase_start = time.monotonic()
            stop = phase_start + phase["duration_seconds"]
            counters = {row["name"]: Counter() for row in dataset}
            histograms = {row["name"]: Counter() for row in dataset}
            telemetry = {"samples": 0, "errors": 0, "maxima": dict.fromkeys(METRICS, 0)}
            stable_since = None
            stable_samples = 0
            next_row = 0

            async def worker() -> None:
                nonlocal next_row
                while time.monotonic() < stop:
                    row = dataset[next_row % len(dataset)]
                    next_row += 1
                    before = time.monotonic()
                    count = counters[row["name"]]
                    count["attempts"] += 1
                    try:
                        code, body = await request(client, row, timeout, json_body="response_equals" in row)
                        count[f"http_{code}"] += 1
                        if code in (429, 503):
                            count["rejections"] += 1
                        elif code in row["success_statuses"] and (
                            "response_equals" not in row
                            or isinstance(body, dict)
                            and all(body.get(key) == value for key, value in row["response_equals"].items())
                        ):
                            count["successes"] += 1
                        else:
                            count["errors"] += 1
                    except (httpx.HTTPError, TimeoutError, ValueError):
                        count["errors"] += 1
                    # Millisecond ceilings are conservative; 120 seconds caps the histogram.
                    bucket = min(120001, math.ceil((time.monotonic() - before) * 1000))
                    histograms[row["name"]][bucket] += 1
                    await asyncio.sleep(max(0, min(phase["pause_seconds"], stop - time.monotonic())))

            async def poll() -> None:
                nonlocal stable_since, stable_samples
                while time.monotonic() < stop:
                    try:
                        sample = await observe()
                        telemetry["samples"] += 1
                        for key in METRICS:
                            telemetry["maxima"][key] = max(telemetry["maxima"][key], sample[key])
                        healthy = all(sample[key] <= phase["metric_maxima"][key] for key in METRICS)
                        if healthy:
                            if stable_since is None:
                                stable_since = time.monotonic() - phase_start
                            stable_samples += 1
                        else:
                            stable_since, stable_samples = None, 0
                    except (httpx.HTTPError, TimeoutError, ValueError):
                        telemetry["errors"] += 1
                        stable_since, stable_samples = None, 0
                    await asyncio.sleep(max(0, min(profile["sample_interval_seconds"], stop - time.monotonic())))

            await asyncio.gather(poll(), *(worker() for _ in range(phase["concurrency"])))
            result = {
                **phase,
                "elapsed_seconds": time.monotonic() - phase_start,
                "telemetry": telemetry,
                "workloads": {},
            }
            for row in dataset:
                name = row["name"]
                counts = counters[name]
                attempts = counts["attempts"]
                rank = math.ceil(attempts * 0.95)
                seen = 0
                p95 = None
                for bucket, count in sorted(histograms[name].items()):
                    seen += count
                    if seen >= rank:
                        p95 = bucket / 1000
                        break
                result["workloads"][name] = {
                    **{key: counts[key] for key in ("attempts", "successes", "errors", "rejections")},
                    "category": row["category"],
                    "p95_seconds_upper_bound": p95,
                    "completed_requests_per_second": attempts / result["elapsed_seconds"],
                    "statuses": {key: value for key, value in counts.items() if key.startswith("http_")},
                }
                if counts["successes"] < phase["min_successes"]:
                    failures.add(f"{phase['name']}/{name}: insufficient successful observations")
                if not attempts or counts["errors"] / attempts > phase["max_error_ratio"]:
                    failures.add(f"{phase['name']}/{name}: error ratio exceeded")
                if attempts and counts["rejections"] / attempts > phase["max_rejection_ratio"]:
                    failures.add(f"{phase['name']}/{name}: rejection ratio exceeded")
                if p95 is None or p95 > phase["p95_seconds"]:
                    failures.add(f"{phase['name']}/{name}: p95 latency exceeded or unmeasured")
            if telemetry["samples"] < 2 or telemetry["errors"]:
                failures.add(f"{phase['name']}: telemetry incomplete or invalid")
            if phase["name"] == "recovery":
                if stable_samples < 2 or stable_since is None or stable_since > profile["max_recovery_seconds"]:
                    failures.add("recovery: metrics did not sustain recovery before deadline")
                else:
                    evidence["recovery_seconds"] = stable_since
            elif any(telemetry["maxima"][key] > phase["metric_maxima"][key] for key in METRICS):
                failures.add(f"{phase['name']}: resource ceiling exceeded")
            if phase["name"] == "overload" and not any(count["rejections"] for count in counters.values()):
                failures.add("overload: no rejection observed")
            return result

        for phase in profile["phases"]:
            evidence["phases"].append(await run_phase(phase))
        try:
            final = await observe()
            evidence["final_metrics"] = {key: final[key] for key in METRICS}
            if any(final[key] > profile["phases"][-1]["metric_maxima"][key] for key in METRICS):
                failures.add("final observation exceeds recovery resource ceilings")
                evidence["recovery_seconds"] = None
        except (httpx.HTTPError, TimeoutError, ValueError):
            failures.add("final target identity/telemetry verification failed")
            evidence["recovery_seconds"] = None
    growth = None if storage_first is None or storage_last is None else storage_last - storage_first
    if growth is None or growth > profile["max_storage_growth_bytes"]:
        failures.add("storage growth exceeded or unmeasured")
    evidence.update(
        storage_growth_bytes=growth,
        elapsed_seconds=time.monotonic() - started,
        failures=sorted(failures),
        passed=not failures,
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    """Run a reviewed profile; return 0 for measured pass, 1 for fail, 2 for input error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", default="SINGLE_USER_API_KEY")
    args = parser.parse_args(argv)

    async def execute(profile: dict, dataset: list) -> dict:
        headers = {}
        if os.environ.get(args.api_key_env):
            headers["X-API-KEY"] = os.environ[args.api_key_env]
        async with httpx.AsyncClient(
            base_url=profile["base_url"], headers=headers, trust_env=False, follow_redirects=False
        ) as client:
            return await run(profile, dataset, client)

    try:
        profile = json.loads(args.profile.read_text())
        dataset = json.loads(args.dataset.read_text())
        validate(profile, dataset)
        # Exclusive creation prevents overwriting another run or an input file.
        with args.output.open("x") as output:
            report = asyncio.run(execute(profile, dataset))
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print("PASS" if report["passed"] else "FAIL")
        return 0 if report["passed"] else 1
    except (OSError, ValueError):
        print("Invalid profile, dataset, or output path; no certification produced.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
