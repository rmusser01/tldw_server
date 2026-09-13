#!/usr/bin/env python3
"""
Validate Email Milestone M2 gate metrics from Prometheus text exposition.

Focuses on the staging lag SLO requirement:
- median email sync lag (p50) must be <= 5 minutes by default.

Supports either:
1) Live metrics fetch from a server endpoint, optionally with a sampling window.
2) Offline metric files (before/after) for repeatable dry-runs.

Passing checks apply only to the supplied metrics. Offline fixtures do not
validate staging, and endpoint mode does not establish deployment provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_SAMPLE_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[+-]?Inf|NaN)$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def _env_first(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _normalize_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return "http://127.0.0.1:8000"
    if not raw.startswith(("http://", "https://")):
        raw = f"http://{raw}"
    return raw.rstrip("/")


def _decode_body(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return str(raw)


def _fetch_metrics_text(
    *,
    base_url: str,
    metrics_path: str,
    timeout: float,
    api_key: Optional[str],
    bearer: Optional[str],
) -> str:
    path = "/" + str(metrics_path or "").lstrip("/")
    url = f"{_normalize_base_url(base_url)}{path}"
    req = Request(url, method="GET")
    req.add_header("Accept", "text/plain")
    if api_key:
        req.add_header("X-API-KEY", api_key)
    if bearer:
        req.add_header("Authorization", f"Bearer {bearer}")
    try:
        with urlopen(req, timeout=timeout) as resp:  # nosec B310
            status = int(resp.getcode())
            body = _decode_body(resp.read() or b"")
    except HTTPError as exc:
        status = int(getattr(exc, "code", 0) or 0)
        body = _decode_body(exc.read() or b"")
        raise RuntimeError(f"metrics fetch failed: status={status} body={body[:400]}") from exc
    except URLError as exc:
        raise RuntimeError(f"metrics fetch URL error: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"metrics fetch error: {exc}") from exc
    if status != 200:
        raise RuntimeError(f"metrics fetch failed: status={status} body={body[:400]}")
    return body


def _unescape_prom_label(value: str) -> str:
    return (
        value.replace(r"\\", "\\")
        .replace(r"\"", '"')
        .replace(r"\n", "\n")
        .replace(r"\t", "\t")
    )


def _parse_labels(raw_labels: str | None) -> Dict[str, str]:
    if not raw_labels:
        return {}
    out: Dict[str, str] = {}
    for match in _LABEL_RE.finditer(raw_labels):
        key = str(match.group(1) or "").strip()
        value = _unescape_prom_label(str(match.group(2) or ""))
        if key:
            out[key] = value
    return out


def _parse_value(raw: str) -> float:
    value = str(raw).strip()
    if value in {"+Inf", "Inf", "+inf", "inf"}:
        return math.inf
    if value in {"-Inf", "-inf"}:
        return -math.inf
    if value == "NaN":
        return math.nan
    return float(value)


def _parse_samples(text: str) -> list[Tuple[str, Dict[str, str], float]]:
    samples: list[Tuple[str, Dict[str, str], float]] = []
    for line in str(text or "").splitlines():
        row = line.strip()
        if (not row) or row.startswith("#"):
            continue
        match = _SAMPLE_RE.match(row)
        if not match:
            continue
        name = str(match.group(1))
        labels = _parse_labels(match.group(2))
        value = _parse_value(str(match.group(3)))
        if math.isnan(value):
            continue
        samples.append((name, labels, value))
    return samples


def _counter_by_status(
    samples: list[Tuple[str, Dict[str, str], float]],
    *,
    metric_name: str,
    provider: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, labels, value in samples:
        if name != metric_name:
            continue
        if str(labels.get("provider") or "") != provider:
            continue
        status = str(labels.get("status") or "").strip().lower()
        if not status:
            continue
        out[status] = out.get(status, 0.0) + float(value)
    return out


def _counter_by_outcome(
    samples: list[Tuple[str, Dict[str, str], float]],
    *,
    metric_name: str,
    provider: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, labels, value in samples:
        if name != metric_name:
            continue
        if str(labels.get("provider") or "") != provider:
            continue
        outcome = str(labels.get("outcome") or "").strip().lower()
        if not outcome:
            continue
        out[outcome] = out.get(outcome, 0.0) + float(value)
    return out


def _histogram_buckets(
    samples: list[Tuple[str, Dict[str, str], float]],
    *,
    metric_name: str,
    provider: str,
) -> Dict[float, float]:
    out: Dict[float, float] = {}
    bucket_name = f"{metric_name}_bucket"
    for name, labels, value in samples:
        if name != bucket_name:
            continue
        if str(labels.get("provider") or "") != provider:
            continue
        le_text = str(labels.get("le") or "").strip()
        if not le_text:
            continue
        if le_text in {"+Inf", "Inf", "inf", "+inf"}:
            le_value = math.inf
        else:
            try:
                le_value = float(le_text)
            except ValueError:
                continue
        out[le_value] = out.get(le_value, 0.0) + float(value)
    return out


def _delta_map(after: Dict[Any, float], before: Dict[Any, float]) -> Dict[Any, float]:
    keys = set(before.keys()) | set(after.keys())
    out: Dict[Any, float] = {}
    for key in keys:
        delta = float(after.get(key, 0.0) - before.get(key, 0.0))
        out[key] = delta if delta > 0 else 0.0
    return out


def _hist_quantile(buckets: Dict[float, float], quantile: float) -> Optional[float]:
    if not buckets:
        return None
    total = float(buckets.get(math.inf, 0.0))
    if total <= 0:
        return None
    target = float(quantile) * total
    running_prev = 0.0
    le_prev = 0.0
    for le_value in sorted(buckets.keys()):
        running = float(buckets.get(le_value, 0.0))
        if running >= target:
            if math.isinf(le_value):
                return le_prev
            if running <= running_prev:
                return le_value
            proportion = (target - running_prev) / (running - running_prev)
            return le_prev + (le_value - le_prev) * proportion
        if not math.isinf(le_value):
            le_prev = le_value
        running_prev = running
    return None


def _load_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}s"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Email M2 gate lag SLO from metrics output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=_env_first("TLDW_BASE_URL", "TLDW_E2E_SERVER_URL", "BASE_URL")
        or "http://127.0.0.1:8000",
    )
    parser.add_argument("--metrics-path", default="/api/v1/metrics/text")
    parser.add_argument("--provider", default="gmail")
    parser.add_argument("--api-key", default=_env_first("SINGLE_USER_API_KEY", "TLDW_API_KEY", "X_API_KEY"))
    parser.add_argument("--bearer", default=_env_first("ADMIN_BEARER", "TLDW_BEARER"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=0,
        help="If >0 and using live endpoint mode, collect before/after snapshots and evaluate deltas.",
    )
    parser.add_argument("--max-median-lag-seconds", type=float, default=300.0)
    parser.add_argument("--min-success-runs", type=int, default=1)
    parser.add_argument("--metrics-file", default=None, help="Offline metrics snapshot text file")
    parser.add_argument(
        "--metrics-file-after",
        default=None,
        help="Optional second snapshot file for delta evaluation with --metrics-file",
    )
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    provider = str(args.provider).strip()
    if not provider:
        print("[FAIL] provider cannot be empty", file=sys.stderr)
        return 2

    before_text: Optional[str] = None
    after_text: str

    if args.metrics_file:
        before_text = _load_text_file(str(args.metrics_file))
        if args.metrics_file_after:
            after_text = _load_text_file(str(args.metrics_file_after))
        else:
            after_text = before_text
            before_text = None
    else:
        after_text = _fetch_metrics_text(
            base_url=args.base_url,
            metrics_path=args.metrics_path,
            timeout=float(args.timeout),
            api_key=args.api_key,
            bearer=args.bearer,
        )
        if int(args.window_seconds) > 0:
            before_text = after_text
            time.sleep(int(args.window_seconds))
            after_text = _fetch_metrics_text(
                base_url=args.base_url,
                metrics_path=args.metrics_path,
                timeout=float(args.timeout),
                api_key=args.api_key,
                bearer=args.bearer,
            )

    after_samples = _parse_samples(after_text)
    before_samples = _parse_samples(before_text or "") if before_text is not None else []

    runs_after = _counter_by_status(
        after_samples, metric_name="email_sync_runs_total", provider=provider
    )
    runs_before = _counter_by_status(
        before_samples, metric_name="email_sync_runs_total", provider=provider
    )
    recovery_after = _counter_by_outcome(
        after_samples,
        metric_name="email_sync_recovery_events_total",
        provider=provider,
    )
    recovery_before = _counter_by_outcome(
        before_samples,
        metric_name="email_sync_recovery_events_total",
        provider=provider,
    )
    lag_after = _histogram_buckets(
        after_samples, metric_name="email_sync_lag_seconds", provider=provider
    )
    lag_before = _histogram_buckets(
        before_samples, metric_name="email_sync_lag_seconds", provider=provider
    )

    if before_text is not None:
        run_counts = _delta_map(runs_after, runs_before)
        recovery_counts = _delta_map(recovery_after, recovery_before)
        lag_buckets = _delta_map(lag_after, lag_before)
        sample_mode = "delta"
    else:
        run_counts = dict(runs_after)
        recovery_counts = dict(recovery_after)
        lag_buckets = dict(lag_after)
        sample_mode = "snapshot"

    success_runs = float(run_counts.get("success", 0.0))
    failed_runs = float(run_counts.get("failed", 0.0))
    skipped_runs = float(run_counts.get("skipped", 0.0))
    total_runs = success_runs + failed_runs + skipped_runs

    lag_p50 = _hist_quantile(lag_buckets, 0.50)
    lag_p95 = _hist_quantile(lag_buckets, 0.95)
    lag_slo_pass = lag_p50 is not None and lag_p50 <= float(args.max_median_lag_seconds)
    success_runs_pass = success_runs >= int(args.min_success_runs)

    if args.metrics_file:
        evidence_source = "offline_fixture"
        evidence_note = "Offline fixture threshold checks only; staging evidence remains unverified."
    else:
        evidence_source = "live_endpoint"
        evidence_note = "Live endpoint metrics; confirm deployment provenance before using as staging evidence."

    print(f"[INFO] mode={sample_mode} provider={provider} evidence_source={evidence_source}")
    print(f"[INFO] {evidence_note}")
    print(
        f"[INFO] runs: success={success_runs:.0f} failed={failed_runs:.0f} "
        f"skipped={skipped_runs:.0f} total={total_runs:.0f}"
    )
    print(
        f"[INFO] lag: p50={_format_seconds(lag_p50)} p95={_format_seconds(lag_p95)} "
        f"target_p50<={float(args.max_median_lag_seconds):.2f}s"
    )
    if recovery_counts:
        ordered = ", ".join(
            f"{key}={int(value)}" for key, value in sorted(recovery_counts.items())
        )
        print(f"[INFO] recovery_events: {ordered}")
    else:
        print("[INFO] recovery_events: none")

    if not success_runs_pass:
        print(
            f"[FAIL] insufficient successful sync samples for evaluation "
            f"(success_runs={success_runs:.0f}, min={int(args.min_success_runs)})"
        )
    if not lag_slo_pass:
        print("[FAIL] median lag SLO not satisfied")
    if success_runs_pass and lag_slo_pass:
        print(f"[PASS] Email M2 lag threshold checks passed for {evidence_source} ({sample_mode})")

    output = {
        "provider": provider,
        "mode": sample_mode,
        "evidence_source": evidence_source,
        "evidence_note": evidence_note,
        "runs": {
            "success": int(success_runs),
            "failed": int(failed_runs),
            "skipped": int(skipped_runs),
            "total": int(total_runs),
        },
        "recovery_events": {k: int(v) for k, v in sorted(recovery_counts.items())},
        "lag_seconds": {
            "p50": lag_p50,
            "p95": lag_p95,
            "threshold_p50_max": float(args.max_median_lag_seconds),
        },
        "checks": {
            "success_runs_min": int(args.min_success_runs),
            "success_runs_pass": bool(success_runs_pass),
            "lag_slo_pass": bool(lag_slo_pass),
        },
        "passed": bool(success_runs_pass and lag_slo_pass),
    }

    if args.output_json:
        Path(str(args.output_json)).write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return 0 if output["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
