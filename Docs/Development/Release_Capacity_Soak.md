# Release capacity and soak measurements

`Helper_Scripts/load_tests/release_soak.py` runs a reviewed, disposable HTTP
workload through steady, overload and recovery phases and writes a JSON report.
It requires authentication and workflow observations, fresh resource metrics and
an immutable target identity. A report passes only the supplied envelope; it
is not a release authorization or an independent provenance attestation.

## Run a profile

From the repository root, activate the project virtual environment:

```bash
source .venv/bin/activate
python -m Helper_Scripts.load_tests.release_soak \
  --profile /tmp/release-profile.json \
  --dataset /tmp/release-dataset.json \
  --output /tmp/release-evidence.json
```

The CLI reads `SINGLE_USER_API_KEY` from the environment and sends `X-API-KEY`.
Use `--api-key-env NAME` for a different environment variable. Do not put the key
in the profile, URL, shell command or report. HTTP redirects and ambient proxies
are disabled; TLS verification remains enabled. Workload and observation paths
must be relative to the configured origin. Use a disposable deployment and
synthetic data: POST workloads repeat and may create many records. Arrange
cleanup before starting; the runner does not infer safe deletion operations.

Exit status is 0 for measured pass, 1 for measured failure, and 2 for invalid
inputs or output paths. The output file must not already exist. Interrupted or
killed runs may leave an incomplete file; only valid JSON with `passed: true`,
three complete phases and no failures represents a completed passing run.

## Example inputs: edit before use

This deliberately illustrative envelope is **not a measured reference limit**.
Replace both identity strings with the verified deployed artifact digest and
source commit; review durations, concurrency, ceilings and workload mix for the
selected hardware and deployment. The collector and workflow paths below are
operator-provided fixtures, not built-in tldw endpoints.

Save as `/tmp/release-profile.json`:

```json
{
  "name": "candidate-linux-amd64",
  "base_url": "http://127.0.0.1:8000",
  "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "telemetry_path": "/release-fixtures/observations",
  "timeout_seconds": 10,
  "sample_interval_seconds": 1,
  "max_sample_age_seconds": 5,
  "max_storage_growth_bytes": 104857600,
  "max_recovery_seconds": 30,
  "phases": [
    {
      "name": "steady", "duration_seconds": 300, "concurrency": 4,
      "pause_seconds": 0.1, "min_successes": 100,
      "max_error_ratio": 0, "max_rejection_ratio": 0,
      "p95_seconds": 2,
      "metric_maxima": {"queue_depth": 20, "db_pool_in_use": 10, "storage_bytes": 1073741824}
    },
    {
      "name": "overload", "duration_seconds": 60, "concurrency": 32,
      "pause_seconds": 0.01, "min_successes": 1,
      "max_error_ratio": 0, "max_rejection_ratio": 0.95,
      "p95_seconds": 5,
      "metric_maxima": {"queue_depth": 200, "db_pool_in_use": 20, "storage_bytes": 1073741824}
    },
    {
      "name": "recovery", "duration_seconds": 60, "concurrency": 4,
      "pause_seconds": 0.1, "min_successes": 20,
      "max_error_ratio": 0, "max_rejection_ratio": 0,
      "p95_seconds": 2,
      "metric_maxima": {"queue_depth": 20, "db_pool_in_use": 10, "storage_bytes": 1073741824}
    }
  ]
}
```

Save as `/tmp/release-dataset.json`:

```json
[
  {
    "name": "authenticated-read", "category": "authentication",
    "method": "GET", "path": "/api/v1/auth/me", "success_statuses": [200]
  },
  {
    "name": "seeded-terminal-workflow", "category": "workflow",
    "method": "GET", "path": "/release-fixtures/terminal-workflow",
    "success_statuses": [200], "response_equals": {"state": "complete"}
  }
]
```

Workloads run round-robin across a shared queue at each phase's concurrency.
Each workload must independently meet `min_successes`, error/rejection ratios
and p95 latency. HTTP 429 and 503 are counted as capacity rejections, separately
from errors, and at least one rejection must occur during overload. This
classification is observable HTTP behavior, not proof that a 503 has a graceful
internal cause; inspect the service's logs and telemetry separately. Other
unexpected statuses, timeouts, malformed expected JSON and mismatched top-level
`response_equals` fields count as errors. Redirects cannot count as success.

Requests measure complete bounded response reads, not TTFT. Every request has a
wall-clock timeout and a 1 MiB response ceiling. The p95 is a conservative
millisecond histogram bound over all attempts, including failures and
rejections. Counters and histogram buckets have bounded memory; raw request or
response content is not retained. Requests already running when a phase ends
are drained within the configured timeout before the next phase begins; reports
include both requested duration and actual elapsed time. Client pacing makes
this a closed-loop concurrency test, not a guaranteed offered request rate.

A successful submission or polling GET does **not** demonstrate completion of
new jobs. For an ingestion or long-running workflow claim, prepare terminal
checks with independently verified data and completion criteria, and retain the
specialized workload producer's evidence. This runner does not dynamically
chain returned job IDs, upload multipart files or perform streaming SSE timing.
Use the existing `chat_streaming_load.py`, `chat_streaming_sweep.py`, Locust
benchmark or embeddings-specific harness for those complementary measurements.
Their success alone does not replace this report's resource/identity checks.

## Observation endpoint contract

Provide an authenticated, same-origin JSON collector endpoint in the disposable
deployment, or expose an operator-owned collector behind its reverse proxy. The
current server has no single authoritative endpoint with all these fields.
Do not map a missing metric to zero or substitute configured pool capacity for
actual usage. Obtain identity from the running artifact/container, queue depth
from the selected Jobs/Scheduler backend, occupied database connections from
the actual pool, and stored bytes from the tested durable-data paths. Keep the
collector implementation, its deployment configuration, hardware description,
measurement scope and artifact-provenance verification alongside the report.
It must timestamp fresh measurements, not re-date a cached or invented value.

Every response must be HTTP 200 and contain:

```json
{
  "artifact_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "sampled_at": 1789050000.25,
  "queue_depth": 3,
  "db_pool_in_use": 2,
  "storage_bytes": 1048576
}
```

`sampled_at` is Unix time in seconds. It must advance, be no older than
`max_sample_age_seconds`, and not exceed the client clock by more than one
second. Identity must match on every observation. The runner validates it
before generating workload traffic, throughout each phase and after draining
the last phase. Each phase needs at least two valid samples; **any failed
observation fails the report**. This is fail-closed evidence binding to the
collector's claim, not cryptographic proof that the collector described the
actual serving workers. An operator must verify that association independently,
particularly behind a load balancer or across multiple services.

Resource maxima apply throughout steady and overload. During recovery, all
metrics must return within their recovery maxima before `max_recovery_seconds`
and remain there for the rest of the sampled phase, including at least two
consecutive valid samples. The terminal observation must also satisfy the
recovery ceilings; a final spike invalidates recovery. `recovery_seconds` measures the start of that final
uninterrupted healthy telemetry suffix, from the recovery phase's start.
HTTP thresholds apply to the whole recovery phase. Net storage growth compares
the first and final valid observations; per-phase storage maxima also bound
peaks. Storage decreases are reported unchanged rather than converted to growth.

## Compare reports and close the release task

The report records normalized JSON SHA-256 hashes for both input files, declared
artifact/source identity, phase thresholds, effective throughput, status counts,
latency bounds, telemetry sample/error counts and maxima, net storage growth,
recovery time and explicit failure reasons. Keep the original inputs and verify
the hashes before comparing results. Compare matching profile/dataset hashes,
collector definition, topology and hardware; different source/artifact identities
are expected when comparing candidates. No version tag or mutable branch name
is accepted in place of the digest and full source revision.

TASK-13013.9 remains open until an operator runs the selected exact artifact,
records a reproducible reference envelope (including ingestion and terminal
workflow evidence), and attaches collector/provenance/hardware information.
The in-process regression suite verifies harness behavior only; it establishes
no supported production concurrency, storage allowance or recovery guarantee.
