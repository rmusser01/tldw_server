# Release capacity and soak harness — TASK-13013.9

## Design and authorized scope

The release Stage 4.2 execution authorizes a small operator-run measurement
harness. Existing `Helper_Scripts/load_tests/chat_streaming_load.py` measures
streaming TTFT/chunk latency; `Helper_Scripts/embeddings_soak_test.py` drives a
Redis-specific synthetic workload and does not enforce release thresholds.
Neither establishes a release-wide capacity envelope. Preserve these specialized
helpers; use the existing HTTPX dependency and asyncio for a bounded HTTP runner.

A JSON profile and JSON dataset specify an immutable artifact SHA-256 and source
revision, three ordered phases (steady, overload, recovery), concurrency,
duration, request pacing/timeouts, success/error/latency thresholds, and queue,
database-pool and storage ceilings. Repeated requests are explicit operator
workloads, with authentication and workflow categories required. HTTP success
measures only the named request; job submission is never reported as completion.
A JSON observation endpoint supplies fresh, timestamped artifact-bound queue,
pool and storage measurements. Missing, stale, mismatched or nonnumeric samples
fail the run. This endpoint may be an operator-owned local collector; the API
currently lacks a single authoritative release-identity/telemetry endpoint.
No fabricated defaults or self-reported digest constitutes provenance proof.

Bound memory with millisecond latency histograms and aggregate telemetry ranges,
counts and final recovery state. Record profile/dataset hashes, exact declared
identity, phase settings and summaries, threshold failures, storage growth and
sustained metric recovery time. Never retain bodies, credentials or request URLs.
Credentials come from an environment variable. Disable redirect following and
ambient proxy settings, constrain dataset paths to the configured origin, and
bound HTTP read size and wall-clock request time.

This is a closed-loop HTTP envelope, not a universal workflow engine. Dynamic
job chaining, streaming TTFT, multipart ingestion and automatic resource cleanup
remain outside this runner; existing specialized helpers cover some of those
measurements. Operators must seed disposable datasets and use terminal-status
requests for long-running workflows. A successful synthetic integration test
certifies the harness only. A real exact-artifact run with reviewed thresholds
and collector provenance is still required to close TASK-13013.9.

## Stage 1: Contract and regressions
**Goal:** Document the profile/evidence contract and write behavioral tests.
**Success Criteria:** Invalid identities/profiles are rejected and regressions
express success, overload, recovery, missing metrics and target drift.
**Tests:** Unit validation and in-process HTTP integration tests.
**Status:** Complete

## Stage 2: Minimal runner and operator example
**Goal:** Implement the runner and document an editable reference profile.
**Success Criteria:** Evidence fails closed, no secret/raw body leakage, bounded
requests and explicit measurement scope.
**Tests:** Run the same regressions against the implementation.
**Status:** Complete

## Stage 3: Verification and handoff
**Goal:** Run focused tests, formatting, lint, Bandit and independent review.
**Success Criteria:** No new findings; task records precise implementation proof
and keeps live capacity certification open.
**Tests:** Focused pytest, Ruff, Black, Bandit and CLI help/example validation.
**Status:** Complete

## Stage 4: Exact-artifact operator certification
**Goal:** Establish a measured, reproducible release operating envelope.
**Success Criteria:** Run reviewed profiles against the actual candidate with
verified collector/artifact association, hardware/topology, ingestion and
terminal-workflow evidence; attach reports and supported limits.
**Tests:** Operator load/soak/overload/recovery run, no private infrastructure
required by the harness.
**Status:** Not Started — no live capacity certification has been performed.

## Implementation verification (2026-09-10)

- 30 focused tests pass, including in-process ASGI HTTP success/overload/recovery,
  invalid profiles, missing/stale metrics, target drift, terminal-state checks,
  timeouts, response bounds, credential-safe CLI output and overwrite rejection.
- Independent root review found terminal resource rechecking missing. The
  final-only queue-spike regression failed on the old implementation and passes
  after checking final metrics against recovery ceilings.
- Higher concurrency without an observed capacity rejection fails; no synthetic
  workload or collector observation certifies release capacity.
- Ruff and Black pass. Production Bandit has zero findings/errors; test Bandit
  has zero findings/errors with intentional test assertions (B101) excluded.
  Reports: `/tmp/bandit_task13013_9_harness.json` and
  `/tmp/bandit_task13013_9_tests.json`.
- CLI help works and documented JSON profile/dataset validate.
- Pytest reports the existing unknown `plugins` config warning. Earlier default
  temporary-directory cleanup touched unrelated stale test garbage and warned;
  final run uses its own `/tmp/task13013-9-pytest-reviewed` directory.
- Command: activate the root virtual environment, then run
  `python -m pytest tldw_Server_API/tests/Helper_Scripts/test_release_soak.py -q
  --override-ini addopts='' --confcutdir=tldw_Server_API/tests/Helper_Scripts
  --basetemp=/tmp/task13013-9-pytest-reviewed`.
- No commit, push, CI workflow change, deployment mutation or live workload run
  was performed. TASK-13013.9 stays In Progress for Stage 4.

### Normal repository fixture verification (supersedes isolated fixture run)

Independent review ran the normal repository fixtures and caught a prohibited
HTTPX constructor patch in the CLI test (29 passed, one failed). The test now
uses an ephemeral loopback HTTP collector and the real unmodified HTTPX client;
no network guard is disabled. The sandbox denies socket binding, so this test
requires a normal local/CI process allowed to bind loopback. The unsandboxed
normal command passed **30 tests, four existing warnings, in 4.67 seconds**:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Helper_Scripts/test_release_soak.py -q \
  --basetemp=/tmp/task13013-9-pytest-normal-local-http
```

Log: `/tmp/pr2761-soak-normal-local-http.log`. No alternate configuration or
`--confcutdir` was used. Both Bandit reports were regenerated and contain zero
findings/errors (test B101 excluded). This real fixture server verifies CLI
credential transport and failure evidence; it is not a release workload or
operator envelope certification.
