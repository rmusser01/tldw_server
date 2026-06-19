# Sandbox Operator Evidence Ingestion Design

**Date:** 2026-06-19
**Status:** Proposed design
**Backlog:** TASK-2391
**Parent design:** `Docs/superpowers/specs/2026-06-18-sandbox-operator-status-consolidation-design.md`
**Scope:** Slice 2 for bounded, advisory host-gated VZ smoke evidence ingestion
into the consolidated sandbox operator-status projection.

## Summary

Slice 1 added the admin-only read-only
`GET /api/v1/sandbox/admin/operator-status` endpoint with an `evidence`
section placeholder. Slice 2 should populate that section from a server-side
configured host-gated VZ smoke evidence bundle when an operator explicitly
configures one.

The existing host smoke wrapper writes a machine-readable evidence file:

```text
<evidence-dir>/host-smoke-evidence.json
```

The existing `summarize-host-e2e-evidence.py` script renders advisory Markdown
from that evidence bundle. The operator-status server must not scrape that
Markdown or import the CLI script as application code. The stable input for this
slice is the evidence directory plus direct known child files, especially
`host-smoke-evidence.json`.

## Goals

- Populate `sections.evidence` in the consolidated operator status when a
  server-side configured evidence directory is present.
- Keep evidence advisory and read-only.
- Reuse the existing host smoke evidence bundle shape instead of creating a new
  artifact contract.
- Keep normal CI portable; no real VZ execution is introduced.
- Avoid arbitrary file probing by disallowing request-supplied evidence paths.
- Preserve privacy boundaries by exposing only bounded metadata, never raw logs.

## Non-Goals

- Do not run host-gated smoke, helper lifecycle commands, repair, cleanup,
  launchd commands, or real VMs.
- Do not create evidence directories or evidence files.
- Do not parse the Markdown GitHub step summary.
- Do not import `tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py`
  into server code.
- Do not accept evidence paths through query parameters, request bodies, or
  per-call headers in this slice.
- Do not recursively scan evidence directories or artifact directories.
- Do not expose raw serial logs, helper stdout/stderr, guest output,
  environment dumps, or arbitrary file contents.
- Do not make stale evidence blocking by default. Strict/expected-evidence mode
  is a later design.

## Input Contract

Slice 2 uses a server-side configured evidence directory path. The recommended
first configuration surface is an environment variable:

```text
TLDW_SANDBOX_VZ_EVIDENCE_DIR=/path/to/evidence
```

An unset variable means evidence is not configured. In that case
`sections.evidence.status` remains `not_configured` and does not affect
`overall_status`.

The configured directory is an operator expectation. If it is set but missing,
unsafe, unreadable, malformed, or oversized, the evidence section should report
that state and the overall status should degrade unless a more severe live
diagnostic condition already dominates.

## Parser Boundary

Add a small server-side parser module, for example:

```text
tldw_Server_API/app/core/Sandbox/operator_evidence.py
```

Responsibilities:

- Resolve and validate the configured evidence directory path.
- Inspect only direct known child files.
- Read only `host-smoke-evidence.json`, bounded by a size cap.
- Return a normalized evidence summary dict for `operator_status.py`.
- Treat every field as untrusted external input.

The parser should not classify the whole sandbox subsystem. It should only
describe evidence availability, parsed metadata, warnings, and stable reasons.
`operator_status.py` remains responsible for projecting that parser output into
the final `sections.evidence`, `recommended_actions`, and `overall_status`.

## Safe Filesystem Handling

The parser must reject or safely report:

- embedded NUL in configured paths
- missing evidence directory
- path exists but is not a directory
- evidence directory symlink
- intermediate symlink when descriptor-safe traversal is available
- unreadable evidence directory
- missing `host-smoke-evidence.json`
- `host-smoke-evidence.json` symlink
- `host-smoke-evidence.json` non-regular file
- `host-smoke-evidence.json` larger than the configured size cap
- malformed UTF-8 or malformed JSON
- top-level JSON values that are not objects
- unsupported schema versions

Use a conservative size cap. The existing summarizer uses `1 MiB`; the server
parser should use the same cap unless implementation review finds a stronger
reason to lower it.

Descriptor-relative `open`/`stat` with no-follow flags should be used where
portable. If the platform cannot provide safe directory file descriptor
operations, the parser should fail closed for configured evidence and report a
stable reason such as `evidence_safe_open_unavailable`. It should not fall back
to unsafe recursive or symlink-following reads.

## Parsed Metadata

The normalized parser output should include only bounded scalar metadata and
small bounded collections:

- configured path status
- `schema_version`
- `created_at`
- computed `age_seconds` when `created_at` is valid
- `smoke_run_id`
- `final_exit_code`, accepting only real integer values and rejecting boolean
  or container values
- skip flags such as `skip_build`, `skip_sign`, and `include_failure_drills`
- phase names with scalar status and exit code
- cleanup status scalar fields
- expected file presence/readability status for known evidence files
- stable parser reason codes

Path metadata in `host-smoke-evidence.json` may be retained only as bounded
pointers already emitted by the smoke wrapper, and only after scalar coercion.
Do not dereference those paths.

Do not expose:

- raw log contents
- raw nested values
- arbitrary keys from JSON
- environment variables
- helper command lines
- guest output

## Evidence Section Projection

`operator_status.build_operator_status()` should accept an optional normalized
evidence summary from the service layer. The first implementation can extend the
function signature with an optional keyword argument:

```python
evidence_summary: Mapping[str, Any] | None = None
```

Recommended section behavior:

| Evidence input | `sections.evidence.status` | Severity | Overall impact |
| --- | --- | --- | --- |
| Env unset | `not_configured` | `info` | none |
| Configured path missing/unsafe/unreadable | `unavailable` | `warning` | degrade |
| JSON missing/malformed/oversized/unsupported | `unknown` | `warning` | degrade |
| Valid JSON, `final_exit_code == 0`, no stale/build/sign skip warnings | `ready` | `info` | none |
| Valid JSON, `final_exit_code == 0`, stale/build/sign skip warnings | `degraded` | `warning` | degrade |
| Valid JSON, `final_exit_code != 0` | `action_required` | `error` | action required |

Recommended evidence fields:

- `configured: bool`
- `source: "host_smoke_evidence"`
- `evidence_dir`
- `schema_version`
- `created_at`
- `age_seconds`
- `smoke_run_id`
- `final_exit_code`
- `phases`
- `expected_files`
- `skip_flags`
- `reasons`

All dynamic values must be scalar-coerced and bounded before inclusion.

## Recommended Actions

If evidence is configured but unavailable or invalid, add a stable action:

```text
inspect_host_gated_evidence
```

If evidence reports a blocking non-zero final exit, add:

```text
run_host_gated_smoke
```

If evidence is stale or reports build/sign skip flags, add:

```text
review_expected_skips
```

Actions should point operators to existing docs or diagnostics where useful.
They must not invoke repair, cleanup, smoke, launchd, or helper lifecycle
operations.

## Staleness

If `created_at` is a valid timestamp, compute `age_seconds` using an injectable
clock in tests. Use a conservative default stale threshold such as 7 days for
classification, but keep it advisory:

- stale valid success evidence -> `degraded`
- stale valid failure evidence -> remains `action_required`
- malformed timestamp -> evidence `degraded` or `unknown` with a stable reason

Do not make stale evidence blocking in this slice.

## Skip Classification

The smoke wrapper records `skip_build`, `skip_sign`, and
`include_failure_drills`. Slice 2 should classify these conservatively:

- `skip_build=true` or `skip_sign=true` means the evidence is still useful but
  incomplete for a full prepared-host proof. Report `degraded` evidence and
  `review_expected_skips`.
- `include_failure_drills=false` is informational by default. The default smoke
  path does not run disruptive failure drills, so this should not degrade status
  unless a later strict/expected-evidence mode explicitly requires those drills.
- Non-boolean skip flag values should become parser reasons and should not be
  treated as truthy.

## Service Integration

`SandboxService.operator_status()` should collect evidence summary through a
small helper only after runtime and macOS diagnostics are gathered. Evidence
collection failure must be isolated to the evidence section, matching Slice 1
partial-failure behavior.

Expected service flow:

1. Gather runtime diagnostics.
2. Gather macOS diagnostics.
3. Read configured evidence summary if configured.
4. Pass all three inputs into `build_operator_status()`.

If evidence parsing raises an expected operational exception, convert it into a
stable section-local reason. Programming errors should continue to propagate
rather than being hidden by broad exception handling.

## API And Schema

No new endpoint is required. Slice 2 extends the existing response shape for:

```text
GET /api/v1/sandbox/admin/operator-status
```

The existing `SandboxAdminOperatorStatusSection` permits section-specific
fields. Add schema documentation or field comments where needed, but avoid
overfitting the response into many one-off Pydantic models until the shape has
stabilized.

## Testing Strategy

Normal CI remains portable.

Add unit tests for the parser:

- unset env returns not configured
- valid evidence bundle parses expected metadata
- embedded NUL path is rejected without crashing
- missing directory reports unavailable
- evidence directory symlink is rejected
- JSON symlink is rejected
- oversized JSON is rejected without reading contents
- malformed JSON does not leak raw JSON
- unsupported schema version reports a stable reason
- nested/raw container values are not exposed
- valid stale evidence computes age with an injectable clock

Add projection/service tests:

- unconfigured evidence does not degrade otherwise ready status
- configured invalid evidence degrades overall status
- successful evidence remains advisory and ready
- non-zero final exit produces evidence `action_required`
- build/sign skip flags produce degraded evidence and `review_expected_skips`
- `include_failure_drills=false` remains informational by default
- evidence parser failure does not hide runtime/macOS sections

Run focused tests:

```text
python -m pytest tldw_Server_API/tests/sandbox/test_operator_status.py \
  tldw_Server_API/tests/sandbox/test_operator_evidence.py \
  tldw_Server_API/tests/sandbox/test_admin_macos_diagnostics.py::test_admin_operator_status_returns_structured_payload \
  tldw_Server_API/tests/sandbox/test_admin_rbac.py::test_admin_endpoints_require_admin_role -q
```

Run Bandit on touched server files.

## Security And Privacy Review

- Keep endpoint admin-only.
- Do not add request path input.
- Do not follow evidence symlinks.
- Do not recursively walk directories.
- Do not create or mutate evidence files.
- Do not expose raw logs or arbitrary JSON keys.
- Keep evidence advisory and label it as such in docs.
- Treat path strings and JSON as untrusted external input.
- Keep strict evidence enforcement as a future separately reviewed design.

## Risks And Mitigations

- Risk: Evidence ingestion becomes arbitrary file probing.
  Mitigation: server-side env path only, no request path input, direct known
  child files only, no recursive scan.
- Risk: Operators confuse old smoke evidence with live readiness.
  Mitigation: include `created_at`, `age_seconds`, and advisory/stale status;
  keep live readiness in runtime and macOS sections.
- Risk: Server code duplicates too much CLI summarizer logic.
  Mitigation: implement only the small machine-readable parser needed by the
  API. If shared behavior grows, later refactor the summarizer and server to a
  common library.
- Risk: Evidence metadata leaks sensitive details.
  Mitigation: scalar allowlist only, bounded strings, no raw logs, no arbitrary
  JSON keys.

## Acceptance Criteria

- Spec clearly uses evidence bundle input, not Markdown summary scraping.
- Spec keeps evidence path configuration server-side/env-only for this slice.
- Spec defines fail-closed handling for unsafe paths, malformed JSON, oversized
  input, symlinks, and unsupported schema.
- Spec defines status/action behavior for unconfigured, invalid, successful,
  stale, skipped, and failed evidence.
- Spec preserves read-only/no-mutation operator-status boundaries.
