# Sandbox API — Quick Guide (Spec 1.0/1.1)

This guide summarizes the Sandbox (code interpreter) API with concise examples. The API supports spec 1.0 and 1.1. Version 1.1 is backward‑compatible and adds optional interactivity and resume features.

Base URL: `/api/v1/sandbox`

Auth: Standard tldw AuthNZ
- Single user: `X-API-KEY: <key>`
- Multi user (JWT): `Authorization: Bearer <token>`

## Runtime support contract

Runtime discovery and runtime preflight are authoritative for the current host.
Use GET `/api/v1/sandbox/runtimes` to discover what this deployment can admit
before choosing a runtime. The current support inventory is maintained in
`Docs/Sandbox/sandbox-runtime-capability-inventory.md`; the isolation and policy
contract is maintained in `Docs/Sandbox/sandbox-security-policy-matrix.md`.

Current runtime identities are `docker`, `firecracker`, `lima`, `vz_linux`,
`vz_macos`, `seatbelt`, and `worktree`. Availability does not imply a security
guarantee:
- Runtime discovery includes machine-readable `boundary_class`,
  `vm_grade_isolation`, `untrusted_eligible`, `isolation_warnings`, and
  `network_policy_contract` fields, plus `normalized_reason_details` and a
  `session_contract` object. Use those fields for client decisions instead of
  parsing prose notes.
- `isolation_warnings` are advisory metadata for client UX and operator
  context. They are not admission rejection reasons by themselves.
- `seatbelt` is host-local. `seatbelt` is not `untrusted`-eligible.
- `worktree` is host-local. `worktree` is not `untrusted`-eligible.
- `vz_macos` real execution is not implemented; it is a scaffold/preflight
  identity until the real runner lands.
- `firecracker`, `lima`, and `vz_linux` are host-gated VM-grade paths and must
  pass their own prerequisites before use.
- `untrusted` workloads require a VM-grade runtime that preflight admits. Do not
  substitute `seatbelt` or `worktree` when a VM-grade runtime is requested.

## Firecracker host prep

If you plan to use the Firecracker runtime, follow the host prerequisites and
smoke-test steps in `Docs/Deployment/Operations/Firecracker_Host_Checklist.md`.

## Lima runtime (macOS/Linux VMs)

Lima is a VM runtime identity backed by Virtualization.framework on macOS or
QEMU on Linux when host prerequisites and enforcement preflight pass.

### Requirements
- Install Lima: `brew install lima` (macOS) or via package manager
- Verify: `limactl version`

### Usage
```json
{
  "spec_version": "1.0",
  "runtime": "lima",
  "base_image": "ubuntu:24.04",
  "command": ["python3", "-c", "print('hello')"],
  "timeout_sec": 300
}
```

### Notes
- VMs use Virtualization.framework on macOS or QEMU on Linux
- Network isolation: `deny_all` only when the Lima enforcer preflight proves
  strict enforcement for this host
- Workspace mounted at `/workspace` inside VM
- Slower startup than containers (~10-30s vs ~1s for Docker)
- Recommended for macOS development when VM-grade Linux isolation is available
- Runtime parity: REST, MCP `sandbox.run`, and ACP use the current runtime enum;
  clients should confirm host support through runtime discovery before dispatch
- Strict fail-closed mode: Lima accepts `deny_all` only when enforcement is
  ready; `allowlist` execution is not supported today
- Platform constraint: Windows/WSL strict Lima enforcement is not supported yet and fails closed

## Trust-Level Tiers

Risk-based isolation profiles auto-apply resource limits based on code trustworthiness.

| Level | Max CPU | Max Memory | Timeout | Network | Use Case |
|-------|---------|------------|---------|---------|----------|
| `trusted` | 8 | 16GB | 600s | allowlist | Verified internal code |
| `standard` | 4 | 8GB | 300s | deny_all | Default for most runs |
| `untrusted` | 1 | 1GB | 60s | deny_all | User-submitted code |

### Session with trust level
```json
{
  "spec_version": "1.0",
  "runtime": "firecracker",
  "base_image": "python:3.11-slim",
  "trust_level": "untrusted"
}
```

### Run with trust level
```json
{
  "spec_version": "1.0",
  "runtime": "firecracker",
  "base_image": "python:3.11-slim",
  "command": ["python", "user_script.py"],
  "trust_level": "untrusted"
}
```

The examples use `firecracker` as a VM-grade runtime placeholder. Replace it
with a VM-grade runtime available on your host, such as `lima` or `vz_linux`
when their preflight checks pass.

When `untrusted` is specified and admitted, the run is automatically constrained to:
- Max 1 CPU
- Max 1GB memory
- Max 60s execution timeout
- Network deny_all (no egress)
- Max 64 PIDs
- Restricted file descriptors (256)

`seatbelt` and `worktree` are host-local runtimes and are rejected for
`untrusted` workloads.

## Feature discovery
GET `/api/v1/sandbox/runtimes`
Response (example):
```
{
  "runtimes": [
    {
      "name": "docker",
      "available": true,
      "reasons": [],
      "normalized_reasons": [],
      "normalized_reason_details": [],
      "default_images": ["python:3.11-slim", "node:20-alpine"],
      "max_cpu": 4.0,
      "max_mem_mb": 8192,
      "max_upload_mb": 64,
      "max_log_bytes": 10485760,
      "queue_max_length": 100,
      "queue_ttl_sec": 120,
      "workspace_cap_mb": 256,
      "artifact_ttl_hours": 24,
      "boundary_class": "container",
      "vm_grade_isolation": false,
      "untrusted_eligible": true,
      "isolation_warnings": [],
      "network_policy_contract": {
        "deny_all": {
          "support_state": "supported",
          "strict_enforcement": true,
          "readiness_source": "config"
        },
        "allowlist": {
          "support_state": "host_gated",
          "strict_enforcement": true,
          "readiness_source": "config"
        }
      },
      "session_contract": {
        "support_state": "supported",
        "reuse_model": "workspace_only",
        "requires_live_health_check": false,
        "recovery_state": "unsupported",
        "repair_state": "unsupported"
      },
      "supported_spec_versions": ["1.0", "1.1"],
      "interactive_supported": false,
      "egress_allowlist_supported": false,
      "store_mode": "memory"
    }
  ]
}
```
Runtime isolation fields mean:
- `reasons`: raw runtime preflight facts for operator diagnostics.
- `normalized_reasons`: stable reason codes derived from raw preflight facts.
- `normalized_reason_details`: structured metadata derived from
  `normalized_reasons`, including category, severity, availability-blocking
  posture, operator action, and message key.
- `boundary_class`: `container`, `host_local`, `vm_grade`, or
  `vm_grade_scaffold`.
- `vm_grade_isolation`: whether the runtime boundary is VM-grade for isolation
  claims, independent of current host availability.
- `untrusted_eligible`: whether policy may admit this runtime for `untrusted`
  workloads when preflight and host readiness also pass.
- `isolation_warnings`: advisory warning codes for client UX and operator
  context. These are not rejection reasons by themselves; admission remains
  governed by runtime preflight, policy, and request validation.
- `network_policy_contract`: static runtime posture for `deny_all` and
  `allowlist`. It describes whether each policy is supported, unsupported,
  scaffold-only, or host-gated, whether strict enforcement is possible, and
  where current readiness should be read from.
- `session_contract`: static runtime posture for session participation, reuse
  model, live health check expectation, and recovery/repair maturity. It does
  not replace `available`, runtime preflight, or admin diagnostics.

Current posture mapping:

| Runtime | `boundary_class` | `vm_grade_isolation` | `untrusted_eligible` | `isolation_warnings` |
| --- | --- | --- | --- | --- |
| `docker` | `container` | `false` | `true` | `[]` |
| `firecracker` | `vm_grade` | `true` | `true` | `[]` |
| `lima` | `vm_grade` | `true` | `true` | `[]` |
| `vz_linux` | `vm_grade` | `true` | `true` | `[]` |
| `vz_macos` | `vm_grade_scaffold` | `false` | `false` | `[]` |
| `seatbelt` | `host_local` | `false` | `false` | `host_local_boundary`, `not_vm_grade_isolation`, `not_untrusted_eligible` |
| `worktree` | `host_local` | `false` | `false` | `host_local_boundary`, `not_vm_grade_isolation`, `not_untrusted_eligible` |

Network policy contract mapping:

| Runtime | `deny_all` | `allowlist` |
| --- | --- | --- |
| `docker` | `supported`, strict, `config` readiness | `host_gated`, strict, `config` readiness |
| `firecracker` | `host_gated`, strict, `runtime_preflight` readiness | `scaffold`, not strict, `runtime_preflight` readiness |
| `lima` | `host_gated`, strict, `runtime_preflight` readiness | `unsupported`, not strict, `not_applicable` |
| `vz_linux` | `host_gated`, strict, `runtime_preflight` readiness | `unsupported`, not strict, `not_applicable` |
| `vz_macos` | `scaffold`, not strict, `runtime_preflight` readiness | `unsupported`, not strict, `not_applicable` |
| `seatbelt` | `unsupported`, not strict, `not_applicable` | `unsupported`, not strict, `not_applicable` |
| `worktree` | `unsupported`, not strict, `not_applicable` | `unsupported`, not strict, `not_applicable` |

`network_policy_contract` is static posture metadata. Current host readiness is
still reported through:

- `strict_deny_all_supported`
- `strict_allowlist_supported`
- `enforcement_ready` (object with `deny_all`/`allowlist`)
- `host` (host capability facts for troubleshooting)

Session contract mapping:

| Runtime | `support_state` | `reuse_model` | Live health check | `recovery_state` | `repair_state` |
| --- | --- | --- | --- | --- | --- |
| `docker` | `supported` | `workspace_only` | `false` | `unsupported` | `unsupported` |
| `firecracker` | `scaffold` | `scaffold` | `false` | `unsupported` | `unsupported` |
| `lima` | `scaffold` | `scaffold` | `false` | `unsupported` | `unsupported` |
| `vz_linux` | `host_gated` | `warm_vm` | `true` | `host_gated` | `host_gated` |
| `vz_macos` | `scaffold` | `scaffold` | `false` | `scaffold` | `scaffold` |
| `seatbelt` | `scaffold` | `workspace_only` | `false` | `unsupported` | `unsupported` |
| `worktree` | `scaffold` | `workspace_only` | `false` | `unsupported` | `unsupported` |

`session_contract` is static posture metadata. Same-session warm runtime reuse
is currently limited to the host-gated `vz_linux` path; host-local runtimes only
participate through workspace-oriented session inputs and do not provide warm
runtime reuse.

## Admin diagnostics and recovery

Admin diagnostics are operator surfaces. They are not substitutes for
client-facing runtime discovery, and diagnostics endpoints do not mutate runtime
state.

### Cross-runtime diagnostics

GET `/api/v1/sandbox/admin/runtime-diagnostics`

This admin-only, read-only endpoint derives from `/api/v1/sandbox/runtimes` and
groups every runtime by readiness posture. It preserves raw `reasons`, additive
`normalized_reasons`, `normalized_reason_details`, isolation warning metadata,
session reuse posture, and a `recommended_action` value for operator triage.

The response includes:

- `source`: currently `feature_discovery`.
- `summary`: counts for `total`, `ready`, `host_gated`, `scaffold`,
  `unavailable`, plus `host_local_warning_runtimes` and
  `repair_supported_runtimes`.
- `runtimes`: one row per runtime with readiness, isolation, network, session,
  repair, and recommended-action fields.
- `startup_warning_summary`: compact current-process startup warning status
  when the app startup-warning registry is present.

Repair support remains scoped to runtimes whose session contract explicitly
advertises it. Today that means `vz_linux`; this endpoint does not add generic
repair or reconciliation behavior for Docker, Firecracker, Lima, `seatbelt`,
`worktree`, or `vz_macos`.

### macOS diagnostics

GET `/api/v1/sandbox/admin/macos-diagnostics`

This admin-only, read-only endpoint exposes macOS/VZ operator details that are
intentionally omitted from public discovery:

- host readiness, including macOS and Apple silicon checks
- helper readiness, protocol, version, and transport metadata
- `vz_linux` and `vz_macos` template readiness
- runtime execution mode and remediation hints
- reconciliation between persisted `vz_linux` session controls and live helper
  VM state
- image-store correlation for templates, persisted run manifests, and dry-run GC
  candidate classification
- helper stdout/stderr log pointers, per-VM serial log pointers, guest-agent
  readiness metadata, and helper-provided resource counters when available
- read-only `recovery_summary` with status, severity, issue codes, counts,
  recommended action, and pointers to the existing dry-run-first admin actions
- `startup_warning_summary` for current-process sandbox startup warnings

Log diagnostics report paths, existence, and byte sizes only. They do not read
or return log file contents.

### Image-store cleanup

GET `/api/v1/sandbox/admin/macos-image-store/cleanup-plan`

Returns a read-only cleanup plan for image-store run directories and manifests,
including planned actions, live-match blockers, planning-only manifests,
inactive runs, and legacy run directories.

POST `/api/v1/sandbox/admin/macos-image-store/cleanup`

Runs the same plan through an explicit admin action. It defaults to
`dry_run=true`; unfiltered mutating cleanup requires `confirm_all=true`.

### VZ reconciliation repair

POST `/api/v1/sandbox/admin/macos-reconciliation/repair`

Runs explicit `vz_linux` repair actions. It defaults to dry-run, skips active
sessions, and can delete stale or unhealthy inactive persisted session-control
rows when requested. Orphan VM termination is never automatic: it requires
`terminate_orphaned_vms=true` and helper metadata proving the VM is owned by
this `tldw` `vz_linux` sandbox control plane.

Helper-unavailable and helper-protocol-mismatch conditions fail closed and block
mutating repair.

## Create a session
POST `/api/v1/sandbox/sessions`
Headers: `Idempotency-Key: <uuid>` (recommended)
Body (1.0):
```
{
  "spec_version": "1.0",
  "runtime": "docker",
  "base_image": "python:3.11-slim",
  "timeout_sec": 300
}
```
Response:
```
{ "id": "<session_id>", "runtime": "docker", "base_image": "python:3.11-slim", "expires_at": null, "policy_hash": "<hash>" }
```

## Start a run (one‑shot or session)
POST `/api/v1/sandbox/runs`
Headers: `Idempotency-Key: <uuid>` (recommended)
Body (1.0):
```
{
  "spec_version": "1.0",
  "runtime": "docker",
  "base_image": "python:3.11-slim",
  "command": ["python", "-c", "print('hello')"],
  "timeout_sec": 60
}
```
Body (1.1 additions — optional):
```
{
  "spec_version": "1.1",
  "runtime": "docker",
  "base_image": "python:3.11-slim",
  "command": ["python", "-c", "input(); print('ok')"],
  "timeout_sec": 60,
  "interactive": true,
  "stdin_max_bytes": 16384,
  "stdin_max_frame_bytes": 2048,
  "stdin_bps": 4096,
  "stdin_idle_timeout_sec": 30,
  "resume_from_seq": 100
}
```
Response (scaffold example):
```
{
  "id": "<run_id>",
  "spec_version": "1.1",
  "runtime": "docker",
  "base_image": "python:3.11-slim",
  "phase": "completed",
  "status_reason_code": "completed",
  "status_reason_details": {
    "code": "completed",
    "category": "success",
    "severity": "info",
    "terminal": true,
    "retryable": false,
    "operator_action": "none",
    "user_message_key": "sandbox.status.completed"
  },
  "exit_code": 0,
  "policy_hash": "<hash>",
  "log_stream_url": "ws://host/api/v1/sandbox/runs/<run_id>/stream?from_seq=100"
}
```

`status_reason_code` and `status_reason_details` are additive and derived from
existing status data. Clients should use the exact returned literals for stable
grouping: `queued`, `starting`, `running`, `completed`, `limits_applied`,
`nonzero_exit`, `policy_failed`, `runtime_unavailable`, `startup_timeout`,
`execution_timeout`, `canceled_by_user`, `killed`, `queue_ttl_expired`,
`runtime_error`, and `unknown`. Details add stable grouping metadata such as
`category`, `severity`, `terminal`, `retryable`, `operator_action`, and
`user_message_key`. Raw `phase`, `message`, and `exit_code` remain available
for display and operator diagnostics.

## Stream logs (WebSocket)
WS `/api/v1/sandbox/runs/{id}/stream`
- Optional query: `from_seq=<N>` (1.1 resume)
- When signed URLs are enabled, include `token` and `exp` query params.
Frames:
- `{ "type": "event", "event": "start" }`
- `{ "type": "stdout"|"stderr", "encoding": "utf8"|"base64", "data": "...", "seq": 123 }`
- `{ "type": "heartbeat", "seq": 124 }`
- `{ "type": "truncated", "reason": "log_cap", "seq": 125 }`
- `{ "type": "event", "event": "end", "data": {"exit_code": 0}, "seq": 126 }`
- Interactivity (1.1): client→server stdin frames `{ "type": "stdin", "encoding": "utf8"|"base64", "data": "..." }`

## Artifacts
- List: GET `/api/v1/sandbox/runs/{id}/artifacts`
- Download: GET `/api/v1/sandbox/runs/{id}/artifacts/{path}`
  - Supports single HTTP Range only. Use `Range: bytes=start-end` or suffix `bytes=-N`.
  - Multiple ranges are not supported; the server returns `416 Range Not Satisfiable` with `Content-Range: bytes */<size>`.
  - Responses include `Accept-Ranges: bytes`. A valid partial response includes `206 Partial Content` and `Content-Range: bytes <start>-<end>/<size>`.

Example:
```
# First 5 bytes
GET /api/v1/sandbox/runs/<id>/artifacts/out.txt
Range: bytes=0-4

HTTP/1.1 206 Partial Content
Accept-Ranges: bytes
Content-Range: bytes 0-4/10
Content-Length: 5

01234

# Unsupported multi-range
GET /api/v1/sandbox/runs/<id>/artifacts/out.txt
Range: bytes=0-1,3-4

HTTP/1.1 416 Range Not Satisfiable
Content-Range: bytes */10
```

## Idempotency conflicts
409, example:
```
{
  "error": {
    "code": "idempotency_conflict",
    "message": "Idempotency-Key replay with different body",
    "details": { "prior_id": "<id>", "key": "<Idempotency-Key>", "prior_created_at": "<ISO8601>" }
  }
}
```

## Strict Lima failure contracts
- `503 runtime_unavailable` when `limactl`/runtime is unavailable or strict host enforcement permissions are unavailable. For explicit `runtime=lima`, `error.details.suggested` is an empty list (no fallback).
  - `error.details.reasons` includes provider preflight reasons (for example `limactl_missing` or `permission_denied_host_enforcement`).
- `422 policy_unsupported` when strict requirements cannot be proven (for example `strict_allowlist_not_supported` or unsupported `network_policy`).

Example `422`:
```
{
  "error": {
    "code": "policy_unsupported",
    "message": "Runtime 'lima' does not satisfy requirement 'allowlist'",
    "details": {
      "runtime": "lima",
      "requirement": "allowlist",
      "reasons": ["strict_allowlist_not_supported"]
    }
  }
}
```

## Health
- Authenticated: GET `/api/v1/sandbox/health` (includes store timings and Redis ping)
- Public: GET `/api/v1/sandbox/health/public` (no auth)

## Egress Policy and DNS Pinning

Some deployments enforce an egress allowlist for sandboxed runs. The Docker runner supports a deny‑all baseline (network=none) and, when enabled, a granular host‑level allowlist using iptables on the DOCKER-USER chain.

Utilities exposed in `tldw_Server_API.app.core.Sandbox.network_policy` help you prepare and manage rules:

- `expand_allowlist_to_targets(raw_allowlist, resolver=..., wildcard_subdomains=("", "www", "api"))`
  - Accepts a mix of CIDR (e.g., `10.0.0.0/8`), literal IPs (`8.8.8.8`), hostnames (`example.com`), wildcard prefixes (`*.example.com`), and suffix tokens (`.example.com`).
  - Resolves hostnames to A records and promotes to `/32`; returns a de‑duplicated list like `['1.2.3.4/32', '10.0.0.0/8']`.

- `pin_dns_map(raw_allowlist, resolver=...)`
  - Returns a mapping `{ host -> [IPs] }` after resolution for observability/debugging.

- `refresh_egress_rules(container_ip, raw_allowlist, label, resolver=..., wildcard_subdomains=...)`
  - Best‑effort revocation + re‑apply: deletes all rules in DOCKER‑USER containing `label` and applies an updated set of `ACCEPT` rules for resolved targets, followed by a final `DROP` for the container IP.

Examples:
```
from tldw_Server_API.app.core.Sandbox.network_policy import (
    expand_allowlist_to_targets, pin_dns_map, refresh_egress_rules
)

# Allowlist with CIDR, IP, wildcard and suffix tokens
raw = ["10.0.0.0/8", "8.8.8.8", "*.example.com", ".example.org"]
targets = expand_allowlist_to_targets(raw)
# e.g., ['10.0.0.0/8', '8.8.8.8/32', '93.184.216.34/32', ...]

# Inspect pinned DNS map (for logs/metrics)
pins = pin_dns_map(raw)
# e.g., {'example.com': ['93.184.216.34', ...], 'example.org': ['203.0.113.10', ...]}

# Apply (or refresh) rules for a given container
apply_specs = refresh_egress_rules(
    container_ip="172.18.0.2",
    raw_allowlist=raw,
    label="tldw-run-<short-id>",
)
```

Notes:
- Suffix tokens (like `.example.com`) behave like wildcards for a few common subdomains plus the apex (configurable).
- If `iptables-restore` is unavailable, the code falls back to iterative `iptables` commands.
- To revoke rules for a finished container, the runner labels and deletes rules by that label.

## Snapshots and Cloning

Save session state and create copies for experimentation.

### Create Snapshot
POST `/api/v1/sandbox/sessions/{id}/snapshot`

Response:
```json
{
  "snapshot_id": "snap-abc123def456",
  "created_at": "2026-01-31T12:00:00Z",
  "size_bytes": 1048576
}
```

### List Snapshots
GET `/api/v1/sandbox/sessions/{id}/snapshots`

Response:
```json
{
  "items": [
    {
      "snapshot_id": "snap-abc123def456",
      "session_id": "sess-xyz789",
      "created_at": "2026-01-31T12:00:00Z",
      "size_bytes": 1048576
    }
  ]
}
```

### Restore Snapshot
POST `/api/v1/sandbox/sessions/{id}/restore`

Body:
```json
{ "snapshot_id": "snap-abc123def456" }
```

Response:
```json
{
  "restored": true,
  "snapshot_id": "snap-abc123def456"
}
```

### Clone Session
POST `/api/v1/sandbox/sessions/{id}/clone`

Body (optional):
```json
{ "new_session_name": "my-experiment" }
```

Response:
```json
{
  "session_id": "new-session-id",
  "cloned_from": "original-session-id"
}
```

### Delete Snapshot
DELETE `/api/v1/sandbox/sessions/{id}/snapshots/{snapshot_id}`

Response:
```json
{ "ok": true, "snapshot_id": "snap-abc123def456" }
```

### Use Cases
- **Safe experimentation**: Create a snapshot before making changes, restore if something breaks
- **Parallel exploration**: Clone a session to try multiple approaches simultaneously
- **State preservation**: Save workspace state across restarts or long-running investigations

## Notes
- Spec versions are validated against server config. Default: `["1.0","1.1"]`.
- Interactivity requires runtime and policy support; fields are ignored otherwise.
- `log_stream_url` may be unsigned; prefer Authorization headers if signed URLs are disabled.
