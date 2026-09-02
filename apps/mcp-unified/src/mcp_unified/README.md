# MCP Unified

MCP Unified is the standalone package boundary for the Model Context Protocol
runtime and gateway being extracted from `tldw-server`.

The package status is `public-alpha`, and the publishing status is `published`.
Released versions are published on PyPI; repository versions remain release
candidates until their protected publish succeeds. The package is built and
tested inside the `tldw-server` repository; the
former internal/experimental phase remains relevant only to earlier releases.

This package does not currently ship an end-user standalone gateway server
launcher. `mcp-unified-gateway` commands manage local configuration or talk to
an already mounted remote gateway supplied by a host application.

## What Is Included

- JSON-RPC gateway runtime primitives for HTTP, WebSocket, and stdio entrypoints.
- Profile presets, profile resolution, and policy result models.
- Claude-style profile permission rule parsing for tool, command, path, domain,
  external MCP, skill, and agent subjects.
- Role presets with compact tooling discovery metadata and progressive
  disclosure categories for suggested next-step tools.
- Gateway-local profile, assignment, external-server, credential-grant, and
  audit storage interfaces.
- Optional SQLite-backed stores for standalone gateway configuration.
- External MCP server registry, runtime lifecycle, process policy, and transport
  helpers.
- Metadata-only tool-use reporting for aggregate profile, model, tool, and
  prompt-version analysis.
- Configurable package-level tool-call hook manager primitives for embedders
  that need ordered pre/post policy, approval, or audit hooks.
- Package-local filesystem advisory lock backends for coordinating
  read-before-mutate workflows. The memory backend is the default.
  An optional SQLite backend can coordinate cooperating processes that point at
  the same local database file.
- Package CLIs: `mcp-unified-gateway` for local config management and remote
  gateway runtime operations, plus `mcp-unified-smoke` for JSON-RPC gateway
  smoke validation.

For hands-on setup and operations, see [USER_GUIDE.md](USER_GUIDE.md).

## Release Status

Inspect the current package metadata:

```bash
mcp-unified-gateway package-info
```

Expected current status:

- package status: `public-alpha`
- publishing status: `published`
- license expression: `GPL-3.0-only`

The package ships `py.typed` as a PEP 561 marker so downstream type checkers can
recognize the inline type annotations when consuming built artifacts.

## Publishing Readiness

Standalone package publishing is live but guarded. Package metadata reports
`public-alpha` and publishing state `published`; repository builds remain
release candidates until their protected publish succeeds.

Run the full internal release candidate gate:

```bash
make mcp-unified-rc
```

For each clean wheel and sdist environment, that gate also installs the exact
official Tier 1 Python SDK pin `mcp==2.0.0` and exercises automatic strict
stdio negotiation at `2026-07-28`, tool discovery, and one tool call. The pin
is the official Python SDK
[`v2.0.0`](https://github.com/modelcontextprotocol/python-sdk/releases/tag/v2.0.0)
release at tag commit `6f69a37`. This is explicit stdio interoperability
evidence, not a claim of full transport conformance: the official conformance
server harness is URL-oriented, and this package does not add a modern HTTP
transport for that harness.

Build artifacts and generate the TestPyPI upload plan without uploading:

```bash
make mcp-unified-publish-dry-run
```

Merging a package version bump to `main` triggers the guarded PyPI publishing
workflow. The workflow runs the RC gate, verifies the version is not already on
PyPI, and then uses the repository's configured trusted publishing environment
rather than a long-lived PyPI token.

Manual TestPyPI and PyPI workflow dispatch remain available for release
rehearsals and operator-driven publishes. Manual live uploads require an
explicit confirmation input and the RC helper's publish opt-in guard.

## Install From PyPI

Install the published package boundary with the gateway extras:

```bash
python -m pip install "mcp-unified[gateway]"
```

Downstream applications should use a compatible-minor pin:

```bash
python -m pip install "mcp-unified[gateway]~=0.3.0"
```

For development tooling, install the optional development extras:

```bash
python -m pip install "mcp-unified[gateway,dev]"
```

## Install From This Repository

Use the package-local project file when testing unpublished repository changes:

```bash
python -m pip install -e "apps/mcp-unified[gateway]"
```

For test and packaging work, install both the gateway runtime and development
tools:

```bash
python -m pip install -e "apps/mcp-unified[gateway,dev]"
```

The package dependency groups intentionally stay small. Heavy `tldw-server`
runtime stacks such as media ingestion, transcription, RAG, and WebUI
dependencies are outside this package boundary.

## Strict Stdio Protocol

The public `mcp_unified.gateway` API implements five pinned MCP revisions over
newline-delimited binary stdio:

| Revision | Lifecycle | Batch requests |
| --- | --- | --- |
| `2026-07-28` | Per-request `_meta`; no initialize session | Rejected |
| `2025-11-25` | `initialize`, then operations | Rejected |
| `2025-06-18` | `initialize`, then operations | Rejected |
| `2025-03-26` | Standalone `initialize`, then operations | Accepted only after initialization |
| `2024-11-05` | `initialize`, then operations | Rejected |

The strict surface owns revision negotiation, validation, projection,
pagination, cancellation, limits, and stdio framing. Existing HTTP/WebSocket
routes are compatibility surfaces with their existing contracts; this release
does not claim modern MCP conformance for HTTP.

Embed strict stdio with caller-owned binary streams, or omit them to use the
process binary adapters:

```python
import asyncio

from mcp_unified.gateway import GatewayLimits, serve_stdio

raise SystemExit(
    asyncio.run(
        serve_stdio(runtime, limits=GatewayLimits(max_in_flight=1))
    )
)
```

The injected runtime and host application own catalogs, authorization, policy,
audit, local files and databases, content, and privacy decisions. The protocol
layer does not expose or duplicate application-local data and never treats
self-reported client identity as authorization.

`GatewayLimits` has these exact defaults:

| Limit | Default | Limit | Default |
| --- | ---: | --- | ---: |
| `max_input_line_bytes` | 1,048,576 | `max_output_line_bytes` | 1,048,576 |
| `max_result_bytes` | 786,432 | `max_json_depth` | 64 |
| `max_in_flight` | 16 | `default_catalog_page_size` | 50 |
| `max_catalog_page_size` | 100 | `max_catalog_items` | 10,000 |
| `max_batch_items` | 100 | `max_requests_per_minute` | 600 |
| `request_burst` | 32 | `max_schema_bytes` | 262,144 |
| `max_schema_depth` | 32 | `max_schema_subschemas` | 1,024 |
| `max_schema_refs` | 256 | `max_schema_pattern_chars` | 4,096 |
| `max_schema_validation_processes` | 4 | `schema_validation_timeout_seconds` | 5.0 |
| `graceful_shutdown_timeout_seconds` | 5.0 | | |

Schema compilation and instance validation run in disposable bounded child
processes. On native Windows, the preflighted schema and complete validation
instance are briefly stored in an owner-only file in the operating-system
temporary directory so the nested stdio server can launch the child reliably.
The file is never logged, is removed during the same bounded child cleanup,
and is not retained after success, failure, timeout, cancellation, or shutdown.
Applications handling data that must never touch temporary storage should
account for this Windows behavior before enabling strict tool calls.

Modern responses use conservative private cache hints
`{"ttlMs": 0, "cacheScope": "private"}`; legacy projections omit modern cache
fields. Errors expose only stable, allowlisted classifications and safe limit
metadata, never raw payloads, paths, credentials, schemas, exception strings,
or private result sizes. If an oversized response cannot fit, the fixed generic
internal-error line is exactly 79 bytes including its newline: an output limit
of 79 emits that one line, while 78 emits nothing rather than truncating data.

Cancellation stops pending asynchronous work and propagates request
cancellation to the runtime. Shutdown is bounded by
`graceful_shutdown_timeout_seconds` and reports incomplete input, output, or
cleanup work on stderr without corrupting protocol stdout. Python cannot kill a
non-returning worker thread; hosts must bound synchronous work, and clients must
escalate from stream close to process terminate and then kill when a child does
not exit within its grace period.

## Quick CLI Check

Validate the CLI is importable and can report package status:

```bash
mcp-unified-gateway package-info
```

List bundled profile presets:

```bash
mcp-unified-gateway list-presets
```

Option A: duplicate a preset, then preview the new stored profile:

```bash
mcp-unified-gateway duplicate-preset project-researcher \
  --profile-id <new-profile-id> --config ./gateway.json

mcp-unified-gateway preview-profile-tools --profile <new-profile-id> \
  --config ./gateway.json
```

Option B: create a profile from JSON, then preview the ID declared inside that
file:

```bash
mcp-unified-gateway create-profile --profile-file ./profile.json \
  --config ./gateway.json

mcp-unified-gateway preview-profile-tools --profile <profile-id-from-json> \
  --config ./gateway.json
```

For a minimal custom profile JSON template and the recommended discovery flow,
see [USER_GUIDE.md](USER_GUIDE.md#3-work-with-profiles).

Run the deterministic in-process smoke scenario:

```bash
mcp-unified-smoke inprocess --json-report -
```

Validate a gateway config file:

```bash
mcp-unified-gateway validate-config ./gateway.json
```

## Package-Local Status

When a host application mounts the package gateway, `GET /mcp/status` returns
best-effort readiness metadata for that package-local mount. It includes package
status (`public-alpha`, `published`), runtime name/version,
profile store persistence, default profile state, admin-auth configured state,
external server counts, warnings, and next actions. It is not the embedded TLDW
Server status endpoint; embedded users should call `/api/v1/mcp/status`.

Build an aggregate tool-use report when reporting is enabled:

```bash
mcp-unified-gateway tool-events report --group-by profile --config ./gateway.json
```

## Policy Explanation

`explain-policy` explains one profile/tool decision before execution. It reports
the effective `allow`, `ask`, or `deny` outcome, reason code, contributing
policy state, and redacted subjects for a hypothetical tool call.
It does not execute filesystem tools or fully validate authored
`policy_document.path_grants`; verify those with safe runtime tool calls against
representative allowed and denied paths.

`preview-profile-tools` previews a profile's effective tool surface across
installed tools and profile recommendations so operators can see which tools are
visible, deferred, blocked, or unavailable before assigning a profile. Pass a
`session_id` when previewing runtime-effective state that includes session-bound
approval grants.

Local CLI examples:

```bash
mcp-unified-gateway explain-policy --profile <profile-id> --tool fs.patch \
  --args-json-file ./patch-args.json --config ./gateway.json

mcp-unified-gateway preview-profile-tools --profile <profile-id> \
  --category filesystem --config ./gateway.json
```

Remote CLI example:

```bash
export MCP_UNIFIED_GATEWAY_URL=http://127.0.0.1:8000/mcp
export MCP_UNIFIED_GATEWAY_ADMIN_KEY=replace-with-admin-key

echo '{"path":"src/app.py"}' | mcp-unified-gateway explain-policy \
  --remote --profile <profile-id> --tool fs.read --args-stdin

mcp-unified-gateway preview-profile-tools --remote --profile <profile-id> \
  --category filesystem --session-id "$MCP_SESSION_ID" --exclude-denied
```

Admin API examples:

```bash
curl -sS -X POST "$MCP_UNIFIED_GATEWAY_URL/policy/explain" \
  -H "X-MCP-Gateway-Admin-Key: $MCP_UNIFIED_GATEWAY_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"<profile-id>","tool_name":"fs.read","arguments":{"path":"src/app.py"}}'

curl -sS -X POST "$MCP_UNIFIED_GATEWAY_URL/profiles/<profile-id>/tool-preview" \
  -H "X-MCP-Gateway-Admin-Key: $MCP_UNIFIED_GATEWAY_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"category":"filesystem","include_denied":true,"session_id":"session-1"}'
```

Policy explanation and preview calls are audited when audit storage is
configured, and responses redact or sanitize sensitive subjects. Raw tool
arguments are not echoed back. Prefer `--args-json-file` or `--args-stdin` over
inline `--args-json` for sensitive arguments so values are not exposed in shell
history or process listings.

## Minimal Gateway Config

```json
{
  "store": {
    "kind": "sqlite",
    "sqlite_path": "./mcp-gateway.db"
  },
  "default_preset_id": "project-researcher"
}
```

Save this as `gateway.json`, then use:

```bash
mcp-unified-gateway validate-config ./gateway.json
```

## Tool-Use Reporting

Tool-use reporting is disabled by default. When enabled, the gateway records
metadata about attempted tool calls so operators can compare how profiles,
models, modes, and tool prompt ids perform over time. Reports expose aggregate
counts, success rates, top reason codes, and latency percentiles.

Reporting deliberately avoids tool arguments, tool result payloads, secret
values, raw exception text, and conversation content. Use a SQLite reporting
store for CLI reporting, export, and cleanup commands.

```json
{
  "store": {
    "kind": "sqlite",
    "sqlite_path": "./mcp-gateway.db"
  },
  "default_preset_id": "project-researcher",
  "tool_use_reporting": {
    "enabled": true,
    "store": {
      "kind": "sqlite",
      "sqlite_path": "./mcp-tool-events.db"
    },
    "retention_max_age_days": 30,
    "retention_max_events": 100000
  }
}
```

See [USER_GUIDE.md](USER_GUIDE.md) for report, export, cleanup, privacy, and
future evaluation workflow details.

## Tool-Call Hooks

The package includes a host-neutral `ConfiguredToolCallHookManager` for
embedding pre/post tool-call hooks through `MCPRuntimeDependencies`. Pre-hooks
run in configured order and stop at the first `deny`, `ask`, or
`approval_required` decision. Post-hooks run after tool completion and continue
after individual post-hook failures so the original tool result or error is
preserved.

```python
from mcp_unified.tool_hooks import (
    ConfiguredToolCallHookManager,
    ToolHookRegistration,
)

hook_manager = ConfiguredToolCallHookManager(
    [
        ToolHookRegistration(
            hook_id="profile-policy",
            before=check_profile_policy,
            after=record_profile_observation,
            order=10,
        )
    ]
)
```

Hook summaries are metadata-only and can be attached to tool-use reporting
events when reporting is enabled. Gateway JSON/admin configuration for hook
registries is intentionally left to a later surface; this slice provides the
package API and protocol/reporting integration.

## Documentation

- [USER_GUIDE.md](USER_GUIDE.md) - package-local user and operator guide.
- `Docs/MCP_UNIFIED_STANDALONE_GATEWAY_ADMIN.md` - repository-level admin and
  release-gate notes.
- `Docs/MCP/Unified/` - `tldw-server` MCP Unified host documentation.

## Local Verification

Run the package boundary and CLI tests:

```bash
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py \
  -q
```

Run the isolated artifact gate used by CI:

```bash
python -m pytest \
  -c apps/mcp-unified/pytest-artifact-gate.ini \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_distribution_metadata_matches_extras \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_sdist_contains_only_package_boundary \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_artifacts_include_typed_marker \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_artifacts_include_package_docs \
  -q
```
