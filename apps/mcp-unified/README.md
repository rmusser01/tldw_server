# MCP Unified

MCP Unified is the standalone package boundary for the Model Context Protocol
runtime and gateway being extracted from `tldw-server`.

The package is currently internal/experimental. It is built and tested inside
the `tldw-server` repository, but it is not published as an independent PyPI
package yet. Treat this directory as the supported package-local integration
surface for early standalone gateway work.

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

- package status: `internal-experimental`
- publishing status: `not-published`
- license expression: `GPL-3.0-only`

The package ships `py.typed` as a PEP 561 marker so downstream type checkers can
recognize the inline type annotations when consuming built artifacts.

## Publishing Readiness

Standalone package publishing is prepared but still guarded. The package
metadata is PyPI-shaped, while the runtime metadata remains
`internal-experimental` and `not-published` until maintainers intentionally
promote it.

Run the full internal release candidate gate:

```bash
make mcp-unified-rc
```

Build artifacts and generate the TestPyPI upload plan without uploading:

```bash
make mcp-unified-publish-dry-run
```

Live TestPyPI or PyPI upload is not part of normal PR CI. It requires the manual
workflow, maintainer-owned credentials, an explicit confirmation input, and the
RC helper's publish opt-in guard.

## Install From This Repository

Use the package-local project file when testing the standalone boundary:

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

## Quick CLI Check

Validate the CLI is importable and can report package status:

```bash
mcp-unified-gateway package-info
```

List bundled profile presets:

```bash
mcp-unified-gateway list-presets
```

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
status (`internal-experimental`, `not-published`), runtime name/version,
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

`preview-profile-tools` previews a profile's effective tool surface across
installed tools and profile recommendations so operators can see which tools are
visible, deferred, blocked, or unavailable before assigning a profile. Pass a
`session_id` when previewing runtime-effective state that includes session-bound
approval grants.

Local CLI examples:

```bash
mcp-unified-gateway explain-policy --profile researcher --tool fs.patch \
  --args-json-file ./patch-args.json --config ./gateway.json

mcp-unified-gateway preview-profile-tools --profile researcher \
  --category filesystem --config ./gateway.json
```

Remote CLI example:

```bash
export MCP_UNIFIED_GATEWAY_URL=http://127.0.0.1:8000/mcp
export MCP_UNIFIED_GATEWAY_ADMIN_KEY=replace-with-admin-key

printf '{"path":"src/app.py"}' | mcp-unified-gateway explain-policy \
  --remote --profile researcher --tool fs.read --args-stdin

mcp-unified-gateway preview-profile-tools --remote --profile researcher \
  --category filesystem --session-id "$MCP_SESSION_ID" --exclude-denied
```

Admin API examples:

```bash
curl -sS -X POST "$MCP_UNIFIED_GATEWAY_URL/policy/explain" \
  -H "X-MCP-Gateway-Admin-Key: $MCP_UNIFIED_GATEWAY_ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"profile_id":"researcher","tool_name":"fs.read","arguments":{"path":"src/app.py"}}'

curl -sS -X POST "$MCP_UNIFIED_GATEWAY_URL/profiles/researcher/tool-preview" \
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
