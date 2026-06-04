# MCP Unified

MCP Unified is the standalone package boundary for the Model Context Protocol
runtime and gateway being extracted from `tldw-server`.

The package is currently internal/experimental. It is built and tested inside
the `tldw-server` repository, but it is not published as an independent PyPI
package yet. Treat this directory as the supported package-local integration
surface for early standalone gateway work.

## What Is Included

- JSON-RPC gateway runtime primitives for HTTP, WebSocket, and stdio entrypoints.
- Profile presets, profile resolution, and policy result models.
- Role presets with compact tooling discovery metadata and progressive
  disclosure categories for suggested next-step tools.
- Gateway-local profile, assignment, external-server, credential-grant, and
  audit storage interfaces.
- Optional SQLite-backed stores for standalone gateway configuration.
- External MCP server registry, runtime lifecycle, process policy, and transport
  helpers.
- A package CLI, `mcp-unified-gateway`, for local config management and remote
  gateway runtime operations.

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

## Install From This Repository

Use the package-local project file when testing the standalone boundary:

```bash
python -m pip install -e "mcp_unified[gateway]"
```

For test and packaging work:

```bash
python -m pip install -e "mcp_unified[dev]"
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

Validate a gateway config file:

```bash
mcp-unified-gateway validate-config ./gateway.json
```

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
  -c mcp_unified/pytest-artifact-gate.ini \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_distribution_metadata_matches_extras \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_sdist_contains_only_package_boundary \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_artifacts_include_typed_marker \
  .github/tests/test_mcp_unified_artifact_gate.py::test_mcp_unified_standalone_artifacts_include_package_docs \
  -q
```
