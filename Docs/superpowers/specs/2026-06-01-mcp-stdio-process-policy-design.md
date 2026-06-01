# MCP Unified Stdio Process Policy Design

Date: 2026-06-01
Status: Revised after design review
Backlog: TASK-586

## Summary

Add an explicit process-execution policy layer for package-owned upstream stdio
MCP transports. The package can now launch real external MCP server processes,
so the next hardening step is to make executable, working-directory, PATH, and
environment inheritance decisions deliberate and testable before adding real
third-party install/update execution.

The policy should live in `mcp_unified`, apply before any subprocess spawn, and
return deterministic safe reason codes when a configured external server
violates deployment policy. This slice should not add package-manager behavior,
daemon control, or WebUI changes.

## Goals

- Enforce configurable executable allowlists before `asyncio.create_subprocess_exec`.
- Support bounded canonical cwd validation for configured external stdio
  servers.
- Restrict inherited environment variable names through a runtime policy in
  addition to each server's `env_allowlist`.
- Make PATH lookup explicit and reject it when deployment policy disables it.
- Reject shell wrapper executables by default unless a host deliberately allows
  them.
- Treat Windows and POSIX executable path comparisons consistently enough that
  shell detection and cwd-root checks do not depend on spelling differences.
- Keep policy failures secret-safe in exceptions, logs, status payloads, and
  audit events.
- Preserve the `mcp_unified` package boundary with no imports from
  `tldw_Server_API`.
- Add focused regression tests for denied and allowed process-policy cases.

## Non-Goals

- No package-manager, marketplace, or automatic install/update execution.
- No durable daemon or remote control client for already-running gateways.
- No per-server persisted process-policy schema in
  `ExternalServerDefinition`.
- No shell parsing, command expansion, glob expansion, or variable expansion.
- No WebSocket upstream transport policy in this slice.
- No resource limits such as CPU, memory, or process groups. Those can be added
  later as another policy dimension.

## Current Foundation

The current package code already provides:

- `mcp_unified.federation.stdio_transport.StdioExternalTransport`
- `mcp_unified.federation.stdio_transport.create_external_transport`
- `mcp_unified.gateway.external_runtime.GatewayExternalRuntimeManager`
- `mcp_unified.gateway.config.GatewayExternalRuntimeBootstrapConfig`
- `mcp_unified.storage.models.ExternalServerDefinition`

The stdio transport already uses `asyncio.create_subprocess_exec` without a
shell, validates non-empty commands, validates configured cwd existence, and
builds the child environment only from `ExternalServerDefinition.env_allowlist`.
The remaining gap is that these checks are mostly syntactic. They do not let a
deployment bound which executables, cwd roots, PATH lookup behavior, or env names
are acceptable.

## Approach Options

### Option A: Runtime Process Policy Object

Add a package-owned policy object, pass it into the stdio transport factory, and
wire it through gateway bootstrap config. The registry continues storing only
server definitions. Deployment owners configure process constraints once at
runtime.

Tradeoff: one more config surface, but it keeps mutable server records separate
from host-level execution policy and avoids schema churn.

### Option B: Persist Policy Per External Server

Add policy fields directly to `ExternalServerDefinition`, store them in SQLite,
and let each server carry its own execution constraints.

Tradeoff: this gives granular control but lets editable registry records modify
their own guardrails unless every mutation path learns new approval semantics.
It also widens the storage schema for a policy that is better treated as a host
deployment boundary.

### Option C: Hardcode Safer Defaults Only

Reject obvious shell wrappers and require absolute executable paths without
adding a configurable policy model.

Tradeoff: simple, but too rigid for real MCP deployments that commonly use
package shims such as `node`, `python`, or `npx`. It also leaves no way for host
applications to express their actual trust boundary.

## Recommended Approach

Use Option A.

This makes the trust boundary explicit while keeping the work narrow. The
runtime policy is injected into package factories and config bootstrap. Server
definitions remain data records; policy remains a deployment control.

## Policy Model

Add the policy and coercion helpers in a sibling module,
`mcp_unified.federation.process_policy`, instead of growing
`stdio_transport.py` further. Re-export the public policy model from
`mcp_unified.federation` for callers that already import federation utilities
from the package root.

```python
@dataclass(frozen=True, slots=True)
class StdioProcessPolicy:
    allowed_executables: tuple[str, ...] = ()
    allowed_cwd_roots: tuple[str | Path, ...] = ()
    allowed_env_names: tuple[str, ...] | None = None
    allow_path_lookup: bool = True
    reject_shell_executables: bool = True
    default_cwd: str | Path | None = None
```

The module should provide a single normalization boundary, for example
`coerce_stdio_process_policy(value)`, that accepts `None`,
`StdioProcessPolicy`, or a mapping loaded from JSON/TOML. It should reject
blank strings, non-string list entries, invalid booleans, and invalid path
values with `ValueError` before runtime bootstrap succeeds. Normalization should
copy caller data and store immutable tuples.

Semantics:

- `allowed_executables`: empty means no allowlist restriction. Entries may be
  executable names such as `python` or canonical absolute paths such as
  `/usr/bin/python3`. Entries with path separators are expanded and resolved
  before comparison; bare-name entries match the normalized executable basename.
  Bare-name allowlists still trust the parent `PATH`, so strict deployments
  should use absolute executable paths and set `allow_path_lookup=false`.
- `allowed_cwd_roots`: empty means cwd only needs to exist. When non-empty, the
  resolved cwd must be under one of the resolved roots. Root and cwd comparison
  should use resolved canonical paths, with platform-appropriate case
  normalization where needed.
- `allowed_env_names`: `None` means the server's `env_allowlist` is the only env
  name filter. A tuple means the server may only request names also present in
  the policy list. An empty tuple means no inherited environment names are
  allowed.
- `allow_path_lookup`: when false, command executables without a path separator
  are rejected. This is stricter than the current PATH allowlist requirement.
- `reject_shell_executables`: when true, common shell wrappers such as `sh`,
  `bash`, `zsh`, `fish`, `cmd`, `cmd.exe`, `powershell`, `powershell.exe`,
  `pwsh`, and `pwsh.exe` are rejected unless the executable is explicitly listed
  in `allowed_executables`.
- `default_cwd`: optional runtime cwd used when a server does not configure
  `cwd`. It is resolved and checked against `allowed_cwd_roots`. If
  `allowed_cwd_roots` is configured and neither the server nor policy provides a
  cwd, transport construction should fail instead of silently inheriting the
  process cwd.

The default policy preserves most current behavior while blocking shell wrappers.
Hosts that need stricter behavior can set `allow_path_lookup=false`,
absolute-path `allowed_executables`, and `allowed_cwd_roots`.

## Transport Enforcement

`StdioExternalTransport` should accept `process_policy: StdioProcessPolicy | Mapping[str, Any] | None`.
During construction it should:

- normalize and copy the policy
- validate `server.transport == "stdio"`
- validate and normalize `server.command`
- apply executable policy before spawn
- resolve the effective cwd using `server.cwd` or `policy.default_cwd`
- ensure cwd exists and satisfies `allowed_cwd_roots` when configured
- detect shell wrappers by normalized basename so `bash`, `/bin/bash`, `cmd.exe`,
  and `C:\Windows\System32\cmd.exe` are covered
- validate requested env names against `policy.allowed_env_names`
- keep child env construction limited to the intersection of server allowlist,
  policy allowlist when configured, and current `os.environ`

PATH lookup rules should remain explicit:

- if the command executable is a bare name and `allow_path_lookup=false`, reject
  it before checking `PATH`
- if the command executable is a bare name and `allow_path_lookup=true`,
  `PATH` must be present in `server.env_allowlist`
- if `policy.allowed_env_names` is configured, `PATH` must also be present in
  that policy allowlist before a bare-name executable may be launched
- if the command executable contains a path separator, compare it to executable
  allowlist entries using a resolved path. Relative executable paths should be
  resolved against the effective cwd when one exists, otherwise against the
  current process cwd only for validation.

Policy denials should raise `StdioExternalTransportError` with safe reason
codes:

- `process_policy_executable_denied`
- `process_policy_shell_denied`
- `process_policy_path_lookup_denied`
- `process_policy_cwd_denied`
- `process_policy_env_denied`

Error `details` may include safe scalar data such as `server_id`, `field`,
`executable_name`, or denied env names. It must not include full command arrays,
environment values, credential values, request bodies, or stderr contents.

## Gateway Config Wiring

Extend `GatewayExternalRuntimeBootstrapConfig` with:

```python
process_policy: StdioProcessPolicy | Mapping[str, Any] | None = None
```

`external_runtime_manager_from_storage()` should accept the same policy and
wrap the transport factory only when a non-`None` config policy is supplied and
the package-owned stdio factory is being used:

```python
def factory(server: ExternalServerDefinition) -> ExternalFederationTransport:
    return create_external_transport(server, process_policy=policy)
```

When no config policy is supplied, keep the existing factory identity
(`manager._transport_factory is create_external_transport`) so current tests and
callers that introspect the default factory remain compatible. The default
`create_external_transport(server)` path should still apply the default
`StdioProcessPolicy`, including shell-wrapper rejection.

If a caller supplies a custom `external_transport_factory`, the package should
not force this stdio policy into it. Custom factories own their own enforcement
boundary. The config validator should still parse and validate policy so bad
gateway config fails early.

The CLI `validate-config` output should include a compact `process_policy`
summary with booleans/counts, not raw paths if that risks leaking deployment
layout. A practical shape is:

```json
{
  "external_runtime": {
    "process_policy": {
      "configured": true,
      "allowed_executables": 2,
      "allowed_cwd_roots": 1,
      "allowed_env_names": 3,
      "allow_path_lookup": false,
      "reject_shell_executables": true,
      "default_cwd": true
    }
  }
}
```

## Data Flow

1. A host loads gateway config from JSON or TOML.
2. `GatewayExternalRuntimeBootstrapConfig` coerces and validates process-policy
   values with `coerce_stdio_process_policy()`.
3. `bootstrap_profile_gateway_from_config()` builds the external runtime
   manager. If config provided a policy, the default stdio transport factory
   closes over that policy; otherwise the existing default factory is reused.
4. `GatewayExternalRuntimeManager.start_server()` loads an
   `ExternalServerDefinition`.
5. The factory creates `StdioExternalTransport(server, process_policy=policy)`.
6. The transport applies process policy before any subprocess is spawned.
7. Allowed transports continue to initialize, discover tools, and execute calls.
8. Denied transports surface safe reason codes through existing lifecycle error
   handling.

## Error Handling

Policy violations should fail before process creation. They should be treated
like other transport start failures by the runtime manager:

- `start_server()` returns or raises existing `external_server_start_failed`
  envelopes at the gateway boundary.
- `_last_errors` may include a safe exception summary with the policy reason
  code.
- Audit payloads should record the high-level failure reason only.
- Logs should not include command arrays or env values.

The transport may include the policy-specific reason code in the chained
exception. The public gateway reason can remain the existing start-failed reason
to preserve API compatibility.

## Tests

Add or extend focused tests under
`tldw_Server_API/app/core/MCP_unified/tests/test_stdio_external_transport.py` and
gateway config tests where appropriate.

Coverage:

- shell wrapper commands are rejected by default
- shell wrapper commands can be allowed only by explicit executable allowlist
- `allow_path_lookup=false` rejects bare executable names
- executable allowlist accepts matching executable names
- executable allowlist rejects non-matching executable names
- cwd roots reject resolved cwd outside configured roots
- cwd roots accept resolved cwd inside configured roots
- `default_cwd` is used when server cwd is omitted
- policy env allowlist rejects a server-requested env name outside the policy
- allowed env values still inherit only from names present in `os.environ`
- config loader validates process-policy mappings and rejects invalid types
- bootstrap passes configured policy into the package stdio factory while
  preserving the default factory identity when policy is omitted
- denied policy errors do not include env values or full command arrays
- runtime-manager start failures caused by policy denial expose safe status and
  `_last_errors` metadata without full command arrays or env values
- Windows-style shell basenames and path normalization are covered at the helper
  level where practical on the current platform
- `mcp_unified.federation.transports` documentation no longer describes the
  package contract as non-spawning now that real stdio transport exists

## Rollout

This is a hardening slice. Existing direct API users can continue constructing
`StdioExternalTransport(server)` without a custom policy, but default shell
wrapper rejection is a deliberate behavior change. Gateway users can opt into
stricter deployment policy through config before enabling real upstream stdio
auto-start.

Real install/update execution remains deferred until this process-policy layer
is in place and reviewed.
