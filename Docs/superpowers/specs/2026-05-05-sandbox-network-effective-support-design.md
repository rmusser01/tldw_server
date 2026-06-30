# Sandbox Network Effective Support Design

## Goal

Stabilize sandbox network policy discovery and admission so `allowlist` is only
reported or admitted when a runtime can strictly enforce that policy right now.

## Context

The sandbox runtime inventory already exposes a static `network_policy_contract`
for `deny_all` and `allowlist`. The contract records whether a policy is
supported, host-gated, scaffold-only, or unsupported, and whether strict
enforcement is possible.

The remaining Phase 2 gap is effective support drift:

- Policy admission uses the static contract but does not check current
  readiness for host-gated/config-gated policies.
- Docker discovery can report allowlist support through older
  `egress_allowlist_supported` wording even when the execution path would fall
  back to `network=none`.
- Firecracker allowlist is scaffold/planned and must not be advertised as
  currently usable, even if operator flags are set.

## Design

Add a single effective-support helper in
`tldw_Server_API/app/core/Sandbox/runtime_capabilities.py`:

```python
runtime_network_policy_effective_support(
    runtime: RuntimeType | str,
    enforcement_ready: Mapping[str, bool] | None,
) -> dict[str, bool]
```

The helper evaluates each policy mode with the same rules:

1. Static contract must be `supported` or `host_gated`.
2. Static contract must have `strict_enforcement=True`.
3. Current readiness for that policy must be true.

This deliberately treats `scaffold`, `unsupported`, and non-strict modes as
false even when raw runtime preflight or environment flags claim readiness.

## Runtime Readiness Inputs

`collect_runtime_preflights()` should provide useful readiness facts for
runtimes that do not have a dedicated preflight object today:

- Docker: `deny_all=True` when Docker is available; `allowlist=True` only when
  Docker is available and both egress enforcement and granular enforcement are
  enabled.
- Firecracker: `deny_all=True` when Firecracker is available; `allowlist=False`
  because the static contract remains scaffold/non-strict.
- Lima, VZ, seatbelt, and worktree continue to use their existing runner
  preflights.

## Admission

`SandboxPolicy._require_network_policy_supported()` should use the effective
support helper and the runtime's current preflight result. A requested policy is
admitted only when the helper returns true.

This preserves existing fail-closed reasons:

- invalid policy -> `unsupported_network_policy`
- unsupported `allowlist` -> `strict_allowlist_not_supported`
- unsupported `deny_all` -> `strict_deny_all_not_supported`

## Discovery

`SandboxService.feature_discovery()` should derive legacy discovery booleans
from the same effective-support helper:

- `strict_deny_all_supported`
- `strict_allowlist_supported`
- runtime-specific `egress_allowlist_supported`

The raw `enforcement_ready` object remains visible for diagnostics, but clients
should treat the strict support booleans as the effective admission signal.

## Non-Goals

- Do not implement granular allowlist support for Firecracker, Lima, VZ, or
  host-local runtimes.
- Do not change the static `network_policy_contract` vocabulary.
- Do not add a new public API field unless tests show existing fields cannot
  express the contract clearly.
- Do not treat Docker's `network=none` fallback as a valid `allowlist`.

## Testing

Add focused tests for:

- Docker allowlist admission fails when granular enforcement is not ready.
- Docker allowlist admission succeeds when granular enforcement readiness is
  explicitly true.
- Firecracker allowlist remains rejected even if raw readiness is true.
- Discovery strict support booleans match effective support for every runtime.
- Docker discovery reports allowlist support only when granular enforcement is
  enabled.
