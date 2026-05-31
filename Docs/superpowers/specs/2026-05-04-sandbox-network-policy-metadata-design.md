# Sandbox Network Policy Metadata Design

**Status:** Approved implementation design.
**Date:** 2026-05-04.
**Task:** TASK-44.

## Goal

Expose a structured, machine-readable network policy contract in sandbox runtime
discovery so clients can distinguish static runtime guarantees from current host
readiness.

This is an additive follow-up to runtime implementation state, normalized reason
codes, and runtime isolation metadata. It should not change admission behavior
or remove existing discovery fields.

## Current Problem

`/api/v1/sandbox/runtimes` already exposes compatibility/readiness booleans:

- `strict_deny_all_supported`
- `strict_allowlist_supported`
- `enforcement_ready`
- `egress_allowlist_supported`

Those fields are useful, but they mix static security posture, current host
preflight, and runtime-specific implementation details. That makes it hard for
clients and ACP callers to know whether a runtime is fundamentally capable of a
policy, merely host-gated, scaffold-only, or explicitly unsupported.

## Design

Add static network policy metadata to `runtime_capabilities.py`, keyed by
`RuntimeType`, with import-time completeness validation.

Each runtime exposes `network_policy_contract` in discovery:

```json
{
  "deny_all": {
    "support_state": "host_gated",
    "strict_enforcement": true,
    "readiness_source": "runtime_preflight"
  },
  "allowlist": {
    "support_state": "unsupported",
    "strict_enforcement": false,
    "readiness_source": "not_applicable"
  }
}
```

The support-state vocabulary reuses the roadmap maturity vocabulary:
`supported`, `unsupported`, `scaffold`, `host_gated`, and `not_applicable`.

The readiness-source vocabulary is intentionally small:

- `runtime_preflight`: current readiness comes from runtime preflight truth
- `config`: readiness depends primarily on server/operator configuration
- `not_applicable`: no runtime readiness is meaningful because unsupported

Existing booleans remain for compatibility. The new contract is the stable,
static policy posture. Existing fields continue to summarize current readiness
or legacy client expectations.

## Runtime Contract

| Runtime | `deny_all` | `allowlist` |
| --- | --- | --- |
| `docker` | `supported`, strict, config/preflight summarized by existing fields | `host_gated`, strict, config |
| `firecracker` | `host_gated`, strict, runtime preflight | `scaffold`, not strict, runtime preflight |
| `lima` | `host_gated`, strict, runtime preflight | `unsupported`, not strict, not applicable |
| `vz_linux` | `host_gated`, strict, runtime preflight | `unsupported`, not strict, not applicable |
| `vz_macos` | `scaffold`, not strict, runtime preflight | `unsupported`, not strict, not applicable |
| `seatbelt` | `unsupported`, not strict, not applicable | `unsupported`, not strict, not applicable |
| `worktree` | `unsupported`, not strict, not applicable | `unsupported`, not strict, not applicable |

## Guardrails

- Do not use `available=true` as proof of a network security guarantee.
- Do not imply `seatbelt` or `worktree` provide strict egress isolation.
- Do not imply `vz_macos` has real strict enforcement before real execution
  exists.
- Do not make Docker look VM-grade; this metadata only describes network
  policy semantics.
- Treat the map like isolation metadata: adding a `RuntimeType` without
  metadata should fail fast at import time.

## Testing

Add focused tests for:

- map completeness against `RuntimeType`
- unknown runtime rejection
- discovery payload includes `network_policy_contract` for every runtime
- host-local runtimes report unsupported strict network policy contracts
- `vz_linux` reports host-gated strict `deny_all` and unsupported `allowlist`
- schema marks the new field required and non-nullable

## Documentation

Update:

- `Docs/API-related/Sandbox_API.md`
- `Docs/Published/API-related/Sandbox_API.md`
- `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- `Docs/Sandbox/sandbox-security-policy-matrix.md`

The docs should state that the contract is static posture metadata, while
runtime preflight remains the source of current host readiness.
