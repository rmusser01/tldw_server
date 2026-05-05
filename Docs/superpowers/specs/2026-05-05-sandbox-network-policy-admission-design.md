# Sandbox Network Policy Admission Design

## Goal

Use the runtime `network_policy_contract` as the shared static admission gate for sandbox sessions and runs. Unsupported or non-strict runtime/network-policy combinations should fail before enqueue or session creation, while runtime preflight remains responsible for current host readiness.

## Current State

Runtime discovery now exposes `network_policy_contract` for each `RuntimeType`, but admission still relies on scattered runtime-specific checks and preflight booleans. This leaves host-local runtimes such as `seatbelt` and `worktree` available for `deny_all` requests even though their contract states strict `deny_all` is unsupported.

## Design

Add a single `SandboxPolicy` helper that validates the requested runtime and normalized network policy against `runtime_network_policy_metadata(runtime)`.

The helper should:

- Normalize empty policy through the existing trust-profile/default flow before validation.
- Accept only `deny_all` and `allowlist`.
- Raise `PolicyUnsupported(..., reasons=["unsupported_network_policy"])` for invalid policy values.
- Select the requested mode metadata (`deny_all` or `allowlist`) from the static contract.
- Raise `PolicyUnsupported(..., reasons=["strict_<mode>_not_supported"])` when `support_state` is `unsupported` or `not_applicable`.
- Raise `PolicyUnsupported(..., reasons=["strict_<mode>_not_supported"])` when `strict_enforcement` is false.
- Allow statically supported or host-gated strict modes to proceed to the existing dynamic runtime preflight layer.

Call this helper from `apply_to_session()` and `apply_to_run()` after profile/default network policy assignment. This ensures direct runs and session creation use the same contract, including inherited trust-profile defaults.

## Non-Goals

- Do not remove execution-time preflights for Lima, VZ Linux, VZ macOS, Seatbelt, or Worktree.
- Do not change Docker's dynamic config checks or egress enforcement behavior.
- Do not implement allowlist support for runtimes whose contract says it is unsupported or scaffold-only.
- Do not infer security guarantees from `available=True`.

## Risk Review

The main behavior change is intentional: `seatbelt` and `worktree` should no longer admit strict `deny_all` requests because their static contract says strict enforcement is unsupported. That can reject some previously accepted local-development paths, but it avoids representing host-local execution as a strict network sandbox.

Docker `allowlist` must remain layered. The static contract can admit it only as a possible strict mode; dynamic Docker/config readiness still determines whether real enforcement is ready.

Session-backed runs must validate after session values are inherited, or admission may check a placeholder request policy rather than the effective session policy.

## Test Strategy

Add focused policy tests that assert:

- Invalid policy values fail with `unsupported_network_policy`.
- `seatbelt` and `worktree` reject `deny_all` and `allowlist`.
- `vz_linux` admits `deny_all` at the static contract layer and rejects `allowlist`.
- Empty policy values are populated from trust profiles before validation.
- Both `apply_to_session()` and `apply_to_run()` enforce the same behavior.
