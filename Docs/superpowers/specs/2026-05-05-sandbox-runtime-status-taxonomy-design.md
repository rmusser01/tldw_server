# Sandbox Runtime Status Taxonomy Design

## Goal

Normalize known sandbox runner status strings into stable `status_reason_code`
values so API clients do not need runtime-specific message matching. This is a
narrow Phase 3 runtime parity slice: it improves the existing derived taxonomy
without changing runner execution behavior, database shape, or raw diagnostic
messages.

## Current State

`run_status_taxonomy.py` already derives a small client-facing reason vocabulary
from `phase`, `message`, `exit_code`, and `resource_usage`. The current mapping
handles common phases, timeout strings, queue TTL expiry, nonzero exits, limit
signals, and some runtime-unavailable strings.

The gap is that some first-class policy failure messages emitted by the service
are currently classified inconsistently. In particular, `vz_linux_policy_failed`
and `vz_macos_policy_failed` are listed in the runtime-unavailable alias set,
which makes policy admission failures look like missing runtime support. Other
policy messages depend on substring matching instead of an explicit alias set.

## Design

Use explicit alias sets inside `run_status_taxonomy.py` for stable categories:

- Policy failures: exact aliases for `lima_policy_failed`,
  `vz_linux_policy_failed`, `vz_macos_policy_failed`, `seatbelt_policy_failed`,
  and `worktree_policy_failed`, plus the existing conservative substring
  fallback for messages containing both `policy` and `failed`.
- Runtime unavailable: exact aliases for runtime availability failures and
  conservative heuristics for runtime/provisioning contexts with unavailable,
  missing, or not-found signals.
- Timeouts: preserve existing startup/execution timeout behavior and make the
  alias intent explicit in tests.
- Fallbacks: genuinely unknown failed messages remain `runtime_error`.

Raw `message` remains unchanged for operators. Only the derived
`status_reason_code` changes.

## Risks And Mitigations

- Risk: broad aliases could hide real runtime errors.
  Mitigation: prefer exact alias sets first and keep heuristic matching narrow.
- Risk: changing VZ policy messages from `runtime_unavailable` to
  `policy_failed` may affect clients already compensating for the old behavior.
  Mitigation: this is the intended taxonomy correction and is additive at the
  schema level; raw messages remain available for compatibility and debugging.
- Risk: adding new reason codes would expand public contract surface too early.
  Mitigation: no new reason codes in this slice.

## Tests

Add focused tests in `test_run_status_reason_codes.py` that prove:

- All known runtime policy failure messages normalize to `policy_failed`.
- Runtime-unavailable aliases still normalize to `runtime_unavailable`.
- Template/provisioning missing messages still normalize to
  `runtime_unavailable`.
- Unrelated missing/artifact/command errors remain `runtime_error`.

## Documentation

Update the sandbox runtime capability inventory to record that the first Phase 3
taxonomy pass centralizes known aliases while leaving richer structured error
metadata and cross-runtime session/recovery parity for later phases.
