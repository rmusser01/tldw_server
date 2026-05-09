# VZ Linux Helper Crash And Host Reboot Recovery Posture Design

**Date:** 2026-05-09
**Status:** Proposed design for implementation planning
**Backlog:** TASK-195
**GitHub:** https://github.com/rmusser01/tldw_server/issues/1459
**Scope:** `vz_linux` helper crash, launchd-managed helper restart posture,
host reboot posture, diagnostics expectations, repair semantics, and future
test/drill guidance.

## Summary

The `vz_linux` runtime now has real helper-backed execution, generation-aware
same-session VM reuse, read-only diagnostics, dry-run-first reconciliation
repair, and manual host-gated drills for helper-side VM invalidation plus
smoke-owned helper restart. The remaining recovery gap is not another generic
repair mechanism. It is a clear operator posture for broader helper crash,
launchd restart, and host reboot events.

This design defines those events as distinct failure modes and keeps the
existing ownership boundaries intact:

- Python owns API behavior, run/session identity, persisted session-control
  rows, image-store manifests, repair authorization, and audit posture.
- The Swift helper owns live VM truth, helper process generation, host/template
  checks, and guest transport observations.
- `launchd` may supervise the helper process, but it does not own sandbox
  session metadata or VM recovery decisions.
- The guest agent owns in-guest readiness and command execution capability only
  after the helper proves the VM is live and reachable.

## Source Documents

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/Sandbox/sandbox-runtime-capability-inventory.md`
- `Docs/Sandbox/macos-runtime-operator-notes.md`
- `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md`
- `Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md`
- `Docs/superpowers/specs/2026-04-29-vz-helper-lifecycle-hardening-design.md`
- `Docs/superpowers/specs/2026-05-09-vz-linux-helper-generation-session-recovery-design.md`
- `Docs/superpowers/specs/2026-05-09-vz-linux-helper-restart-recovery-drill-design.md`
- `Docs/superpowers/specs/2026-05-09-vz-linux-host-failure-drills-design.md`
- `tools/macos-vz-helper/PROTOCOL.md`

## Current Baseline

Already implemented behavior:

- `vz_linux` session reuse persists VZ session-control metadata.
- The helper exposes `helper_instance_id` and `helper_started_at` generation
  details.
- The runner reuses a session VM only when helper live status, ownership/session
  metadata, and helper generation agree.
- Helper unavailable and helper protocol mismatch fail closed and preserve
  session-control rows.
- Reachable stale VM truth, unhealthy VM status, ownership/session mismatch, or
  generation drift clears stale session-control metadata before provisioning a
  fresh VM.
- Admin macOS diagnostics are read-only and summarize reconciliation,
  observability, image-store, startup warnings, and recovery posture.
- Reconciliation repair is explicit, admin-only, dry-run-first, skips active
  sessions, and blocks mutating repair when helper truth is unavailable or
  protocol-mismatched.
- Host-gated smoke can run manual failure drills for a drill-owned terminated
  VM and for a smoke-owned helper restart lease.

Known remaining gap:

- The repo does not yet define accepted behavior for arbitrary helper crashes,
  launchd-managed helper restarts, or host reboot events as an operator
  contract.
- Current real-host drills prove a narrow helper restart under smoke-harness
  ownership, not launchd behavior or host reboot.
- Diagnostics can report stale persisted rows after helper truth is available,
  but they should not mutate state or guess what happened during a reboot.

## Goals

1. Define helper crash, launchd restart, and host reboot as separate recovery
   modes.
2. State which component owns truth for helper process identity, live VM state,
   persisted session-control rows, image-store state, and guest readiness.
3. Preserve fail-closed behavior when helper state is ambiguous.
4. Preserve dry-run-first repair semantics and explicit operator action.
5. Identify portable tests, host-gated/manual drills, and operator-only
   procedures for future implementation slices.
6. Keep this as a posture/design slice that future code PRs can reference.

## Non-Goals

1. No networking changes or vmnet allowlist implementation.
2. No helper boot path rewrite or replacement of the current Swift helper.
3. No guest protocol changes or tldw-agent replacement.
4. No `vz_macos` real execution or `vz_macos` warm-session semantics.
5. No generic cross-runtime repair automation.
6. No automatic launchd install/load/unload behavior in this slice.
7. No destructive host reboot CI or scheduled reboot workflow.
8. No automatic deletion of session-control rows during diagnostics.

## Recovery Modes

### Mode 1: Helper Crash Or Manual Stop Without Replacement

Definition: the helper process is gone, its socket is absent or not accepting
connections, and Python cannot get helper live VM truth.

Expected posture:

- `vz_linux` runs that need helper truth fail closed.
- Persisted session-control rows are preserved.
- Admin diagnostics report helper unavailable and recovery unavailable.
- Mutating reconciliation repair is blocked because helper truth is unavailable.
- Operator action is to restore helper readiness first, then re-run diagnostics
  or retry the session run.

Why preserve rows: Python cannot know whether a helper outage is transient, a
protocol setup problem, a permissions problem, or an operator-managed restart in
progress. Deleting rows at this point would destroy potentially recoverable
control-plane state based on absence of evidence.

### Mode 2: Helper Crash Followed By Direct Manual Restart

Definition: the old helper process is gone, a new helper process starts on the
accepted socket, and the new helper has a different helper generation.

Expected posture:

- The replacement helper owns a fresh live VM registry.
- Existing persisted session-control rows are reuse candidates only after live
  helper status is reachable.
- If `get_vm_status(stored_vm_id)` is absent, unhealthy, ownership/session
  mismatched, or generation-mismatched, the next run clears that row and
  provisions a fresh VM.
- Diagnostics classify rows whose VM IDs are missing from `list_vms()` as stale
  once helper truth is available.
- Dry-run repair may plan stale-row deletion for inactive sessions.
- Mutating repair may delete only inactive stale/unhealthy rows after the
  operator has inspected the dry-run plan.

This is the behavior current generation-aware reuse and manual helper-restart
drills are meant to protect.

### Mode 3: Launchd-Managed Helper Restart

Definition: `launchd` restarts the helper after crash, kickstart, login, or
manual operator action. Process supervision comes from launchd, but the helper
still reports its own generation through the repo-owned protocol.

Expected posture:

- `launchd` is only the process supervisor. It is not a sandbox repair engine.
- `vz-helperctl.py status` should verify helper binary, pid/socket/log path
  safety, entitlements status, ping, protocol version, helper version, and
  launchd plist match when a plist path is configured.
- Python should treat the replacement helper like any other new helper
  generation.
- Session-control rows are preserved while helper readiness is ambiguous and
  cleared only after reachable helper truth proves staleness.
- Diagnostics should not assume that launchd restart implies a clean VM
  teardown or a valid warm VM.

Future launchd implementation should keep service management operator-first:
generate/validate plist scaffolding before adding any explicit bootstrap,
bootout, or kickstart command. Automatic repair must not be attached to launchd
restart.

### Mode 4: Host Reboot

Definition: the macOS host reboots. Helper process identity, helper in-memory VM
registry, VM processes, virtiofs state, guest readiness, and socket state should
be treated as lost until proven otherwise.

Expected posture:

- Persisted Python session-control rows may survive because they are stored in
  SQLite/Postgres.
- Image-store template manifests and run manifests may survive because they are
  filesystem state.
- Live VM truth is unavailable until the helper is running and can answer
  `list_vms()` and `get_vm_status()`.
- Existing persisted session-control rows should be treated as stale candidates,
  not automatically deleted during startup or diagnostics.
- Once helper truth is reachable, rows whose VM IDs are absent from helper live
  state should be reported as stale.
- The next session run may clear a stale row and provision a fresh VM when
  helper truth is reachable and protocol-compatible.
- Dry-run repair may plan inactive stale-row deletion after diagnostics.
- Mutating repair remains explicit and admin-owned.
- Guest-agent readiness from before reboot has no value. New guest readiness
  must be observed through the replacement helper and fresh VM/agent handshake.

Host reboot should not become a hidden startup repair path. Startup warning
collection can report helper unavailable, protocol mismatch, or reconciliation
drift, but it should not delete rows or terminate VMs.

## Ownership Truth Matrix

| State | Owner | Recovery rule |
| --- | --- | --- |
| Helper process identity | Swift helper reports generation; `vz-helperctl`/launchd observe process facts | Trust helper generation only after protocol-compatible ping/status. Do not infer generation from socket path alone. |
| Live VM registry | Swift helper | Use `list_vms()` and `get_vm_status()` as live truth only when helper is reachable and protocol-compatible. |
| Session-control rows | Python sandbox store | Preserve while helper truth is ambiguous; delete only after reachable live truth proves stale or through explicit repair. |
| Image-store templates and run manifests | Python image store | Treat as durable provenance, not proof that a VM is live. GC/cleanup remains explicit and dry-run-first. |
| Guest-agent readiness | Guest agent via helper transport | Any pre-crash/pre-reboot readiness is invalid unless the current helper observes the current VM and guest readiness again. |
| Launchd service state | Operator/launchd | Useful for process supervision and status, not sufficient for session reuse or repair decisions. |
| Socket and pid files | Operator lifecycle tooling plus helper socket safety | Necessary for safe helper access, never proof of warm VM validity. |

## Preserve Versus Clear Rules

Preserve persisted session-control rows when:

- helper socket is absent or unavailable
- helper ping fails due to unavailable helper
- helper protocol version mismatches Python expectation
- helper status/list calls fail before returning trustworthy live VM facts
- startup diagnostics run before helper readiness is established

Clear persisted session-control rows when:

- helper is reachable and protocol-compatible
- the row is inactive or belongs to the session currently attempting reuse
- the stored VM ID is absent from helper live state
- the stored VM is unhealthy
- helper ownership/runtime/session metadata does not match `tldw/vz_linux` and
  the requested session
- stored helper generation and live helper generation are both present and do
  not match
- legacy/missing generation cannot be compensated by matching live
  ownership/session metadata

Skip automatic clearing when:

- the session is active in reconciliation context
- helper truth is unavailable or protocol-mismatched
- the row references state outside the `vz_linux` ownership model
- the operation is diagnostics-only

## Diagnostics And Repair Contract

Diagnostics should remain read-only:

- report helper unavailable/protocol mismatch as recovery unavailable
- report stale rows only after helper live state is available
- report owned, unknown, and foreign orphan VMs separately
- point to dry-run repair and image-store cleanup-plan endpoints when relevant
- avoid reading helper logs or serial logs inline
- avoid deleting rows, terminating VMs, or running image-store cleanup

Repair should remain explicit:

- default to dry-run
- require admin authorization
- block mutation when helper unavailable or protocol-mismatched
- skip active sessions
- delete inactive stale/unhealthy session-control rows only when requested
- terminate orphan VMs only when metadata proves owned `tldw/vz_linux` state and
  `terminate_orphaned_vms=true` is explicitly requested
- never infer reboot cleanup solely from persisted rows

## Test And Drill Strategy

### Portable Tests

Portable tests can cover:

- runner preserves session-control rows when helper unavailable during reuse
- runner preserves session-control rows on helper protocol mismatch
- runner clears stale rows when reachable helper status is absent/unhealthy
- runner clears rows on helper generation mismatch
- reconciliation reports stale rows only when helper live VM list is available
- reconciliation blocks mutating repair when helper unavailable/protocol
  mismatched
- `vz-helperctl.py status` reports socket/pid/log/plist/protocol checks without
  starting or stopping the helper
- launchd plist rendering/validation remains dry-run/operator-owned
- startup warning policy reports helper unavailable as warning and protocol
  mismatch as blocking without mutating sandbox state

### Host-Gated Manual Drills

Prepared Apple silicon host drills can cover:

- current drill-owned VM termination and fresh provisioning
- current smoke-owned helper restart lease and fresh provisioning
- future direct helper crash drill using `vz-helperctl.py start`/`stop` rather
  than pytest owning the process
- future launchd restart drill after explicit launchd bootstrap/bootout/kickstart
  tooling exists
- future stale socket/pid recovery after helper crash, limited to private
  operator-owned paths

Manual drills should stay opt-in until they are boring across repeated prepared
host runs. Scheduled host-gated CI should not reboot hosts or enable destructive
failure drills by default.

### Operator-Documented Procedures

Host reboot should remain operator-documented until there is a safe prepared
runner procedure:

1. Reboot the host.
2. Start or verify the helper through the managed operator workflow.
3. Run `vz-helperctl.py status` and macOS sandbox diagnostics.
4. Inspect reconciliation and recovery summary.
5. Run reconciliation repair in dry-run mode if stale rows are reported.
6. Apply mutating repair only after reviewing the plan.
7. Run real host smoke to verify ephemeral execution and same-session fresh VM
   provisioning.

This sequence is safer than hiding reboot cleanup in app startup.

## Future Implementation Slices

Recommended follow-up order:

1. Update operator docs with the crash/reboot procedure and expected
   diagnostics reasons.
2. Add portable tests for any missing preserve-versus-clear cases.
3. Add a `vz-helperctl.py` managed restart/status drill that does not depend on
   pytest owning the helper process.
4. Add explicit launchd bootstrap/bootout/kickstart scaffolding only after the
   managed helper command can validate and cleanly stop/start helpers.
5. Add a manual host reboot checklist to host-gated policy before considering
   any automated reboot runner.

## Design Review

### Potential Issue: Treating Helper Restart As Proof That Old VMs Are Gone

The replacement helper's empty registry is not proof about every host resource,
but it is the only repo-owned live VM truth. The safe action is to avoid reusing
old session-control rows and provision fresh only after reachable helper truth
shows the old VM is absent or stale.

### Potential Issue: Auto-Deleting Rows On Helper Unavailable

This would make transient helper startup or socket permission problems
destructive. The design preserves rows until helper truth is available or an
operator explicitly repairs state.

### Potential Issue: Launchd Becomes Hidden Automation

Launchd should supervise only the helper process. Recovery decisions remain in
Python diagnostics, runner reuse checks, and explicit repair endpoints.

### Potential Issue: Reboot Leaves Filesystem State That Looks Valid

Image-store manifests and session-control rows are durable provenance, not live
VM proof. After reboot, live VM and guest readiness must be re-observed through
the current helper before reuse.

### Potential Issue: Host Reboot Tests Are Too Disruptive For CI

Keep reboot as an operator-documented manual procedure until a dedicated
prepared runner can tolerate disruptive reboot tests and preserve logs
reliably.

### Potential Issue: Over-Generalizing To Other Runtimes

The detailed repair contract remains `vz_linux`-specific because it depends on
helper-owned VM metadata and generation truth. Other runtimes should advertise
unsupported or scaffold recovery states until they can prove equivalent
ownership.

## Acceptance Mapping

- AC1: helper crash, launchd restart, and host reboot are defined as distinct
  modes with explicit ownership truth.
- AC2: portable tests, host-gated/manual drills, and operator-only procedures
  are separated in the test strategy.
- AC3: preserve-versus-clear rules retain fail-closed and dry-run-first repair
  semantics.
- AC4: non-goals exclude networking, helper boot rewrites, guest protocol
  changes, `vz_macos`, and generic repair automation.
- AC5: this spec links to issue #1459 and records design review findings for
  follow-up implementation planning.
