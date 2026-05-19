# Sandbox Cleanup And Recovery Contract Tests Design

**Status:** Approved implementation design.
**Date:** 2026-05-05.
**Backlog:** `TASK-62`.
**Scope:** Portable cleanup/recovery contract tests for sandbox sessions and
host-local/runtime paths.

## Related Guidance

- `Docs/Sandbox/sandbox-architecture-doctrine.md`
- `Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md`
- `Docs/Sandbox/sandbox-runtime-capability-inventory.md`

## Goal

Add local-first contract coverage for cleanup and recovery behaviors that should
stay true across developer machines without requiring Apple Silicon
Virtualization.framework, Docker, Firecracker, Lima, or another prepared host.

The slice turns Phase 4 reliability doctrine into tests for portable seams:

- failed run setup must not leak runtime workspaces
- timeout/cancel paths must clear runtime tracking and temporary directories
- session deletion must clean durable workspace state through service and
  orchestrator seams
- stale cached session metadata must not authorize new session-backed runs
- host-local runtimes must continue to report workspace-only session reuse and
  no repair/recovery contract

## Non-Goals

- Do not run real `vz_linux` VMs in this PR.
- Do not expand macOS repair beyond existing `vz_linux` admin flows.
- Do not add warm runtime reuse for `seatbelt` or `worktree`.
- Do not introduce a new cross-runtime repair API.
- Do not weaken fail-closed policy or runtime availability checks to make tests
  easier.

## Design Review

The main risk is writing tests that overfit private implementation details while
pretending to prove runtime behavior. This design limits private-state
assertions to cleanup internals that are already explicit lifecycle contracts,
such as worktree active-run maps and durable session roots. Public service and
orchestrator seams are preferred for session recovery checks.

Another risk is conflating "session participates in API workflow" with "warm
runtime reuse". The runtime inventory says host-local runtimes are
`workspace_only`, so tests should assert that distinction directly instead of
adding any helper-health or repair semantics to `worktree`/`seatbelt`.

The final risk is broadening the PR into runtime implementation work. The
baseline already contains cleanup behavior for several paths. If a new contract
test passes immediately, this PR should keep it as regression coverage and avoid
unnecessary production edits.

## Test Strategy

Use focused unit/contract tests:

- `test_worktree_runner.py` for portable worktree timeout cleanup with fake
  subprocess and temporary worktree directories.
- `test_session_store_durability.py` for durable session deletion and stale
  metadata behavior across service/orchestrator instances.
- `test_runtime_inventory_contract.py` for explicit host-local no-warm-reuse
  and no-repair/no-recovery session contract metadata.

The tests must not require real Docker, VZ, Firecracker, Lima, or host-gated
network tooling. Existing real-host smoke remains under host-gated suites.
