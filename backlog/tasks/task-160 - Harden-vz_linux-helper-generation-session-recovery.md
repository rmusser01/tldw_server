---
id: TASK-160
title: Harden vz_linux helper-generation session recovery
status: Done
assignee: []
created_date: '2026-05-09 05:34'
updated_date: '2026-05-09 18:43'
labels:
  - sandbox
  - vz_linux
  - recovery
  - lifecycle
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1414'
  - 'https://github.com/rmusser01/tldw_server/pull/1406'
  - 'https://github.com/rmusser01/tldw_server/pull/1397'
documentation:
  - Docs/Sandbox/sandbox-architecture-doctrine.md
  - Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md
  - tldw_Server_API/app/core/Sandbox/README.md
  - >-
    Docs/superpowers/specs/2026-05-09-vz-linux-helper-generation-session-recovery-design.md
  - >-
    Docs/superpowers/plans/2026-05-09-vz-linux-helper-generation-session-recovery.md
  - >-
    Docs/superpowers/plans/2026-04-27-vz-linux-lifecycle-recovery-hardening-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next narrow sandbox recovery slice after PR #1406: make vz_linux session reuse detect helper identity or generation drift after a helper restart and clear stale session-control metadata before provisioning a fresh VM. Keep scope focused on helper/session-control recovery; do not add host reboot automation, launchd install/load behavior, networking changes, or broad repair automation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design spec defines the helper-generation/session-control recovery contract and explicitly reviews lifecycle ownership and scope risks before implementation.
- [x] #2 vz_linux session reuse can distinguish healthy same-helper reuse from stale control state after helper identity or generation drift.
- [x] #3 Stale generation metadata is cleared or replaced before provisioning a fresh VM, without deleting session-control rows on helper unavailable or helper protocol mismatch.
- [x] #4 Focused host-independent tests cover healthy reuse, generation mismatch recovery, helper-unavailable fail-closed behavior, and protocol-mismatch fail-closed behavior.
- [x] #5 Operator docs and host-gated smoke expectations are updated only where needed to explain the new recovery signal and limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write and review the helper-generation session recovery design spec.
2. Create a staged implementation plan after the design spec is accepted.
3. Implement helper-owned generation details in the Swift helper and Python models.
4. Extend VZ session-control persistence across in-memory, SQLite, Postgres, and orchestrator layers.
5. Harden `VZLinuxRunner` session reuse to compare helper generation and preserve rows on ambiguous helper failures.
6. Add focused host-independent tests and minimal operator documentation updates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Design spec added at `Docs/superpowers/specs/2026-05-09-vz-linux-helper-generation-session-recovery-design.md`.
- Implementation plan added at `Docs/superpowers/plans/2026-05-09-vz-linux-helper-generation-session-recovery.md`.
- PR opened at https://github.com/rmusser01/tldw_server/pull/1414.
- PR review fixes: plan now normalizes whitespace-only helper generation values to `None` and explicitly requires `MacOSVirtualizationHelperProtocolError` in the runner's controlled failure exception tuple.
- Current evidence: helper ping/status has no generation signal, and persisted VZ session control stores only VM/template/workspace readiness.
- Design decision: helper generation must be helper-owned rather than Python-synthesized; Python should preserve session-control rows when helper availability/protocol truth is ambiguous.

Implemented helper-owned generation details in Swift helper responses and PROTOCOL.md.

Persisted helper_instance_id/helper_started_at across VZ session-control store interfaces, in-memory, SQLite, Postgres, SQLite migrations, and orchestrator facade.

Hardened VZLinuxRunner reuse so same-generation healthy VMs are reused, generation/status/metadata drift reprovisions, and helper unavailable/protocol mismatch fail closed without deleting or overwriting session-control state.

Verification: swift test --filter 'PingTests|HelperServiceVMTests' passed; pytest test_vz_linux_runner.py test_vz_linux_session_control_store.py test_store_sqlite_migrations.py passed; Bandit touched Python scope wrote /tmp/bandit_vz_helper_generation.json with 0 findings.

PR 1420 review pass plan: add runner docstrings and metadata None guard; move optional nonempty string normalization to a shared Sandbox helper; make Postgres session-control column migrations fail fast with contextual logging; run focused tests and Bandit before pushing.

PR 1420 review verification: focused pytest passed with 27 passed and 2 skipped; git diff --check passed; Bandit review scope wrote /tmp/bandit_vz_helper_generation_review.json with 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented helper-generation-aware vz_linux session recovery. The Swift helper now emits per-process helper generation details, VZ session-control persistence stores them, and VZLinuxRunner reuses session VMs only when live helper metadata and generation match while preserving rows on helper unavailable/protocol mismatch. Focused Swift/Python tests, README operator notes, SQLite migration coverage, and Bandit verification were completed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
