---
id: TASK-160
title: Harden vz_linux helper-generation session recovery
status: In Progress
assignee: []
created_date: '2026-05-09 05:34'
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
- [ ] #1 A design spec defines the helper-generation/session-control recovery contract and explicitly reviews lifecycle ownership and scope risks before implementation.
- [ ] #2 vz_linux session reuse can distinguish healthy same-helper reuse from stale control state after helper identity or generation drift.
- [ ] #3 Stale generation metadata is cleared or replaced before provisioning a fresh VM, without deleting session-control rows on helper unavailable or helper protocol mismatch.
- [ ] #4 Focused host-independent tests cover healthy reuse, generation mismatch recovery, helper-unavailable fail-closed behavior, and protocol-mismatch fail-closed behavior.
- [ ] #5 Operator docs and host-gated smoke expectations are updated only where needed to explain the new recovery signal and limitations.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

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
- Current evidence: helper ping/status has no generation signal, and persisted VZ session control stores only VM/template/workspace readiness.
- Design decision: helper generation must be helper-owned rather than Python-synthesized; Python should preserve session-control rows when helper availability/protocol truth is ambiguous.
<!-- SECTION:NOTES:END -->
