---
id: TASK-2384
title: Design sandbox operator/admin status consolidation
status: Done
labels:
- sandbox
- vz_linux
- docs
- operator-ux
documentation:
- Docs/superpowers/specs/2026-06-18-sandbox-operator-status-consolidation-design.md
- Docs/superpowers/specs/2026-05-02-sandbox-module-roadmap-design.md
- Docs/Sandbox/sandbox-runtime-capability-inventory.md
- Docs/Sandbox/macos-runtime-operator-notes.md
modified_files:
- Docs/superpowers/specs/2026-06-18-sandbox-operator-status-consolidation-design.md
- Docs/Sandbox/sandbox-runtime-capability-inventory.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a roadmap-backed design for a read-only sandbox operator/admin status consolidation layer, focused on VZ evidence/lifecycle readiness signals without adding helper mutation, repair mutation, launchd bootstrap, host reboot automation, or new runtime truth sources.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec documents read-only operator/admin status consolidation scope and non-goals.
- [ ] #2 Spec identifies existing sources of truth and avoids duplicating runtime-specific logic in clients.
- [ ] #3 Spec covers testing, security, and rollout boundaries for a future implementation slice.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-only slice. The approved option is a read-only operator/admin status consolidation spec. The spec anchors to existing runtime discovery, runtime diagnostics, macOS diagnostics, image-store probe, reconciliation recovery summary, startup warning summary, and optional evidence summary artifacts. Inline design review checked for source-of-truth drift, mutation boundaries, evidence-ingestion risk, and status-classification ambiguity. Verification exposed a pre-existing inventory docs-contract drift in the portable session-contract gate wording; the inventory Current Gaps section was minimally updated to satisfy the existing gate without changing production behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the read-only sandbox operator/admin status consolidation design spec. The spec defines sources of truth, non-goals, status sections, classification rules, evidence-handling safety, implementation slices, tests, and security boundaries. During verification, the existing portable runtime capability gate exposed inventory docs-contract drift; the inventory Current Gaps section was minimally updated to document the portable session-contract gate and host-gated recovery boundary. Verification: python -m pytest tldw_Server_API/tests/sandbox/test_runtime_capability_gate.py -q passed with 8 tests. Bandit is not applicable because this slice changes markdown/backlog docs only; an attempted Bandit run against markdown produced a parse error and no Python findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Backlog task is linked to the spec and final verification notes.
- [ ] #2 Design spec is committed on an isolated branch from dev.
- [ ] #3 Spec review/design risk review issues are addressed or recorded.
- [ ] #4 No production behavior changes are introduced in the design-only slice.
<!-- DOD:END -->
