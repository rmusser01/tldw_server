---
id: TASK-2248
title: Backfill Security request-edge middleware ADR
status: To Do
dependencies:
- TASK-2247
labels:
- docs
- process
- adr
- security
modified_files:
- Docs/ADR/
- Docs/ADR/inventory/2026-06-03-decision-inventory.md
- tldw_Server_API/app/core/Security/README.md
- backlog/tasks/
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill the first bounded Security ADR from TASK-2247 evidence. Scope the accepted decision to request-edge Security middleware only: normal startup installs setup access guard/CSP and security headers, RequestIDMiddleware and DrainGateMiddleware are always installed, CSP is path-sensitive, production defaults security headers on when ENABLE_SECURITY_HEADERS is absent, and caveats are explicit for test mode, security-header disablement, HSTS opt-in/HTTPS behavior, and relaxed Setup CSP/eval defaults. Do not include outbound egress or secret/serialization policy in this ADR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create the next accepted ADR under `Docs/ADR/` for Security request-edge middleware using the standard ADR template and TASK-2247 evidence.
- [ ] #2 Keep accepted claims scoped to request-edge middleware startup wiring, request IDs, drain gate, setup guard/CSP, security headers, and documented caveats.
- [ ] #3 Update `Docs/ADR/README.md`, the INV-029 inventory row, and relevant Security README backlink after ADR creation.
- [ ] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Use TASK-2247's confirmation audit as the evidence boundary. Create ADR-019 unless another ADR number has appeared on dev. Keep egress/SSRF and secrets/serialization out of scope except as alternatives/follow-up. Update the ADR index, inventory row, and `tldw_Server_API/app/core/Security/README.md` backlink. Run Markdown/link checks and targeted Security middleware tests before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
