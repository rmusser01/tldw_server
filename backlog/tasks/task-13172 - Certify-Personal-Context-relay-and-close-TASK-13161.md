---
id: TASK-13172
title: Certify Personal Context relay and close TASK-13161
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 03:35'
updated_date: '2026-09-05 02:17'
labels:
  - personal-context
  - sync
  - security
  - certification
dependencies:
  - TASK-13166
  - TASK-13167
  - TASK-13168
  - TASK-13169
  - TASK-13170
  - TASK-13171
references:
  - >-
    backlog/tasks/task-13161 -
    Relay-ordered-Personal-Context-authority-publications-through-Sync-V2.md
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove the remediated encrypted Personal Context relay through the production server factory and close TASK-13161 only after its complete acceptance and quality evidence is independently verified.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Production server-factory and TestClient flows prove direct canonical mutation, client ingress, authority publication, encrypted pull, and exact-once replay through real persistence boundaries.
- [x] #2 After-commit relay failure is isolated from the accepted ingress response, durable debt survives restart, recovery runs before a later push or pull, and the authority result is published exactly once.
- [x] #3 The crash, race, poison, receipt, exact-budget, activation-proof, conflict, and retention matrices from TASK-13166 through TASK-13171 pass together without weakening their assertions.
- [x] #4 The single authoritative Personal Context dataset invariant is enforced and tested across configuration and runtime lookup.
- [x] #5 No client-ingress row, hidden pending authority row, plaintext protected value, wrapped key, or content-derived diagnostic can leave its authorized boundary, and version-zero behavior remains compatible.
- [x] #6 Targeted pytest suites, Ruff, Bandit, artifact scans, and git diff checks pass; the full repository suite is not run unless the user explicitly requests it.
- [x] #7 TASK-13161 is marked Done only after independent specification and code-quality review accepts every original criterion and Definition-of-Done item; the final report, documentation, and any incident-backed lesson are updated.
- [x] #8 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md remains the governing decision.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a production-factory certification test that uses real temporary Personalization and Sync databases, production personal_context_service_for_user() and sync_v2_service_for_user() dependencies, FastAPI wiring, TestClient, and unique privacy canaries. 2. Establish the focused RED or GREEN against the approved production base; if RED identifies a production defect, stop and report the owning task before any production edit. 3. Prove ordered direct create/update publication, hidden client ingress with exact receipt, accepted HTTP behavior across injected after-commit relay failure, durable restart recovery before later push and pull, verified exact-once home-authority egress/decryption, checkpoint progression, privacy boundaries, version-zero/budget compatibility, and the single authoritative dataset invariant across bootstrap and runtime. 4. Run the exact 14-file targeted remediation matrix, scoped Ruff/Bandit/diff checks, and stable-path artifact/canary scans; do not run the full suite. 5. Produce the Phase A SDD certification report and progress evidence and commit a review candidate while keeping TASK-13172 and TASK-13161 In Progress with closure pending two independent reviews. 6. After both reviews approve in a later closure phase, complete task records and close TASK-13161. ADR required: no new ADR. backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs canonical authority, relay journaling, encryption, purge custody, activation, and the external-backup boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Certified the production Personal Context authority relay through real factory dependencies, FastAPI TestClient, durable Personalization and Sync stores, restart recovery, exact-once encrypted egress, checkpoint progression, privacy boundaries, and the single-authority-dataset invariant. Final candidate 8ee4c2227df533dbff3dea303c7838c1ba01d4d6 passed 25/25 certification tests with genuine PostgreSQL executed and no skip; the exact 14-file matrix passed 773/773; the affected bootstrap/store gate passed 241 with 2 existing skips and only eight approved-head baselines deselected. Scoped Ruff with no cache, Bandit, git diff, and the three-phase artifact/canary inventory passed. Both final independent reviewers approved with no Critical, Important, or Minor findings. ADR-002 remains governing; no new ADR was required. Artifact claims cover application-owned custody only and exclude external or operator backups, exported recovery bundles, and prior-process memory. Protocol version remains 0; no schema or public API was added.

Published with the certified relay in PR #2868 against dev: https://github.com/rmusser01/tldw_server/pull/2868. Verification applies to implementation 8ee4c2227d; subsequent changes only close and link task records.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-13172 certification and independent review are complete. Candidate 8ee4c2227d is approved, all targeted gates are green, documented baseline skips and custody limits are retained, and TASK-13161 is closed from the accepted child evidence.
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
