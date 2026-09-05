---
id: TASK-13172
title: Certify Personal Context relay and close TASK-13161
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 03:35'
updated_date: '2026-09-04 23:55'
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
- [ ] #1 Production server-factory and TestClient flows prove direct canonical mutation, client ingress, authority publication, encrypted pull, and exact-once replay through real persistence boundaries.
- [ ] #2 After-commit relay failure is isolated from the accepted ingress response, durable debt survives restart, recovery runs before a later push or pull, and the authority result is published exactly once.
- [ ] #3 The crash, race, poison, receipt, exact-budget, activation-proof, conflict, and retention matrices from TASK-13166 through TASK-13171 pass together without weakening their assertions.
- [ ] #4 The single authoritative Personal Context dataset invariant is enforced and tested across configuration and runtime lookup.
- [ ] #5 No client-ingress row, hidden pending authority row, plaintext protected value, wrapped key, or content-derived diagnostic can leave its authorized boundary, and version-zero behavior remains compatible.
- [ ] #6 Targeted pytest suites, Ruff, Bandit, artifact scans, and git diff checks pass; the full repository suite is not run unless the user explicitly requests it.
- [ ] #7 TASK-13161 is marked Done only after independent specification and code-quality review accepts every original criterion and Definition-of-Done item; the final report, documentation, and any incident-backed lesson are updated.
- [ ] #8 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md remains the governing decision.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a production-factory certification test that uses real temporary Personalization and Sync databases, production personal_context_service_for_user() and sync_v2_service_for_user() dependencies, FastAPI wiring, TestClient, and unique privacy canaries. 2. Establish the focused RED or GREEN against the approved production base; if RED identifies a production defect, stop and report the owning task before any production edit. 3. Prove ordered direct create/update publication, hidden client ingress with exact receipt, accepted HTTP behavior across injected after-commit relay failure, durable restart recovery before later push and pull, verified exact-once home-authority egress/decryption, checkpoint progression, privacy boundaries, version-zero/budget compatibility, and the single authoritative dataset invariant across bootstrap and runtime. 4. Run the exact 14-file targeted remediation matrix, scoped Ruff/Bandit/diff checks, and stable-path artifact/canary scans; do not run the full suite. 5. Produce the Phase A SDD certification report and progress evidence and commit a review candidate while keeping TASK-13172 and TASK-13161 In Progress with closure pending two independent reviews. 6. After both reviews approve in a later closure phase, complete task records and close TASK-13161. ADR required: no new ADR. backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs canonical authority, relay journaling, encryption, purge custody, activation, and the external-backup boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase A review round 1 is being remediated; TASK-13172 and TASK-13161 remain In Progress and closure-only AC/DoD remain unchecked pending independent re-review. The initial candidate's authority bind fence is retained, including same-target idempotence, exact-one fail-closed runtime lookup, and a pre-created-candidate concurrency precondition. Review RED/GREEN added the missing pre-effect boundary: sole-authority resolution, deterministic conditional default creation, domain enrollment/state creation, watermark capture, and authoritative binding now share one rollback-capable Sync transaction, so an existing non-default authority is reused without default creation and an interleaved rejection leaves no dataset/domain side effect. Certification diagnostics use content-free boolean/fixed-digest comparisons; a forced-failure subprocess proves captured pytest output excludes all protected canaries. The combined lifecycle uses production `main.app` router registration, leaves both production service dependency functions intact, overrides authentication only, and proves factory/cache traversal. The repository PostgreSQL fixture executed a real synchronized two-connection bind race (1 passed in 4.72s) with exactly one winner. The stable artifact run passed 1/1 in 26.20s at `/tmp/task13172-review1-green-artifacts-rerun`, scanning active Personalization/Sync/Notes DB/WAL/SHM files before both backend resets and at final-active state, plus controlled content-free log, diagnostic, migration snapshot, and application-owned SQLite backup fixtures. Plaintext, ingress, diagnostic-marker, and raw-key canaries were absent; wrapped material appeared only in the authorized Sync DB/WAL or encrypted backup boundary. Its path-only inventory records explicit custody limits and excludes external/operator backups, exported recovery bundles, and prior-process memory without claiming deletion outside application custody. Final 14-file/static/self-review gates and a fix-round commit remain pending. No full suite is authorized. ADR required: no new ADR; ADR-002 continues to govern.
Review-round-1 final candidate evidence: post-self-review certification passed 9/9 with 4 warnings in 114.19s at `/tmp/task13172-review1-final-certification`, with PostgreSQL required and executed. The exact 14-file matrix passed 757/757 across groups 162+93+215+287. Ruff passed; Bandit exited 0 with existing parser/nosec warnings; `git diff --check` passed. The final phase-aware inventory is `/tmp/task13172-review1-final-certification/test_production_http_relay_deb0/certification-evidence/artifact-inventory.json`; an independent scan found no protected canary, while wrapped material was classified only in authorized Sync DB/WAL and controlled encrypted backup custody. Eight additional owning-suite failures were reproduced 8/8 at detached approved head `cabdaf36f2`: one known fake-PostgreSQL binding baseline and seven legacy bootstrap activation-proof baselines. Their test files and the production activation gate are unchanged by this candidate; none was weakened or fixed. Self-review corrected nondeterministic set hashing and explicitly checks that rejected bootstrap does not invoke key wrapping. No full suite was run. Phase A remains pending independent re-review; TASK-13172 and TASK-13161 remain In Progress and all closure-only AC/DoD items remain unchecked.
The final evidence paragraph supersedes the preceding in-progress checkpoint's statement that gates and the fix-round commit were still pending; only the independent re-review and closure phase remain pending.
Review-round-2 remediation adds central pre-effect validation for every Personal Context authority target and stricter reuse validation for an existing deterministic Chatbook default ID. Workspace-scoped, archived, wrong-policy, invalid-default-marker, and malformed-generation targets now fail closed before dataset/domain/binding/key/envelope/cursor/link mutation; valid non-default authority reuse and same-target idempotence remain intact. The certification fixture now requires active Sync/Notes WAL+SHM observations at both pre-reset phases and retires every tracked factory-created Sync backend, including the final active instance, before proving caches empty. Final post-self-review certification passed 18/18 with genuine PostgreSQL at `/tmp/task13172-round2-final-post-self-review`; the exact 14-file matrix passed 766/766; affected bootstrap/store coverage passed 241 with 2 existing skips and only the 8 approved-head baselines explicitly deselected; Ruff, Bandit, diff, and the 39-record artifact/canary inventory passed. Self-review strengthened the rejection digest to cover every dataset column, then passed the 12-case invariant set. Phase A remains pending same-reviewer specification and code/security re-review; TASK-13172 and TASK-13161 remain In Progress and all closure-only AC/DoD boxes remain unchecked.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
