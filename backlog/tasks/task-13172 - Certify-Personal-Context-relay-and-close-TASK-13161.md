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
Phase A review candidate prepared; TASK-13172 and TASK-13161 remain In Progress pending two independent approvals. Initial focused RED proved TASK-13172 AC#4 defects: second authoritative dataset bootstrap mutated the target and wrote a key, and legacy duplicate runtime lookup selected a first row. The approved narrow fix moves the authoritative check into one Sync DB transaction, locks existing owner datasets in deterministic order (PostgreSQL FOR UPDATE; SQLite write transaction), rejects sibling or malformed active authority state before target/domain/key/envelope/cursor/link mutation, keeps same-dataset replay idempotent, and replaces both first-match runtime loops with exact-one fail-closed lookup. The concurrency claim requires candidate datasets to preexist; bootstrap uses one deterministic default ID. Production HTTP certification uses real per-user Personalization/Sync/ChaCha DBs, both FastAPI routers, production factories, TestClient, explicit after-commit failures, two backend/cache restarts with distinct service/backend identities, later push and pull recovery, exact canonical/Sync ingress receipts, encrypted hidden ingress, one applied home-authority decrypted egress, repeat-pull no-duplicate checkpoint, version 0, zero-limit rejection, and unique privacy canaries. Verification: focused 5 passed/6 warnings/11.83s at /tmp/tldw-task-13172-certification-focused-final; exact 14-file matrix 753 passed across groups 214+239+300; Ruff passed; Bandit exit 0 with existing parser/nosec warnings; git diff --check passed. Artifact scan covered 12 DB files; plaintext, ingress, diagnostic, and raw-key canaries absent; wrapped key present only in authorized Sync_v2.db. No WAL/SHM/log file/diagnostic file/migration snapshot/application backup was produced. External/operator backups, exported recovery bundles, and prior process memory remain excluded per TASK-13169. No full suite run. ADR required: no new ADR; ADR-002 governs. Closure-only AC/DoD remain unchecked until both reviews approve.
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
