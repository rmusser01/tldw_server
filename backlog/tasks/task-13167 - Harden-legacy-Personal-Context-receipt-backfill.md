---
id: TASK-13167
title: Harden legacy Personal Context receipt backfill
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 03:32'
updated_date: '2026-09-04 06:44'
labels:
  - personal-context
  - sync
  - security
  - relay
dependencies:
  - TASK-13160
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
Remediate TASK-13161 by permitting legacy empty wire-identity backfill only after the stored receipt and decrypted canonical source prove the complete historical publication identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An empty legacy wire identity is accepted only after exact comparison of every old receipt fact plus decrypted source domain, role, operation, batch size, canonical digest, profile, generation, object, internal version, manifest revision, wire version, batch, and sequence.
- [x] #2 Validation compares the current authoritative manifest and journal binding before any receipt mutation.
- [x] #3 Backfill is one transactional compare-and-set from empty only, checks the affected row count, and every subsequent use follows the strict modern identity path.
- [x] #4 Any mismatch, corrupt ciphertext, changed key, or stale manifest leaves the receipt unchanged and performs no mutation, acknowledgement, or replay.
- [x] #5 Real SQLite tests cover matching legacy rows and record, scope, manifest, pending-state, terminal-state, corrupt-ciphertext, and changed-key mismatches.
- [x] #6 Errors, logs, persistence metadata, and test diagnostics remain content-free for protected profile data.
- [x] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs publication identity and receipt integrity.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED legacy mismatch tests. 2. Validate the complete stored receipt against decrypted source and current manifest. 3. Backfill the empty wire identity with one checked CAS. 4. Run targeted security and regression checks. 5. Self-review and close the task. ADR required: no new ADR; ADR-002 governs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented fail-closed legacy empty-wire receipt validation at the existing publication-journal seam. The validator authenticates and compares exact SQLite types and values across the receipt, source row, manifest sibling, deterministic batch shape and relay state, canonical historical manifest and parent lineage, and current manifest before one checked empty-only CAS. Review round 1 restored the pre-TASK-13167 modern nonempty-receipt source verification path and added real-SQLite record, scope, manifest, pending-proposal, and terminal-proposal coverage. Review round 2 replaced raw protected-body assertion operands with digests and fixed parse diagnostics. Review round 3 made the diagnostic canary itself fail content-free: all safety properties reduce to one boolean, unsafe results use a fixed no-trace pytest failure, and parse helpers raise outside the Pydantic handler so no cause or context is retained. Rounds 2 and 3 changed tests only; production behavior is unchanged. No schema, migration, repository API, dependency, or new ADR was required; ADR-002 governs. Verification: the canary passed; 39 focused receipt tests and 60 combined publication tests passed with 7 existing runtime/configuration warnings; Ruff passed; Bandit passed with no findings; git diff --check passed. The full suite was not run per the task plan; no blockers remain.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Legacy Personal Context receipt backfill now proves complete authenticated historical publication identity before changing only an empty wire version, while modern nonempty receipts retain their original replay path.
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
