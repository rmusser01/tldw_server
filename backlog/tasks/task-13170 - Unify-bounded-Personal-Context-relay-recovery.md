---
id: TASK-13170
title: Unify bounded Personal Context relay recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 03:34'
updated_date: '2026-09-04 16:15'
labels:
  - personal-context
  - sync
  - recovery
  - relay
dependencies:
  - TASK-13169
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
Remediate TASK-13161 by routing every Personal Context pull shape through one exact, bounded recovery coordinator with safe watermarks and no hidden-row leakage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One recovery coordinator handles legacy, signed, mixed-domain, and requested-domain-subset pulls without divergent recovery semantics.
- [x] #2 Each pull spends one exact shared budget of at most 100 inspected raw or canonical rows and at most 100 milliseconds across source selection, decryption, relay, and raw Sync scanning.
- [x] #3 The coordinator never calls a source lookup with a zero limit; exactly 100 inspected rows returns a valid completed or pending result and row 101 is deferred.
- [x] #4 The deadline is enforced during incremental source lookup and decryption, not only between batches.
- [x] #5 Page-plus-one lookahead, true exhaustion, hidden-ingress filtering, pending barriers, and per-stream signed safe watermarks cannot skip or expose eligible rows.
- [x] #6 Non-Personal-Context domains and requested subsets retain their delivery order and cursors while eligible authority rows are restored and decrypted only after successful recovery.
- [x] #7 Real tests cover exact source-only, raw-only, and combined 100 and 101 boundaries, deadline expiry, multiple batches, hidden prefixes, pending barriers, lookahead, subsets, and mixed legacy and signed datasets.
- [x] #8 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs bounded recovery and Personal Context egress.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED deadline-race tests that expire during lease/current validation before stage, finalize, and publication-state writes. 2. Add RED legacy and signed tamper tests for missing or malformed routing, generation type drift, and authority-role tampering while preserving proven canonical client ingress and intended stale-row handling. 3. Add the minimum shared deadline rechecks and fail-closed authenticated safe-skip classification required by those tests. 4. Run the Task 5 budget and affected relay/transport/service regressions, Ruff, Bandit, and diff integrity checks. 5. Update TASK-13170 and the task-5 report, self-review the authority/watermark boundary, and commit the scoped review fix. ADR required: no new ADR; ADR-002 already governs this correction.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added one mutable Personal Context recovery budget per pull and passed the same object through publication source selection/decryption, relay, legacy raw scanning, signed stream scanning, and page-plus-one lookahead.
- Enforced the absolute deadline and remaining-row allowance before each query, source decrypt, raw classification, and authority restoration; zero-limit source calls are impossible and inspected-but-unsafe rows remain watermark barriers.
- Preserved legacy and per-stream signed cursors, requested-domain subsets, hidden-ingress filtering, conflict/pending barriers, Notes ordering, existing cleanup authorization, and inactive ongoing-sync version 1.
- Added deterministic boundary, multi-batch, deadline, mixed-mode, subset, lookahead, and watermark regression coverage. Targeted tests passed: budget `17`, transport `17`, relay recovery `32`, relay compatibility `6`, and Sync service `165`. Ruff and `git diff --check` passed; Bandit exited 0 with only existing parser/accepted `nosec` warnings.
- Review round 1 closed the remaining deadline and watermark races. Relay now rechecks the same absolute deadline after every successful lease/current-row validation, before stage, publication-state writes, and finalization. Both pull scanners classify Personal Context rows through one fail-closed boundary: only an exact canonical ingress receipt or cleanup's complete content-free structural shape is permanently hidden; malformed, type-drifted, role-tampered, and unattested stale authority rows remain barriers.
- Review RED evidence was `6 failed` for deadline flips inside current-row validation, `8 failed` for missing/malformed routing, generation type drift, and role tampering across legacy and signed pulls, and `1 failed` for a deadline crossed during finalization before the batch-completion write. Final focused verification passed: budget `39`, transport `17`, relay recovery `32`, relay compatibility `6`, and Sync service `165`. Scoped Ruff and `git diff --check` passed; Bandit exited 0 with only its existing comment-parser and accepted `nosec B608` warnings. The deterministic integrated deadline fixture was corrected from a 100 ns unit scale to the service's 100 ms scale before the final GREEN run.
- ADR required: no. ADR-002 already governs bounded recovery and Personal Context egress. No schema, migration, dependency, public protocol, activation, broad cleanup, or generalizable new lesson was introduced.
- Known skip: the full repository suite was not run, per the task's targeted-verification scope.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
