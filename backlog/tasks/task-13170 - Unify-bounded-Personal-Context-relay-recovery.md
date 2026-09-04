---
id: TASK-13170
title: Unify bounded Personal Context relay recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 03:34'
updated_date: '2026-09-04 18:12'
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
1. Trace every Personal Context authority proof read through publication and Sync stores, including semantic, manifest, and ingress-backed authority. 2. Add RED boundary/deadline tests that count each returned proof row and forbid later proof queries, restoration, exposure, or cursor advancement once the shared 100-row/100 ms budget closes. 3. Add RED legacy mixed-pull coverage proving include_own_changes=True returns a valid non-PC row whose device_id is NULL. 4. Thread the existing shared budget/deadline through the verified proof chain or reuse already-loaded rows, and minimally correct the nullable-device visibility predicate without weakening TASK-13166/TASK-13167. 5. Run focused RED to GREEN, the Task 5 budget/transport/relay/recovery/service matrices and targeted mixed tests, then Ruff, Bandit, and diff checks. 6. Update TASK-13170 and the task-5 report, self-review, commit one scoped fix, and confirm a clean worktree. ADR required: no new ADR; ADR-002 and the existing authority verification contracts govern these bounded enforcement corrections.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added one mutable Personal Context recovery budget per pull and passed the same object through publication source selection/decryption, relay, legacy raw scanning, signed stream scanning, and page-plus-one lookahead.
- Enforced the absolute deadline and remaining-row allowance before each query, source decrypt, raw classification, and authority restoration; zero-limit source calls are impossible and inspected-but-unsafe rows remain watermark barriers.
- Preserved legacy and per-stream signed cursors, requested-domain subsets, hidden-ingress filtering, conflict/pending barriers, Notes ordering, existing cleanup authorization, and inactive ongoing-sync version 1.
- Added deterministic boundary, multi-batch, deadline, mixed-mode, subset, lookahead, and watermark regression coverage. Targeted tests passed: budget `17`, transport `17`, relay recovery `32`, relay compatibility `6`, and Sync service `165`. Ruff and `git diff --check` passed; Bandit exited 0 with only existing parser/accepted `nosec` warnings.
- Review round 1 closed the remaining deadline and watermark races. Relay now rechecks the same absolute deadline after every successful lease/current-row validation, before stage, publication-state writes, and finalization. Both pull scanners classify Personal Context rows through one fail-closed boundary: only an exact canonical ingress receipt or cleanup's complete content-free structural shape is permanently hidden; malformed, type-drifted, role-tampered, and unattested stale authority rows remain barriers.
- Review RED evidence was `6 failed` for deadline flips inside current-row validation, `8 failed` for missing/malformed routing, generation type drift, and role tampering across legacy and signed pulls, and `1 failed` for a deadline crossed during finalization before the batch-completion write. Final focused verification passed: budget `39`, transport `17`, relay recovery `32`, relay compatibility `6`, and Sync service `165`. Scoped Ruff and `git diff --check` passed; Bandit exited 0 with only its existing comment-parser and accepted `nosec B608` warnings. The deterministic integrated deadline fixture was corrected from a 100 ns unit scale to the service's 100 ms scale before the final GREEN run.
- Review round 2 removed the remaining trust in mutable applied authority metadata. Pull now spends the same shared budget on an exact acknowledged canonical source lookup, authenticates the journal row, then reuses the TASK-13166 authority tag and cross-store receipt verifier before restoration. Personal Context own-device rows are no longer pre-excluded by SQL, restoration is fenced again after successful decrypt, and relay compensation/lease operations stop before any subsequent read or write once the deadline closes. Legitimate empty-wire authority remains on the existing verified receipt path; no weaker fallback was added.
- Round-2 focused RED evidence was `15 failed, 8 passed`: six failures covered batch identity, server origin, and encrypted-ingress relabeling across legacy/signed pulls; four covered own-device barriers; two covered post-restore expiry; and three covered relay compensation/renewal expiry. The eight already-green cases confirmed existing profile, generation, key, and role guards. Final focused verification was `23 passed, 39 deselected`; full targeted matrices passed: budget `62`, transport `17`, relay recovery `32`, relay compatibility `6`, and Sync service `165`. The exact boundary uses 96 relay rows plus a raw/source authority pair plus one attested hidden ingress for 100 total; row 101 remains deferred and no zero-limit lookup occurs. Transport authority fixtures now use the real encrypted journal-to-relay provenance path. Scoped Ruff and `git diff --check` passed; Bandit exited 0 with only existing comment-parser and accepted `nosec B608` warnings.
- Review round 3 closes unmetered authority-proof reads. The pull-owned budget now charges every returned acknowledged source, Sync ingress receipt, canonical receipt, publication batch, companion manifest, origin publication, origin Sync envelope, and origin-base Sync envelope—even when that I/O closes the deadline. Each later read is pre-fenced, two-row uniqueness probes require two remaining slots, and the already-authenticated source/origin identities are reused instead of reread. Semantic ingress consumes six rows including the raw authority; manifest ingress consumes eight, reaching a valid exact-100 result after 92 relay rows while relay row 93 defers proof row 101 without exposure or cursor movement.
- Round-3 corrected RED evidence was `11 failed, 1 passed`: four proof-budget/row-101 cases exposed or undercounted authority, six publication/Sync deadline cases launched later proof reads or failed to charge returned rows, and legacy nullable-device delivery failed while its signed counterpart already passed. A focused self-review RED (`1 failed`) then proved acknowledged-source I/O expiry also needed returned-row charging. Final matrices passed: budget `75`, authority identity `62`, transport `17`, relay recovery `32`, relay compatibility `6`, and Sync service `165`. Scoped Ruff and `git diff --check` passed; Bandit exited 0 with only its existing comment-parser and accepted `nosec B608` warnings.
- Review round 4 closes the last two unmetered proof boundaries. Hidden-ingress attestation now receives the same pull budget in legacy and signed scans, fences the Sync receipt query before and after I/O, and charges every returned receipt once before validating it. One attested hidden ingress therefore costs two rows: its raw envelope and its receipt. After 98 relay rows, raw row 99 plus valid receipt row 100 may advance safely; after 99 relay rows, raw row 100 is a barrier and receipt row 101 is not queried. Missing, mismatched, malformed, or post-I/O-expired receipts never advance or expose the hidden ingress.
- Companion-manifest proof now uses the existing publication-row primary key `(profile_id, profile_publication_sequence, batch_ordinal)` at the canonical last ordinal. That outer lookup can return at most one row, while a `NOT EXISTS` sibling-manifest guard preserves duplicate-role rejection; the returned exact row is charged before its role, batch, generation, ordinal, and version facts are accepted. Wrong-ordinal substitutes and duplicate manifest roles remain barriers. Legacy and signed semantic/purge authority chains now complete exactly at relay `94` + raw `95` + acknowledged source `96` + Sync receipt `97` + canonical receipt `98` + publication batch `99` + companion manifest `100`; relay `95` defers the companion as row `101` without another query or watermark movement.
- Round-4 corrected RED evidence was `16 failed, 12 passed`: all 16 failures mapped to uncharged/unfenced hidden-ingress receipts or exact-row-100 companion manifests across legacy and signed pulls. The focused GREEN run passed `28`; the full recovery-budget file passed `103`; authority identity, transport, relay recovery, and relay compatibility passed `117` combined (`62 + 17 + 32 + 6`); Sync service passed `165`. Scoped Ruff and `git diff --check` passed; Bandit exited 0 with only its existing comment-parser and accepted `nosec B608` warnings.
- Review round 5 makes duplicate-manifest uniqueness a common receipt proof for semantic, purge, and manifest-source authority. The already-budgeted publication-batch query now returns an exact correlated manifest-role count; its one batch row is charged before the count is strictly validated as integer `1`. Manifest authority reuses its authenticated source row without another read or charge, while an out-of-range sibling manifest fails closed just like the semantic/purge paths. The composite-PK one-slot companion lookup, wrong-ordinal barriers, deadline fences, and exact-100/101 accounting are unchanged.
- Round-5 RED was `2 failed, 6 passed`: legacy and signed manifest authority were exposed across a duplicate sibling, while clean exact-100, row-101, and batch-I/O deadline guards already passed. Focused GREEN passed `8`; the full recovery-budget file passed `109`; authority identity, transport, relay recovery, and relay compatibility passed `117` combined (`62 + 17 + 32 + 6`); Sync service passed `165`. Scoped Ruff and `git diff --check` passed; Bandit exited 0.
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
