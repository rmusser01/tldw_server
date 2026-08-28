---
id: TASK-13134
title: Persist immutable character-conversation behavior snapshots
status: In Progress
assignee: []
created_date: 2026-08-28 05:06
updated_date: 2026-08-28 08:41
labels:
- character-chat
- api
- persistence
- roleplay-resume
dependencies: []
references:
- https://github.com/rmusser01/tldw_chatbook
- backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md
documentation:
- Docs/superpowers/plans/2026-08-27-character-conversation-behavior-snapshot-contract.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist a complete historical behavior snapshot for each newly created character conversation so later completion can reproduce saved character behavior without rereading mutable cards, presets, lorebooks, exemplars, or other source records.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every production character-conversation creator uses one atomic factory that stores a versioned, digested behavior snapshot for every participant or rolls back; explicitly unsupported/active-Sync paths remain non-resumable and cannot advertise per-conversation readiness.
- [x] #2 The snapshot classifies and captures every immutable character-behavior input while excluding credentials, portrait binaries, and other secrets; mutating or deleting source records cannot alter the stored behavior.
- [x] #3 Conversation detail exposes snapshot status, schema version, digest, monotonic settings_version, monotonic history_version, and message/tail identity-version fences; legacy missing or invalid snapshots remain explicit and are never silently backfilled.
- [x] #4 Settings version 1 materializes the effective creation-time provider, model, and explicit sampling values or marks Resume ineligible before append; later deployment-default changes cannot alter completion. Behavior-affecting settings materialize referenced preset, overlay, lore/world-book, memory, participant, and related values when applied and increment settings_version on every mutation; every message-history add/edit/delete/restore/branch/tail mutation advances history_version transactionally.
- [x] #5 Targeted migration, transaction, all-creator, active-Sync readiness, source/deployment-default mutation isolation, incomplete-effective-settings, ancestor-history mutation, legacy-status, multi-participant, size-bound, and authorization tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define and test the canonical version-1 behavior snapshot and complete input classification.
2. Recheck the current schema head, allocate the next free version for SQLite and PostgreSQL, then route every production character-conversation creator through atomic snapshot creation or explicit non-resumable readiness.
3. Materialize creation-time effective provider/model/sampling plus later behavior settings, add monotonic history_version across central message mutations, expose coherent snapshot/settings/history/tail fences, run targeted tests and Bandit, and close TASK-13134 before TASK-13135 begins.

ADR required: yes
ADR path: backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md
Reason: This changes persistent schema, historical-data policy, and conversation behavior authority.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 ADR for behavior-snapshot storage and authority is accepted and linked.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 canonical snapshot checkpoint completed. TDD evidence: initial collection RED because the module was absent; fail-closed/hardening RED reached 10 failing new cases; final targeted suite passed 49 tests. Independent specification review PASS and code-quality review PASS after hardening. Targeted Ruff, Bandit, and git diff checks passed. Commits: 2b17ba3819 and 5bae47f9b6.
Stage 2 storage checkpoint completed. Allocated ChaCha schema v55 from verified v54 head for SQLite and PostgreSQL; added fail-closed immutable snapshot storage, explicit legacy missing reads, settings/history fences, caller-owned transaction seams, and exactly-once prompt-history fencing including image, metadata/pin, and atomic Sync append/tombstone paths. TDD began with 9/9 migration failures and successive focused RED cases for integrity/atomicity gaps. Final evidence: required migration/PostgreSQL suite 43 passed, existing Sync tests 9 passed, writable-path message/hydration regressions 33 passed, independent specification review PASS, independent quality/security review PASS, Ruff/Bandit/diff checks passed with only documented pre-existing lint debt. Commits: c9dbdf1bde, 2788284c4a, ff4ea7ac80, c15b8c4173.
Stage 3 creation/readiness checkpoint completed. Added one atomic resumable character-conversation factory, immutable multi-participant behavior snapshots with prompt/generation provenance, creation-time effective provider/model/per-field sampling materialization, explicit non-resumable legacy/unsupported paths, detail-only authoritative readiness and fences, reserved readiness mutation protection, and credential-safe snapshot/settings boundaries. TDD began with the absent factory and then focused RED cases for configuration, drift, rollback, aliases, tampering, auth, list/detail boundaries, and secret classification. Final evidence: independent unfiltered canonical-plus-creation integration run 136 passed; broader unfiltered Task 3 run 267 passed with 3 skips before the final alias-only delta; implementer compatibility run 191 passed with 3 skips; Ruff introduced no new findings, Bandit reported zero issues, compileall/diff checks passed, and definitive specification and quality reviews both PASS. Commits: 5619f0f4e7, bd2ac72ce8, 1e25dd5d7d, 569b11bf52, 00eb0e9833, cbf50150d0, 4c2e051e95.
Stage 4 materialized-settings and coherent-fence checkpoint completed. Behavior settings now embed versioned canonical authority for effective provider/model/sampling, scope-correct presets, overlays, participants, greetings, world books, author notes, and owner-scoped persona memory; every successful behavior mutation advances `settings_version`, while central history mutations advance `history_version` exactly once. Resume readiness fails closed unless materialized authority matches the immutable snapshot, PostgreSQL source reads use bounded stable double-collection, and PostgreSQL preset/world-book reads require tenant proof. The atomic factory derives ownership from the scoped database and rejects conflicting caller identity. Final controller evidence: 273 targeted tests passed in 184.85 seconds; Bandit, compileall, and diff checks passed; Ruff introduced no findings beyond the documented pre-existing baseline. Definitive independent specification review PASS and quality review PASS reported no critical or important findings. PostgreSQL-specific authorization/isolation coverage uses PostgreSQL-shaped interleaving fakes rather than a live PostgreSQL service; a full repository sweep was not run because repository policy requires targeted verification unless explicitly requested. No TASK-13135 work was introduced. Commits: 384d309b07, 73f65b1986, 9b47435996, ad794f09f0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
