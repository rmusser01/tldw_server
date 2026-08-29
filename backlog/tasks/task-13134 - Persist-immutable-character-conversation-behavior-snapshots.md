---
id: TASK-13134
title: Persist immutable character-conversation behavior snapshots
status: Done
assignee: []
created_date: '2026-08-28 05:06'
updated_date: '2026-08-29 04:02'
labels:
  - character-chat
  - api
  - persistence
  - roleplay-resume
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook'
  - >-
    backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md
documentation:
  - >-
    Docs/superpowers/plans/2026-08-27-character-conversation-behavior-snapshot-contract.md
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 canonical snapshot checkpoint completed. TDD evidence: initial collection RED because the module was absent; fail-closed/hardening RED reached 10 failing new cases; final targeted suite passed 49 tests. Independent specification review PASS and code-quality review PASS after hardening. Targeted Ruff, Bandit, and git diff checks passed. Rebased commits: aa5722f90e and a2acc9d8c0.
Stage 2 storage checkpoint completed. After rebasing onto `origin/dev` at 1ad2f1e5b3, preserved current-dev schema v55-v63 and allocated the next free ChaCha schema v64 from verified v63 for SQLite and PostgreSQL. Added fail-closed immutable snapshot storage, explicit legacy missing reads, settings/history fences, caller-owned transaction seams, and exactly-once prompt-history fencing including image, metadata/pin, and atomic Sync append/tombstone paths. TDD began with migration failures and successive focused RED cases for integrity/atomicity gaps; post-rebase v64 compatibility began with 9 focused failures. Final rebased migration/schema evidence: 65 passed and 3 existing live-PostgreSQL-dependent tests skipped; the current-dev v61 migration file passed 12 tests. Independent specification and quality/security reviews found no code-level blocker. Rebased commits: 08cc13f643, d5142b3b75, 6bb8adb33d, 158cc959aa, and compatibility commit 420b754d8d.
Stage 3 creation/readiness checkpoint completed. Added one atomic resumable character-conversation factory, immutable multi-participant behavior snapshots with prompt/generation provenance, creation-time effective provider/model/per-field sampling materialization, explicit non-resumable legacy/unsupported paths, detail-only authoritative readiness and fences, reserved readiness mutation protection, and credential-safe snapshot/settings boundaries. TDD began with the absent factory and then focused RED cases for configuration, drift, rollback, aliases, tampering, auth, list/detail boundaries, and secret classification. Final pre-rebase evidence included 136 canonical-plus-creation integration passes and 267 broader passes with 3 skips; the post-rebase matrix below revalidated these paths against current dev. Rebased commits: 9b4ef46f17, 73326b2bdb, 3260ae4dc5, d6d5cc202b, 93af9114d7, 70f0d0c51e, and c17338cc1e.
Stage 4 materialized-settings and coherent-fence checkpoint completed. Behavior settings now embed versioned canonical authority for effective provider/model/sampling, scope-correct presets, overlays, participants, greetings, world books, author notes, and owner-scoped persona memory; every successful behavior mutation advances `settings_version`, while central history mutations advance `history_version` exactly once. Resume readiness fails closed unless materialized authority matches the immutable snapshot, PostgreSQL source reads use bounded stable double-collection, and PostgreSQL preset/world-book reads require tenant proof. The atomic factory derives ownership from the scoped database and rejects conflicting caller identity. Review hardening routed settings writers through one owner-scoped transactional state loader, fenced derived summary writes against both settings and history, kept greeting materialization bound to the current primary-character identity, and made pin edits follow PostgreSQL message -> metadata -> conversation lock order. Shared metadata and RAG merges now create/lock the metadata row before reading so a stale merge cannot erase a concurrent pin. Final controller evidence after rebasing onto `origin/dev` at `1ad2f1e5b3`: the exact six-file contract matrix passed 377 tests with 13 warnings in 424.42 seconds; Bandit exited 0 across all touched production modules; compileall and `git diff --check` passed. Ruff reported no findings in the new TASK files or the final metadata store/test correction and no new findings versus the current-dev baseline in existing files. Independent final reviews returned SPEC PASS and QUALITY PASS. PostgreSQL-specific authorization, isolation, and interleaving coverage uses PostgreSQL-shaped fakes rather than a live PostgreSQL service; a full repository sweep was not run because repository policy requires targeted verification unless explicitly requested. No TASK-13135 work was introduced. Rebased commits: eb8cfaefad, 81f956be7b, 6dd7f5885a, 5f7e343687, 048ee4f250, 97f1b7267e, 0241106db6, 78e2379804, 64a044781a, 4e1cd7b8d8, 503d1a8143, d65cecce5e, and 3bac7d9aa5.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

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
