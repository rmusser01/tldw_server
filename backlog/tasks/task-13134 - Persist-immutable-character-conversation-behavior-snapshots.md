---
id: TASK-13134
title: Persist immutable character-conversation behavior snapshots
status: In Progress
assignee: []
created_date: 2026-08-28 05:06
updated_date: 2026-08-28 06:50
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
- [ ] #1 Every production character-conversation creator uses one atomic factory that stores a versioned, digested behavior snapshot for every participant or rolls back; explicitly unsupported/active-Sync paths remain non-resumable and cannot advertise per-conversation readiness.
- [ ] #2 The snapshot classifies and captures every immutable character-behavior input while excluding credentials, portrait binaries, and other secrets; mutating or deleting source records cannot alter the stored behavior.
- [ ] #3 Conversation detail exposes snapshot status, schema version, digest, monotonic settings_version, monotonic history_version, and message/tail identity-version fences; legacy missing or invalid snapshots remain explicit and are never silently backfilled.
- [ ] #4 Settings version 1 materializes the effective creation-time provider, model, and explicit sampling values or marks Resume ineligible before append; later deployment-default changes cannot alter completion. Behavior-affecting settings materialize referenced preset, overlay, lore/world-book, memory, participant, and related values when applied and increment settings_version on every mutation; every message-history add/edit/delete/restore/branch/tail mutation advances history_version transactionally.
- [ ] #5 Targeted migration, transaction, all-creator, active-Sync readiness, source/deployment-default mutation isolation, incomplete-effective-settings, ancestor-history mutation, legacy-status, multi-participant, size-bound, and authorization tests pass.
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
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 ADR for behavior-snapshot storage and authority is accepted and linked.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage 1 canonical snapshot checkpoint completed. TDD evidence: initial collection RED because the module was absent; fail-closed/hardening RED reached 10 failing new cases; final targeted suite passed 49 tests. Independent specification review PASS and code-quality review PASS after hardening. Targeted Ruff, Bandit, and git diff checks passed. Commits: 2b17ba3819 and 5bae47f9b6.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
