---
id: TASK-13163
title: Resolve Personal Context conflicts through the batched Sync API
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-03 13:40'
updated_date: '2026-09-05 21:42'
labels:
  - personal-context
  - sync
  - conflicts
  - security
dependencies:
  - TASK-13162
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the existing batched Sync conflict API for ongoing Personal Context conflicts while preserving both immutable candidates and routing every mutating decision through canonical Personalization authority. Ordinary conflicts and semantic-key collisions remain narrowly frozen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Personal Context push conflict creates or reuses a deterministic protected home-authority candidate before returning a terminal conflict result.
- [ ] #2 Conflict records carry expected local and remote envelope IDs; stale reviews are rejected without mutating canonical state or resolving the generic conflict.
- [ ] #3 The batched endpoint implements skip, overwrite, and duplicate_rename only, with canonical overwrite, merge payload, and duplicate decisions routed through PersonalContextService.
- [ ] #4 Mutating decisions use idempotent Personalization replay receipts so interruption cannot duplicate a version, manifest advance, publication batch, merge, or renamed record.
- [ ] #5 Ordinary conflicts freeze one object; key collisions freeze both object IDs and only the contested semantic-key slot while unrelated objects continue.
- [ ] #6 Candidate retention, replay, stale-review, key-collision, batch-partial-failure, authorization, and plaintext-canary tests pass.
- [ ] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs conflict authority and retention.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md. Reason: implement approved ongoing-sync conflict authority and retention. Follow Task 2 in Docs/superpowers/plans/2026-09-03-personal-context-ongoing-sync-02-server-activation-conflict-purge.md after auditing current activation/publication seams. First verify current behavior, identify durable encrypted receipt/freeze storage and migration requirements, and refine exact integration steps without changing approved wire actions. Then add failing candidate retention and replay/stale-review tests, implement canonical batched resolution plus exact freezes, run targeted SQLite/PostgreSQL/canary/authorization tests, and obtain spec and quality review. Keep ongoing_sync_version=0. Isolated branch starts at the verified TASK13192 converter fix 6363466d07; no merge is claimed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Continuation checkpoint: approved detail plan Docs/superpowers/plans/2026-09-05-personal-context-conflict-resolution-detail.md adds missing canonical encrypted journal/freeze and Sync retention ownership; independent plan review approved. Existing ongoing wire-contract baseline: 12 passed. Implementation paused before code edits for an owner decision: on a semantic-key collision between distinct local and shared object IDs, does overwrite/Keep local authorize retiring the shared record and installing the local ID, or must that shared record remain and the local copy use duplicate_rename? Approved spec defines duplicate_rename but does not resolve this destructive identity choice. No default deletion, action restriction, or capability enablement introduced. Investigation found generic encrypted object_versions could own private journals under existing purge/rotation inventory; verify in implementation after choice. TASK13192 committed6363466d07 is this isolated branch base, not yet merged.

Owner clarification resolved: user explicitly chooses deconfliction outcome; no automatic local/server winner. Keep shared, keep local values, reviewed merge, or explicitly distinct keep-both. For same-key distinct IDs, keep-local/merge explicitly targets established shared canonical identity; incoming duplicate is accounted for by exact receipt, not silently installed alongside it. Spec and detail plan updated; continuation authorized. Supersedes prior paused-decision checkpoint.
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
