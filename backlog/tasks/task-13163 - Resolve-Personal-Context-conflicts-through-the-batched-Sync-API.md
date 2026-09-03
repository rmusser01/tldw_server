---
id: TASK-13163
title: Resolve Personal Context conflicts through the batched Sync API
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-03 13:40'
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

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
