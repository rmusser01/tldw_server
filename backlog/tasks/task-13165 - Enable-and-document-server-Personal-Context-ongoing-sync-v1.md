---
id: TASK-13165
title: Enable and document server Personal Context ongoing sync v1
status: To Do
assignee:
  - '@codex'
created_date: '2026-09-03 13:41'
labels:
  - personal-context
  - sync
  - documentation
  - rollout
dependencies:
  - TASK-13163
  - TASK-13164
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Gate ongoing_sync_version=1 on the complete server publication, relay, activation, continuity, conflict, cleanup, and purge implementation, then publish accurate operator, API, and developer documentation. This is the server rollout and certification boundary, not a new synchronization mechanism.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Readiness advertises ongoing_sync_version=1 only when every required storage, relay, identity, activation, continuity, conflict, cleanup, and generation-fence component is operational; otherwise bounded blockers accompany version 0.
- [ ] #2 Server integration tests cover direct API and agent mutations, ingress acceptance, relay recovery, activation, conflicts, restrictive cleanup, purge, restart, and capability downgrade without a Chatbook process-local callback.
- [ ] #3 The generated wire artifact and Shared Profile Core pin remain exact and reproducible, with contract fixtures covering the oldest and newest advertised compatible versions.
- [ ] #4 Operator, API, and developer guides explain trigger-based delivery, activation, status semantics, retry/relay attention, conflicts, cleanup, device removal, purge, recovery limits, and future-client integration.
- [ ] #5 Strict documentation generation, focused Personal Context and Sync suites, security checks, diff hygiene, and independent review pass.
- [ ] #6 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md and the approved ongoing-sync specification govern rollout.
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
