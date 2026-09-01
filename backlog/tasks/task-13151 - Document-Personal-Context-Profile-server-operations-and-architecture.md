---
id: TASK-13151
title: Document Personal Context Profile server operations and architecture
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-01 14:47'
updated_date: '2026-09-01 18:09'
labels: []
dependencies: []
references:
  - >-
    https://github.com/rmusser01/tldw_chatbook/blob/dev/Docs/superpowers/specs/2026-08-31-personal-context-documentation-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish accurate server operator and developer documentation for the canonical Personal Context peer, authenticated API, encrypted storage, and current Sync-v2 boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An operator guide covers authentication, master-key setup, Chatbook linking, export, server purge behavior, and current operational limitations.
- [ ] #2 A developer guide maps Shared Core parity, per-user storage, real key-custody ownership, services, API routes, Sync-v2 adapters, conflict metadata, purge fencing, the ten-item extension checklist, and targeted tests.
- [ ] #3 The existing Personal Context API reference accurately distinguishes shipped Sync-v2 support from missing server-origin publication and purge acknowledgement.
- [ ] #4 User, developer, and API indexes plus MkDocs navigation make the guides discoverable and cross-link stable Chatbook documentation.
- [ ] #5 Generated published documentation is reproducible and strict MkDocs, endpoint, custody, bootstrap, materializer, composed-app, contract, link, and diff checks pass after the final rebase.
- [ ] #6 Offline/queued, locked, incompatible, version-conflict, first-link semantic-collision, post-link semantic-collision, and purge-pending guidance is explicit and consistent with Chatbook.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebase and inventory merged behavior.
2. Add server operator guide.
3. Add developer guide and correct API reference.
4. Add indexes and MkDocs navigation.
5. Final rebase, regenerate curated docs, strict validation.
6. Complete notes and open docs-only PR.

ADR required: no
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: Documentation only; the existing Personal Context authority, Sync, and encryption ADR applies.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
