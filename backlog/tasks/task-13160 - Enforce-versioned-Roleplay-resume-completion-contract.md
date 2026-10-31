---
id: TASK-13160
title: Enforce versioned Roleplay resume completion contract
status: To Do
assignee: []
created_date: '2026-08-28 05:06'
updated_date: '2026-09-05 01:01'
labels:
  - character-chat
  - api
  - streaming
  - roleplay-resume
dependencies:
  - TASK-13159
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
Expose an exact capability-gated character completion contract that consumes the immutable conversation behavior snapshot, binds generation to authoritative settings and message fences, and persists assistants without silent replay or implicit branching.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Authenticated capability discovery advertises roleplay_resume_contract_version >= 1 with required snapshot_completion, fenced_completion, idempotent_user_append, and nonstream_assistant_persist features; route or OpenAPI presence alone never proves support.
- [ ] #2 Caller-selected user message IDs are idempotent for identical appends; a new append atomically compare-and-swaps expected snapshot digest, settings version, prior tail ID/version, history_version, and resume eligibility, drift inserts no row/branch with structured 409, conflicting ID reuse returns structured 409, and authoritative ID/version/fences are returned for reconciliation.
- [ ] #3 Snapshot-required completion accepts no prompt append or current-card behavior overrides and atomically checks expected snapshot digest, settings version, history_version, exact input user ID, tail ID, and tail version before provider dispatch.
- [ ] #4 Non-streaming commit and optional streamed persist compare-and-swap the server-issued generation fence including history_version; earlier-message edit/delete or any concurrent mutation creates no assistant or implicit branch and returns recoverable generated content with saved false.
- [ ] #5 Non-streaming persistence is a required base feature; streaming is advertised only with valid dedicated server-only current/secondary signing material and uses a stable assistant ID plus an opaque short-lived HMAC grant bound to owner, scope, conversation, authoritative user parent/input, full generation fence, and final content digest; the client-known single-user API key, public/default/placeholder material, and missing/short secrets cannot enable or sign the feature.
- [ ] #6 Structured missing/invalid snapshot, policy, drift, saved, unknown, validation-degraded, grant-tamper/expiry, cross-user, cross-conversation, and idempotent identical-replay outcomes are documented and covered by targeted endpoint and concurrency tests.
- [ ] #7 Current card, preset, lore/world-book, note/memory, participant, exemplar, settings, ancestor-message edit/delete, append-race, active-Sync readiness, and mutation-during-provider tests prove historical isolation and fail-closed persistence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-09-04: With explicit user approval, renumbered this task from its colliding former ID TASK-13135 to TASK-13160. Its snapshot prerequisite is now TASK-13159. Scope and acceptance criteria are unchanged; implementation remains pending the prerequisite merge.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 The accepted server ADR and implementation plan are linked.
<!-- DOD:END -->
