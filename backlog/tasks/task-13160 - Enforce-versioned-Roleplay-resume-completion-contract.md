---
id: TASK-13160
title: Enforce versioned Roleplay resume completion contract
status: In Progress
assignee: []
created_date: 2026-08-28 05:06
updated_date: 2026-09-05 01:01
labels:
- character-chat
- api
- streaming
- roleplay-resume
dependencies:
- TASK-13159
references:
- https://github.com/rmusser01/tldw_chatbook
- backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md
documentation:
- Docs/superpowers/plans/2026-08-27-character-conversation-behavior-snapshot-contract.md
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-09-04: With explicit user approval, renumbered this task from its colliding former ID TASK-13135 to TASK-13160. Its snapshot prerequisite is now TASK-13159. Scope and acceptance criteria are unchanged; implementation remains pending the prerequisite merge.
2026-10-10: Continue the approved all-ten chat UX queue after atomic Edit commit69e594c495. ADR required: no new decision; the accepted backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md governs historical behavior, full-history fences, immutable append and fenced assistant commit. Governing implementation plan: Docs/superpowers/plans/2026-08-27-character-conversation-behavior-snapshot-contract.md, remaining Stages4/5. Current snapshot storage/materialization prerequisite is included in latest dev. New bounded plan: Docs/Design/IMPLEMENTATION_PLAN_chat_roleplay_resume_13160.md. TASK13160 also has an unrelated Personal Context task filename: backlog-py resolves the Enforce-versioned-Roleplay file, and its exact path is asserted before mutation; no Personal Context record is edited. Required base is implemented and verified before advertising capability. Optional stream persistence stays unadvertised until its dedicated grant and commit checks pass. No execution result or full resume acceptance is claimed.
Stage1 foundation qualified: strict caller-ID append and exact receipt replay; original stored receipt bound to original cid/message/parent/version and complete prior+1 fence, with eight real corruption negatives. Authenticated API tests moved to tests/Chat_NEW/integration/test_roleplay_resume_completion.py to reuse real credentialed_test_client; earlier Character-package missing-fixture errors are not RED evidence. Qualified API prerequisite RED: 7 failures for absent route (26.43s). Qualified replay regression RED: 8 failures DID NOT RAISE, 8 official PostgreSQL skips (9.43s). Stage1 final matrix: 57 passed, 27 official PostgreSQL unavailable skips, 56.15s. Includes actual auth/no-auth, foreign-owner and explicit workspace refusal, active Sync refusal, quota rollback/replay at cap, two independent connections identical-ID concurrency, same-owner other-conversation ID conflict, exact multiline text, later edit/delete replay, original full-state drift and H1 read coexistence. Docker daemon read-only availability query timed out after 10s; initial combined run was stopped at a hung test-owned Docker subprocess; reruns use the official TLDW_TEST_NO_DOCKER=1 switch with fixture-reported unavailable skips. No PostgreSQL correctness claim. Route is implemented but Roleplay capability remains unadvertised until protected generation and CAS commit are complete. Stage2 snapshot-only generation next; native/package/physical AT and participant timing remain outstanding.
Stage1 incremental commit gate: owned formatting preserves Python AST; all 8 authored Python scopes pass Ruff, production Bandit6 scopes has zero findings, diff check passes. Final post-format receipt/replay/concurrency/H1 subset13 passed13 official PG unavailable skips42 deselected14.22s. Independent reviewer A bounded CLEAR for owner/scope, immutable replay integrity, atomic cap/rollback and H1 coexistence. Reviewer B identified Stage2 requirement: validate saved resource semantics before initial append as well as generation; no capability is advertised by the Stage1 foundation. Governing accepted Roleplay ADR002, ADR049/050/059 unchanged; no new dependency/table/durable authority.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
