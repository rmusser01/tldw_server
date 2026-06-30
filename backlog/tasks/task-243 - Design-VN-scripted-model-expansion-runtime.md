---
id: TASK-243
title: Design VN scripted model expansion runtime
status: Done
assignee: []
created_date: '2026-05-10 19:52'
updated_date: '2026-05-10 21:00'
labels:
  - vn
  - design
  - scripted-generation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design spec for GitHub issue #1535: backend-owned model expansion and regeneration for VN scripted_story sessions, plus the follow-on WebUI generation-history inspector path. This is design-first before implementation and should split backend/runtime and WebUI work into separate PR-sized paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design defines backend generation requests, immutable revisions, active revision semantics, confirmation/cancel/regenerate/activate commands, idempotency, failures, moderation, checkpoints, and generated-choice behavior.
- [x] #2 Design defines the dedicated generation-history and debug APIs plus the WebUI generation-history route expectations.
- [x] #3 Design explicitly splits backend runtime/API implementation from WebUI inspector implementation.
- [x] #4 Spec is written under Docs/superpowers/specs and reviewed before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed and patched spec after two reviewer passes. Addressed generation profile snapshot map/profile_key semantics, generation_id ownership via vn_play_generations, active_revision_id checkpoint/restore semantics, read-time activation overlay, fail-closed moderation, idempotent model-call recovery, strict nested output schemas, exact offset pagination envelope, and debug endpoint path/auth/audit.

Verification: git diff --check passed after the review-fix patch. Final focused subagent review found no blocker/high findings. Bandit skipped because this patch only updates design documentation and Backlog metadata, not executable code.

PR #1549 review pass: live review sweep found two unresolved Gemini threads about active-revision overlay integration with scene-state derivation and deterministic V1 activation blocking. Reopening task to address them.

PR #1549 review fixes: clarified active-revision overlay integration with current scene derivation and replaced vague activation dependency analysis with deterministic V1 blocking when material downstream scene/script events exist. git diff --check passed; Bandit remains skipped because changes are documentation/backlog-only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed PR #1549 review fixes. The spec now requires scene derivation to substitute active revision output/resolved visuals instead of replaying inactive generation event payloads, and V1 revision activation now has a deterministic downstream-material-event blocking rule.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
