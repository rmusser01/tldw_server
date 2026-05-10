---
id: TASK-229
title: Address PR 1516 VN platform API review findings
status: Done
assignee: []
created_date: '2026-05-10 15:22'
updated_date: '2026-05-10 16:57'
labels:
  - vn
  - code-review
  - api
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1516'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable code-review findings on PR 1516 for the VN platform API branch. Scope is limited to verified review issues on the existing PR branch, especially scripted generation/regeneration placeholders, VN asset cleanup blocker wiring, and maintainability cleanup called out by reviewers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Verified actionable PR 1516 review comments are either fixed or explicitly rejected with technical rationale.
- [x] #2 Scripted generation and regeneration behavior is made production-appropriate for V1 or clearly gated so clients cannot mistake placeholders for live model output.
- [x] #3 VN asset cleanup blocker wiring protects generated files referenced by VN scripts, sessions, or save slots where the current backend can verify those references.
- [x] #4 Focused tests cover the review-fix behavior and existing VN Play/VN Assets tests continue to pass.
- [x] #5 Bandit and whitespace checks pass on the touched backend scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed PR #1516 review follow-up: removed scripted generation placeholders by requiring literal persisted text/regeneration text for V1 script opcodes; added cleanup blocker provider wiring for published script manifests plus active play sessions/checkpoints; tightened VN asset idempotency to claim/replay/release with required keys for mutating operations; fixed upload duplicate in-progress claim handling and import-commit enqueue-failure claim release; added VN policy backend/conversion hardening; serialized save-slot creation through the session-action lock; replayed duplicate non-completed turn requests as errors instead of 200 payloads; and pinned scripted-session policy checks to published script/pack metadata with mismatch rejection.

Verification: python -m pytest tldw_Server_API/tests/VN_Assets tldw_Server_API/tests/VN_Play tldw_Server_API/tests/VN_Platform tldw_Server_API/tests/VN_Scripts tldw_Server_API/tests/VN_Policy -q passed with 467 passed, 5 warnings. Focused frontend Vitest passed for VN asset/play tests with 32 passed. compileall passed for touched backend/test modules. Bandit over touched backend scope wrote /tmp/bandit_vn_pr1516.json with 0 results. git diff --check passed.

Reopened after final review-thread sweep found additional actionable items: debug-state response schema, octet-stream content handling, and prompt tokenizer exception narrowing.

Final review-thread sweep addressed remaining prompt tokenizer fallback, debug-state response typing, and VN asset content MIME contract issues. Verification after final fixes: focused prompt/generation slice 34 passed; full VN backend suite 470 passed, 5 warnings; focused VN frontend Vitest 32 passed; compileall passed; Bandit /tmp/bandit_vn_pr1516.json 0 results; git diff --check passed.

Post-push hardening for remaining non-outdated review threads: made legacy create_idempotency_record conflict-tolerant for same payloads and moved save-slot checkpoint, event, slot upsert, and action completion into a single repository transaction. Verification after hardening: focused idempotency/save-slot tests 17 passed; full VN backend suite 472 passed, 5 warnings; compileall passed; Bandit /tmp/bandit_vn_pr1516.json 0 results; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved PR #1516 review findings across VN assets, play, policy, scripts, and platform API docs/frontend contracts. Final pass added explicit scripted debug-state response schema, raw content vs preview MIME behavior, network-safe tokenizer fallback without a blanket exception, conflict-tolerant legacy idempotency writes, and atomic save-slot create commits. Verification passed: full VN backend suite 472 passed, focused frontend VN Vitest 32 passed, compileall, Bandit 0 findings, and git diff --check.
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
