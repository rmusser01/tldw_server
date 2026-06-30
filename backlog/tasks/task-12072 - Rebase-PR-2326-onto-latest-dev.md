---
id: TASK-12072
title: Rebase PR 2326 onto latest dev
status: Done
references:
- https://github.com/rmusser01/tldw_server/pull/2326
modified_files:
- tldw_Server_API/app/api/v1/endpoints/chatbooks.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/tests/Services/test_router_groups_contract.py
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
- tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the Explainer workspace PR branch onto the latest origin/dev, resolve conflicts from the Explainer review follow-up commit, verify the rebased branch, and push the rewritten PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR branch is rebased onto the current origin/dev head.
- [x] #2 All rebase conflicts are resolved in favor of preserving current dev behavior plus Explainer additions.
- [x] #3 Focused verification passes or any remaining blockers are documented.
- [x] #4 Rebased branch is force-pushed safely to the PR branch.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Fetched latest `origin/dev` at `256e87804b587a8d899f728096bdd5c636e32557` and rebased `codex/explainer-workspace-pr` onto it.
- Resolved review-followup conflicts by keeping latest dev behavior and reapplying Explainer additions: combined Chatbook `QuotaExceededError` and sanitized `ExportError` handling, retained new writing annotation and Explainer exceptions, aligned router contract tests to current workspace migration/membership/eligibility specs, and preserved structured cockpit layout seeding in the real-server E2E spec.
- Skipped the old `fix: sanitize tts history write failure logs` replay because latest dev already contains the safer `request_id={}` logging behavior and matching sanitizer test; retaining the old commit would have added only stale task metadata.
- Verification: conflict files had no markers; `git diff --check` passed; focused backend pytest passed 197 tests; Explainer Vitest passed 14 tests; ESLint on the conflicted real-server E2E spec exited with 0 errors and existing `no-explicit-any` warnings; Bandit on backend conflict files reported 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2326 onto latest `origin/dev`, resolved all conflicts, dropped the redundant audio sanitizer replay because dev already contains the equivalent fix, and verified the rebased branch with focused backend/frontend/security checks.
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
