---
id: TASK-12098
title: Rebase PR 2573 on dev and address review feedback
status: Done
labels:
- codex
- review
- webui
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2573
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up task for PR #2573 to rebase against latest dev, inspect external review feedback and CI, implement minimal valid fixes, verify, and push.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR branch is rebased on latest dev and pushed.
- [ ] #2 All still-valid PR review comments are addressed or documented with a reason.
- [ ] #3 Focused tests and security scan for touched scope are run and recorded.
- [ ] #4 Unrelated local dirty worktree changes are preserved.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2573 on latest dev and addressed the remaining valid review feedback. Fixed runtime single-user auth normalization and placeholder handling, browser env-auth opt-out behavior, embedded character image data URL validation, env-only custom OpenAI provider listing, explicit tracked character switch routing, and mobile cockpit rail restore affordances. Verification: focused shared UI regression tests passed (134 tests); runtime bootstrap test passed (24 tests); LLM provider readiness pytest passed (6 tests); Bandit on llm_providers.py and provider_config_resolution.py passed with 0 findings. Additional checks: frontend ESLint targeted command exited 0 with warnings only; shared UI TypeScript compile needs NODE_OPTIONS=--max-old-space-size=8192 and still fails on unrelated baseline type errors outside this PR scope; targeted Prettier check would rewrite many touched TS files, so no formatting churn was applied. Unrelated local dirty changes were preserved and left unstaged.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
