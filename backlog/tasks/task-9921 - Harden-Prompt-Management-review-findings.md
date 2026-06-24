---
id: TASK-9921
title: Harden Prompt Management review findings
status: Done
updated_date: 2026-06-24 04:29
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and fix validated findings from the current Prompt_Management module review. Scope is tldw_Server_API/app/core/Prompt_Management and focused tests only; avoid unrelated workspace changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Validated and fixed Prompt_Management review findings: gated unsafe ProgramEvaluator execution behind explicit PROMPT_STUDIO_ALLOW_UNSAFE_CODE_EVAL plus project metadata operator gate; fixed exact legacy placeholder rendering; allocated optimizer variant versions from target name max version; made Prompt Studio Jobs worker reject payload user_id fallback and bound/close per-user caches; fixed PromptsInteropService search signature handling; removed deprecated auth exports from prompt_studio package; removed generated __pycache__/.pyc files. Verification: focused regression group 7 passed; nearby unit set 21 passed; program evaluator integration controls 5 passed; interop search regression 1 passed; cleanup retest 3 passed; py_compile touched modules passed; Bandit touched Prompt Management scope returned 0 findings.
Reopening task to rebase PR #2445 onto latest dev and address validated PR review comments on cache eviction, DB close error visibility, and unsafe code evaluation production safeguards.
PR #2445 follow-up: rebased codex/prompt-management-review-fixes-pr onto origin/dev at 3f3221aa8fca86b55d314e8868fd02ff627e1b7c. Addressed validated review comments by deferring Prompt Studio DB connection close for active job owners, logging DB close failures, and adding a production-like environment acknowledgement gate for unsafe ProgramEvaluator subprocess execution. Updated Prompt Studio README with the new safety gates. Verification: jobs_worker/program_evaluator unit tests 21 passed; broader Prompt Studio unit regression group 26 passed; program evaluator integration controls 5 passed; Prompts interop search regression 1 passed; py_compile touched modules passed; Bandit touched Python files returned 0 findings.
Final PR #2445 rebase update: origin/dev advanced again, so the PR branch was rebased a second time onto 0ca69b4b4476fba779ad7098848ae7b2a0bcde06. Final review-feedback commit after rebase is 4ebd748b3b84fcec61dcba9c905dd742f02004a7.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR #2445 is rebased onto latest dev (0ca69b4b4476fba779ad7098848ae7b2a0bcde06) and review feedback is addressed with active-job cache close deferral, close-failure logging, and production-specific unsafe code-eval gating plus focused regression coverage.
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
