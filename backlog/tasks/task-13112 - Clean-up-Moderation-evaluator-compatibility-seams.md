---
id: TASK-13112
title: Clean up Moderation evaluator compatibility seams
status: In Progress
created_date: 2026-08-24 04:57
dependencies:
- TASK-13011
labels:
- Moderation
- refactor
- behavior-preserving
priority: medium
references:
- Docs/superpowers/specs/2026-08-01-moderation-shared-models-extraction-design.md
- Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md
- tldw_Server_API/app/core/Moderation/moderation_service.py
documentation:
- Docs/superpowers/specs/2026-08-23-moderation-compatibility-seams-cleanup-design.md
- Docs/superpowers/plans/2026-08-23-moderation-compatibility-seams-cleanup.md
modified_files:
- Docs/superpowers/specs/2026-08-23-moderation-compatibility-seams-cleanup-design.md
- Docs/superpowers/plans/2026-08-23-moderation-compatibility-seams-cleanup.md
- tldw_Server_API/app/core/Moderation/moderation_service.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
- backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md
updated_date: 2026-08-24 07:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove repository-unused private ModerationService delegates superseded by PolicyEvaluator while preserving all public facade behavior, dynamic dispatch through _evaluate_text_core(), policy_types() compatibility, and regex/redaction semantics. This is a strict structural cleanup following TASK-13011.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repository and test usage of candidate private ModerationService evaluator delegates is documented.
- [ ] #2 Only private delegates with no required production or extension-point role are removed.
- [ ] #3 Public ModerationService signatures and behavior remain unchanged.
- [ ] #4 _evaluate_text_core() and policy_types() compatibility remain unchanged.
- [ ] #5 Characterization, evaluator, caller-contract, compilation, and security checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute `Docs/superpowers/plans/2026-08-23-moderation-compatibility-seams-cleanup.md` using TDD and subagent-driven development: (1) remove five class/static direct evaluator shims, (2) remove three instance scan shims and duplicate private characterization, (3) run the exact compilation, quality, security, and caller matrix, and (4) complete independent spec/quality review and PR-readiness gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created isolated worktree `codex/moderation-compatibility-seams` from current `origin/dev` (`2c3589fa09`).

Repository compatibility inventory and the exact eight-method removal set are documented in the design spec. Independent review retained `_evaluate_action_internal()` because it preserves public `evaluate_text()` dispatch, documented the intentional private-surface compatibility risk, made the absence invariant class-local, specified the exact verification matrix, removed a duplicate endpoint gate, and corrected whitespace.

Pre-change verification passed: 318 unit Moderation tests, 89 Guardian tests, 16 Chat tests, 12 selected Workflow tests, and 1 targeted Audio test.
The revised design is user-approved. The implementation plan was written and self-reviewed against the spec: exact retained pytest nodes collect successfully, compilation precedes test cleanup in both implementation stages, placeholder and whitespace scans are clean, and all eight deletions map to retained direct-evaluator coverage.
Task 1 removed the five class/static direct PolicyEvaluator shims. The new ModerationService.__dict__ absence test failed before deletion and passed after deletion. Direct evaluator and service delegation suites passed.
Task 1 spec compliance review: approved with no issues. Task 1 code quality review: approved with no critical, important, or minor findings. Reviewer verification passed compilation, 313 Moderation unit tests, 78 evaluator/delegation tests, retained dispatch and policy type checks, Ruff, Bandit, and `git diff --check`. Black would reformat the same production file at the base SHA, so that formatting result is pre-existing and not part of this cleanup.
Task 2 removed the three instance scan shims and wrapper-only duplicate characterization. The class-local absence test completed a red/green cycle. Direct evaluator, public service dispatch, signature, and policy_types() compatibility tests passed.
Task 2 quality review found the numeric-string chunk-geometry gap, and the direct evaluator test now preserves max_scan_chars="2" geometry.
Task 2 spec compliance review approved the exact three-shim removal and retained dispatch boundaries with no issues.
Task 2 code quality review identified one coverage gap: the deleted service-private numeric-string test asserted max_scan_chars="2" chunk geometry, while the direct evaluator replacement only checked a no-match result with "1". Commit c601412b7c restores the exact [(0, 2)] geometry assertion with one evaluator and the same limits. Fresh re-review approved with no remaining findings; 134 evaluator-boundary tests, Ruff, and diff checks passed.
Task 3 stability, quality, and security verification completed after rebase.

Rebase evidence:
- Worktree: /Users/appledev/Documents/GitHub/tldw_server/.worktrees/moderation-compatibility-seams
- Branch: codex/moderation-compatibility-seams
- Current origin/dev: 83fa300fc1b0e77e81219af2abe5b4ddc2c85069
- git merge-base origin/dev HEAD: 83fa300fc1b0e77e81219af2abe5b4ddc2c85069
- Pre-tracking HEAD: 83f489876e7828e7e1fbd42cd043f6d6ed95b57d
- Initial verification status was clean and the branch was 7 commits ahead of origin/dev.

Fresh Task 3 evidence:
- Removed-method scope rg exited 1 with zero matches for all eight approved names. Retained-dispatch rg returned exactly two definitions: _evaluate_text_core at line 743 and _evaluate_action_internal at line 766.
- py_compile passed for moderation_service.py and the three changed evaluator test files.
- Black passed the three changed tests: 3 files would be left unchanged.
- Black separately reported moderation_service.py would be reformatted (exit 1). The same Black version/check against origin/dev's moderation_service.py snapshot at /tmp/moderation_service_origin_dev_task_13112.py also reported it would be reformatted (exit 1), confirming pre-existing base/head parity. The service was not reformatted.
- Ruff passed all four files: All checks passed.
- Full Moderation unit gate: 299 passed, 605 warnings in 11.87s. The count is consistent with the approved removal of wrapper-specific duplicate tests and addition of the two absence invariants from the original 318-test baseline.
- Guardian supervised-policy gate: 89 passed, 184 warnings in 11.67s.
- Chat moderation integration gate: 16 passed, 81 warnings in 26.43s.
- Workflow moderation-adapter gate: 12 passed, 45 deselected, 35 warnings in 8.81s.
- Exact Audio retention/redaction node: 1 passed, 10 warnings in 9.11s.
- Bandit completed with exit 0. JSON report /tmp/bandit_task_13112.json: 965 LOC, 0 findings/results, 0 errors, 0 high/medium/low/undefined severity, 0 high/medium/low/undefined confidence, 0 nosec lines, and 0 skipped tests.
- git diff --check origin/dev...HEAD passed with no output.
- git diff --name-only origin/dev...HEAD contained exactly the seven approved files: plan, design, TASK-13112, moderation_service.py, and the three evaluator tests.
- Pre-tracking git status --short was clean.
- Full branch diff self-review found no issues: production changes delete only the exact eight approved methods; retained dispatch and policy_types() coverage remains; no policy_evaluator.py or caller files changed; deleted wrapper-specific characterization maps to direct evaluator coverage.

Stage 3 is complete. Stage 4 independent review and final PR-readiness verification is now in progress. TASK-13112 remains In Progress and is not marked Done.
Rebase SHA mapping: pre-rebase c601412b7c is d7a5468f6f on the current branch.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
