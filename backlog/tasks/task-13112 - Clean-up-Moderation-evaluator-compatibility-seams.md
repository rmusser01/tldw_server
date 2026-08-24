---
id: TASK-13112
title: Clean up Moderation evaluator compatibility seams
status: Done
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
updated_date: 2026-08-24 07:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove repository-unused private ModerationService delegates superseded by PolicyEvaluator while preserving all public facade behavior, dynamic dispatch through _evaluate_text_core(), policy_types() compatibility, and regex/redaction semantics. This is a strict structural cleanup following TASK-13011.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Repository and test usage of candidate private ModerationService evaluator delegates is documented.
- [x] #2 Only private delegates with no required production or extension-point role are removed.
- [x] #3 Public ModerationService signatures and behavior remain unchanged.
- [x] #4 _evaluate_text_core() and policy_types() compatibility remain unchanged.
- [x] #5 Characterization, evaluator, caller-contract, compilation, and security checks pass.
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
Task 3 stability, quality, and security verification completed after the first rebase.

First verification rebase evidence (historical):
- Worktree: /Users/appledev/Documents/GitHub/tldw_server/.worktrees/moderation-compatibility-seams
- Branch: codex/moderation-compatibility-seams
- First verification base origin/dev (historical): 83fa300fc1b0e77e81219af2abe5b4ddc2c85069
- First verification merge-base with origin/dev (historical): 83fa300fc1b0e77e81219af2abe5b4ddc2c85069
- First verification pre-tracking HEAD (historical): 83f489876e7828e7e1fbd42cd043f6d6ed95b57d
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

Stage 3 is complete. Stage 4 independent review and final PR-readiness verification followed.
Rebase SHA mapping: pre-rebase c601412b7c -> first-rebase d7a5468f6f -> current second-rebase 6c316af721. The mapping was verified before final tracking: all three commits have the same stable patch ID, 4bd77a2a1b83d0a7bc275d8265fdfdc612dd589f.
Quality-review correction: the deleted service characterization preserved both numeric-string max_scan_chars="2" geometry and the successful-match boundary, while the direct evaluator replacement retained only geometry plus a long-text no-match. Added the exact direct assertion `evaluator.find_match_span(re.compile("x"), "xx", limits) == (0, 1)` in `test_direct_numeric_string_limits_are_coerced_for_evaluation()`, keeping the existing `[(0, 2)]` geometry and no-match assertions. This is characterization-only coverage: the focused test passed on its first run, confirming existing behavior; no production red phase or production change was expected or needed. Fresh results: changed-test py_compile exit 0; exact node 1 passed, 8 warnings; policy_evaluator + characterization + delegation suites 133 passed, 272 warnings; Black check 1 file unchanged; Ruff check passed; `git diff --check` passed with no output.

Independent Task 4 review outcomes:
- Independent spec review: APPROVE with no findings. It confirmed the approved scope and retained boundaries; residual uncertainty remains because unknown external callers of the intentionally removed private methods cannot be ruled out.
- Independent quality review initially reported a P2 coverage gap because successful numeric-string boundary matching was not retained directly. Commit 8b51d70c642a3f597ccd5c35726296bbd5ba9529 added `find_match_span(re.compile("x"), "xx", limits) == (0, 1)`.
- Quality re-review: APPROVE. The original P2 is resolved; numeric-string geometry, successful boundary matching, and the long no-match behavior are all retained.
- Quality re-review also reported P3 stale rebase metadata in this task. The P3 is resolved by relabeling the 83fa300fc1 first-verification base and merge-base as historical, recording the verified three-SHA mapping, and adding the second/final rebase evidence below.

Second/final rebase evidence:
- Fresh `git fetch origin dev` completed successfully.
- Current origin/dev: 21aed4cc0d1e9e2e2a34fc84307bbd1d3b879871
- Current `git merge-base origin/dev HEAD`: 21aed4cc0d1e9e2e2a34fc84307bbd1d3b879871
- Task 4 pre-tracking HEAD: 8b51d70c642a3f597ccd5c35726296bbd5ba9529
- Branch was clean and 9 commits ahead of origin/dev before final tracking.

Fresh final Task 4 matrix:
- Removed-method rg: exit 1, zero matches for all eight approved names (0.00s). Retained-method rg: exactly `_evaluate_text_core` at line 743 and `_evaluate_action_internal` at line 766 (0.00s).
- `py_compile` passed moderation_service.py and all three changed evaluator tests with no output (0.06s).
- Black passed the three changed tests: 3 files would be left unchanged (0.34s).
- Black on moderation_service.py reported it would be reformatted (exit 1, 0.41s). Black against the fresh current-origin/dev snapshot `/tmp/moderation_service_origin_dev_task_13112_final.py` reported the same result (exit 1, 0.42s), confirming pre-existing base/head parity. No mass-formatting was performed.
- Ruff passed all four files: All checks passed (0.11s).
- Full Moderation unit glob: 299 passed, 605 warnings in 14.79s (24.24s wall).
- Guardian supervised-policy suite: 89 passed, 184 warnings in 14.08s (21.85s wall).
- Chat moderation integration suite: 16 passed, 81 warnings in 30.16s (38.16s wall).
- Workflow moderation-adapter selection: 12 passed, 45 deselected, 35 warnings in 9.65s (17.51s wall).
- Exact Audio retention/redaction node: 1 passed, 10 warnings in 9.66s (18.37s wall).
- Bandit completed with exit 0 in 0.21s and wrote `/tmp/bandit_task_13112_final.json`. Metrics: 965 LOC, 0 findings/results, 0 errors, 0 high/medium/low/undefined severity, 0 high/medium/low/undefined confidence, 0 nosec lines, and 0 skipped tests.
- `git diff --check origin/dev...HEAD` passed with no output (0.01s).
- `git diff --name-only origin/dev...HEAD` contained exactly seven approved files: the implementation plan, design spec, TASK-13112, moderation_service.py, and the three changed evaluator tests.
- Pre-tracking `git status --short` was clean.
- The final merge-base equaled current origin/dev at 21aed4cc0d1e9e2e2a34fc84307bbd1d3b879871.
- Final branch-diff self-review found no findings. The production diff is deletion-only for the exact eight wrappers; `policy_evaluator.py` and caller files are unchanged; public signatures, tuple ordering, dynamic dispatch, and policy_types coverage remain; direct evaluator tests retain the unique numeric-string geometry, successful-match boundary, and long no-match behavior.

Residual risk and PR gate:
- Unknown external consumers of the intentionally removed private names cannot be ruled out.
- `moderation_service.py` has pre-existing Black nonconformance shared by current origin/dev; it was intentionally not mass-formatted in this structural cleanup.
- No technical gates were skipped and there are no known blockers.
- No PR was created or pushed. PR creation still requires a human-written `Change summary` explaining what changed and why those implementation choices were made.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exactly eight repository-unused private service wrappers were removed. The public facade, dynamic dispatch, policy_types compatibility, and evaluator semantics were retained. Unique numeric-string geometry and successful-match behavior are retained directly in PolicyEvaluator coverage. The cleanup removes redundant private callable surface while preserving supported behavior and active extension boundaries.
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
