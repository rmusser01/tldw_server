---
id: TASK-13877
title: Implement Moderation policy_types retirement
status: Done
labels:
- moderation
- refactor
- behavior-preserving
dependencies:
- TASK-13876
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retire PolicyCompiler.policy_types() and PolicyEvaluator.policy_types() per the approved structural design. Replace internal hook lookups with the existing private canonical model aliases while preserving supported ModerationService behavior, import boundaries, runtime type-hint behavior, and canonical result types. The undocumented subclass model-substitution seam and direct hook calls are intentionally removed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PolicyCompiler and PolicyEvaluator no longer define or call policy_types().
- [x] #2 Compiler policy/rule construction and evaluator checks/results use the existing private canonical model aliases.
- [x] #3 Legacy-named subclass policy_types() methods are ignored by representative compile/evaluate operations.
- [x] #4 Clean-process import tests exercise real compile/evaluate operations without loading moderation_service.
- [x] #5 Focused and downstream Moderation tests pass after compilation-first verification.
- [x] #6 Ruff, scoped Black check, Bandit, git diff --check, and source audit pass on the touched scope.
- [x] #7 Current PR #3176 receives fresh requesting-code-review, all exact-head/current-base required statuses, and a verified normal merge while retaining the requester-provided Change summary and disclosed compatibility limit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-20 current identity: TASK-13436 -> TASK-13877; pinned source PR #3176 at a20c5e0f386893758d9b079a2b1be1adbaa2b89e. Original IDs now belong to unrelated Jobs/quota/evaluation records on dev, which must remain unchanged. Fresh global published maximum was TASK-13870; our already-created VN records reserve TASK-13871 through TASK-13874, so this work uses TASK-13875 through TASK-13877. Created/edited only through backlog-py; no manual identity edit.

ADR required: no. Searched Docs/ADR/README.md and existing ADRs; no moderation-specific ADR governs this internal cleanup. Neutral Moderation/models.py ownership was already implemented by PR #2929. This change removes the obsolete import-cycle workaround without changing that ownership, supported ModerationService/public HTTP contracts, persistence or security policy. The intentional removal of undocumented direct hook calls/subclass model substitution is explicitly retained in the approved design and PR compatibility notes. ADR-002 and ADR-059 govern tracking recreation; no accepted rationale is rewritten.

Original frontmatter (historical identity/metadata):
```yaml
---
id: TASK-13436
title: Implement Moderation policy_types retirement
status: Done
created_date: 2026-10-03 21:45
dependencies:
- TASK-13435
labels:
- moderation
- refactor
- behavior-preserving
priority: medium
documentation:
- Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md
- Docs/superpowers/plans/2026-10-03-moderation-policy-types-retirement-implementation-plan.md
modified_files:
- tldw_Server_API/app/core/Moderation/policy_compiler.py
- tldw_Server_API/app/core/Moderation/policy_evaluator.py
- tldw_Server_API/tests/unit/test_moderation_models_characterization.py
- tldw_Server_API/tests/unit/test_moderation_models_imports.py
- tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
updated_date: 2026-10-04 23:05
references:
- https://github.com/rmusser01/tldw_server/pull/3176
---
```

Historical implementation notes (verbatim):

Execution started with the user-selected subagent-driven workflow. The implementation will follow the committed plan at db1fb1497b with spec-compliance and code-quality review checkpoints.
Pre-edit focused baseline: 119 passed, 250 warnings in 1.77s on Python 3.14.3.
TDD red: the two inverse legacy-hook tests collected successfully and both failed on exact canonical-type assertions: compiler returned ReplacementPolicy and evaluator returned ReplacementResult (2 failed, 16 warnings). Green: py_compile passed for all five changed Python files; inverse tests 2 passed; focused suite 115 passed; all Moderation unit tests 293 passed; Guardian 89 passed; Chat integration 17 passed; Workflow moderation adapters 12 passed (47 deselected); Audio redaction 1 passed. Quality: Ruff passed, Black check passed after scoped formatting of the touched compiler file, Bandit reported 0 findings (/tmp/bandit_TASK-13436.json), and git diff --check passed. Source audit: zero policy_types references in production and exactly two local legacy fixture definitions across the three changed test files. Self-review found no parsing, scanning, ranking, redaction, exception, service-dispatch, model, or moderation_service behavior changes; the diff is limited to the approved six-file write set.
Code-quality review requested stronger inverse-hook coverage: both legacy hooks returned canonical PatternRule, so a partial migration of rule selection could escape. Reopened to substitute an incompatible replacement rule and exercise compiler rule construction plus evaluator snippet, redaction, counted redaction, and evaluation paths.
Correction evidence: strengthened both inverse fixtures with incompatible ReplacementRule classes. Compiler coverage now detects policy and rule slot regressions for global and user compilation. Evaluator coverage now exercises build_sanitized_snippet, redact_text, redact_text_with_count, and evaluate_text with canonical rule/result identity assertions. Verification: changed test py_compile passed; inverse tests 2 passed; focused suite 115 passed; Ruff passed; Black check passed; source audit found zero production policy_types references and exactly two local test fixture definitions; git diff --check passed. Self-review confirmed only the characterization test and task record changed; production is untouched.
Final controller verification against origin/dev 4c4f197f68481664c58d4553bbfbb45dae157e28: branch already contained current dev; py_compile passed; focused suite 115 passed; Moderation unit suite 293 passed; combined Guardian/Chat/Audio run 107 passed; Workflow moderation adapters 12 passed (47 deselected); Ruff passed; Black left 5 files unchanged; Bandit reported 0 findings and 0 errors (/tmp/bandit_TASK-13436_final.json); diff check and ancestry passed; source audit found zero production references and exactly two local regression fixtures; worktree was clean. Spec review approved, code-quality review finding was corrected and re-approved, and final whole-branch review found no actionable issues. Residual risk remains limited to the documented intentional compatibility break for unknown external policy_types callers/subclasses and private-alias rebinding.
PR preparation on 2026-10-04: rebased cleanly onto origin/dev bf8f2ad6a4; post-rebase py_compile passed; focused suite 115 passed; Moderation unit suite 293 passed; combined Guardian/Chat/Audio run 107 passed; Workflow adapters 12 passed (47 deselected); Ruff and Black passed; Bandit reported 0 findings and 0 errors (/tmp/bandit_TASK-13436_pr.json); ancestry, diff, and source audits passed. Opened PR #3176 against dev. The PR Change summary remains intentionally marked as awaiting human-authored wording required by the AI-generated PR merge gate.
The human requester supplied the required Change summary on 2026-10-04. It was inserted verbatim into PR #3176, replacing the merge-gate placeholder while preserving the technical, compatibility, verification, tracking, and automated reviewer sections. The human-authored summary merge gate is now satisfied.
PR follow-up: addressing Qodo's missing annotations and docstrings in the two legacy-hook regression fixtures, then refreshing against current dev.
Addressed both Qodo rule violations with explicit constructor/test/hook annotations and concise docstrings in the two legacy-hook fixtures. Compilation, 12 characterization tests, Ruff, Black formatting, and Bandit (excluding expected test assertions B101) passed. Production code is unchanged.
Rebased cleanly onto latest origin/dev. Post-rebase verification: compilation passed; 400 combined moderation/Guardian/Chat/Audio tests and 12 Workflow moderation adapter tests passed (412 total). Ruff, Black check, scoped production Bandit, and git diff --check passed.

Original implementation-plan section (historical):
Follow the approved implementation plan: establish the two inverse subclass regression tests red, replace hook-dependent import tests with real operations, make the minimal private-alias production edits, then run compilation-first focused, downstream, formatting, lint, security, and source-audit gates.

Resumed qualification: implementation remains unchanged; fresh tests/lint/Bandit, caller audit, ID/dependency preservation, independent review and serial hosted integration are pending. Existing failed full-suite shards concern a stale privilege snapshot, a VN responsiveness timing assertion, and an email upload fixture forbidden outbound/background call. Inspect them against current dev before any fix; do not weaken assertions or infer a moderation cause from unrelated failures. Current plan: Docs/superpowers/plans/2026-10-03-moderation-policy-types-retirement-implementation-plan.md.
2026-10-20 pre-rebase qualification of unchanged PR source a20c5e0f386893758d9b079a2b1be1adbaa2b89e: compilation passed; complete moderation plus Guardian/Chat/Audio suites 400 passed, zero skips (7 existing warnings), Workflow moderation adapters 12 passed / 47 deselected (6 warnings). First combined run had 84 Guardian setup errors because the chosen /private/tmp base lay outside trusted database roots; rerun under the OS TMPDIR passed all 400 with no guard changes. Fresh inverse control temporarily restored both original-base compiler/evaluator files: both legacy-hook regression tests failed as intended; finally restored exact PR bytes, then both tests passed. All5 scoped Python files remain byte-identical to the original PR head. Current caller audit found only the2 intentional legacy test fixture methods and no production references. Ruff and Black pass. Production Bandit0findings/0errors; changed-test Bandit excluding pytest B101 reports3LOW subprocess warnings (B404/B603), exactly matching original-base import-isolation tests, with0new findings/0errors. Deterministic metadata/section/checklist/dependency/canonical/ADR preservation checks pass. Current-base qualification, independent requesting-code-review, fresh hosted CI and normal merge remain pending; AC7 and overall completion remain open.
2026-10-20 current-base qualification: clean fifteen-patch rebase onto dev 2e6ceab9fb97c5712533be670504a81a24525489 completed at 41ab9efadb55268dc1d1e520f01ac3e2c6deaf03; final range-diff marks every prior patch unchanged. Fresh global other-open-PR check confirms TASK-13875 through TASK-13877 remain unused, the three unrelated existing ID colliders are byte-identical, all original metadata/section/checklist content is retained, the new dependency chain is correct, and authored ADR bytes are unchanged. Compilation passed. Complete moderation/Guardian/Chat/Audio selection: 400 passed, zero skips (7 existing warnings); Workflow moderation: 12 passed / 47 deselected (6 warnings). The three previously failed CI cases plus four canonical-format ratchet tests passed together: 7 passed (4 warnings). This confirms the old privilege snapshot, VN responsiveness and email upload failures are absent on this PR’s current base; it is not a full-shard or whole-suite claim. Fresh caller audit finds no production hook references and only the two intentional legacy fixture methods. Ruff/Black and applicable pre-commit/diff checks pass. Fresh production Bandit has zero findings/errors; changed tests retain the same three low-severity subprocess findings as original baseline, with no new findings/errors (B101 excluded only for pytest assertions). All five Python files match original PR head a20c5e0f386893758d9b079a2b1be1adbaa2b89e; prior inverse negative/green evidence therefore retains its unchanged-source meaning. Independent pre-rebase requesting-code-review approved d1eed58196 without findings. Final rebased-head review, protected publication, hosted current-head/current-base gates and verified normal merge remain pending; AC7 stays open. The human-provided Change summary will remain verbatim.
2026-10-20 verified completion: PR #3176 merged normally as e521605198dd311b94d2c63416bf07db3e5e7a0a from independently reviewed head 9ceeb6f3cdc9bdb5914983f048cf632b42da8f96 at 2026-10-20T16:44:57Z. All seven required statuses passed on current dev base 2e6ceab9fb97c5712533be670504a81a24525489; no unresolved review threads and queue mode unset. Exact-head requesting-code-review found no Critical, Important or Minor issues; requester-provided Change summary retained verbatim. Fresh local qualification remains 400 connected tests, 12 Workflow cases, and 7 historical-failure/canonical-format controls passing. All five Python files match original PR head a20c5e0f386893758d9b079a2b1be1adbaa2b89e; no new Bandit findings. Four report-only AuthNZ shards failed before checkout/tests because mirror.gcr.io PostgreSQL image pulls exhausted the public mirror quota (jobs 118746618347, 118746618506, 118746618612, 118746618782 in run 39456743840). The single-job retry was refused by GitHub HTTP403 while the parent workflow remains running. This is not a whole-suite pass claim and no required status was waived. This task-only closeout is carried by final batch documentation PR #3285; it does not amend the merged implementation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed PolicyCompiler.policy_types() and PolicyEvaluator.policy_types(), including the evaluator tuple cache and unused runtime policy import. Compiler/evaluator runtime operations now bind directly to the existing canonical private model aliases, while TYPE_CHECKING annotations, public runtime namespaces, service facade identities, and supported Moderation behavior remain covered. Direct hook calls, subclass model substitution, and evaluator tuple-cache identity are intentionally unsupported; private alias rebinding remains unsupported.

Current completion: 2026-10-20 verified completion: PR #3176 merged normally as e521605198dd311b94d2c63416bf07db3e5e7a0a from independently reviewed head 9ceeb6f3cdc9bdb5914983f048cf632b42da8f96 at 2026-10-20T16:44:57Z. All seven required statuses passed on current dev base 2e6ceab9fb97c5712533be670504a81a24525489; no unresolved review threads and queue mode unset. Exact-head requesting-code-review found no Critical, Important or Minor issues; requester-provided Change summary retained verbatim. Fresh local qualification remains 400 connected tests, 12 Workflow cases, and 7 historical-failure/canonical-format controls passing. All five Python files match original PR head a20c5e0f386893758d9b079a2b1be1adbaa2b89e; no new Bandit findings. Four report-only AuthNZ shards failed before checkout/tests because mirror.gcr.io PostgreSQL image pulls exhausted the public mirror quota (jobs 118746618347, 118746618506, 118746618612, 118746618782 in run 39456743840). The single-job retry was refused by GitHub HTTP403 while the parent workflow remains running. This is not a whole-suite pass claim and no required status was waived. This task-only closeout is carried by final batch documentation PR #3285; it does not amend the merged implementation.
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
