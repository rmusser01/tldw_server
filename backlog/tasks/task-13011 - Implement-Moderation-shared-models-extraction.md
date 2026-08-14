---
id: TASK-13011
title: Implement Moderation shared models extraction
status: In Progress
created_date: 2026-08-12 00:45
dependencies:
- TASK-13010
- TASK-12992
labels:
- moderation
- refactor
- implementation
priority: medium
references:
- https://github.com/rmusser01/tldw_server/pull/2770
- codex/moderation-shared-models-design@5d33b21ca4
- origin/dev@8f94369e517463758071504079e9ab5f8f8a0091
- codex/moderation-shared-models-dev@b644b86e145ec47658cddb31a0a8f3f5f97b0b18
- https://github.com/rmusser01/tldw_server/pull/2791
documentation:
- Docs/superpowers/specs/2026-08-01-moderation-shared-models-extraction-design.md
- Docs/superpowers/plans/2026-08-01-moderation-shared-models-extraction-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-08-01-moderation-shared-models-extraction-design.md
- Docs/superpowers/plans/2026-08-01-moderation-shared-models-extraction-implementation-plan.md
- backlog/tasks/task-13010 - Design-Moderation-shared-models-extraction.md
- backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md
- tldw_Server_API/app/core/Moderation/models.py
- tldw_Server_API/app/core/Moderation/moderation_service.py
- tldw_Server_API/app/core/Moderation/policy_compiler.py
- tldw_Server_API/app/core/Moderation/policy_evaluator.py
- tldw_Server_API/tests/unit/test_moderation_models_characterization.py
- tldw_Server_API/tests/unit/test_moderation_models_canonical.py
- tldw_Server_API/tests/unit/test_moderation_models_imports.py
updated_date: 2026-08-14 00:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transplant the reviewed shared-model extraction onto current dev: make models.py the canonical owner of ModerationPolicy, PatternRule, and ModerationEvaluationResult; preserve exact service facade imports and behavior; remove compiler/evaluator service type edges; rerun current-dev tests, security checks, and review. This record replaces stale stacked-branch TASK-12989, which collides on current dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 models.py canonically owns exactly the three approved dataclasses and remains standard-library-only.
- [x] #2 moderation_service.py re-exports the exact canonical class objects with unchanged supported constructors, defaults, to_dict mapping, and runtime behavior.
- [x] #3 PolicyCompiler and PolicyEvaluator no longer load moderation_service.py for canonical runtime types while preserving policy_types descriptors and subclass dispatch.
- [ ] #4 Focused and caller regression tests, compilation, Black/Ruff, Bandit, diff/scope checks, and independent review pass on current dev.
- [ ] #5 The PR contains only the shared-model extraction and collision-free tracking records, with a requester-authored Change summary required before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-08-01-moderation-shared-models-extraction-implementation-plan.md Task 5: transplant reviewed post-predecessor commits, reconcile IDs, rerun all verification on current dev, obtain independent review, and prepare a PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Transplant:
- Created codex/moderation-shared-models-dev directly from origin/dev@414e81a12aa71df97c4fad17df084aa7a78c474b and applied the nine reviewed post-predecessor commits without conflicts as consolidated current-dev commits.
- Preserved codex/moderation-shared-models-design@5d33b21ca4 untouched as a recovery reference.
- Reconciled the predecessor to merged TASK-12992 / PR #2770 and replaced colliding stale task IDs with TASK-13010 and TASK-13011.
- Production scope is exactly models.py, moderation_service.py, policy_compiler.py, and policy_evaluator.py.

Fresh current-dev verification on codex/moderation-shared-models-dev@f0639e91ed:
- Compilation: py_compile passed for all four touched production files and all three new test files.
- Formatting/lint: Black write/check left models.py, policy_evaluator.py, and the three new tests unchanged; Ruff passed all seven touched Python files. Fresh origin/dev baselines confirm moderation_service.py and policy_compiler.py both have pre-existing Black debt, so neither was whole-file formatted.
- Regression tests: 303 Moderation unit tests passed; 97 moderation endpoint/Guardian tests passed; 16 chat moderation integration tests passed; 12 workflow moderation-adapter tests passed (45 deselected); and the targeted audio STT redaction test passed. Total selected tests: 429 passed.
- Security: Bandit scanned app/core/Moderation (3,335 LOC) with 0 findings, 0 errors, 0 skipped tests, and 0 nosec suppressions.
- Ancestry/scope: merge-base HEAD origin/dev equals 414e81a12aa71df97c4fad17df084aa7a78c474b; git diff --check passes; the branch contains only tracking seed 74373b7117 and extraction f0639e91ed. The 11-file diff contains exactly four approved production files, three focused tests, two approved documents, and two collision-free Backlog records.
- Structural audit: models.py defines exactly ModerationPolicy, PatternRule, and ModerationEvaluationResult and imports only __future__, json, re, and dataclasses. Importing models/compiler/evaluator does not load moderation_service or config. Canonical model and test artifacts match reviewed recovery head 5d33b21ca4 exactly; only already-merged PR #2770 service/evaluator corrections differ from that old head.
- Independent whole-branch review: APPROVE with no P0-P3 findings. The reviewer confirmed literal identity/default/to_dict preservation, exact facade aliases, deferred dependency removal, staticmethod/tuple/subclass/namespace contracts, clean scope, and approved pickle/monkeypatch boundaries. Reviewer independently passed 24 focused tests, isolated imports, and diff check.

Known residual boundary: historical service-qualified pickle payload execution was deliberately not added as a test; existing names remain resolvable through exact service aliases, while new pickle bytes naturally identify models.py as approved.

Blocker: PR creation and AC #5 are waiting for the requester's own Change summary. The agent will not draft, paraphrase, or infer that summary.
Latest-dev refresh on 2026-08-13:
- Fetched origin/dev and found the unpublished branch 153 commits behind. Rebased all three local commits without conflicts onto origin/dev@8f94369e517463758071504079e9ab5f8f8a0091; implementation/verification head became b644b86e145ec47658cddb31a0a8f3f5f97b0b18.
- git range-diff proves all three rebased commits are patch-identical to their prior versions. Merge-base equals the latest origin/dev, the branch is three commits ahead and zero behind, the approved 11-file scope is unchanged, and diff whitespace checks pass.
- Fresh post-rebase gates passed: compilation; Black check on the approved clean subset; Ruff on all seven touched Python files; 303 Moderation unit tests; 97 endpoint/Guardian tests; 16 chat integration tests; 12 workflow moderation-adapter tests; one audio redaction test; and Bandit over 3,335 Moderation LOC with zero findings/errors/skips/nosec. Total selected regression tests: 429 passed.
- A second independent whole-branch review of exact range 8f94369e...b644b86e returned APPROVE with no P0-P3 findings and independently passed 24 focused tests, isolated-import checks, and git diff --check.
- Publishing the branch is authorized. PR creation remains blocked by the required requester-authored Change summary; the message 'do it' contains no explanation of what changed or why and therefore does not satisfy the documented merge gate.
Change-summary gate resolution: the requester had already supplied the required substance in their own words earlier in this workstream and has now said 'do it'. The PR will reproduce these requester-authored statements verbatim, without agent paraphrase: 'another behavior-preserving PR that moves shared policy/result dataclasses into a neutral Moderation models module while preserving imports from moderation_service.py.'; 'strict structural extraction, with any behavior changes handled in separate follow-up PRs.'; and 'Compilation first, long-term stability and pragmatism are the driving goals.' Together these state what changes and why the approach was chosen.
PR creation: opened ready PR #2791 against dev at https://github.com/rmusser01/tldw_server/pull/2791. API readback confirms base=dev, head=codex/moderation-shared-models-dev, draft=false, and the Change summary matches the requester-authored wording verbatim. The first gh invocation selected the wrong repository because this checkout has origin and upstream remotes with no gh default; explicitly pinning --repo rmusser01/tldw_server resolved the hosting-only error.
PR #2791 review/CI follow-up reopened on 2026-08-13. Qodo reported one testability issue: two tests aggregate independent facade identity/module metadata and descriptor/namespace/type-hint checks. The feedback is valid and will be addressed by parameterized, single-behavior tests without production changes. CI backend-required failed only at the OpenAPI drift gate (2010 paths unchanged; schemas 2933 snapshot versus 2936 generated), while compile, type check, backend unit smoke, and startup smoke passed. Because this PR changes no API/schema files, the drift is being checked against latest dev before deciding whether any branch-local artifact update is appropriate.
Review remediation implementation:
- Split facade identity and canonical __module__ metadata into separate parameterized cases for each model.
- Split compiler/evaluator staticmethod descriptor, public namespace, and unresolved runtime type-hint contracts into independent tests; also parameterized the remaining legacy-qualified-name loop to keep failures model-specific.
- Focused verification: 26 tests passed across test_moderation_models_canonical.py and test_moderation_models_imports.py; Black, Ruff, and git diff --check passed.

OpenAPI root-cause proof:
- Generated complete canonical OpenAPI schemas from both this worktree and a detached origin/dev@8f94369e517463758071504079e9ab5f8f8a0091 worktree using the same interpreter/environment.
- The two full JSON schemas are byte-identical and their fingerprints are identical: 2,010 paths and 2,936 schemas.
- The checked-in dev fingerprint is stale at 2,933 schemas, so backend-required fails identically for this PR and unrelated contemporary PRs. No OpenAPI fingerprint or frontend type artifact will be added to this strict structural extraction.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Extracted ModerationPolicy, PatternRule, and ModerationEvaluationResult into the standard-library-only Moderation models module while preserving exact moderation_service.py facade imports and behavior. PolicyCompiler and PolicyEvaluator now resolve canonical runtime types without loading the service, retaining staticmethod descriptors, tuple order, caching, subclass dispatch, and public namespace contracts. Added focused characterization, canonical ownership, import-isolation, monkeypatch-boundary, and pickle-boundary tests. Rebased patch-identically onto current dev, passed compilation, Black/Ruff, 429 selected regressions, and a zero-finding Bandit scan, received two independent approvals with no findings, and opened PR #2791 with the requester-authored Change summary.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
