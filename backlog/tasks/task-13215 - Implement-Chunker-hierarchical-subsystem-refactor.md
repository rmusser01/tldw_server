---
id: TASK-13215
title: Implement Chunker hierarchical subsystem refactor
status: In Progress
assignee: []
created_date: 2026-09-07 19:26
updated_date: 2026-10-03 00:21
labels:
- chunking
- refactor
- implementation
dependencies:
- TASK-13112
references:
- TASK-13112
documentation:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
priority: high
modified_files:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
- backlog/completed/task-13112 - Design-Chunker-hierarchical-subsystem-refactor.md
- backlog/archive/tasks/task-13113 - Implement-Chunker-hierarchical-subsystem-refactor.md
- backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md
- tldw_Server_API/app/core/Chunking/chunker.py
- tldw_Server_API/app/core/Chunking/process_text/models.py
- tldw_Server_API/app/core/Chunking/process_text/dispatch.py
- tldw_Server_API/app/core/Chunking/hierarchical/
- tldw_Server_API/tests/Chunking/
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Chunker hierarchical subsystem design in an isolated worktree based on current origin/dev. Extract spans, leaves, tree building, grouping, flattening, models, and service coordination while preserving the approved public, output, fallback, aliasing, logging, and call-trace contracts. No production edits begin until TASK-13112 is approved and finalized.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The implementation branch is reconciled with current origin/dev and focused baseline characterization is rerun before production edits
- [x] #2 Frozen characterization tests cover option handling, leaf call traces, spans, malformed trees, flatten aliasing, logging, signatures, and import boundaries
- [ ] #3 The hierarchical package is extracted with the approved component interfaces and dependency direction while public hierarchy behavior remains compatible
- [ ] #4 The approved private span and header-title helpers are removed and process_text imports the shared span function directly
- [ ] #5 Focused and complete Chunking tests, compileall, Ruff, scoped Black, Bandit, and git diff --check pass with results recorded
- [ ] #6 The PR remains non-merge-ready until the human requester supplies the required Change summary explaining what changed and why
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md task-by-task: (1) reconcile current origin/dev and rerun the focused baseline before production edits; (2) freeze public signatures, composition, option, call-trace, logging, malformed-tree, aliasing, and regex contracts; (3) extract passive models and shared paragraph spans while migrating process_text; (4) extract and activate leaf construction; (5) extract the tree builder and per-call service coordination; (6) extract and activate grouping; (7) extract flattening and complete public delegation; (8) enforce dependency boundaries, run the complete Chunking/static/security gates, obtain final code review, and prepare the PR handoff against dev. Each structural stage uses red-green tests and a focused commit. Any behavior correction must satisfy the approved correction gate and land in a separate fix commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
This record supersedes the colliding active TASK-13113 after rebasing onto origin/dev introduced the unrelated completed TASK-13113 record for PR 2808. The approved design and detailed implementation plan remain unchanged. 2026-09-07 Task 1 baseline reconciliation: refreshed origin/dev at 01b516d4805ddf5eb7cddac35a11ef0e062facc5; the required scoped diff from 4958cfed65d3c6e9baa43ea47e2b155fed204e13 through current origin/dev across tldw_Server_API/app/core/Chunking and tldw_Server_API/tests/Chunking was empty, so no approved hierarchy/process_text contract was affected. Rebased the isolated branch without conflicts. The focused baseline suite collected 92 tests and completed with 91 passed, 1 skipped, 0 failures, and 196 warnings in 3.01s. Environmental output was limited to the existing no-.env fallback to config.txt, legacy single-user API-key warning, isolated test database fallback, and emitted OpenTelemetry spans. No production files were edited; Bandit is not applicable to this documentation/tracking-only task. Tracking reconciliation completed on 2026-09-07: Backlog allocated TASK-13215, the official CLI archived the old active implementation TASK-13113 without modifying the unrelated completed PR-2808 TASK-13113, and the approved spec, implementation plan, and completed TASK-13112 design record now reference TASK-13215.

2026-09-07 second baseline reconciliation: origin/dev was e3174f1ad9f6dd0b11e4ecb20d48c1c4090d3bfe; the required scoped diff from 01b516d4805ddf5eb7cddac35a11ef0e062facc5 through current origin/dev across tldw_Server_API/app/core/Chunking and tldw_Server_API/tests/Chunking was empty. Rebasing preserved the five workstream commits, including the completed Task 1 commit now at fa93165f7b, and produced the required 0/5 branch relationship. The exact focused seven-file suite collected 92 tests and completed with 91 passed, 1 skipped, 0 failures, and 196 warnings in 1.86s. Environmental output remained limited to the existing no-.env fallback to config.txt, legacy single-user API-key warning, isolated test database fallback, and emitted OpenTelemetry spans. No production files were edited; Task 1 checkboxes remain complete.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-09-07 Task 2 contract freeze completed. Added tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py, tldw_Server_API/tests/Chunking/test_hierarchy_malformed_contracts.py, and tldw_Server_API/tests/Chunking/test_hierarchical_spans.py (29 test functions, 54 collected cases) and updated only Task 2 in Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md. Focused pytest: 54 passed, 0 failed, 119 warnings in 1.35s. Ruff: All checks passed. Black --check: 3 files would be left unchanged. git diff --check: exit 0 with no output. Bandit on the three pytest files reported only the expected B101 assert-use rule (57 low-severity findings, 0 medium/high); rerun excluding B101 completed with 0 findings and 0 errors. Self-review confirmed deterministic instance fakes prevent leaf, rewrite, LLM, or external calls; every temporary Loguru sink is removed in finally; metadata fixtures are read-only and preserve the identities under test; malformed/span expectations match the observed unextracted baseline. No baseline discrepancy was found and no production code was changed.
2026-09-07 Task 2 specification-review correction: acceptance criterion #2 remains unchanged in wording but is now unchecked because it also includes import boundaries, which belong to Task 3 and have not been implemented. Task 2's characterization portion is complete and its implementation-plan checkboxes remain accurately checked; this tracking correction does not start Task 3. Strengthened the Task 2 tests to record sanitize_output truthiness before method resolution, cover sanitize_output removal on the semantic rewrite branch as well as metadata and ordinary branches, and assert identity for template and method_options forwarded by the public flat wrapper.
Task 2 specification-review verification: focused three-file suite collected 56 cases and completed with 56 passed, 0 failed, and 123 warnings in 1.17s. Ruff reported All checks passed. Black --check reported all 3 files would be left unchanged. Bandit excluding the expected pytest B101 assertion rule completed with 0 findings and 0 errors. git diff --check exited 0 with no output. The change scope contains only tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py and this official TASK-13215 record; no production file or Task 3 file changed.
Task 2 quality-review correction: hardened safe_search lookup-failure interception, exact public-wrapper call count, words/sentences/tokens metadata fallback and outer-retry traces, all missing/invalid start/end offset cases, and exact single-record fallback logging. Focused suite: 70 passed, 0 failed, 151 warnings in 1.25s; Ruff and Black --check passed; Bandit excluding B101 reported 0 findings/errors; git diff --check passed. Only the two Task 2 test files and this tracking record changed; no production or Task 3 files changed. AC #2 remains unchecked pending Task 3 import-boundary work.
2026-09-07 Task 3 shared models and paragraph spans completed test-first. RED evidence: after editing tests only, the two-file hierarchy run exited 2 during collection with two expected ModuleNotFoundError errors because tldw_Server_API.app.core.Chunking.hierarchical did not yet exist. Direct GREEN: 67 passed, 0 failed, 145 warnings in 1.09s. Required five-file GREEN: 151 passed, 1 skipped, 0 failed, 315 warnings in 1.75s; the existing Thai optional-dependency skip was preserved.

Created tldw_Server_API/app/core/Chunking/hierarchical/__init__.py, models.py, and spans.py. Modified tldw_Server_API/app/core/Chunking/chunker.py, process_text/models.py, process_text/dispatch.py, tests/Chunking/test_hierarchical_spans.py, tests/Chunking/test_hierarchy_refactor_contracts.py, tests/Chunking/test_process_text_components.py, and Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md. The span helper's executable AST matches the pre-extraction method after normalizing only relative-import depth and type annotations. Chunker and process_text now use module-level call-time symbols, and ProcessTextContext no longer exposes the private span helper.

Checks: Ruff passed on all 9 touched Python files. Black --check left the 8 scoped new/process/test files unchanged; separate line-range checks left both changed chunker.py lookup lines unchanged, avoiding unrelated formatting churn in the legacy file. compileall passed for hierarchical, chunker.py, and process_text. Bandit scanned 2,652 production LOC with 0 findings and 0 errors. git diff --check passed. Import inspection found no forbidden hierarchical dependency and no import cycle. AC #2 is now fully satisfied: Task 2 covered option handling, leaf call traces, spans, malformed trees, flatten aliasing, logging, and signatures; Task 3 adds the remaining absolute/relative AST import-boundary coverage. AC #3-#5 remain open because later extraction and full-suite tasks are not complete.
2026-09-07 Task 3 quality-review correction: repaired the AST import-boundary characterization in tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py without production changes. The helper now preserves ImportFrom module/name structure, rejects Chunker (including aliases) and wildcard imports from exactly the parent Chunking package, and uses exact-or-descendant module matching instead of substring matching. Regression coverage now includes relative, absolute, aliased, wildcard, exact, descendant, similarly named allowed modules, and future leaves/grouping upper-layer cases. Review RED: 11 failed, 46 passed, 125 warnings in 1.13s, exposing five missed parent-package re-export/wildcard cases and six false positives for harmless similarly named modules. Focused GREEN: 62 passed, 0 failed, 135 warnings in 1.06s. Required five-file GREEN: 168 passed, 1 skipped, 0 failed, 349 warnings in 1.64s; the existing Thai optional-dependency skip remains. Ruff passed; Black --check left the file unchanged; compileall passed; Bandit on the changed test excluding expected pytest B101 reported 0 findings and 0 errors; git diff --check passed. AC #2 remains checked because the complete import-boundary criterion is now covered by the repaired tests. Task 3 plan checkboxes remain complete.
2026-09-07 Task 4 hierarchical leaf extraction completed test-first on base origin/dev 6cd2745f696af04668a61c20b84ab8a9e69ca5e4 (starting HEAD 10ca821435913a73005bf8a08814b4e9a2dac0fd); no fetch, rebase, or merge was performed. Direct RED after creating only tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py: collection stopped with 0 items and 1 expected ModuleNotFoundError because tldw_Server_API.app.core.Chunking.hierarchical.leaves was missing. Active-wiring RED after creating leaves.py: the focused lookup contract failed once with AttributeError because chunker.py did not yet expose build_leaf_block. Direct GREEN: 34 passed, 79 warnings, 0 failed in 0.75s. Required five-file GREEN after final formatting: 109 passed, 230 warnings, 0 failed in 1.27s. Task 3 five-file regression suite after final formatting: 169 passed, 1 skipped, 351 warnings, 0 failed in 1.64s; the existing Thai optional-dependency skip remains.

Files: created tldw_Server_API/app/core/Chunking/hierarchical/leaves.py and tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py; modified tldw_Server_API/app/core/Chunking/chunker.py, tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py, and Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md. leaves.py now owns the unchanged rewrite, metadata, bounded rolling mapping, and nested/outer fallback branches; chunker.py creates one HierarchyTextViews and ResolvedHierarchyOptions per call and performs only leaf append coordination through the module-level build_leaf_block lookup. The data-driven lower-layer AST boundary actively scanned leaves.py and passed.

Checks/review: Ruff passed on all four touched Python files. Black --check left leaves.py and both touched test files unchanged; line-range Black --check left every changed chunker.py statement unchanged, while a whole-file probe continued to report only unrelated legacy formatting in chunker.py and was intentionally not applied. compileall passed for hierarchical and chunker.py. Bandit scanned 2,203 touched production LOC with 0 findings and 0 errors. git diff --check passed before tracking finalization. Scope/import/duplication self-review found no P0-P3 findings: leaves.py imports no Chunker, process_text, service, builder, grouping, or flatten module, and the duplicate local leaf body/rewrite set is absent from chunker.py. AC #2 remains checked; AC #3/#4/#5 and all later task-plan steps remain open because Tasks 5-8 and the complete final suite are not complete.
2026-10-02 Task 4 interrupted-work recovery (America/Los_Angeles): starting HEAD 10ca821435913a73005bf8a08814b4e9a2dac0fd; recovered the six existing staged Task 4 files in place. The previous implementer's 2026-09-07 RED/GREEN evidence remains historical evidence, not a new RED run; implementation and staging were complete but the commit was interrupted by a usage limit. No production/test changes were needed in recovery, and no reset, fetch, rebase, merge, or Task 5 work was performed.

Fresh verification, each command run after source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate:
- python -m pytest tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py tldw_Server_API/tests/Chunking/test_hierarchical_rewrite_offsets.py tldw_Server_API/tests/Chunking/test_offsets_additional.py tldw_Server_API/tests/Chunking/test_chunking_regressions.py -q: 109 passed, 0 failed, 230 warnings in 1.94s.
- python -m pytest tldw_Server_API/tests/Chunking/test_hierarchical_spans.py tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py tldw_Server_API/tests/Chunking/test_process_text_components.py tldw_Server_API/tests/Chunking/test_process_text_refactor_equivalence.py tldw_Server_API/tests/Chunking/test_thai_tables_spans.py -q: 169 passed, 1 skipped, 0 failed, 351 warnings in 2.49s. Existing optional skip: PyThaiNLP not available.
- python -m ruff check on chunker.py, hierarchical/leaves.py, test_hierarchical_leaves.py, and test_hierarchy_refactor_contracts.py: All checks passed.
- python -m black --check on leaves.py and both touched test files: 3 files unchanged. python -m black --check --line-ranges 26-27 --line-ranges 445-453 --line-ranges 462-465 tldw_Server_API/app/core/Chunking/chunker.py: unchanged; no whole-file legacy formatting was applied.
- python -m compileall -q tldw_Server_API/app/core/Chunking/hierarchical tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/tests/Chunking/test_hierarchical_leaves.py tldw_Server_API/tests/Chunking/test_hierarchy_refactor_contracts.py: exit 0.
- python -m bandit -r tldw_Server_API/app/core/Chunking/chunker.py tldw_Server_API/app/core/Chunking/hierarchical/leaves.py -f json -o /tmp/bandit_TASK-13215_task4_recovery_2026-10-02.json: exit 0; 2,203 production LOC, 0 findings, 0 errors, 0 skipped tests.
- git diff --check --cached: exit 0.

Scoped self-review found no concrete Task 4 problem: rewrite-method membership, metadata invalid-offset handling, exact logs, rolling search/clamping, nested/outer fallback behavior, freshness/non-mutation, call-time lookup, and the active leaves.py AST dependency boundary match the approved requirements. Existing test environment warnings/config fallback and OpenTelemetry output remain. Only Task 4 plan steps are finalized by the recovery commit; TASK-13215 remains In Progress, later Task 5-8 work and full final gates remain open. Commit subject: refactor: extract hierarchical leaf construction; the commit includes this tracking evidence and the Task 4 plan update.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
