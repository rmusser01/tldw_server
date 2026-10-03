---
id: TASK-13422
title: Implement Chunker hierarchical subsystem refactor
status: In Progress
created_date: 2026-10-03 01:30
dependencies:
- TASK-13421
labels:
- chunking
- refactor
- implementation
priority: high
references:
- TASK-13421
- backlog/tasks/task-13215 - Implement-Chunker-hierarchical-subsystem-refactor.md
- https://github.com/rmusser01/tldw_server/pull/3095
documentation:
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
modified_files:
- tldw_Server_API/app/core/Chunking/chunker.py
- tldw_Server_API/app/core/Chunking/hierarchical/
- tldw_Server_API/app/core/Chunking/process_text/models.py
- tldw_Server_API/app/core/Chunking/process_text/dispatch.py
- tldw_Server_API/tests/Chunking/
- Docs/superpowers/specs/2026-08-23-chunker-hierarchical-subsystem-refactor-design.md
- Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md
updated_date: 2026-10-03 02:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the user-approved compatibility-first hierarchical subsystem refactor. This replacement task resolves the active TASK-13215 collision with unrelated upstream Writing work after October 2 baseline reconciliation; approved scope and completed implementation stages are unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The implementation branch is reconciled with current origin/dev and focused baseline characterization is rerun before production edits
- [x] #2 Frozen characterization tests cover option handling, leaf call traces, spans, malformed trees, flatten aliasing, logging, signatures, and import boundaries
- [x] #3 The hierarchical package is extracted with the approved component interfaces and dependency direction while public hierarchy behavior remains compatible
- [x] #4 The approved private span and header-title helpers are removed and process_text imports the shared span function directly
- [x] #5 Focused and complete Chunking tests, compileall, Ruff, scoped Black, Bandit, and git diff --check pass with results recorded
- [ ] #6 The PR remains non-merge-ready until the human requester supplies the required Change summary explaining what changed and why
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Continue Tasks7-8 in the approved plan after completed Tasks1-6 and specification/quality approvals: extract flatten traversal and complete public delegation test-first; enforce boundaries and resolve only hierarchy-local typing issues without behavior changes; run focused/full Chunking and static/security gates; final independent reviews; push and create draft PR against dev with human-written Change summary merge blocker. No behavior corrections pre-approved.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-02 tracking reconciliation: supersedes ambiguous TASK-13215, retaining complete historical evidence at the explicit old record path. New design authority is TASK-13421. Pinned origin/dev86e287fee7bfa1a1588639232e35db3666851ded; isolated branch codex/chunker-hierarchical-refactor-design at3e34078eef78111633a210cda4a476baa25abbeb. Tasks1-6 complete with separate specification and quality approvals. Task5 regression caused by erroneous controller grouping-passthrough instruction was validated and reverted in b8d015c582; exact seven-key envelope remains. Task6 required155pass/65contracts, all static gates/Bandit0. Full pre-grouping suite669pass/1PyThaiNLPskip/1747warnings/47.45s; initial tokenizer cache failure resolved by priming cl100k_base in one network-enabled existing test, with no code/skip changes. Two informational hierarchy-local mypy errors in spans.py122/127 reserved for final cleanup. Task7 worker interrupted before code edits when ID collisions discovered; one start note remains in the historical task, unrelated upstream records unchanged. Resume same Task7 worker after tracking reconciliation. Draft PR creation authorized; keep task In Progress until human-written Change summary gate satisfied.
The previously completed acceptance criteria 1,2,4 are retained from the old implementation record; remaining package extraction/full final verification/human-summary gates remain open. Forward-looking spec and plan references now use unique TASK-13421/TASK-13422 IDs. Old records and historical IDs remain provenance, with no further mutation by ambiguous ID.
2026-10-02 Task 7 resumed from clean tracking-migration HEAD 52abf0bc014c0a5b38db225ec3224532fc44b5f0, same production baseline as 3e34078eef; origin/dev remains 86e287fee7bfa1a1588639232e35db3666851ded. Use this unique task only; historical TASK-13215/TASK-13112 records are controller-owned. Direct missing-module RED, then active-wiring RED precede delegation. Found two frozen Task 6 lookup tests in test_hierarchical_grouping.py target chunker module helpers; requested narrow allowlist expansion to migrate those lookups to hierarchical.flatten while preserving all assertions. Other Task 7 work proceeds within six Python files plus plan/task only; no Task 8 changes.
2026-10-02 Task 7 RED/GREEN progress: tests-only direct file run exited 2 with the expected ImportError for absent hierarchical.flatten (12 warnings, 1.04s; /tmp/task7_direct_red.log). After moving the exact traversal body with only dedentation and self.normalize_chunk_type -> supplied callback, direct GREEN: 36 passed, 84 warnings, 1.23s (/tmp/task7_direct_green.log). Before public/service wiring, five new active contracts failed as expected (66 deselected, 22 warnings, 1.14s; /tmp/task7_wiring_red.log): public lookup still bypassed the sentinel and HierarchyService had no flatten method. Added the exact non-dict guard-before-callback service delegate, unchanged public signature/docstring wrapper, removed unused grouping imports. Plan Task7 Steps1-4 checked; full integration/static gates pending. No typing cleanup, package helper changes, or flat-composition changes.
2026-10-02 Task 7 verification before scope clarification: delegated direct/public/package three-file GREEN is 125 passed, 262 warnings, 2.29s (/tmp/task7_delegation_green.log). Exact required 14-file run is NOT GREEN: 346 passed, 1 established PyThaiNLP skip, 2 failed, 711 warnings, 3.20s (/tmp/task7_required_green.log). Both failures are test_hierarchical_grouping.py::test_flatten_uses_call_time_grouping_helpers[False/True], which still patch removed chunker module imports. User was asked to authorize only migrating those frozen lookup targets to hierarchical.flatten, outside the six-file Task 7 allowlist. Did not weaken assertions, retain unused imports, edit out-of-scope files, or commit the failing suite. Steps5-6 remain open in the plan; task remains In Progress pending controller clarification.
Gates: Ruff all six Python paths passed after fixing one new-test E731 lambda assignment. Black reformatted only new flatten.py/test_hierarchical_flatten.py; checks pass on the five scoped paths and changed chunker.py ranges26/425-427. Compileall all six paths passed. Bandit scan of chunker.py/flatten.py/service.py:1682 production LOC,0 findings,0 errors (/tmp/bandit_TASK-13422_task7.json). git diff --check passed. Read-only normalized AST self-review proves exact current traversal after dedentation and callback-name substitution, unchanged public tree/flat composition and package flatten helper/catch tuple, unchanged service constructor/build_tree; flatten.py has no self/context references and imports only grouping/standard library. No production behavior correction or Task8 cleanup. Known spans.py typing issues unchanged.
2026-10-02 Controller-authorized narrow Task 7 extension BEFORE file edits: independently validated the two failing frozen test_flatten_uses_call_time_grouping_helpers cases at test_hierarchical_grouping.py lines211-226 patch Chunking.chunker.merge_texts/group helpers, while the approved public wrapper no longer owns/imports these symbols. This is an active lookup migration necessitated by extraction, not a baseline behavior correction. Authorization permits this seventh Python path ONLY to retarget the test import/patches to Chunking.hierarchical.flatten, retaining every frozen call/expected assertion and continuing to invoke the public Chunker method. Update Task7 Files/git-add entries to include it, then rerun exact14-file suite/static/security/diff gates and commit. No further user reconfirmation, unrelated edits, or Task8 work.
2026-10-02 Task 7 flatten extraction complete; controller-authorized narrow seventh Python file lookup migration resolves the interim scope blocker. Exact required14-file final GREEN:348 passed,1 established PyThaiNLP skip,0 failures,711 warnings,3.20s (/tmp/task7_final_green.log). RED evidence remains recorded above: absent module ImportError; five active service/public wiring failures; two frozen Task6 tests using stale lookup targets before authorized migration. Direct coverage: malformed dictionary matrix/public guard separation, ancestry/section paths/empty and nested titles, per-call callback and type fallback, preserved indices/totals, header buffering/header-only sections, element/by-kind weights, recursive no-duplicate traversal, nonmutation/shallow metadata copies/nested identities/shared ancestry lists.
Final gates after seventh-path migration: Ruff all seven Python paths passes; Black --check all six scoped new/test/service files and separate chunker.py changed ranges26/425-427 passes with no legacy whole-file formatting; compileall all seven paths passes. Final Bandit on chunker.py/flatten.py/service.py:1682 production LOC,0 findings/errors (/tmp/bandit_TASK-13422_task7_final.json). git diff --check passes. Fresh normalized AST self-review proves unchanged moved traversal after only callback lookup replacement, unchanged public tree/flat composition and package helper explicit exception tuple, unchanged one-assignment service constructor/build_tree. Frozen Task6 grouping statements/assertions identical except import/public Chunker owner lookup; rest of test_hierarchical_grouping.py identical to baseline. No actionable extraction issue found.
Changed paths: created hierarchical/flatten.py and tests/Chunking/test_hierarchical_flatten.py; modified chunker.py, hierarchical/service.py, tests/Chunking/test_hierarchy_refactor_contracts.py, test_hierarchy_malformed_contracts.py, and the explicitly authorized test_hierarchical_grouping.py lookup only, Task7 section in approved plan, and this unique record. Task7 Steps1-6 finalized by commit subject refactor: extract hierarchical flattening. Starting HEAD52abf0bc014c0a5b38db225ec3224532fc44b5f0 and pinned origin/dev86e287fee7bfa1a1588639232e35db3666851ded unchanged before commit. No fetch/rebase/push, main-checkout edits, unrelated/historical task mutation, validation/deepcopy/error handling correction, or Task8 typing cleanup. Existing environment/config/deprecation warnings remain; missing-module RED emitted known closed-stream Loguru diagnostics. Two spans.py122/127 typing issues remain reserved for Task8. TASK-13422 remains In Progress: overall final verification/reviews and human-written PR Change summary gates remain open.
2026-10-02 Task8 preflight before production cleanup: verified clean HEAD b38c0cd74816d78d80d72faed2ae07dc134d8e7c, branch codex/chunker-hierarchical-refactor-design and unchanged pinned origin/dev86e287fee7bfa1a1588639232e35db3666851ded. Fresh exact mypy RED /tmp/task13422_mypy_red.txt exits1:13 hierarchy-local errors across spans.py122/127, grouping.py127/214/239, flatten.py51/52/53/68/121/123/124/139 (7 source files). Span fallback will use identity cast(int, code_fence_start), not assert/guard, preserving even None on exceptional append; rename template-rule kind to avoid classifier optional inference conflict. Additional grouping/flatten errors independently validated as dynamic dict.get values (including malformed values/second-read semantics); narrowly annotate affected local text/metadata/config/weight values Any rather than coercing or adding validation. Existing frozen malformed/component contracts retained. No behavior correction. Ownership search shows only absence assertions plus builder.py private _extract_header_title definition/calls: approved Task5 explicitly moves helper here and spec320/570 requires builder ownership, so do not delete/rename it merely to satisfy contradictory Task8 broad search expectation; clarify plan wording. No outer-owner hierarchy imports. Steps8-10 and independent reviews remain controller-owned; task stays In Progress.
2026-10-02 Task8 preflight GREEN: exact Task7 14-file suite348passed/1establishedPyThaiNLPskip/711warnings/3.08s (/tmp/task13422_preflight_focused.txt); full normal-sandbox Chunking777passed/1same skip/1963warnings/47.84s (/tmp/task13422_preflight_full.txt), both exit0. Separate AST boundary selection24passed/47deselected/60warnings/1.17s (/tmp/task13422_preflight_ast.txt). Exact compileall and Ruff exit0; scoped Black exit0,15files unchanged (no whole-file legacy formatting). Exact informational mypy exit0 residual output: Success: no issues found in 7 source files (/tmp/task13422_mypy_green.txt). No process_text baseline changes. Exact Bandit JSON /tmp/bandit_task_13422.json inspected:0findings,0errors,2822LOC; high/medium/low/undefined severity and confidence all0; nosec0,skipped_tests0. Existing config/deprecation warnings remain, no environment workaround/newskip/install. Self-review normalized ASTs for spans/grouping/flatten identical to starting HEAD after stripping annotations, undoing local template_kind rename, and unwrapping identity cast; frozen public argument ASTs and flat composition unchanged against pinned origin/dev; package helper byte-identical. Ownership AST contracts pass; builder private title helper intentionally retained per Task5/spec, plan expectation clarified. No dormant duplicate body, new guards, coercions, broad suppressions, exception handling changes or behavior corrections. Branch/working diffchecks pass. Exact changed files: hierarchical/spans.py,grouping.py,flatten.py; approved plan; only current task. Steps1-7/completion evidence updated; Steps8-10 remain pending independent spec/quality reviews then separate final doc/evidence pass and controller PR. Task In Progress. Historical tracking/main checkout untouched. Incremental preflight commit planned: refactor: clarify hierarchical span annotations, not Step9 final commit.
2026-10-02 final Task8 Steps8/9 evidence: controller explicitly reports separate final whole-branch SPEC and QUALITY APPROVED on immutable86e287fee7bfa1a1588639232e35db3666851ded..ae72a4702f1baeeac70f5abb35ec512f769a0152, no actionableP0-P3 findings/no code changes required. Actual controller full log read (/tmp/task13422_final_controller_tests.txt):777passed,1optionalPyThaiNLPskip,1963warnings,48.76s. Skip source is test_thai_sentence_with_pythainlp_if_available, reason PyThaiNLP not available; no new skip. Actual controller BanditJSON read (/tmp/bandit_task_13422_controller.json):0findings,0errors,2822LOC, all severity/confidence counts0,nosec0,skipped_tests0. Controller additionally reports independent Ruffpass/Black15unchanged/compileallpass/mypy0issues7sourcefiles/diffcheckclean; consistent with directly verified preflight evidence above. Exact mypy residual: Success: no issues found in 7 source files. Focused14file preflight348passed/1skip/711warnings3.08s; AST24passed/47deselected/60warnings1.17s. Existing config/deprecation warnings remain. Main checkout, production/tests and historical records untouched in this final pass. Only plan/current task evidence edited; no further broad tests required because reviewed code unchanged. Fresh branch diffcheck passes; final docs-only staged scope/diffcheck required before commit. Plan Steps8/9 and final review checklist checked; Step10 draftPR against dev pending controller. AC3/4/5 checked;AC6 remains open;DoD2-6 checked,DoD1 open;status In Progress. Explicit merge blocker: human requester must write their own Change summary explaining what changed and why these architecture/compatibility choices were selected. AI technical summary is not that human-owned summary. Pending historical archive exception user answer remains controller-owned; no old record edits/archive. Final evidence commit subject test: verify hierarchical subsystem refactor (this docs-only commit), SHA to be reported to controller after creation.

Verified branch commit inventory through reviewed code head:
5e4ecb0b966ee6779592e2d46f717e42ca157aa8 docs: design Chunker hierarchical refactor
8e21d2730d1053b87245276c7019e8e686c7bee9 docs: tighten hierarchical refactor contracts
90c9bbe8ea21970330a4a57a2d932bf601eecb85 docs: define hierarchical refactor delivery gates
6be96260918f7e1ae619d5861eb0bbefe06aa8de docs: plan hierarchical subsystem refactor
d3c28256c981e0e709fdea2a02a09badcef9f568 docs: reconcile hierarchical refactor baseline
3bb3fc13731c2f65e27d40e2b7447cf36e42a1b9 docs: refresh hierarchical refactor baseline
b0e24d0632bf71cd7c4c698b17dbf0ac3ab36211 docs: preserve hierarchical task provenance
4587d002a07716f6fffdf3e5baed08699b48e83d test: characterize hierarchical chunking contracts
c18db2bd66ef70a2cfda97340bc226d9421773a9 test: strengthen hierarchical compatibility contracts
9a0f02f729d920ca1004cfece306748f90af5e07 test: harden hierarchical characterization coverage
6b3b7f857bd907977bb55b31f3dfa2ab1fbb1dcc refactor: share hierarchical paragraph spans
72a957c1b403c4823376bbde3c4eefc5a13dddf0 test: tighten hierarchical import boundaries
b397c524c2619250cc26679c3df8716d9858a6ab refactor: extract hierarchical leaf construction
38c3e232fb96776ffefb84dc02e676b88023da24 docs: reconcile hierarchical refactor baseline before tree extraction
198fb5204cf4301e27cd9904a923cc7062a6e78c refactor: extract hierarchical tree builder
b8d015c5821e23091023d70972f426953fdaa766 fix: preserve hierarchical tree envelope during extraction
3e34078eef78111633a210cda4a476baa25abbeb refactor: extract hierarchical grouping
52abf0bc014c0a5b38db225ec3224532fc44b5f0 docs: reconcile Chunker workstream tracking identities
b38c0cd74816d78d80d72faed2ae07dc134d8e7c refactor: extract hierarchical flattening
ae72a4702f1baeeac70f5abb35ec512f769a0152 refactor: clarify hierarchical span annotations

Final changed-file inventory: hierarchy package __init__/models/spans/leaves/builder/grouping/flatten/service; public chunker.py; process_text/models.py and dispatch.py; frozen hierarchy/malformed/spans/leaves/builder/grouping/flatten contracts plus narrow process_text component seam migration; approved spec/plan and historical tracking reconciliation provenance. This final pass touches only Docs/superpowers/plans/2026-08-24-chunker-hierarchical-subsystem-refactor.md and backlog/tasks/task-13422 - Implement-Chunker-hierarchical-subsystem-refactor.md. No final behavior correction. Prior b8d015c582 restores exact baseline envelope after extraction regression, not baseline defect correction.
2026-10-02 Step10 handoff: pushed codex/chunker-hierarchical-refactor-design and created draft PR #3095 against dev: https://github.com/rmusser01/tldw_server/pull/3095. PR includes module/compatibility rationale, focused348pass1skip/full777pass1skip, compile/Ruff/Black/mypy/Bandit evidence, spec/plan links, and explicit human-written Change summary merge blocker. App artifact attached. Code at ae72a4702 and final docs evidence at5225c7fcf; controller independently reran full tests777pass1skip1963warnings48.76s and all static/security checks, Bandit0findings/errors2822LOC. Final specification and quality reviews approved with no remaining actionable code findings. Task stays In Progress; AC6human gate remains unchecked. Historical colliding task records remain preserved pending requester's scoped archive preference, unrelated upstream records untouched. Worktree retained for review; no merge performed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved compatibility-first hierarchy decomposition into passive models/protocols, shared spans, leaf construction, tree building, grouping, flatten traversal, and live-context service coordination. Chunker retains its public signatures and flat composition; process_text uses the shared span function directly. Focused boundaries and direct contracts improve reviewability without changing observed offsets, option precedence, call multiplicity, exception/logging behavior, malformed-input handling, or shallow identities. Type cleanup uses localized annotations, an identity cast and a local rename; b8d015c582 restores the baseline envelope after an extraction regression, not a baseline behavior correction. Both final whole-branch reviews approved. Focused348passed/1optional skip; full777passed/1PyThaiNLPskip; compileall/Ruff/scopedBlack/mypy pass; Bandit0findings/0errors2822LOC; diff checks clean. Draft PR #3095 is open against dev: https://github.com/rmusser01/tldw_server/pull/3095. Worktree preserved. MERGE BLOCKER: the human requester must write their own Change summary explaining what changed and why these boundaries and compatibility choices were selected; this AI technical summary does not satisfy that gate. Task remains In Progress with AC6/overallDoD1 open. Historical colliding records are preserved pending the requester's scoped archive preference; unrelated upstream records untouched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
