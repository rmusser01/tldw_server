---
id: TASK-15416
title: Implement Chunker shared execution subsystem refactor
status: In Progress
labels:
- chunking
- refactor
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the requester-approved five-stage shared execution plan using Native execution and independent whole-branch review. Preserve the accepted live-context boundary and all existing path-specific behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Freeze current public signatures, docstrings, options, cache, metadata and generator behavior before extraction
- [ ] #2 Actively delegate text, metadata and generator paths through a live execution context without behavioral changes
- [ ] #3 Share equivalent preparation and strategy setup with arbitrary user options kept separate from internal keywords
- [ ] #4 Verify real cold imports, six CI matrix assignments, full Chunking, static checks, ratchets and touched-scope Bandit
- [ ] #5 Obtain independent whole-branch review and disposition all findings before publication; do not merge without separate authority and a new human Change summary
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved spec Docs/superpowers/specs/2026-11-02-chunker-shared-execution-refactor-design.md; plan Docs/superpowers/plans/2026-11-02-chunker-shared-execution-refactor.md; accepted ADR-086 Docs/ADR/086-chunker-execution-live-context-boundary.md. ADR required: yes, existing accepted ADR-086 governs; no new architecture decision. Design TASK-15414 and planning TASK-15415 are completed records, not dependencies. Requester explicitly approved Native execution followed by independent whole-branch review. Use only existing isolated worktree on codex/chunker-execution-refactor-design. HEAD38776160fb; reconciled live dev5937f33065 has no scoped source/test/policy/ADR changes relative to approved baseline3ab1ce5d. Two dev bookkeeping commits are unrelated; preserve branch and defer integration rebase under queue policy. Fresh full Chunking baseline796passed1existingPyThaiNLPskip2000warnings42.12s (/tmp/chunker_execution_implementation_baseline.txt). HEAD/latest-dev/all39openPR file lists scanned with pagination: max15415, allocated15416. Main/bookkeeping/unrelated history remain untouched.
Stage1 characterization frozen on unchanged production code:103passed280warnings3.00s (/tmp/chunker_execution_task1_frozen.txt). Initial81case full suite877passed1establishedskip2206warnings40.14s; final full rerun follows stage commit via official task-done. Ruff/Black/compileall, shardguard0newuncovered, taskIDguard and Bandit0findings/errors (onlypytestB101excluded) pass. Test scaffolding corrections traced before freeze; no production behavior correction. Python3.13+ compiler docstring indentation differs from3.12, so independent literal text compared with inspect.cleandoc while source docstrings remain untouched. Task-start heading parser needed cosmetic Task N headings retaining Stage N; no plan content/scope change. Current TASK15415 planning handoff markedDone after explicit Native approval.
Stage2 active text migration: meaningful RED18fail (absent service/hooks/wiring), initialGREEN121pass then final123pass332warnings4.85s. Full pre-final-group suite918pass1establishedskip2292warnings42.80s; official postcommit full run follows. Normalized AST proves moved plain body identical after live-context/hook substitutions and local type annotations; all other facade methods, public signature and original source docstring unchanged. Context/hooks/cache protocols introduced, service constructor inert; real4cold-import processes pass. Ruff/Black/compileall/newpackage mypy3files and production/test Bandit0findings/errors pass. Compatibility probe caught an invalid proposed typing-only suppress expansion: grouped exceptions differ on3.12/3.13/3.14. Exact pre-extraction method reproduces originalTypeError. Focused regression RED with expanded tuple, GREEN after restoring original nested tuple; six narrow arg-type ignores preserve runtime behavior, no blanket suppression or inherited correction. All strategy outputs/identities and observability remain unchanged.
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
