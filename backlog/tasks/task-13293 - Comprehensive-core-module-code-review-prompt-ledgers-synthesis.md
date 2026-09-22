---
id: TASK-13293
title: 'Comprehensive core-module code review: prompt, ledgers, synthesis'
status: Done
assignee: []
created_date: '2026-09-22 04:42'
labels:
  - docs
  - review
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tracking task for the read-only comprehensive code review of the ten largest `app/core` modules plus the `api/v1/endpoints` layer (~800,000 of 1,686,000 LOC in `tldw_Server_API/app`).

Deliverables committed:
- `Docs/Development/Used_Prompts/Code_Review/Comprehensive_Core_Module_Code_Review.md` — the reusable review prompt, filed alongside the existing UX review prompts.
- `Docs/superpowers/reviews/<module>/` — per-module staged ledgers for api-endpoints, authnz, chat, db-management, evaluations, ingestion-media-processing, llm-calls, mcp-unified, rag, sync, tts (78 files, ~19,600 lines).
- `Docs/superpowers/reviews/2026-09-21-core-module-duplication-synthesis.md` — cross-module dedupe, ranked findings, and the `core/Utils` migration plan.

No source file was modified. Additive edits to two existing ledger READMEs (`rag/`, `db-management/`) register the 2026-09-21 extension passes.

Central finding: three independently-discovered High-severity defects share one shape — a dual-backend pair where one side is broken and only the working side is tested. Filed as TASK-13290, TASK-13292 and (proposal) EP-1.

Defects filed from this review: TASK-13287, TASK-13288, TASK-13289, TASK-13290, TASK-13291, TASK-13292. Remaining findings in the synthesis are proposals, not tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Review prompt committed under Docs/Development/Used_Prompts/Code_Review/
- [ ] #2 Per-module ledgers and the cross-module synthesis committed under Docs/superpowers/reviews/
- [ ] #3 No source file modified by the review
- [ ] #4 Verified defects filed as individual Backlog tasks
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Read-only review of 10 modules. 6 defect tasks filed for independently-verified findings. Synthesis records 8 cross-module clusters with dedupe verdicts, ~2,300 LOC of zero-risk deletions, and 8 corrections to the review's own premises.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
- [ ] #7 Acceptance criteria completed
- [ ] #8 Final summary added
<!-- DOD:END -->
