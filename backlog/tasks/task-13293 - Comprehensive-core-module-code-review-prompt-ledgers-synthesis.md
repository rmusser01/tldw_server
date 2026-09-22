---
id: TASK-13293
title: 'Comprehensive core-module code review: prompt, ledgers, synthesis'
status: Done
assignee: []
created_date: '2026-09-22 04:42'
updated_date: '2026-09-22 05:01'
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
COMPLETE FILED SET (44 tasks). The description above was written mid-filing and says "remaining findings are proposals"; that is superseded - all 40 ranked synthesis findings are now filed.

Pre-existing / filed during the review: 13287 13288 13289 13290 13291 13292 13294 13295 13296 13297
Filed from the ranked table afterwards: 13300 13301 13302 13306 13307 13308 13309 13310 13314 13315 13316 13317 13318 13319 13322 13323 13324 13325 13326 13327 13328 13329 13330 13331 13332 13333 13334 13335 13336 13337 13338 13339 13340 13341

TASK-13287 gained an addendum: the regex fix alone does not fix the Chat path, because NetworkError is absent from _CHAT_ORCHESTRATOR_PROVIDER_EXCEPTIONS.

Stage 0 of the migration plan (zero-risk items) is being executed under the individual task IDs.
<!-- SECTION:NOTES:END -->

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
