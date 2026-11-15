---
id: TASK-15505
title: Implement and verify the large political PDF research workflow
status: Done
labels:
- research
- pdf
- ux
- uat
dependencies:
- TASK-15504
references:
- Docs/Design/POLITICAL_PDF_VIP_FIXES_2026_11_15.md
- Docs/Reviews/POLITICAL_PDF_VIP_RETEST_2026_11_15.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved priority sequence from TASK-15504 for retained originals, page provenance, scoped QA, focused retrieval, complete chapter summaries, portable research and NN/g heuristic tracking. Rerun the same922pagePDF and DeepSeek through ingestion, summaries, cited answers and export.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implement linked priority fixes with regression coverage and reviewed source provenance
- [x] #2 Complete real PDF and DeepSeek browser walkthrough including chapter coverage and exported research
- [x] #3 Maintain NN/g heuristic and severity ledger with evidence and retest disposition
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User explicitly approved implementation and end-to-end verification on2026-11-15. Baseline a1ec6cc5d1. This task owns design, implementation plan, integration and UAT; child units own product changes. Preserve unrelated original-checkout work.
Managed worktree /Users/macbook-dev/.codex/worktrees/political-pdf-vip-fixes/tldw_server2 on codex/political-pdf-vip-fixes. Three parallel implementation units own provenance, scoped RAG and chapter Jobs; coordinator owns cited-page review/export. Plan and NN/g severity ledger created. ADR assessment: no new ADR; extend existing owners under ADR-003,007,065.
Final verification: integrated frontend TypeScript exit0; all29 changed production Python files Bandit0; focused regressions and independent reviews pass. Full repository suite, vector retrieval, participant usability study and full accessibility conformance not run. Final report records quality failures and all limits. Completed task plan removed per repository policy; durable design/retest evidence retained.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented prioritized PDF research fixes and completed the same-PDF/DeepSeek desktop walkthrough, including fresh ingestion, all 30 selected chapter summaries, individual cited QA, evidence review, Notes/text export and completed ZIP roundtrip. Maintained 43 NN/g findings with severity, evidence and retest disposition. Exact provenance passes; overview/comparison semantic errors remain TASK-15505.6 and mobile/draft feedback remains TASK-15505.5. Temporary credentials and services removed; no pending isolated jobs. Full report: Docs/Reviews/POLITICAL_PDF_VIP_RETEST_2026_11_15.md.
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
