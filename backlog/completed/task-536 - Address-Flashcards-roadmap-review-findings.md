---
id: TASK-536
title: Address Flashcards roadmap review findings
status: Done
assignee: []
created_date: '2026-05-28 02:00'
updated_date: '2026-05-28 02:01'
labels:
  - ux
  - flashcards
  - planning
  - docs
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Patch the Flashcards UX remediation roadmap after review: clarify Study Pack Source ID phase ownership, allow invalid-import recovery to proceed right after evidence refresh, prevent Phase 0 from landing unresolved failing tests, tighten grep/test naming guidance, add minimal responsive checks to early phases, and clean up completed Backlog task storage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Roadmap resolves the reviewed sequencing, ambiguity, and verification risks without changing the approved outcome-based structure.
- [x] #2 Completed roadmap Backlog record is moved or otherwise cleaned up so the active task tree is not polluted by a Done planning task.
- [x] #3 Documentation-only verification is recorded for ASCII, key sections, and whitespace.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Patched the Flashcards remediation roadmap after review. Changes: clarified that Phase 1 labels the current Study Pack Source ID path while Phase 6 owns real source picker/search; allowed the high-severity invalid-import recovery sub-slice to proceed after Phase 0 instead of waiting for Phase 1; added Phase 0 RED/fixme/reproduction guidance so unresolved failing tests are not merged; added grep/test-title guidance to prevent no-op Playwright commands; added minimal narrow-width smoke checks to Phases 1-3; and moved completed TASK-535 into completed storage. Verification: confirmed roadmap is 634 lines; confirmed targeted review-fix strings with rg; confirmed ASCII-only content; ran git diff --check on touched files with no whitespace findings. Bandit/tests skipped because this task changes planning and Backlog documentation only.
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
