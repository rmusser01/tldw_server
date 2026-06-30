---
id: TASK-387
title: Address PR 1724 cache review comments
status: Done
assignee: []
created_date: '2026-05-15 19:16'
updated_date: '2026-05-15 19:20'
labels:
  - llm-cache
  - pr-review
  - cost-control
dependencies: []
documentation:
  - 'https://github.com/rmusser01/tldw_server/pull/1724'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review and address unresolved PR #1724 comments about Anthropic cache breakpoints and usage metadata redaction. Verify against code before changing behavior, keep fixes scoped, and update the PR after pushing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Anthropic cache intent handling can mark up to four eligible cache breakpoints without exposing prompt content
- [x] #2 Usage metadata redaction preserves numeric prompt-like usage counters while still redacting prompt text and secrets
- [x] #3 Focused regression tests cover both review findings
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Addressed Gemini review threads by adding Anthropic multi-breakpoint cache marking up to the documented maximum of four eligible text blocks and by preserving nested numeric usage counters under prompt-like metadata keys while redacting free-form prompt strings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed both unresolved PR #1724 review findings. Anthropic cache intent handling now marks up to four eligible system text cache breakpoints, with a bounded provider-hint override for fewer breakpoints. Raw usage metadata redaction now preserves nested numeric usage counters under prompt-like keys such as input/text while still redacting prompt-like free-form strings and secrets. Verified with focused red/green tests, 106 touched regression tests, py_compile, git diff --check, and Bandit with no findings.
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
