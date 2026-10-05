---
id: TASK-13450
title: Fix WebUI memory growth in comparisons audio PDFs media previews and caches
status: In Progress
created_date: 2026-10-05 01:17
priority: high
documentation:
- Docs/Design/WEBUI_MEMORY_FIXES_2026_10_04.md
updated_date: 2026-10-05 02:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address all seven findings from the Chrome 30GB UAT memory review and create a PR against dev. Bound diff allocations; terminate retired workers; own TTS Blob URLs and cancel late responses; virtualize PDF canvases; bound and cancel media buffering; evict expired and excess shared cache entries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Diff computation has a shared allocation bound and workers terminate when retired.
- [x] #2 TTS resources are released on replacement Stop and unmount; late results cannot restart playback.
- [x] #3 PDF canvas count and media preview buffering remain bounded and retired downloads abort.
- [x] #4 Chat and character caches evict expired and excess entries.
- [ ] #5 Regression tests and relevant checks are recorded and a PR targets dev.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed implementation stages 1-4 in IMPLEMENTATION_PLAN_webui_memory_fixes.md; final verification and publication against dev remain in stage 5.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
All seven reviewed resource/allocation issues addressed using existing dependencies and authenticated transport. Added cancellation and cleanup regressions for comparisons, synthesis, streaming audio, PDF navigation/search, media downloads and shared caches. Final read-only reviewer found no important issues. Native Chrome fixture with 1000 mixed-size PDF pages: 24 immediate repeated jumps between 500 and 1000 preserved selection and mounted at most 5 page canvases; thumbnails and internal links/low zoom also checked. Synthetic fixture does not reproduce or prove resolution of the reported 30GB UAT peak. Embedded media preview ceiling is 64MiB; shared caches have 32-entry/8MiB payload ceilings. Production frontend typecheck passed; touched-scope lint has no errors or new warnings versus dev. Final affected tests are running; prior 116-file run had 1083 passing and 7 failures, with those same 7 failures independently reproduced on clean dev. Repo-wide UI attempt stopped after about 50 minutes without a complete report. Bandit on the TypeScript touched scope reported no findings and scanned no Python files, so does not validate TypeScript. Original dirty UAT checkout remains untouched.
Final affected run completed: 117 files, 1094 tests, 1087 passing and the same 7 clean-dev characters-list-all failures. Final frontend typecheck exit 0; installed ESLint exit 0 with no new findings versus baseline; final PDF group 56 tests pass. Git diff whitespace check passes. origin/dev advanced to ba553fdc51 with backend Claims changes only; the fix branch will rebase onto it before publication.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
