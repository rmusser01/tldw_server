---
id: TASK-13450
title: Fix WebUI memory growth in comparisons audio PDFs media previews and caches
status: Done
created_date: 2026-10-05 01:17
priority: high
documentation:
- Docs/Design/WEBUI_MEMORY_FIXES_2026_10_04.md
updated_date: 2026-10-05 02:43
references:
- https://github.com/rmusser01/tldw_server/pull/3192
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
- [x] #5 Regression tests and relevant checks are recorded and a PR targets dev.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stages 1-5 complete: bounded comparisons and worker retirement; audio ownership/cancellation; virtualized PDF and bounded authenticated media; expiring bounded shared caches; final regression/type/lint/browser checks and dev PR publication. Completed per-task implementation plan removed after publication; durable design and verification remain in Docs/Design/WEBUI_MEMORY_FIXES_2026_10_04.md.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
All seven reviewed resource/allocation issues addressed using existing dependencies and authenticated transport. Added cancellation and cleanup regressions for comparisons, synthesis, streaming audio, PDF navigation/search, media downloads and shared caches. Final read-only reviewer found no important issues. Native Chrome fixture with 1000 mixed-size PDF pages: 24 immediate repeated jumps between 500 and 1000 preserved selection and mounted at most 5 page canvases; thumbnails and internal links/low zoom also checked. Synthetic fixture does not reproduce or prove resolution of the reported 30GB UAT peak. Embedded media preview ceiling is 64MiB; shared caches have 32-entry/8MiB payload ceilings. Production frontend typecheck passed; touched-scope lint has no errors or new warnings versus dev. Final affected tests are running; prior 116-file run had 1083 passing and 7 failures, with those same 7 failures independently reproduced on clean dev. Repo-wide UI attempt stopped after about 50 minutes without a complete report. Bandit on the TypeScript touched scope reported no findings and scanned no Python files, so does not validate TypeScript. Original dirty UAT checkout remains untouched.
Final affected run completed: 117 files, 1094 tests, 1087 passing and the same 7 clean-dev characters-list-all failures. Final frontend typecheck exit 0; installed ESLint exit 0 with no new findings versus baseline; final PDF group 56 tests pass. Git diff whitespace check passes. origin/dev advanced to ba553fdc51 with backend Claims changes only; the fix branch will rebase onto it before publication.
Published PR #3192 against dev and attached it to the Codex chat. Final code review reports no important findings. Source commit rebased onto ba553fdc51; no frontend changes intervened. All requested fixes are published. Repository-wide suite completion and reproduction of the original 30GB UAT peak remain unverified, as documented in the PR. Merge requires the requester to author the human Change summary; no merge performed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all seven reviewed WebUI memory allocation/resource issues using existing dependencies and authentication. Added regression coverage; 1087 tests pass with seven clean-dev baseline failures across 117 files. Frontend typecheck and touched-scope lint pass with no new findings. Chrome 1000-page PDF fixture retained navigation with at most five continuous page canvases. Opened and attached https://github.com/rmusser01/tldw_server/pull/3192 targeting dev; documented the 64MiB media-preview ceiling and UAT profiling limitations.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
