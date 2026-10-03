---
id: TASK-13450
title: Fix WebUI memory growth in comparisons audio PDFs media previews and caches
status: In Progress
created_date: 2026-10-05 01:17
priority: high
documentation:
- Docs/Design/WEBUI_MEMORY_FIXES_2026_10_04.md
updated_date: 2026-10-06 03:14
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
- [ ] #6 PR review findings are resolved, the branch is rebased on latest dev, checks pass and PR 3192 is merged with the requester Change summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
PR follow-up: (1) rebase latest dev and reproduce Qodo PDF-scroll/diff findings; (2) apply minimal fixes with regression tests and return annotations; (3) review, verify, respond to review threads, add requester Change summary and merge when current-head checks pass. Follow-up plan: IMPLEMENTATION_PLAN_webui_memory_pr3192_review.md.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
All seven reviewed resource/allocation issues addressed using existing dependencies and authenticated transport. Added cancellation and cleanup regressions for comparisons, synthesis, streaming audio, PDF navigation/search, media downloads and shared caches. Final read-only reviewer found no important issues. Native Chrome fixture with 1000 mixed-size PDF pages: 24 immediate repeated jumps between 500 and 1000 preserved selection and mounted at most 5 page canvases; thumbnails and internal links/low zoom also checked. Synthetic fixture does not reproduce or prove resolution of the reported 30GB UAT peak. Embedded media preview ceiling is 64MiB; shared caches have 32-entry/8MiB payload ceilings. Production frontend typecheck passed; touched-scope lint has no errors or new warnings versus dev. Final affected tests are running; prior 116-file run had 1083 passing and 7 failures, with those same 7 failures independently reproduced on clean dev. Repo-wide UI attempt stopped after about 50 minutes without a complete report. Bandit on the TypeScript touched scope reported no findings and scanned no Python files, so does not validate TypeScript. Original dirty UAT checkout remains untouched.
Final affected run completed: 117 files, 1094 tests, 1087 passing and the same 7 clean-dev characters-list-all failures. Final frontend typecheck exit 0; installed ESLint exit 0 with no new findings versus baseline; final PDF group 56 tests pass. Git diff whitespace check passes. origin/dev advanced to ba553fdc51 with backend Claims changes only; the fix branch will rebase onto it before publication.
Published PR #3192 against dev and attached it to the Codex chat. Final code review reports no important findings. Source commit rebased onto ba553fdc51; no frontend changes intervened. All requested fixes are published. Repository-wide suite completion and reproduction of the original 30GB UAT peak remain unverified, as documented in the PR. Merge requires the requester to author the human Change summary; no merge performed.
Requester authorized rebase, all Qodo/PR issue remediation and merge, and supplied the human Change summary in chat. Qodo posted four findings: missing comparison-helper return types; pre-metadata user-scroll snap-back; ordinary revisions becoming full replacements due to a 256-edit limit; scroll-reported page changes triggering navigation snap-back. Existing required checks on prior head are green. ADR required: no new ADR for this follow-up; it repairs behavior within the existing comparison and PDF lifetimes without changing module boundaries, public APIs, dependencies or authentication. ADR-059 governs task editing; ADR-006 governs touched-scope security validation.
Rebased without conflicts onto latest dev 5775d3fbbe. Reproduced Qodo behavior findings red before fixing: ordinary revisions lost all unchanged lines; fallback discarded common edges; actual worker budget too low; pre-metadata reader scroll jumped back; momentum feedback triggered navigation. Main budget1024/100ms and worker2048/1000ms retain finite bounds, fallback reconstructs both sides with common edges. PDF uses native scroll for fixed offsets to avoid virtualizer retries; distinguishes scroll feedback, preserves pre-metadata input and explicit destinations, clears markers on mode exit and cancels queued frames before geometry/mode transitions. Independent reviewer reproduced two additional mode/frame races; both fixed with red-to-green coverage and no remaining Important findings. Follow-up run: 52 files, 246 tests pass and one ContentViewer.analysis-markdown failure; same failure reproduced independently on clean latest dev. Native Chrome synthetic 1000-page fixture:24 rapid500/1000 jumps correct, max5 canvases; manual top20000 retained, page25 selected,6 canvases. Frontend typecheck passed before final callback cleanup; final typecheck running. Final touched lint checks running; prior zero errors and one unchanged documentId warning. Bandit0findings/0Python files in TS scope. Requester Change summary added verbatim with attribution to PR body; merge gate satisfied by requester-owned text.
Final review-fix frontend typecheck exits0. Installed touched-scope lint exits0:zero errors, one unchanged documentId warning. All23 PDF/diff memory regression cases pass in the52-file affected run;246/247 tests pass, and the sole Markdown presentation failure is confirmed in clean latest-dev code with identical dependencies. Git diff check passes. Existing temporary per-task plan remains local until final merge; durable design contains the implementation/verification notes.
2026-10-06 02:25 UTC: All seven required dev statuses passed on published 64c1b7e879, with no new findings or unresolved threads. Dev advanced to 8aeddeee36 only through backend router-contract test isolation and its task record. Rebased all three PR commits onto that tip without conflicts; range-diff confirms unchanged patches and the complete apps tree is identical to the verified head. All 23 PDF/diff memory regressions pass again after the rebase. Publication will rerun required gates on the new actual head; no merge or gate bypass yet.
2026-10-06 03:05 UTC: All seven required gates also passed on 8e014d7e637, with no unresolved or new review findings. Dev advanced through approved Media UX PR 3194 to 587cd8e9fe. Rebased all four PR commits without conflicts; range-diff preserves their patches. The only shared touched file is TldwApiClient.ts, where dev adds scoped cancellation to media operations and this PR bounds chat/character caches elsewhere. Integration verification: 16 test files, 450 tests pass; fresh production frontend typecheck exits 0; whitespace check passes. Read-only integration review found no actionable Important issues and confirmed authentication/scope checks, cancellation and cache/resource lifetimes remain intact. The official local Backlog writer resolves this memory-fix record despite an incoming task-ID collision; the unrelated Media audit record is untouched. The rebased publication still needs actual-head required gates before normal merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all seven reviewed WebUI memory allocation/resource issues using existing dependencies and authentication. Added regression coverage; 1087 tests pass with seven clean-dev baseline failures across 117 files. Frontend typecheck and touched-scope lint pass with no new findings. Chrome 1000-page PDF fixture retained navigation with at most five continuous page canvases. Opened and attached https://github.com/rmusser01/tldw_server/pull/3192 targeting dev; documented the 64MiB media-preview ceiling and UAT profiling limitations.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
