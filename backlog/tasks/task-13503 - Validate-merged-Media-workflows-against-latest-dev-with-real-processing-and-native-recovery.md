---
id: TASK-13503
title: Validate merged Media workflows against latest dev with real processing and
  native recovery
status: Done
labels:
- media
- validation
- ux
dependencies:
- TASK-13500
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate the remaining Media workstream evidence gaps on latest dev 7226596cca5b875ad6e121a15133f69600152258 after PR 3194. Use isolated local state and existing processing/Notes APIs. Distinguish real ingestion, local model/provider processing and indexing from fixture evidence. Exercise actual Notes Trash recovery, keyboard/mobile/accessibility paths; do not claim screen-reader speech or novice participant studies without actual execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Record exact dev revision, isolated configuration, inputs, commands and outcomes in a validation report.
- [x] #2 Exercise actual single and multiple-source ingestion, saved content review, recovery and retrieval; record concrete processing/provider limitations.
- [x] #3 Exercise Notes Trash recovery against actual API through the browser and check keyboard/mobile accessibility.
- [x] #4 Attempt native screen-reader validation where permitted and state human participant validation limits honestly.
- [x] #5 Stop task-owned processes and verify original checkout and user library were untouched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requested directly by the user after merge and cleanup. Plan: IMPLEMENTATION_PLAN_media_live_validation_20261005.md. Report: Docs/Reviews/2026-10-05-media-live-validation.md. ADR required: no; validation reuses existing API, persistence and provider rules without changing architecture. ADR-059 governs canonical Backlog editing; ADR-026 governs actual URL retrieval. No new dependency or external messaging authorized.
Live API: real Markdown and URL extraction saved media; whisper-tiny transcribed spoken audio; explicitly loaded MLX Qwen3-0.6B-4bit produced persisted analysis; genuine Redis embedding worker completed MiniLM vector generation and semantic/FTS retrieval returned media 1. Notes native bulk Trash -> reload -> Restore -> reload preserved IDs/title/content and advanced version 1 -> 2 -> 3. Keyboard j and slash focus passed; typing j in search did not navigate. Native VoiceOver was attempted with user-approved foreground use but spoken output could not be verified; VoiceOver process ended and ChatGPT foreground restored. Follow-up findings TASK-13504/13505/13506.
Final evidence: Docs/Reviews/2026-10-05-media-live-validation.md and output/playwright/media-live-validation-20261005/{receipts.json,4 screenshots}. Latest remote dev remained 7226596cca5b875ad6e121a15133f69600152258 through final check. Actual retry submitted only the failed PDF; corrected PDF subsequently saved as media 7. Existing scoped tests 288/288 passed with the established CI timeout; production Chrome extension build passed 37.7s. Temporary plan completed and removed per AGENTS. Bandit non-code skip: only report, task records and selected evidence changed. No app/Python source changed. Test browsers, WebUI/API/Redis workers and isolated Redis stopped; dedicated ports had no listeners. Task dependency symlinks removed; managed Whisper weights preserved in /private/tmp/media-live-validation-20261005. Original checkout porcelain bytes matched the before snapshot exactly and stayed codex/post2970-uat-20260920. VoiceOver process absent; original ChatGPT foreground restored.
2026-10-06 PT merge follow-up: requester approved updating PR #3204 against current dev before merge. MERGE_QUEUE is unset, so use the manual current-base procedure for this PR only. Current remote PR head is d5b853deb06565f7a656299a6bbab217fd1ede10; fetched dev is 005802bdb070fd68e087c4db3f831c33bef07c39. No changed file overlaps the 238 intervening dev commits. Required gates are green on the old head; the broader media-ingestion-new-integration shard reports 239 passed, 1 skipped and one teardown error in the unchanged offline email fixture. Current dev already includes stronger tripwire diagnostics and daily-accounting isolation for that fixture. Rebase the authorized PR, run affected UI suites and the failed email integration file on the resulting source, then confirm exact-head/current-base required checks before landing. ADR required: no; existing architecture and merge policy remain in effect. No new application behavior is planned.
Current-dev integration verified on rebased source 5a8ea9b4b0bb367745f7a17647c48fed1dd1ee1c, base 005802bdb070fd68e087c4db3f831c33bef07c39. No rebase conflicts; all 26 PR application files preserve their pre-rebase content. 18 Media files/464 tests passed in 56.18s; 46 email integration tests passed in 9.80s, including the prior failed case and the new dev fixture guards. Existing dev supplies daily-accounting isolation; no new application patch. WebUI typecheck and lint (0 errors, 180 baseline warnings), Chrome production (48.4s, 49.5 MB), WebUI production/token/budget (595.6 KB under 600 KB) pass. Initial local Node 26 WebUI build exhausted its 4 GB heap; successful rerun used the existing 8 GB typecheck allowance through NODE_OPTIONS, without code/config edits. Current-base task ratchet: 4 tests pass. Bandit: 0 findings, 24 unsupported TypeScript parse errors; no TS security pass claimed. Four dependency links and generated caches/tracing copies removed. Report/receipts updated; original browser/API/model and accessibility evidence retains its earlier revision boundary. User-owned summary remains verbatim, requester authorized updating and merging; final landing waits for seven protected checks on the published head and unchanged dev. ADR required: no; merge queue remains unset.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the original real ingestion, model/indexing, saved-review and Notes recovery validation at its recorded revision. Follow-up UX findings TASK-13504 through TASK-13507 are implemented and verified in PR #3204. Current-dev integration onto 005802bdb070 preserves all 26 PR application files; 464 Media tests and 46 email integration tests, WebUI types/lint and both production builds pass. The prior failed email case passes with the fixture repair already present on dev. VoiceOver speech and human participant studies remain unverified; original live evidence was not rerun on the new base. Temporary dependencies/build artifacts are cleaned. Human-owned summary and merge authorization are recorded; final landing waits for exact-head protected checks and an unchanged dev tip.
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
