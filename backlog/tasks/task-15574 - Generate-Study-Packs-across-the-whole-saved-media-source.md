---
id: TASK-15574
title: Generate Study Packs across the whole saved media source
status: Done
dependencies:
- TASK-15569
references:
- https://github.com/rmusser01/tldw_server/pull/3504
- https://github.com/rmusser01/tldw_server/pull/3512
- Docs/Design/Flashcards.md
- Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Study Pack path clips long evidence to the first 12,000 characters and makes one model call. Whole-media selections must generate from bounded passages through the end of the saved item.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Whole-media generations include the last part of a long saved document and validate citations against each model input.
- [x] #2 Source snapshots and per-job inference remain bounded, and regeneration retains whole-media selection.
- [x] #3 Explicit excerpts, timestamps and chunks preserve existing scope.
- [x] #4 Chapter sets show current card counts and keep all generation passes for one chapter together in the same deck.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Existing ADR-003 Jobs ownership and ADR-025 provider routing govern this internal change. Keep the existing API, worker and persistence transaction. Build and test the generation plan, then publish the reviewed public dev update.
The persisted synthetic ebook reproduced loss of the final chapter at the old evidence bound. The preview and worker now share the chapter and Markdown heading plan and a paragraph fallback. The drawer defaults to all sections, lets users narrow the selection, and shows the generation-pass limit. The worker rejects stale scopes, preserves canonical citation spans and owner context, and commits the deck only after generation and validation complete. The generated API contract was refreshed under ADR-069. Existing ADR-003 and ADR-025 boundaries govern the job and provider.
Verification passed with 162 backend Study Pack tests, 84 focused UI tests, a final 16-test drawer rerun, WebUI typecheck, scoped ESLint and Ruff, and OpenAPI drift checks. Bandit introduced no new findings. The 42 backend skips were PostgreSQL-dependent because the fixture could not reach a service. The change is verified in local tests with synthetic source text.
Expand the draft with a deck browser that reads stable chapter metadata from generated citations. Keep All cards available and preserve deck scheduling. Add a regression with several passes in the same chapter.
Reviewed the chapter browser additions under existing ADR-069. A saved media ID, section-plan revision, and section index identify each set. The owner-scoped browse endpoint reads the current deck and ignores SQLite device identifiers when matching citations. The regression covers two passes of the same chapter plus a separate final chapter, deleted cards, edits from a second device, and distinct chapters in different revisions. Verification passed with 163 backend Study Pack tests and 209 focused UI tests across 18 suites. WebUI typecheck, scoped ESLint and Ruff, OpenAPI drift, and diff checks passed. Bandit introduced no new findings. Local PostgreSQL was unreachable, so 42 PostgreSQL-dependent tests skipped. The chapter UI was verified in component tests.
Server PR #3625 exposed a shared UI race when a media revision arrived. The checkboxes now render directly from the current section selection, so the first render of revised labels clears old choices. The regression checks that first commit as well as the exact submitted locator. The existing API and revision contract still apply.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Large saved documents can be previewed by section before generation. The user can select all sections or a subset. A stored synthetic ebook verified full and chapter-only scope, and grouped cards from two passes of the first chapter into the same set. The Study tab now browses All cards and chapters with current card counts within one deck.
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
