---
id: TASK-15574
title: Generate Study Packs across the whole saved media source
status: Done
dependencies:
- TASK-15569
references:
- https://github.com/rmusser01/tldw_server/pull/3504
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
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Existing ADR-003 Jobs ownership and ADR-025 provider routing govern this internal change. Keep the existing API, worker and persistence transaction. Build and test the generation plan, then publish the reviewed public dev update.
The persisted synthetic ebook reproduced loss of the final chapter at the old evidence bound. The preview and worker now share the chapter and Markdown heading plan and a paragraph fallback. The drawer defaults to all sections, lets users narrow the selection, and shows the generation-pass limit. The worker rejects stale scopes, preserves canonical citation spans and owner context, and commits the deck only after generation and validation complete. The generated API contract was refreshed under ADR-069. Existing ADR-003 and ADR-025 boundaries govern the job and provider.
Verification passed with 162 backend Study Pack tests, 84 focused UI tests, a final 16-test drawer rerun, WebUI typecheck, scoped ESLint and Ruff, and OpenAPI drift checks. Bandit introduced no new findings. The 42 backend skips were PostgreSQL-dependent because the fixture could not reach a service. The change is verified in local tests with synthetic source text.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Large saved documents can be previewed by section before generation. The user can select all sections or a subset. Both the full synthetic ebook and a final-chapter-only selection produced correctly scoped citations.
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
