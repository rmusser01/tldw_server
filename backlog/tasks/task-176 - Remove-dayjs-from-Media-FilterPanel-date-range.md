---
id: TASK-176
title: Remove dayjs from Media FilterPanel date range
status: Done
assignee:
  - codex
created_date: '2026-05-09 18:46'
updated_date: '2026-05-09 19:00'
labels:
  - webui
  - dependencies
  - issue-1346
  - media
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1436'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue issue #1346 by replacing the shared UI Media FilterPanel date range control's dayjs/Ant Design RangePicker value path with native platform date inputs. Scope is limited to FilterPanel date-range state conversion, focused tests, the dependency audit, and task metadata. Leave ReadingList and Items Dayjs value-contract surfaces for later slices.

Expected impact estimate for this narrow replacement: reduce active shared UI dayjs package-import lines from 5 to 4 and remove one RangePicker/Dayjs runtime filter surface from Media. No direct manifest, lockfile, install-size, or bundle-size reduction is expected until dayjs is removed from the remaining ReadingList and Items surfaces, and Ant Design still owns dayjs transitively.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FilterPanel no longer imports or references dayjs or Ant Design DatePicker/RangePicker for media date range filtering.
- [x] #2 The media date range editor uses native date inputs that display existing ISO start/end dates as YYYY-MM-DD values and allow clearing each side independently.
- [x] #3 Changing dates emits start-of-day ISO for startDate and end-of-day ISO for endDate, while clearing both emits null values for both fields.
- [x] #4 Focused tests cover displayed native date values, start/end conversion, independent clearing, and the remaining dayjs import scan expectation.
- [x] #5 Issue #1346 audit notes are updated to reflect the reduced shared UI dayjs import count and remaining deferred ReadingList/Items DatePicker value-contract surfaces.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create an isolated worktree from current origin/dev on branch codex/webui-media-native-date-range-1346.
2. Mirror TASK-176 into the worktree so the task metadata travels with the PR branch.
3. Write focused failing tests for the Media FilterPanel native date inputs: displayed YYYY-MM-DD values, start/end ISO conversion, independent clearing, and remaining dayjs import scan count.
4. Replace FilterPanel's dayjs/AntD RangePicker path with native date inputs and small date conversion helpers that parse YYYY-MM-DD into local start/end day boundaries before ISO serialization.
5. Update Docs/Design/WebUI_Dependency_Audit.md and TASK-176 notes/criteria to record the reduced shared UI dayjs import count and remaining ReadingList/Items surfaces.
6. Run focused Vitest, dayjs import scan, WebUI compile/lint/targeted TypeScript filter, git diff --check, and Bandit skip note for JS-only touched scope before committing and opening a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified PR #1427 merged into dev at a443c105a41431b8dea56074f1e9cf3418f948af on 2026-05-09T18:45:52Z.

Corrected the expected import-line impact from 5->3 to 5->4 because ReadingList and Items each retain separate runtime and type dayjs imports after the Media slice.

Implemented the Media FilterPanel date range replacement in branch codex/webui-media-native-date-range-1346: removed the combined dayjs/Dayjs import and Ant Design RangePicker usage from FilterPanel, added native labeled date inputs, and preserved local start/end day-boundary ISO emission plus independent clearing behavior.

Focused red/green verification: `bunx vitest run src/components/Media/__tests__/FilterPanel.test.tsx src/components/Media/__tests__/FilterPanel.dayjs-imports.test.ts --maxWorkers=1` first failed on missing native labels and five remaining dayjs imports, then passed with 12 tests after implementation.

Exact dayjs scan now returns four remaining shared UI import lines: ReadingItemsList.tsx runtime/type imports and ItemsWorkspace.tsx runtime/type imports. Media FilterPanel no longer appears in the direct dayjs import scan.

WebUI compile passed: `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile` generated 138 static pages and token sync reported OK. `bun run lint` exited 0 with the existing 131-warning baseline. `git diff --check` exited 0.

TypeScript baseline note: `node_modules/.bin/tsc --noEmit --project tsconfig.json --pretty false` still exits 1 on existing EmbeddingsModelSelectionConfig.tsx and lib/api/vnPlay.ts errors; filtering /tmp/task176_tsc.log for FilterPanel, components/Media, Media/__tests__, task-176, and WebUI_Dependency_Audit returned no matches.

Bandit skipped because this slice changed TypeScript, documentation, and Backlog metadata only; no Python files were modified.

Correction after rerunning the TypeScript check post-test-harness tweak: `node_modules/.bin/tsc --noEmit --project tsconfig.json --pretty false` exits 1, not 2, on the existing EmbeddingsModelSelectionConfig.tsx and lib/api/vnPlay.ts baseline errors; the task-specific filter still returns no matches.

Opened PR #1436 against dev: https://github.com/rmusser01/tldw_server/pull/1436
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced the Media FilterPanel date range control's dayjs/Ant Design RangePicker path with native labeled date inputs while preserving the existing controlled MediaDateRange contract. The new helpers format existing ISO values as YYYY-MM-DD, emit local start-of-day/end-of-day ISO values when edited, and preserve the opposite boundary when either side is cleared.

Added focused component coverage for native date display, start/end conversion, and independent clearing, plus a static shared-UI dayjs import guard that now expects only the deferred ReadingList and Items surfaces. Updated the issue #1346 dependency audit to record the active shared UI dayjs import count dropping from 5 to 4 and to keep direct dependency removal deferred until ReadingList/Items are migrated.

Verification: focused Vitest suite passed after the required red run, exact dayjs import scan returns only ReadingItemsList and ItemsWorkspace runtime/type imports, WebUI compile exited 0 and generated 138 static pages with token sync OK, WebUI lint exited 0 with the existing 131-warning baseline, git diff --check exited 0, and the TypeScript baseline still exits 1 on existing EmbeddingsModelSelectionConfig.tsx/lib/api/vnPlay.ts diagnostics with no task-scope matches. Bandit was skipped because no Python files changed.
<!-- SECTION:FINAL_SUMMARY:END -->
