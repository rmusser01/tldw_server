---
id: TASK-188
title: Remove dayjs from remaining ReadingList and Items WebUI date surfaces
status: Done
assignee: []
created_date: '2026-05-09 20:06'
updated_date: '2026-05-09 20:27'
labels:
  - webui
  - dependencies
  - issue-1346
  - dayjs
dependencies:
  - TASK-176
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1436'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue GitHub issue #1346 after the merged Media FilterPanel cleanup by removing the final direct shared-UI dayjs imports from ReadingItemsList and ItemsWorkspace. Scope is limited to replacing those Ant Design Dayjs value/display paths with native date handling, updating focused tests and the issue #1346 dependency audit, and removing direct dayjs manifest declarations only if current import scans confirm no direct package imports remain. Preserve existing filter semantics and table display behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ReadingItemsList no longer imports or references dayjs or Dayjs while preserving existing read/unread/archived/search/status/date filter behavior.
- [x] #2 ItemsWorkspace no longer imports or references dayjs or Dayjs while preserving existing item date range filter behavior and published date display labels.
- [x] #3 Focused tests cover native date input/display behavior and the shared-UI dayjs import guard now expecting zero direct dayjs package imports.
- [x] #4 apps/packages/ui, apps/tldw-frontend, and apps/extension package manifests no longer declare dayjs after direct import scans confirm it is unused by repo source.
- [x] #5 Docs/Design/WebUI_Dependency_Audit.md records the reduced dayjs import count, manifest outcome, and remaining transitive Ant Design/ExcelJS/Mermaid caveat.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in isolated worktree branch codex/webui-final-dayjs-native-dates-1346. ReadingItemsList and ItemsWorkspace now use native date inputs backed by shared date-input helpers for local start/end day ISO conversion. ItemsWorkspace published_at display now uses native Date formatting and preserves invalid raw labels. The exact shared UI dayjs import guard now expects zero direct imports.

Removed direct dayjs declarations from apps/packages/ui/package.json, apps/tldw-frontend/package.json, and apps/extension/package.json after exact active-code import scans across apps returned no direct package imports. Regenerated apps/bun.lock with scripts disabled; dayjs remains only through transitive lockfile ownership from Ant Design, ExcelJS, Mermaid, and optional picker peer metadata.

Verification: focused Vitest first failed on missing native ReadingList/Items date inputs and four remaining dayjs imports, then passed with 3 files and 10 tests after implementation. Exact active-code dayjs import scan returned no matches, direct manifest declaration scan returned no matches, WebUI lint exited 0 with the existing 131-warning baseline, WebUI compile exited 0 and generated 138 static pages with token sync OK, git diff --check exited 0. TypeScript baseline still exits 2 on existing EmbeddingsModelSelectionConfig.tsx and lib/api/vnPlay.ts errors with no task-scope matches. Bandit skipped because no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed the remaining direct dayjs usage from the WebUI/shared UI dependency surface. ReadingList and Items date filters now use native date inputs with shared local day-boundary helpers, Items published-date labels use native Date formatting, the shared UI import guard now requires zero direct dayjs imports, and direct dayjs declarations were removed from the WebUI, shared UI, and extension manifests with the lockfile refreshed. The audit records that dayjs remains only as a transitive dependency of packages such as Ant Design, ExcelJS, and Mermaid.
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
