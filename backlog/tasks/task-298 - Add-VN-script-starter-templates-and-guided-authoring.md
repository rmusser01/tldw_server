---
id: TASK-298
title: Add VN script starter templates and guided authoring
status: Done
assignee: []
created_date: '2026-05-12 05:57'
updated_date: '2026-05-12 06:33'
labels:
  - vn
  - webui
  - api
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1604'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
  - 'https://github.com/rmusser01/tldw_server/pull/1606'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1604: make VN script authoring easier by adding backend-owned starter templates that API clients and the bundled WebUI can use to create draft VN scripts, while preserving backend validation and publish as the authority. This should reduce blank-page friction without introducing a visual node editor or text DSL.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 API clients can list supported starter templates with stable IDs, labels, descriptions, required capabilities, and preview-safe metadata.
- [x] #2 API clients can create a VN script draft from a template through a documented backend endpoint or request shape.
- [x] #3 Template-created drafts validate through the existing diagnostics path and publish through the existing publish path.
- [x] #4 The /vn-scripts WebUI can start from a template and still exposes the JSON editor and diagnostics panel for full control.
- [x] #5 Tests prove template output is deterministic, user-owned, and cannot smuggle unsupported runtime directives or unsafe policy changes.
- [x] #6 Relevant documentation or implementation notes describe the template contract for custom frontends.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation will proceed in an isolated worktree on branch codex/vn-script-templates-1604. Scope: backend-owned VN script starter template catalog and create-from-template endpoint/request shape, TypeScript API/types, /vn-scripts WebUI template picker, focused backend/frontend tests, documentation/plan updates, and verification including Bandit for touched Python scope.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added backend-owned starter template catalog endpoints under `/api/v1/vn/vn-scripts/templates`.
- Added WebUI template selection while preserving the blank/custom JSON creation path.
- Documented the custom frontend contract in `Docs/API-related/VN_PLATFORM_API.md` and the VN platform API spec.
<!-- SECTION:NOTES:END -->

## Verification

<!-- SECTION:VERIFICATION:BEGIN -->
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/VN_Scripts -q` -> 47 passed, 5 warnings.
- `bunx vitest run __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-scripts/VNScriptsWorkbench.test.tsx` from `apps/tldw-frontend` -> 23 passed.
- `bunx eslint components/vn-scripts/VNScriptsWorkbench.tsx lib/api/vnScripts.ts types/vn-scripts.ts __tests__/vn-scripts/vnScriptsApi.test.ts __tests__/vn-scripts/VNScriptsWorkbench.test.tsx` from `apps/tldw-frontend` -> passed.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/vn_scripts.py tldw_Server_API/app/api/v1/schemas/vn_script_schemas.py tldw_Server_API/app/core/VN_Scripts -f json -o /tmp/bandit_vn_script_templates.json` -> passed, 0 findings.
- `git diff --check` -> passed.
- Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1606
<!-- SECTION:VERIFICATION:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented backend-owned VN script starter templates and guided authoring for #1604. The API now exposes a sanitized template catalog plus create-from-template flow that stores normal script drafts through existing validation. The WebUI now loads the catalog, offers a compact starter selector, and immediately hydrates returned template drafts while keeping full JSON editing, diagnostics, validation, save, and publish controls available.
<!-- SECTION:FINAL_SUMMARY:END -->
