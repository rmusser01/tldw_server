---
id: TASK-420
title: Address PR 1820 review comments
status: Done
labels:
- review
- sources
- webui
- backend
modified_files:
- apps/packages/ui/src/components/Option/Sources/SourceForm.tsx
- apps/packages/ui/src/hooks/keyboard/useKeyboardShortcuts.ts
- apps/packages/ui/src/hooks/keyboard/useShortcutConfig.ts
- apps/packages/ui/src/services/tldw/server-capabilities.ts
- apps/packages/ui/src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx
- apps/packages/ui/src/hooks/keyboard/__tests__/useShortcutConfig.test.ts
- apps/packages/ui/src/services/__tests__/server-capabilities.test.ts
- tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py
- tldw_Server_API/app/core/Ingestion_Sources/access_policy.py
- tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_path_browser.py
- tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py
- backlog/tasks/task-405 - Implement-folder-to-notes-Sources-UI-exposure.md
- backlog/tasks/task-406 - Add-Sources-server-path-picker.md
- Docs/superpowers/plans/2026-05-17-sources-server-path-picker-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve still-actionable CodeRabbit, Qodo, and Gemini review comments on PR #1820, then verify and update the PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Still-actionable PR #1820 review comments are addressed in code, tests, or current task/plan text.
- [x] #2 Focused frontend and backend regression tests pass.
- [x] #3 Bandit and git diff whitespace checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1820 review comments: stabilized SourceForm reset dependencies, suppressed mode-navigation shortcuts while editable elements are focused, hardened shortcut config storage coercion, fetched local-directory source entitlements for fallback source discovery, offloaded backend directory browsing filesystem work from the async route, added missing endpoint return annotations and helper docstrings, made malformed rollout percentages fail closed, made the path-browser test order independent, and cleaned current PR task/plan absolute-path and wording nits.

Verification:
- `bunx vitest run src/components/Option/Sources/__tests__/SourceForm.test.tsx src/hooks/keyboard/__tests__/useModeNavigationShortcuts.test.tsx src/hooks/keyboard/__tests__/useShortcutConfig.test.ts src/services/__tests__/server-capabilities.test.ts` -> 4 files, 57 tests passed.
- `project venv python -m pytest tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_path_browser.py tldw_Server_API/tests/Ingestion_Sources/integration/test_ingestion_sources_access_policy.py tldw_Server_API/tests/Ingestion_Sources/unit/test_access_policy.py -q` -> 37 passed, 5 warnings.
- `project venv python -m bandit -r tldw_Server_API/app/api/v1/endpoints/ingestion_sources.py tldw_Server_API/app/core/Ingestion_Sources/access_policy.py tldw_Server_API/app/api/v1/schemas/ingestion_sources.py -f json -o /tmp/bandit_pr1820_review_fixes.json` -> exit 0, no findings.
- `git diff --check` -> exit 0.
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
