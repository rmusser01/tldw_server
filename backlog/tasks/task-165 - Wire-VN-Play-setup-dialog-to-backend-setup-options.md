---
id: TASK-165
title: Wire VN Play setup dialog to backend setup options
status: Done
assignee:
  - codex
created_date: '2026-05-09 15:58'
updated_date: '2026-05-09 17:00'
labels:
  - vn-play
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1407'
  - 'https://github.com/rmusser01/tldw_server/pull/1413'
  - 'https://github.com/rmusser01/tldw_server/pull/1419'
documentation:
  - Docs/API-related/VN_PLAY_API.md
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 VN Play new-session dialog loads character and asset-pack setup state from GET /api/v1/vn-play/setup-options instead of doing client-side readiness fan-out.
- [x] #2 Character and pack selectors render backend-derived labels, defaults, compatibility, trust, warning, and empty-state data while preserving the existing VNPlaySessionCreate payload.
- [x] #3 Manual ID entry remains available when the backend setup-options request fails.
- [x] #4 Focused WebUI tests cover setup-options API usage, payload creation, warning rendering, fallback behavior, and the absence of per-pack readiness fan-out in the setup dialog.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add frontend VN Play setup-options TypeScript types and an API client wrapper in apps/tldw-frontend/lib/api/vnPlay.ts.
2. Refactor NewSessionDialog to consume backend setup options, request updated options when selected character/content rating changes, and keep manual fallback for setup endpoint failures.
3. Update VN Play component/API tests and smoke mocks to use /vn-play/setup-options instead of separate character/pack/readiness setup calls.
4. Run focused frontend tests, lint/diff checks, and Bandit on touched backend-adjacent scope if backend files are touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification: focused Vitest passed (2 files, 21 tests); ESLint passed on touched frontend files; git diff --check passed; Playwright smoke passed after approved local dev-server bind; Bandit skipped because this slice touched only frontend TypeScript/TSX, e2e mock, docs, and Backlog task files with no Python backend code.

PR opened: https://github.com/rmusser01/tldw_server/pull/1419

PR review pass: live review surface has five unresolved threads. Treating Gemini reset-state, Qodo warning acknowledgement, and Qodo duplicate setup-options refetch as actionable. Qodo listCharacters/listVNAssetPacks findings conflict with the chosen backend setup-options API direction and will be answered as non-actionable design mismatch.

PR review fixes: reset all new-session form fields on reopen; added frontend acknowledgement for high-risk setup warnings and persisted settings.setup_acknowledgements on create; suppressed the extra setup-options request caused by applying backend default character IDs. Verification after review fixes: focused Vitest passed (2 files, 24 tests); ESLint passed on touched frontend files; git diff --check passed; Playwright VN Play smoke passed after approved local dev-server bind. Bandit remains skipped because no Python files are touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Wired the VN Play new-session dialog to the backend setup-options API and handled PR review feedback by adding form reset coverage, high-risk setup warning acknowledgement metadata, and deduping backend-default refetches. Non-actionable reviewer suggestions to restore listCharacters/listVNAssetPacks were kept out because this slice intentionally moved setup selection behind the backend setup-options API for custom frontend/API-server parity.
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
