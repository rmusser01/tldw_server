---
id: TASK-12030
title: Fix WebUI single-user auth persistence and settings reload
status: Done
assignee: []
created_date: '2026-06-25 21:00'
updated_date: '2026-06-25 21:08'
labels:
  - webui
  - auth
  - settings
  - ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 1 of the WebUI audit remediation roadmap: a single-user API key saved in Settings must remain available after reload, direct navigation, and route entry without requiring NEXT_PUBLIC_X_API_KEY.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Saving /settings/tldw writes one canonical browser config record and compatible legacy keys needed by existing request helpers.
- [x] #2 Env-provided credentials are not silently overridden by stale browser storage.
- [x] #3 A fresh browser can save connection settings, reload /chat, and make authenticated requests.
- [x] #4 API key values are not exposed in app-owned diagnostics, copied payloads, logs, accessible names, or generated regression artifacts.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 1 implementation task for WebUI audit auth persistence. Planned touched files: apps/tldw-frontend/hooks/useConfig.tsx, apps/tldw-frontend/lib/authStorage.ts, apps/tldw-frontend/lib/api.ts, apps/packages/ui/src/components/Option/Settings/tldw.tsx, apps/packages/ui/src/services/tldw/TldwApiClient.ts, apps/tldw-frontend/hooks/__tests__/useConfig.networking.test.tsx, apps/tldw-frontend/lib/__tests__/api.credentials.test.ts, apps/tldw-frontend/e2e/workflows/settings.spec.ts.

Implemented in the Stage 1 worktree. Actual touched files: apps/tldw-frontend/hooks/useConfig.tsx, apps/tldw-frontend/hooks/__tests__/useConfig.networking.test.tsx, apps/packages/ui/vitest.setup.ts.

Behavior: ConfigProvider now reads canonical tldwConfig plus legacy keys, keeps env API credentials ahead of browser storage, persists user-saved single-user keys to canonical tldwConfig and legacy apiKey, listens for tldw:config-updated from the shared settings client, and avoids creating canonical tldwConfig on an otherwise unconfigured first mount. Test setup now replaces partial runner localStorage with a Storage-compatible shim.

Verification: bun run test:run hooks/__tests__/useConfig.networking.test.tsx lib/__tests__/api.credentials.test.ts passed (17 tests). bun run test:run ../packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts ../packages/ui/src/services/tldw/__tests__/request-core.quickstart.test.ts ../packages/ui/src/components/Option/Settings/__tests__/tldw-connection-status.test.ts passed (6 tests). Direct eslint on touched files passed with 0 errors; it reported existing any-cast warnings in apps/packages/ui/vitest.setup.ts and a Next pages-directory notice from running at apps root. Bandit was run from the repo venv path against touched TS files; Bandit cannot parse TS/TSX and reported syntax parse errors with 0 findings. Vitest still emits the pre-existing Bun --localstorage-file warning.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed WebUI single-user auth persistence for Stage 1. Saved browser credentials now survive reload through the shared tldwConfig path, settings update events refresh the live Next.js config, request helpers continue to receive compatible legacy keys, and environment credentials keep priority over stale browser storage. Focused auth/config tests and nearby shared-client checks pass; lint has no errors; Bandit is not applicable to the touched TypeScript files.
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
