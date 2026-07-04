---
id: TASK-12127
title: Fix WebUI auth persistence after runtime credential scrub
status: Done
priority: High
references:
- https://github.com/rmusser01/tldw_server/issues/2590
modified_files:
- apps/tldw-frontend/extension/shims/runtime-bootstrap.ts
- apps/tldw-frontend/__tests__/extension/runtime-bootstrap.test.ts
- apps/tldw-frontend/pages/_app.tsx
- apps/tldw-frontend/__tests__/app/app-layout.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix GitHub issue #2590: manually entered single-user WebUI auth is stripped from persisted tldwConfig and lost on the second hard reload when no runtime-config or build-time auth re-supplies it. Product decision: users should stay authenticated across hard reloads without re-entering the key.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Manually entered single-user API key remains available to runtime auth after a second hard reload.
- [x] #2 Persisted tldwConfig does not regain clear-text apiKey/accessToken fields.
- [x] #3 Runtime-config and build-time auth precedence remains unchanged.
- [x] #4 Focused frontend regression tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added a sessionStorage-backed runtime auth bridge for manually entered single-user API keys. The bootstrap captures a valid manual key before scrubbing tldwConfig, keeps it only for the current browser session, and rehydrates the runtime override on later hard reloads when no runtime-config or build-time auth is present. The session fallback also covers blank or placeholder apiKey values persisted by Settings, sessionStorage access failures now log sanitized warnings, and the derivation logic is isolated in a helper. The shell auth gate treats runtime auth material as authenticated so header/sidebar chrome remains visible after the scrub. Added coverage for clearing the session key when stored auth switches away from single-user mode.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed issue #2590 for the confirmed same-browser-session scope. Manual single-user auth survives a second hard reload through runtime session auth while tldwConfig remains scrubbed, including when Settings persists a blank single-user apiKey. Added regression coverage for the second-reload runtime bootstrap path, blank-key session rehydrate path, mode-switch session clear path, and WebUI shell gate. Verification: bunx vitest run __tests__/extension/runtime-bootstrap.test.ts __tests__/app/app-layout.test.tsx --config vitest.config.ts (45 tests passed); bunx eslint extension/shims/runtime-bootstrap.ts __tests__/extension/runtime-bootstrap.test.ts pages/_app.tsx __tests__/app/app-layout.test.tsx (0 errors, 1 pre-existing no-explicit-any warning); git diff --check passed. Full bun run typecheck currently fails in unrelated e2e files: e2e/fixtures/knowledge-qa-live.ts and e2e/workflows/tier-2-features/flashcards.spec.ts. Bandit skipped because touched code is frontend TypeScript, not Python.
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
