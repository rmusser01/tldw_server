---
id: TASK-122
title: Replace WebUI axios with fetch helpers for issue 1346
status: Done
assignee:
  - '@codex'
created_date: '2026-05-08 03:19'
updated_date: '2026-05-23 16:32'
labels:
  - webui
  - dependencies
  - cleanup
dependencies:
  - TASK-104
  - TASK-117
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
  - 'https://github.com/rmusser01/tldw_server/pull/1375'
documentation:
  - Docs/Design/WebUI_Dependency_Audit.md
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-webui-axios-fetch-replacement-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the axios replacement slice from the approved WebUI dependency trimming sequence. Replace first-party WebUI axios usage with a fetch-backed client while preserving auth, CSRF, base URL mutation, credentials, timeout, request history, redirects, retry-after mapping, and FormData behavior. Replace the shared UI ElevenLabs axios usage with a local external-origin fetch helper. Remove direct axios declarations only after compatibility tests pass and the Bun lockfile is updated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The first-party apiClient methods and default api baseURL mutation compatibility are preserved without importing axios.
- [x] #2 Fetch-backed request handling preserves auth headers, CSRF injection, browser credentials, per-request headers, withCredentials overrides, signals, timeout behavior, FormData handling, request history, 401 redirect logic, 403 CSRF normalization, and retry-after mapping.
- [x] #3 The ElevenLabs service uses fetch for voices, models, and speech ArrayBuffer responses while preserving API-key headers and timeout semantics.
- [x] #4 Direct axios declarations are removed from the audited WebUI/shared UI/extension manifests when no direct import/type usage remains, and apps/bun.lock is updated consistently.
- [x] #5 Focused tests cover the compatibility surface and fail before implementation where practical; lint, typecheck, compile/build, changed tests, git diff --check, and Bandit rationale are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write focused compatibility tests for the WebUI fetch-backed client and ElevenLabs fetch helper. 2. Implement the WebUI fetch client with axios-compatible public methods and baseURL defaults. 3. Replace exported axios-derived types with local request config and metadata types. 4. Replace ElevenLabs axios calls with a local fetch helper. 5. Remove axios declarations from package manifests, regenerate apps/bun.lock, run verification, update this task, commit, push, and open a PR against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan for axios replacement in Docs/superpowers/plans/2026-05-08-webui-axios-fetch-replacement-implementation-plan.md. Scope is first-party WebUI fetch client, local request config types, ElevenLabs external-origin fetch helper, manifest/lockfile axios removal, and focused compatibility verification.

Implemented fetch-backed first-party WebUI API client and local request config types; preserved baseURL defaults mutation, auth/API key/session/CSRF headers, credentials overrides, timeout/signal handling, response parsing, request history, retry-after mapping, and 401/403 normalization.

Replaced ElevenLabs service axios usage with fetch helpers for voices, models, speech ArrayBuffer responses, xi-api-key headers, JSON payloads, status errors, and timeout errors.

Removed direct axios declarations from apps/tldw-frontend/package.json, apps/packages/ui/package.json, and apps/extension/package.json; apps/bun.lock still contains axios only as optional peer of @vueuse/integrations via vitepress per bun pm why axios.

Verification: bun install --frozen-lockfile passed; direct axios import/type/declaration guards returned no matches; git diff --check passed; focused Vitest passed for api-client.fetch, api.credentials, useConfig.networking, and ElevenLabs tests; bun run lint passed with existing 127-warning baseline; bunx tsc --noEmit -p tsconfig.json --pretty false passed; NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile passed; extension bun run compile passed.

Changed-test sweep note: NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bunx vitest run --changed=origin/dev failed 82 tests across unrelated UI suites. Representative failures reproduced with this patch stashed on clean dev ancestry: ReviewTab.queue-state.test.tsx fails on RecentStudySessions undefined query data, and FlashcardsWorkspace.connection-state.test.tsx fails on missing old demo heading text. Treated as baseline failures, not introduced by this slice.

Bandit skipped because this slice touches TypeScript, package manifests, lockfile, Backlog task metadata, and plan documentation only; no Python files changed.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1375.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced direct WebUI/shared UI axios usage with fetch-backed helpers. The first-party API client now preserves auth, CSRF, session, credentials, timeout/signal, response parsing, request history, retry-after, and baseURL mutation behavior without importing axios, while ElevenLabs now uses a small external-origin fetch helper for voices, models, and speech ArrayBuffer responses. Removed direct axios declarations from the audited WebUI, shared UI, and extension manifests and regenerated apps/bun.lock. Focused compatibility tests, lint, typecheck, frontend compile, extension compile, frozen install, direct axios guards, and git diff checks were recorded; the broad changed-test sweep still hits unrelated dev-baseline UI test failures that reproduce with this patch stashed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Closeout Notes

<!-- SECTION:CLOSEOUT:BEGIN -->
Closed after verifying PR #1375 merged into `dev` on 2026-05-09 at merge commit `95eb67138716a63fc0b1b99dfd97611f0806da32`. The task already recorded completed acceptance criteria, completed Definition of Done items, implementation notes, verification evidence, and final summary; this closeout only corrects the stale Backlog status.
<!-- SECTION:CLOSEOUT:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
