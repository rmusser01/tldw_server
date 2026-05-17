---
id: TASK-418.11
title: Implement WebUI capability and error states
status: Done
labels:
- ux
- webui
- extension
- implementation
- states
priority: high
parent_task_id: TASK-418
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first WP2 capability/error-state remediation slice for the WebUI/extension. Scope: shared user-language capability states and first route adopters /sources, /scheduled-tasks, and /integrations. Preserve existing tables/forms/dense controls, keep raw endpoint/status details behind diagnostics, and do not change backend APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared state primitive expectations are locked with focused tests or documented as already covered.
- [x] #2 A pure capability-state mapping helper is added only if two or more first-adopter routes would otherwise duplicate mapping logic.
- [x] #3 /sources top-level unavailable/error/empty states use shared user-language state UI, with raw endpoint details only in diagnostics.
- [x] #4 /scheduled-tasks top-level unavailable/error/degraded states use shared user-language state UI, with raw endpoint details only in diagnostics.
- [x] #5 /integrations top-level unsupported/error states use shared user-language state UI, with provider-card details left scoped to cards unless they leak raw route state.
- [x] #6 Focused Vitest route/component tests pass for changed state primitives and first adopters.
- [x] #7 Browser QA or Playwright evidence is recorded for /sources, /scheduled-tasks, and /integrations, with any environment gaps documented.
- [x] #8 Later route-family adopters are listed in the task notes instead of silently skipped.
- [x] #9 No backend API changes or broad visual redesign are included.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Baseline: created clean worktree from `origin/dev` at `a1d24c7f4` after PR #1830 merge. Main checkout remains dirty and unrelated. The committed WP2 plan/spec are present; the referenced audit markdown is missing from the clean `origin/dev` worktree but exists as an untracked file in the main checkout, so this implementation branch treats the committed spec and child plan as authoritative and does not copy unrelated untracked audit artifacts into this PR.

Shared state foundation: `state-primitives.test.tsx` passed at baseline (7 tests), so no primitive production change was needed. Added a focused locking test proving raw endpoint details can live in diagnostics while primary copy stays user-language. Added pure `capability-state.ts` after a red test failed because the helper did not exist. Helper maps common capability failures to existing design-system state keys and builds diagnostics for method/endpoint/status/server/raw message. Verification: `bunx vitest run src/components/ui/state/__tests__/capability-state.test.ts src/components/ui/state/__tests__/state-primitives.test.tsx` passed 12 tests.

/sources adoption: replaced the unsupported capability guard, query error state, and empty state with shared `StatePanel` states generated through `buildCapabilityState`. Raw `GET /api/v1/sources`, status, and raw messages now render only inside the diagnostics region; the primary page copy uses user-language unavailable/empty messaging and gives retry/create/server-health actions. Updated source workspace and route-guard tests, including the router/connection-state test harness. Verification: `bunx vitest run src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx` passed 11 tests.

/scheduled-tasks adoption: replaced top-level unsupported endpoint, list-query failure, and partial-overview warnings with shared `StatePanel` states. Primary copy now says unavailable/degraded in user terms; `GET /api/v1/scheduled-tasks`, status codes, server URL, and partial raw messages are diagnostics. The existing reminder table/editor workflow remains unchanged. Verification: `bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx` passed 13 tests.

/integrations adoption: replaced top-level personal unsupported and overview-load failures with shared `StatePanel` states. Personal/workspace integration endpoint paths, status codes, server URL, and raw overview errors are diagnostics; provider cards, drawers, and workspace policy panels remain in their existing scoped UI. The overview query now surfaces the shared error state immediately instead of retrying the same route-level failure three times; the retry path is explicit through the state action. Verification: `bunx vitest run src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx src/routes/__tests__/integrations-route.test.tsx` passed 15 tests.

Design-system guard follow-up: `bun run verify:design-system-state` initially found one remaining touched-file AntD product-state import in `IntegrationManagementPage.tsx` for the Telegram linked actors warning. Added coverage and migrated that warning to a degraded shared state with diagnostics, then removed stale baseline entries for the migrated `/sources`, `/scheduled-tasks`, and `/integrations` AntD states. Verification: targeted Telegram linked-actor test passed; `bun run verify:design-system-state` passed with only pre-existing allowed legacy exceptions.

Combined verification: `bunx vitest run src/components/ui/state/__tests__/capability-state.test.ts src/components/ui/state/__tests__/state-primitives.test.tsx src/components/Option/Sources/__tests__/SourcesWorkspacePage.test.tsx src/routes/__tests__/option-sources-route-guards.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/routes/__tests__/scheduled-tasks-route.test.tsx src/components/Option/Integrations/__tests__/IntegrationManagementPage.test.tsx src/routes/__tests__/integrations-route.test.tsx` passed 8 files / 52 tests. `bun run verify:design-system-state` passed after stale migrated baseline entries were removed. `git diff --check` passed after the final Backlog update. `bunx tsc --noEmit -p tsconfig.json` was attempted and failed on pre-existing package-wide TypeScript errors outside this slice, including evaluations recipe config sample typing, Workspace Studio capability key narrowing, and shortcut config persistence typing.

Browser QA evidence: dev server was started with `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -H 127.0.0.1 -p 18002`. Playwright opened `http://127.0.0.1:18002/sources`, `http://127.0.0.1:18002/scheduled-tasks`, and `http://127.0.0.1:18002/integrations`. All three snapshots were blocked at the WebUI readiness/auth state (`Checking server readiness`, disabled `Waiting` button) because the local API on `127.0.0.1:8000` returned 401 for `/openapi.json`, notifications, and `/api/v1/ingestion-sources/capabilities`. Snapshot evidence: `.playwright-cli/page-2026-05-17T23-21-06-988Z.yml`, `.playwright-cli/page-2026-05-17T23-28-25-790Z.yml`, and `.playwright-cli/page-2026-05-17T23-28-36-573Z.yml`. The changed page states are therefore verified through DOM/component tests rather than live authenticated browser observation in this environment.

Later route-family adopters intentionally left for follow-up slices: `/admin`, `/agents`, `/agent-tasks`, `/acp-playground`, `/settings/model`, `/evaluations`, `/mcp-hub`, `/skills`, `/tts`, `/speech`, and `/data-tables`. This slice did not change backend APIs, route names, navigation structure, tables/forms, or broad visual design.

Security verification: Bandit was not run because the touched implementation scope is TypeScript/React UI, JSON baseline metadata, and this Backlog task. No Python backend code or security-sensitive server path was changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first WP2 capability/error-state remediation slice for `/sources`, `/scheduled-tasks`, and `/integrations`. The branch adds a shared capability-state mapper, adopts the existing design-system `StatePanel` for top-level unavailable/error/degraded/empty states, keeps raw endpoint/status details inside diagnostics, and removes stale design-system baseline exceptions for the migrated pages. Browser QA was attempted for all three target routes but the local authenticated API blocked route entry with 401 responses, so live page-state observation remains an environment gap; focused DOM/component tests cover the changed state behavior.
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
