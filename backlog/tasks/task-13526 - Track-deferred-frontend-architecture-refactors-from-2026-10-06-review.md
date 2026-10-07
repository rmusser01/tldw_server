id: TASK-13526
title: Track deferred frontend architecture refactors from 2026-10-06 review
status: To Do
labels:
- frontend
- architecture
- tech-debt
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Umbrella follow-up for the architecture findings from the 2026-10-06 frontend performance/architecture review (TASK-13511 implemented the tractable performance fixes; these are the larger structural refactors that were deliberately deferred out of that PR to keep it reviewable).

Priority order (highest leverage first, per the review):
1. Unify the two API clients: web shell `lib/api.ts` (807 lines, ApiError) vs shared `TldwApiClient.ts` (8,707 lines, TldwApiError) + `background-proxy.ts`; every timeout/retry/cancellation fix currently lands twice.
2. Move OpenAPI codegen to a workspace package consumed by both clients; the shared client runs on ~8k hand-copied Pydantic types + a manually maintained ClientPath union (opt-in `verify:openapi` only). Web shell already has `generate-api-types.mjs` + fingerprint drift gate — generalize it.
3. Align dependency majors across shells: zustand 5 (web) vs 4.5 (extension), react 18.3.1 vs 18.2.0, marked 17 vs 15 (web violates the package's own peer range). Add a CI peer-range check.
4. Consolidate the three vitest configs that each run the same 2,230 packages/ui test files; the alias resolution map is duplicated across ~6 files.
5. Split `packages/ui` (~900k source lines) along existing seams: `@tldw/api-client` (services/tldw), `@tldw/state` (store), `@tldw/extension-entries` (entries/); enforce with a boundary lint. Includes moving pure logic out of components/ to fix services/stores -> components layering inversions.
6. Dedupe auth (lib/auth.ts vs TldwAuth.ts, both actively patched) and UI primitives (web components/ui vs package primitives/design-system).
7. Split store monoliths (store/workspace.ts 4,065 lines; connection.tsx dynamic-import workaround for the god client) — largely resolved by (1).
8. Asset work: subset 2.4MB TTF fonts to WOFF2 (both shells), slim `_locales` (11MB domain JSONs), narrow pdf.worker web_accessible_resources match.
9. ResearchWorkspace ChatPane virtualization (PlaygroundChat already virtualized via VirtualChatTimeline); extract composer draft into a leaf component (R6); merge ResearchWorkspace dual 5s pollers.
10. Decide fate of packages/voice-assistant-sdk (zero importers) — needs maintainer decision (roadmap?).

NOTE: created manually because the backlog CLI installation was removed from this machine mid-session (binary and node_modules/backlog.md both vanished; `backlog task create` had also been crashing with "Maximum call stack size exceeded" before removal). Per AGENTS.md exception path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each refactor above is either implemented or explicitly descoped with rationale on this record
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
