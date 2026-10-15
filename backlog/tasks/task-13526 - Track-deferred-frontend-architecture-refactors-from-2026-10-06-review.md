id: TASK-13526
title: Track deferred frontend architecture refactors from 2026-10-06 review
status: In Progress
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
11. Split tiny URL predicates from the heavy parse functions in `parser/{twitter,amazon,wiki}.ts` so consumers needn't duplicate host-regex guards around lazy imports (from the TASK-13511 implementation).
12. Add a force-bypass for the 60s `getProvidersStatus` TTL cache after provider-settings saves.
13. Fix the `@tldw/ui` package `exports` map subpaths — bun-run extension unit tests cannot resolve `@tldw/ui/...` imports today (pre-existing; 3 resolveApiPath test failures).
Status notes (2026-10-13 pass): item 3 (dependency-major alignment) is ALREADY RESOLVED on dev by sibling program work — react 18.3.1, marked ^17, property-information ^7, zustand ^5 all match across shells and the ui peer ranges. Item 13 (exports map): the map served nothing (css entry unused — imported relatively; state entry covered by tsconfig paths), so it was removed, and the extension's WXT tsconfig.json (previously an untracked local file) is now tracked so fresh checkouts resolve the `@`/`~`/`@tldw/ui` aliases. Remaining blocker for the 3 resolveApiPath bun tests is NOT the exports map: under `bun test`, `wxt/browser` imported transitively through packages/ui resolves to wxt's browser.d.ts (types condition) even though plain `bun run` from apps/extension resolves browser.mjs correctly; needs a wxt/browser shim module for non-extension contexts or a bun resolution fix (3-attempt rule applied; probes documented in the PR).

NOTE: created manually because the backlog CLI installation was removed from this machine mid-session (binary and node_modules/backlog.md both vanished; `backlog task create` had also been crashing with "Maximum call stack size exceeded" before removal). Per AGENTS.md exception path.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR check (2026-10-09, per .agents/skills/adr-assessment): ADR required: yes — Docs/ADR/065-frontend-api-contract-package.md (Proposed). Reason: establishes the durable rule that one neutral workspace package (@tldw/api-types) owns the frontend API contract artifacts and no second contract source may be introduced.

Stage 1 started 2026-10-09 (branch codex/api-client-unification-13526, plan Docs/Plans/2026-10-09-api-contract-package-13526-implementation-plan.md): shared api-types package, web-shell migration, ClientPath derivation. Deconfliction note: sibling WebUI perf program TASK-13520-13525 (W0-W5) covers perf mechanics only; W1 additionally lists ChatPane virtualization which is therefore left to that program. Overlap warning for that program: W1/W2/W4/W5 scopes partially shipped already by TASK-13511 (PR #3210).
Stage 1 MERGED 2026-10-11 via PR #3219 (merge commit 590e7e579): @tldw/api-types package shipped with committed paths.d.ts (2,113 paths) + derived ClientPath; carried fixes for the blanket *.json gitignore, Dockerfile.webui workspace enumeration, and the pre-existing TASK-13510 mypy breakage in admin_audit_service.py. Stage 2 (error model) MERGED 2026-10-11 via PR #3232 (merge commit c65df167b9): canonical ApiError lives in packages/ui api-error.ts, TldwApiError is a strict subclass, web re-exports the canonical class; the web transport delegation was deliberately deferred after a seam analysis: tldwRequest returns a parsed envelope while web policy needs the raw Response, and the two auth pipelines draw from different credential domains (tldwConfig vs web authStorage) — full delegation needs resolveTldwRequestPlan extraction, a credential-strategy interface in TldwRequestRuntime, and a raw-response mode with onResponseHeaders/buildError hooks. Stage 3 (TldwApiClient domain split) remains.
Transport Stage A shipped 2026-10-14 (branch codex/web-transport-delegation-13526, plan Docs/Plans/2026-10-14-web-transport-delegation-stage-a-13526-implementation-plan.md): web client adopts the shared timeout policy (deriveRequestTimeout seeded with the web 30s default so generic paths keep today's behavior while per-path floors — slides 120s, model-metadata 60s — apply as in the extension) and the shared Retry-After parser (adds HTTP-date support).
Transport Stage B1 shipped 2026-10-14 (branch codex/request-core-plan-extraction-13526): resolveTldwRequestPlan extracted from performTldwRequest (pure pre-flight: validation/transport/URL/guards/content-type/shouldSkipAuth — zero behavior change, extension suites untouched-green) and TldwRequestRuntime gained the optional Stage-B credential strategy (resolveCredentials) that replaces default auth resolution when provided (default inert). Also realigned the server-capabilities voice-conversion guard with the Stage-1 derived union (it still asserted the deleted hand-maintained literal list — tolerated until now via the CI failure-ratchet baseline). Remaining for B2: raw-response mode + onResponseHeaders/buildError hooks + web adoption of the plan/credential strategy.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
