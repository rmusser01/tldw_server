---
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
10. RESOLVED 2026-10-19: packages/voice-assistant-sdk deleted (14 tracked files, zero importers verified across apps/ config/source/Dockerfile; workspace entry removed from apps/package.json).
11. Split tiny URL predicates from the heavy parse functions in `parser/{twitter,amazon,wiki}.ts` so consumers needn't duplicate host-regex guards around lazy imports (from the TASK-13511 implementation).
12. Add a force-bypass for the 60s `getProvidersStatus` TTL cache after provider-settings saves.
13. Fix the `@tldw/ui` package `exports` map subpaths — bun-run extension unit tests cannot resolve `@tldw/ui/...` imports today (pre-existing; 3 resolveApiPath test failures).
Status notes (2026-10-13 pass): item 3 (dependency-major alignment) is ALREADY RESOLVED on dev by sibling program work — react 18.3.1, marked ^17, property-information ^7, zustand ^5 all match across shells and the ui peer ranges. Item 13 (exports map): the map served nothing (css entry unused — imported relatively; state entry covered by tsconfig paths), so it was removed, and the extension's WXT tsconfig.json (previously an untracked local file) is now tracked so fresh checkouts resolve the `@`/`~`/`@tldw/ui` aliases. Remaining blocker for the 3 resolveApiPath bun tests is NOT the exports map: under `bun test`, `wxt/browser` imported transitively through packages/ui resolves to wxt's browser.d.ts (types condition) even though plain `bun run` from apps/extension resolves browser.mjs correctly; needs a wxt/browser shim module for non-extension contexts or a bun resolution fix (3-attempt rule applied; probes documented in the PR).

NOTE: created manually because the backlog CLI installation was removed from this machine mid-session (binary and node_modules/backlog.md both vanished; `backlog task create` had also been crashing with "Maximum call stack size exceeded" before removal). Per AGENTS.md exception path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each refactor above is either implemented or explicitly descoped with rationale on this record
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR check (2026-10-09, per .agents/skills/adr-assessment): ADR required: yes — Docs/ADR/069-frontend-api-contract-package.md (Proposed). Reason: establishes the durable rule that one neutral workspace package (@tldw/api-types) owns the frontend API contract artifacts and no second contract source may be introduced.

Stage 1 started 2026-10-09 (branch codex/api-client-unification-13526, plan Docs/Plans/2026-10-09-api-contract-package-13526-implementation-plan.md): shared api-types package, web-shell migration, ClientPath derivation. Deconfliction note: sibling WebUI perf program TASK-13520-13525 (W0-W5) covers perf mechanics only; W1 additionally lists ChatPane virtualization which is therefore left to that program. Overlap warning for that program: W1/W2/W4/W5 scopes partially shipped already by TASK-13511 (PR #3210).
Stage 1 MERGED 2026-10-11 via PR #3219 (merge commit 590e7e579): @tldw/api-types package shipped with committed paths.d.ts (2,113 paths) + derived ClientPath; carried fixes for the blanket *.json gitignore, Dockerfile.webui workspace enumeration, and the pre-existing TASK-13510 mypy breakage in admin_audit_service.py. Stage 2 (error model) MERGED 2026-10-11 via PR #3232 (merge commit c65df167b9): canonical ApiError lives in packages/ui api-error.ts, TldwApiError is a strict subclass, web re-exports the canonical class; the web transport delegation was deliberately deferred after a seam analysis: tldwRequest returns a parsed envelope while web policy needs the raw Response, and the two auth pipelines draw from different credential domains (tldwConfig vs web authStorage) — full delegation needs resolveTldwRequestPlan extraction, a credential-strategy interface in TldwRequestRuntime, and a raw-response mode with onResponseHeaders/buildError hooks.  Stage 3 investigation 2026-10-19 (branch codex/tldwclient-domain-split-13526, reverted before PR): the composition root already exists (13 domain mixins applied via Object.assign; the review's "split into existing domain modules" is largely DONE). The god-file's remaining bulk is ~3.3k lines of BASE methods whose names are shadowed by domain mixin methods (191 methods measured with an object-scoped detector) — deleting them wholesale broke 28 tests (timeouts, not missing symbols), so the base/domain override relationships are not clean prototype shadowing for every pair: the declaration-merged interface types the BASE signatures while runtime binds the DOMAIN implementations, and some pairs evidently differ in behavior/contract. Safe path for a future slice: per-method deletion behind the full TldwApiClient suite (176 tests green baseline verified), starting with the unambiguous large spans (listCharactersPage 188L, exportDataTable 182L, createCharacter 164L, listChatMessages 162L), plus a prototype-ownership guard test asserting `TldwApiClient.prototype.<name> === <domain>.<name>` for each claimed override. The connection.tsx dynamic-import workaround needs a narrow config-service extraction (getConfig/updateConfig/initialize out of the client graph) — splitting alone cannot kill it because the composition root statically imports every domain.
Stage 3 EXECUTED 2026-10-19 (branch codex/13526-remainder-burndown): the shadowed-base deletion landed safely — 193 dead base methods removed (-3,250 lines; 8,888 -> 5,649) using RUNTIME mixin keys (Object.keys of each domain mixin via a vitest dump — regex parsing had produced false positives) plus a new prototype-ownership guard test pinning every domain method to its prototype slot (with the documented createChatCompletion carve-out). Bisect identified getCharacter as the one name whose base copy is NOT safely shadowed (28 timeout failures when deleted; excluded, root cause for the follow-up). Three cross-domain helper names (normalizeCharacterListResponse, getCharacterListIdentity, normalizeChatSummary) are re-exposed through a small TldwApiClientCharacterHelpers interface contribution; two call sites (ExportMenu exportDataTable arity, adminQueries getSystemStats options) were updated to the true domain signatures that base copies had masked. Full battery 242/242, web tsc clean, extension compile clean.
Remainder disposition (2026-10-19 burndown): config-service extraction (the connection.tsx workaround killer) is DEFERRED — it needs the config lifecycle (getConfig/updateConfig/initialize, 400+ lines incl. the ensureConfigForRequest machinery at TldwApiClient.ts:1992) moved to a narrow module without breaking the auth-commit/config-guidance suites; too surgical to bolt onto this PR. vitest-config consolidation and font/locale subsetting remain open (infra/asset scale, each a standalone PR). The wxt/browser bun-test quirk remains documented with probes (needs a non-extension shim or upstream fix).
Transport Stage A shipped 2026-10-14 (branch codex/web-transport-delegation-13526, plan Docs/Plans/2026-10-14-web-transport-delegation-stage-a-13526-implementation-plan.md): web client adopts the shared timeout policy (deriveRequestTimeout seeded with the web 30s default so generic paths keep today's behavior while per-path floors — slides 120s, model-metadata 60s — apply as in the extension) and the shared Retry-After parser (adds HTTP-date support).
Current chat integration TASK13586 renumbers only this proposed API decision to free ADR069 after local/dev/open-PR scan; accepted independent Notes ADR065 remains byte-identical. API policy rationale and status remain Proposed.
Transport Stage B1 shipped 2026-10-14 (branch codex/request-core-plan-extraction-13526): resolveTldwRequestPlan extracted from performTldwRequest (pre-flight validation/transport/URL/guards/content-type/shouldSkipAuth) and TldwRequestRuntime gained the optional credential strategy resolveCredentials; default behavior retained. Server-capabilities guard remains aligned with derived ClientPath and canonical generated paths using its existing shell-independent path. Remaining B2: raw-response mode, onResponseHeaders/buildError hooks and web adoption. Governing API-contract ADR069 remains Proposed; accepted independent Notes ADR065 unchanged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
