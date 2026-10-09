# Implementation Plan: Frontend Performance & Architecture Remediation (2026-10-06)

Branch: `codex/frontend-perf-arch-remediation-20261006` (off `origin/dev` @ 7ba48f251e)
Worktree: `.worktrees/frontend-perf-arch-remediation`
Backlog: umbrella task (TASK id TBD — update when created)
Source review: 2026-10-06 four-reviewer audit of `apps/` (WebUI + WXT extension + `packages/ui`). Performance/architecture scope; bug/security findings were covered by `apps/FRONTEND_AUDIT.md` (2026-07-02) and are out of scope here.

NOTE for implementers: file:line references below came from the review pass; verify by symbol/search when editing — do not trust line numbers blindly. All paths are relative to `apps/` unless prefixed.

## Stage 1: Extension content-script weight (E1, E2)
**Goal**: Stop taxing every browsed page and the eager panel bundle.
**Changes**:
1. `extension/entrypoints/copilot-popup.content.tsx` + `packages/ui/src/entries/copilot-popup.content.tsx`: run top-frame only (drop `all_frames: true` from the content-script definition). Verify the background's frame targeting (`info.frameId`) still works.
2. Make the heavy module a real split chunk instead of an inlined "dynamic" import: ensure `import()` in the entry wrapper produces a separate chunk (WXT supports split content-script chunks via `web_accessible_resources`); verify in built output that `content-scripts/copilot-popup.js` no longer contains `TldwApiClient`. If WXT's build inlines it regardless, restructure so the every-page script is a minimal stub (message listener only) that `import()`s the popup logic from a `web_accessible_resources` chunk on first message.
3. `packages/ui/src/parser/default.ts` consumers: convert static `import ... from "@/parser/default"` to `await import()` at point of use in `services/web-clipper/content-extract.ts`, `libs/get-html.ts`, `libs/get-tab-contents.ts` (and their importers `hooks/useMessage.tsx`, `hooks/chat-modes/tabChatMode.ts`) so cheerio/Readability/Turndown leave the every-page web-clipper script and the eager sidepanel chunk.
**Success Criteria**: built `content-scripts/copilot-popup.js` and `web-clipper.js` contain no cheerio marker and are drastically smaller; sidepanel eager closure no longer includes cheerio; manifest has no `all_frames: true` for copilot-popup.
**Tests**: existing extension unit tests + new vitest asserting manifest content-script definition (matches, allFrames) and lazy-loading behavior; `bun run build:chrome` + artifact size comparison recorded in task notes.
**Status**: Complete

## Stage 2: MV3 background efficiency (E3, E5, alarm, storage batching)
**Goal**: Cold start stops fan-out; ingest polling stops N+1.
**Changes**:
1. `packages/ui/src/entries/shared/background-init.ts`: capabilities fetch on cold start uses persisted cache (no `forceRefresh: true`); `warmModels` no longer passes forced refresh (drop `refresh_openrouter=true` from cold start + hourly alarm — keep an explicit user-triggered forced path); persist the models force-cooldown timestamp so MV3 suspension cannot reset it (`services/tldw/TldwModels.ts`).
2. OpenAPI drift check (`background-init.ts:201` area): raise TTL (>= 24h) or trigger only on install/config-change; do not download full `/openapi.json` on routine wakes.
3. `tldw:model-warm` alarm: `periodInMinutes: 60` → `360`, handler no-ops when catalog is fresh.
4. `packages/ui/src/db/models.ts`: replace `get(null)` full-storage read with prefix-scoped reads (or single-key catalog blob); batch per-model `chrome.storage.local.set` writes into one set.
5. `packages/ui/src/services/tldw/ingest-jobs-orchestrator.ts`: poll the batch endpoint (`GET /api/v1/media/ingest/jobs`) once per cycle instead of one sequential request per job.
**Success Criteria**: simulated cold start issues ≤1 capability/model request served from cache; a 3-job ingest poll cycle issues exactly 1 HTTP request; alarm period 360 in installed manifest.
**Tests**: unit tests for orchestrator (existing) updated/extended to assert single batch call; new tests for models-db batching and cooldown persistence; background-init unit tests for gating.
**Status**: Complete

## Stage 3: Sidepanel streaming + small shared fixes (E4, R8, R9, R10)
**Goal**: Sidepanel chat stops per-token full-list setState; cheap wins land.
**Changes**:
1. `packages/ui/src/hooks/useMessage.tsx`: all 4 streaming chunk loops (≈ lines 659, 685-696, 1044, 1632, 2214) adopt the 80ms trailing-flush scheduler already used by `hooks/chat/useChatActions.ts` (`STREAMING_UPDATE_INTERVAL_MS = 80`, flush replaces only the streaming message's entry; final flush on stream end so no content is lost).
2. `packages/ui/src/components/Common/Markdown.tsx`: hoist `remarkPlugins`/`rehypePlugins` arrays and the `components` map to module scope (pass theme/settings via module-level getter or context), removing per-render closure rebuilds.
3. Precompute `previousUserMessageByIndex` map (single backward pass, `useMemo`) in `components/Sidepanel/Chat/body.tsx` and `components/Option/Playground/PlaygroundChat.tsx` instead of O(n) `getPreviousUserMessage(index)` per row per render.
4. Debounce keyword-suggestion search (~300ms via existing `useDebouncedValue`/`useDebounce` pattern) in `components/Review/MediaReviewFilterSidebar.tsx`, `ReviewPage.tsx`, `ViewMediaPage.tsx`; ignore stale responses.
**Success Criteria**: streaming 200 chunks into a 50-message sidepanel conversation issues ≤ ceil(duration/80ms) setState calls; markdown render does not allocate new plugin arrays per render; keyword typing issues ≤1 request per 300ms.
**Tests**: new unit test asserting throttled flush cadence for useMessage streaming (fake timers); existing Markdown/Review tests pass.
**Status**: Complete

## Stage 4: Web app load path (W1, W4, W2-partial)
**Goal**: Kill the serial auth→health render gate and per-mount refetch storms.
**Changes**:
1. `apps/tldw-frontend/pages/_app.tsx` + `components/networking/ServerReadinessGate.tsx`: start the `/health` check in parallel with `auth/me` resolution (single gate phase), and render children behind the gate with a non-blocking banner on failure instead of a full-screen spinner — EXCEPT keep blocking behavior on cold first-visit when no auth state exists (preserve existing UX for "server down at startup").
2. `packages/ui` Playground mount (`Option/Playground/Playground.tsx:462-475`): drop `forceRefresh: true` from `fetchChatModels` on mount.
3. `services/tldw/TldwApiClient.ts#getProvidersStatus`: add module-level cache (TTL ~60s, in-flight dedup — reuse `apiSend` pattern) since 12+ components call it independently.
4. CSS co-location: move `react-pdf.css` import out of `pages/_app.tsx` into the PDF viewer components; move `katex.min.css` out of the shared path into the Markdown/lazy consumers if it can ride the async chunk CSS.
**Success Criteria**: authenticated reload of `/chat` performs 0 blocking sequential pre-content round trips beyond auth/me; second mount within TTL performs 0 `/llm/models` and 0 `/config/providers` requests; `_app` CSS shrinks (verify via `check-bundle-budget`).
**Tests**: update/extend `_app`/readiness-gate vitest tests for parallel gate; new cache test for providers status.
**Status**: Complete

## Stage 5: Chat runtime (R1-partial, R2, R3, R4, R5)
**Goal**: Streaming re-render cost becomes O(visible rows), not O(conversation).
**Changes**:
1. R1: `packages/ui/src/hooks/useMessageOption.tsx` + `useMessage.tsx`: use zustand shallow/atomic selectors internally (pattern: `hooks/chat/useChatBaseState.ts:67`) so the hook only re-renders consumers when slices they read change; convert single-field consumers (`Common/ModelSelectOption.tsx:19`, `Common/PromptSearch.tsx:31`, `Common/Settings/ActorPopout.tsx:28`, `Common/ChatSidebar/LocalChatList.tsx:122`, `Option/Settings/system-settings.tsx:35`) to granular `useStoreMessageOption(s => s.field)` selectors.
2. R2: `React.memo` `components/Common/Playground/Message.tsx` (`PlaygroundMessage`) with a custom comparator (message identity + streaming/processing flags + relevant id-level props); in `Option/Playground/PlaygroundChat.tsx` and `Sidepanel/Chat/body.tsx` hoist per-row callbacks to stable `useCallback`s (dispatch-by-id map) and memoize per-row derived objects (`buildMessageResearchActions`).
3. R3: virtualize the web chat message list in `PlaygroundChat.tsx` using the existing `@tanstack/react-virtual` pattern from `Sidepanel/Chat/body.tsx:136` (dynamic row heights via `measureElement`, overscan ~3-5); keep `useSmartScroll` behavior working (scroll-to-bottom during streaming). ResearchWorkspace ChatPane virtualization is follow-up if this stage risks ballooning.
4. R4: `components/Option/KnowledgeQA/KnowledgeQAProvider.tsx:2503-2536`: throttle delta dispatches with the 80ms trailing-flush scheduler; move `parseCitations`/trust normalization into the flush.
5. R5: `services/tldw/chat-request-debug.ts`: clone lazily (on first read) or gate behind debug flag — no unconditional `JSON.parse(JSON.stringify(payload))` per request.
**Success Criteria**: with React Profiler in tests, streaming 100 chunks into a 50-message conversation re-renders only the streaming row (+composer-independent widgets unchanged); completed rows do not re-render; KnowledgeQA dispatches throttled to ~12/s max; chat send performs 0 synchronous full-payload clones unless debug preview open.
**Tests**: render-count regression tests (Profiler or render spy) for memoized rows; KnowledgeQA throttle test with fake timers; chat-request-debug lazy-clone test; existing Playground suite (`test:playground:*` scripts) green.
**Status**: Complete

## Stage 6: Cleanup (A9 partial)
**Changes**: `git rm` `extension/script.js`, `extension/test-js.js` (verified unreferenced); add `.watchlists-e2e-report.json` to `apps/extension/.gitignore`; fix `apps/DEVELOPMENT.md` (`extension/entries/` → `entrypoints/`; web `components/` description); do NOT delete `packages/voice-assistant-sdk` (flagged for maintainer decision in PR).
**Tests**: grep for references before removal; docs-only otherwise.
**Status**: Complete

## Stage 7: Verification
`bun install` fresh; run: packages/ui vitest (full), tldw-frontend web vitest, extension vitest + `build:chrome` (+ artifact size diff table), tldw-frontend `typecheck` and `compile` (bundle budget gate), `verify:openapi` still green. Bandit: N/A (no Python touched — record in task). Self-review diff; update plan statuses; update Backlog task with touched files + verification results.
**Status**: Complete

## Stage 8: Follow-ups + PR
File Backlog follow-up tasks for deferred items: (a) unify API clients + move OpenAPI codegen to shared package (A2/A3); (b) `packages/ui` package split + boundary lint (A1/A6); (c) vitest workspace consolidation + alias single-source (A5); (d) dependency major alignment zustand/marked/react across shells (A4); (e) auth + UI primitives dedupe (A7); (f) store slice splits (A8); (g) font subsetting/woff2 + `_locales` slimming + `web_accessible_resources` narrowing (E6/W5); (h) ResearchWorkspace ChatPane virtualization; (i) composer draft leaf extraction (R6); (j) ResearchWorkspace dual-poller merge. Push branch; `gh pr create --base dev`; PR body notes AI-authored → human `Change summary` required before merge per repo policy.
**Status**: Complete

## Explicitly deferred (not silently dropped)
Architecture refactors A1–A8, E6/W5 asset work, R6, ChatPane virtualization, W3 prerender, dual-poller merge — each filed as a Backlog follow-up in Stage 8 with the review evidence. `voice-assistant-sdk` deletion needs a maintainer decision (roadmap?).
