# tldw-frontend + Extension Audit

This document is a working log for assessing and improving the `tldw-frontend` web app and the WXT browser `extension`. Both are thin shells over the shared `packages/ui` code, so most findings live there and affect **both** clients.

---

## 0. Scope & Goals

- **Date / Reviewer(s)**: 2026-07-02, deep code audit (8 parallel reviewers + manual verification of every Critical/High).
- **Frontend version / branch**: `dev`.
- **Primary goal**: Inherited-codebase risk audit — find real bugs and "ticking time bombs" in structure/design for a maintainer without deep frontend history. **This is an assessment; nothing here has been changed in code.**
- **Method**: Read the hotspot files (8k-LOC API client, 4k-LOC stores, MV3 background worker, auth layer, chat pipeline, render/XSS sinks). Each finding cites `file:line`. Criticals/Highs were re-read and refutation-checked; a prior explorer claim of "21 GB build artifacts in git history" was **disproven** (`.git` is 905 MB, zero tracked build artifacts) and dropped.
- **Out of scope**: backend (`tldw_Server_API`), UX polish (covered by `ux-audit-v3/` and `QA_PAGE_REVIEW_CHECKLIST.md`), performance profiling.

### How to read severity
- **Critical** — silent data loss or credential exposure happening on normal, everyday use.
- **High** — a real bug that breaks a feature or strands the UI under common conditions, or a security gap with a plausible trigger.
- **Medium** — intermittent breakage, fragility, or a maintenance trap that will bite later.
- **Low** — edge cases and latent traps.

---

## 0.5 Findings summary (ranked)

| # | Sev | One-line | Where |
|---|-----|----------|-------|
| C1 | **Critical** | Successful non-streaming chat replies containing the word "error"/"exception"/a file path are silently replaced with `"Chat completion failed."` | `packages/ui/src/services/tldw/TldwApiClient.ts:156-217,2660` |
| C2 | **Critical** | Live auth headers + login-response tokens are written to `localStorage` request-history (200 entries) and never cleared, even on logout | `apps/tldw-frontend/lib/api.ts:468-470`, `lib/history.ts:16-28` |
| H1 | High | `javascript:` DOM-XSS: ~10 hand-rolled source/citation anchors render URLs with no protocol allowlist, and the web app ships **no CSP** | `packages/ui/.../MessageSource.tsx:80,183` (+9 sites) |
| H2 | High | MV3 service-worker suspension orphans in-memory ingest / quick-ingest / auth-replay sessions; "will retry automatically" never happens | `packages/ui/src/entries/background.ts:444-446,695-706,1716` |
| H3 | High | `tldw:upload` / `tldw:stream` attach the API key/bearer to **any** absolute URL — no origin allowlist (the request path has one; these don't) | `packages/ui/src/entries/background.ts:1055-1073,3234-3257` |
| H4 | High | No reachable token refresh in the browser — refresh code exists but is wired only in the extension; expiry mid-session misreports "backend unavailable" | `packages/ui/src/services/background-proxy.ts:835-847`, `TldwApiClient`/`request-core` |
| H5 | High | Stop doesn't actually abort the network stream in normal/RAG chat (signal never threaded); a module-singleton controller makes concurrent streams cancel each other (breaks Compare mode) | `models/ChatTldw.ts:181-223`, `services/tldw/TldwChat.ts:443-446` |
| H6 | High | 5-second connection timeout silently **replays a non-idempotent chat POST** → duplicate generation + duplicate saved messages (triple-confirmed) | `services/background-proxy.ts:1236-1243,1320-1324` |
| H7 | High | Connection store spreads a 20s-old state snapshot over concurrent updates and has a racy overlap guard → UI flips to "disconnected" after a good check; onboarding jumps back a step | `packages/ui/src/store/connection.tsx:594-711,949-1014` |
| H8 | High | Workspace store's rehydrate mutates state in place with no `set()` → subscribers never notified → empty workspace / eternal loading gate | `packages/ui/src/store/workspace.ts:3875-3970` |
| H9 | High | `wxt-browser` storage shim: `clear()` wipes the **entire origin** localStorage, and `local`/`sync`/`session` areas all collide on the same keys | `apps/tldw-frontend/extension/shims/wxt-browser.ts:108-215` |
| H10 | High | Plasmo `useStorage`/`watch()` shims never propagate changes across instances → settings toggles don't apply until a full page reload | `apps/tldw-frontend/extension/shims/plasmo-storage*.ts` |
| H11 | High | One transient `404` permanently disables folder sync (persisted `folderApiAvailable:false`, never reset) until the user clears localStorage | `packages/ui/src/store/folder.tsx:314,401,843` |
| H12 | High | Default 10s timeouts abort normal LLM generations and TTS mid-flight → "Network error" while the server keeps working | `services/tldw/request-core.ts:95-100`, `background-proxy.ts:31-32,271-281` |
| — | **Config** | `strict:false`, `ignoreBuildErrors:true`, and disabled newer `react-hooks` lint (`set-state-in-effect` et al.) let real bugs ship silently | see §9 |
| — | **Config** | 8 of 9 persisted stores have no `version`/`migrate`; frontend vs extension dependency skew on shared code | see §6, §9 |

Medium/Low findings and the full "verified-OK" list are in §4–§10.

---

## 0.6 Remediation status (2026-07-02)

Every Critical and High **bug** finding above was **fixed** in this pass, each with focused unit tests. The two remaining items are not bug fixes: the **config-hardening** task (12102, TS-strict) is a partial/phased migration, and a few **residual refinements** (H1 CSP `unsafe-eval`, 12103 dead-tree removal) need out-of-band verification — both are called out explicitly in the rows below and in "Residuals", so "fixed" refers to the defects, not to those tracked follow-ups. Backlog tasks `task-12091`…`task-12103` track the work.

**Verification:** the new + affected suites all pass — 127 tests across 14 packages/ui files, 6 (C2 redaction), 11 (shim/nav), and 23 existing store/audio/client regression tests. The only red test is a **pre-existing** `workspace.ts` quota-warning test that fails identically on baseline `dev` (confirmed via `git stash` by two reviewers) and is unrelated to these changes.

| # | Status | Notes |
|---|--------|-------|
| C1 | ✅ Fixed | Sanitizer removed from the runtime path (`chat-rag.ts`) and base class; buggy helper deleted. |
| C2 | ✅ Fixed | Centralized redaction in `history.ts`; logout clears history. |
| H1 | ✅ Fixed | Shared `safeExternalUrl`/`openExternalUrl` at all sinks + CSP added and **tightened**: `'unsafe-inline'` dropped from `script-src` (the one trusted inline script is SHA-256-hash-allowlisted), plus `X-Content-Type-Options`/`Referrer-Policy`/`X-Frame-Options`/`Permissions-Policy`. `'unsafe-eval'` retained (WASM) as a documented follow-up. ⚠️ **Verify in a browser before merge** — confirm no CSP violations on the main routes and the dev/error overlays (I can't run the browser here). |
| H2 | ✅ Fixed | Session state persisted to `chrome.storage.session` + `chrome.alarms` backstop. Quick-ingest **remote-job polling now resumes** after a worker restart (alarm-driven, from persisted batch records); an in-flight multipart *upload* is non-resumable by design and instead reports interrupted so the UI isn't stuck, and post-restart cancel always finds the session. |
| H3 | ✅ Fixed | Origin allowlist + `sender.id` guard on `tldw:upload`/`tldw:stream`. Guard logic **consolidated** into a single canonical `absolute-url-guard.ts` (request-core + background-proxy now import it; request-core's diagnostic warnings preserved via optional hooks) — no more triplication. |
| H4 | ✅ Fixed | Web `refreshAuth` wired with single-flight; re-armable timer; FormData-retry bug fixed. |
| H5 | ✅ Fixed | Signal threaded, per-call controllers, ownership-guarded resets, early-throw + abort-path fixes. Regenerate-abort-before-first-token now discards the empty variant and restores the prior active variant/index. |
| H6 | ✅ Fixed | 5s→config-derived timeout; non-idempotent POSTs throw `StreamInterruptedError` instead of being replayed. **F10 closed:** the `stream_transport_interrupted` sentinel is now surfaced through the normal/RAG token pipeline (captured + re-emitted in `ChatTldw.stream`), so a post-first-byte truncation is finalized as *interrupted* (parity with character chat), never saved as complete. |
| H7, H8, H11 | ✅ Fixed | Functional `set` + synchronous guard (connection); hydration published via `set` (workspace); availability flag no longer persisted, with self-healing `merge` (folder). |
| H9, H10 | ✅ Fixed | Per-area isolation + memory-only `session` + scoped `clear()`; cross-instance watch bus + `useStorage` subscription; dynamic-route `useSearchParams`. |
| H12 | ✅ Fixed | Generation-endpoint timeout 10s→120s; messaging-ack decoupled (10s→130s); body read bounded; TTS timeout in `synthesizeSpeech`. |
| Config (12102) | ◑ Partial | Done: `version`/`migrate` baseline on the 8 unversioned stores; added a `typecheck` script (`tsc --noEmit`); corrected the audit's `rules-of-hooks` claim (it is already enabled). **Blocked/phased:** removing `ignoreBuildErrors` / enabling `strict` requires first clearing ~47 **pre-existing** `tsc` errors (in unrelated Watchlists components, at the current loose settings) — that cleanup is separate from audit remediation. Enabling the newer `react-hooks` rules (`set-state-in-effect` et al.) is deferred to avoid a large, noisy fix in this PR. |
| Dead code (12103) | ✅ Done | Web auth stack (`useAuth`/`useConfig`/`Header`/`Layout`/`useIsAdmin`) **deleted** (5 files + their exclusive tests; docs updated; live `lib/*`/`WebLayout` left intact). The `extension/routes/` tree was **kept** — investigation showed it's runtime-unused but parity-test-maintained (not deletable without migrating ~22 tests); documented via `_RUNTIME_UNUSED.md`. |

**Residuals — nearly all closed in a follow-up pass.** Fixed since the first draft: F10 partial-stream marking (H6), quick-ingest resume (H2), guard-helper consolidation (H3), regenerate-abort discard (H5), CSP `script-src` tightening + extra security headers (H1). **What genuinely remains** (each a larger effort or needing out-of-band verification, none reintroducing a defect):
- **12102** — removing `ignoreBuildErrors` / enabling TS `strict` is blocked on ~50 **pre-existing** `tsc` errors in unrelated code (measured; must be cleared first); enabling the newer `react-hooks` rules (`set-state-in-effect` et al.) is deferred to avoid a large noisy fix. A `typecheck` script was added so the team can burn the baseline down.
- **H1** — dropping `'unsafe-eval'` from the CSP needs per-feature (WASM/OCR/tokenizer) browser verification; and the tightened CSP overall should get a quick browser smoke before merge (dev/error overlays especially).
- **12103** — the `extension/routes/` mirror is runtime-unused but kept intentionally in sync by ~22 parity tests; removing it needs those tests migrated first.

---

## 1. Environment & Tooling

- Bun workspace at `apps/`. Web app: **Next.js 16, pages router**. Extension: **WXT 0.20, Manifest V3**. Both import shared code from `packages/ui/src` via `@`/`~` aliases.
- CI runs lint + changed-only Vitest + Playwright e2e (`frontend-required.yml`, `e2e-required.yml`). Gaps: no coverage gate, TypeScript errors do not fail the build (see §9), extension e2e not gated on the main frontend job.
- Local `.next` build output on disk is large (~21 GB) but **not** in git — a `git clean`/prune housekeeping note, not a repo problem.

---

## 2. High-Level Architecture

- **Thin-shell pattern**: `tldw-frontend/pages/*` and `extension/entrypoints/*` are wrappers; the real components, hooks, services, and 59 Zustand stores live in `packages/ui`. A bug in `packages/ui` ships to both clients.
- **State**: Zustand (59 stores; biggest are `workspace.ts` ~4k LOC and `connection.tsx` ~1.3k LOC) + React Query for server state + a few React contexts.
- **Two auth/request stacks** (this seam is the source of several bugs):
  1. **Web-only** — `tldw-frontend/lib/api.ts` + `lib/auth.ts` + `hooks/useAuth.tsx`/`useConfig.tsx`. Used by ~9 modules (VN workbench, connectors, VLM, characters, research runs).
  2. **Shared** — `packages/ui/src/services/tldw/*` routed through `background-proxy.ts` (`bgRequest`/`bgStream`/`bgUpload`). Used by everything else. In the extension these go through the MV3 background worker; in the web build they fall back to direct fetch.
- **Positive**: extension manifest permissions are tight (`host_permissions` is just `api.github.com`), `externally_connectable` is unset (arbitrary sites can't message the worker), and the main markdown/rich-text render pipeline is properly sanitized. See "Verified OK" lists.

---

## 3. API & Backend Alignment

Not re-audited for endpoint drift here (backend is out of scope). One correctness note surfaced: `TldwApiClient.waitForExportReady:6964` treats the server's `export_status="none"` as a failure and discards the real status/detail (`file_artifacts_service.py:387` emits it).

---

## 4. Auth, Roles & Security

### C2 (Critical) — Credentials persisted to localStorage, survive logout
`lib/api.ts:468` runs `applyBrowserHeaders` (adds `Authorization`/`X-API-KEY`/`X-CSRF-Token`), then `:470` passes those headers into `buildRequestHistoryConfig`; `recordSuccess` (`:384-404`) stores `requestHeaders` **and** `responseBody` (which for `/auth/login` includes the `access_token`) into `localStorage['tldw-request-history']` — 200 entries, no redaction (`lib/history.ts:16-28`). `clearRequestHistory` exists but is **never called**, and logout (`lib/auth.ts:203-213`) doesn't touch the key. **Result**: bearer tokens and API keys sit in plaintext localStorage indefinitely and survive logout — readable by any XSS (see H1) or anyone on a shared machine. **Fix**: redact `authorization`/`x-api-key`/`x-csrf-token` and login-response bodies before storing; clear the history key on logout.

### H1 (High) — `javascript:` DOM-XSS via source/citation anchors, no CSP backstop
The web app ships **no Content-Security-Policy** (verified: no `headers()` in `next.config.mjs`, no `middleware`). `MessageSource.tsx:80` reads `const url = source?.url` and `:183` renders `<a href={url} target="_blank" rel="noopener noreferrer">` — `rel`/`target` do not stop `javascript:`. The same unvalidated `<a href={url}>` appears in ~9 more places (research sources, watchlist sources/runs/outputs/alerts, reading list, items, processed). `source.url` is attacker-influenceable: poisoned ingested-page metadata (yt-dlp title/URL, scraped feed), a malicious web-search/research citation, or a crafted API/JSON response. **Result**: clicking a "source" link runs script on the app origin → session/token theft (and C2 makes the loot valuable). The markdown renderer already blocks this via `urlTransform`; these hand-rolled anchors bypass it. **Fix**: a shared `safeExternalUrl()` (allowlist `http`/`https`/`mailto`, else no-op) at these anchors and the `window.open(url)` sites; add a CSP.

Related same-family: **OutputPreviewDrawer.tsx:325** `safeHtml = sanitizedHtml || content` re-injects raw HTML into a same-origin `blob:` tab when DOMPurify returns empty (Medium); **notes `sanitizeUrl`** (`notes-manager-utils.ts:834`) lets a control-char scheme (`java\tscript:`) through (Low–Medium, inside `contentEditable`).

### H3 (High) — Extension attaches credentials to arbitrary absolute URLs
The `tldw:upload` (`background.ts:1055-1073,1143-1168`) and `tldw:stream` port (`:3234-3257`) handlers treat any `path` starting with `http` as absolute and unconditionally add `X-API-KEY`/`Authorization`/`X-TLDW-Org-Id`, then `fetch(url)` — with **no origin allowlist**. The normal request path (`request-core.ts:248,363,387`) *does* gate this (`absoluteOriginAllowlistFromConfig` + `shouldSkipAuth` on cross-origin). So a caller posting `{path:"https://attacker/x"}` gets the user's API key sent to the attacker. **Reachability**: `externally_connectable` is unset, so this needs an extension-context caller — i.e. a buggy or compromised content script (which run on every `http(s)` page). That's why this is High, not remotely-triggerable Critical. Message handlers also don't validate `sender.id` (defense-in-depth gap). **Fix**: apply the same allowlist/`shouldSkipAuth` gate in the upload/stream handlers; add a `sender.id === browser.runtime.id` check.

### H4 (High) — No reachable token refresh in the browser
Refresh-and-retry lives in `request-core.ts:470-514` but requires `runtime.refreshAuth`, which is wired **only** in the extension background (`background.ts:1248-1263`). The web direct fallback passes only `{getConfig}` (`background-proxy.ts:835-847`), and `TldwAuth`'s pre-expiry timer is armed only inside `login`/`verifyMagicLink` and is lost on page reload (`TldwAuth.ts:384-401`). **Result**: a multi-user user who logs in and reloads will, on token expiry, have every request 401 and the UI misreport "backend unavailable" while a valid refresh token sits unused — recovery needs manual re-login. **Fix**: pass a single-flighted `refreshAuth` into the web direct fallback; re-arm the refresh timer on load.

### Medium/Low (auth)
- **Half-wired dead web auth stack** (Medium, maintenance trap): the web `AuthProvider` (`hooks/useAuth.tsx`) and web `ConfigProvider` (`hooks/useConfig.tsx`) are **never mounted** (the `ConfigProvider` in `AppProviders.tsx:80` is antd's, not this one). Their only consumers — `components/layout/Header.tsx` → `components/layout/Layout.tsx` — are imported by nothing. So `useAuth`/`useConfig` would throw if rendered, `api.defaults.baseURL` is never synced from user config, and several "latent" auth bugs never fire. It's ~500 lines of realistic-looking code a maintainer will mistake for live. **Recommendation**: delete or clearly quarantine it.
- **Refresh single-flight is per-context** (`TldwAuth.ts:231-261`): the mutex is only in the extension background; the UI-context auto-refresh timer can race a 401 refresh and persist a rotated/dead token.
- **Redaction is key-name-only** (`background-proxy.ts:209-236`): a JWT/stack/SQL string carried in a *value* (under `detail`, `message`, a string array, or a bare `text/plain` body) is not scrubbed and surfaces to logs/UI.
- **Fetches follow redirects** (no `redirect:"manual"`): browsers strip `Authorization` cross-origin but **not** the custom `X-API-KEY`, so an open redirect can leak it.
- **CSRF** read from `document.cookie` can't see a cross-origin API host's cookie → mutating requests 403 with an unfixable "refresh the page" message; also thrown as a plain `Error`, so `err.status === 403` checks miss (`lib/api.ts:229-236,506-519`).
- **STT token in WebSocket URL query string** (`background.ts:2817`) → lands in server access logs.

### Verified OK (security)
Tight manifest permissions; `externally_connectable` unset; request path enforces the absolute-URL allowlist and strips auth cross-origin; origin comparison resists `user@host` tricks; failed refresh does **not** clear tokens (no logout storm); non-idempotent POSTs are not replayed on messaging timeout; the main markdown pipeline (react-markdown, no `rehype-raw`, `urlTransform` allowlist), st_compat rich text (marked + DOMPurify with `FORBID_TAGS`/`on*` stripping), Mermaid (`securityLevel:strict`), CodeBlock (`iframe sandbox` opaque origin + postMessage token), `JsonViewer` (escape-before-highlight), and the copilot popup (Shadow DOM `textContent` only) are all properly sanitized; the web-clipper Readability→cheerio→Turndown pipeline strips raw HTML before it becomes markdown.

---

## 6. State Management & Data Fetching

### H7 (High) — Connection store: stale-snapshot clobber + racy overlap guard
`connection.tsx` `checkOnce` captures `currentState` at the top (`:595`), runs a health check for up to 20s, then every terminal `set()` does `{...currentState, ...}` (`:949-1014`) — silently reverting any `setConfigPartial`/`markFirstRunComplete`/`setUserPersona` that fired meanwhile (onboarding jumps back a step; `hasCompletedFirstRun` flips back to false). The overlap guard reads `isChecking` at `:598` but sets it at `:698` after five `await`s, so concurrent callers (poller, ChatPane, QuickIngestWizardModal, chat-history hook) both run and the last, stale finisher wins — UI flips to "disconnected" right after a good check. **Fix**: use functional `set((s) => …)`; set the in-flight guard synchronously before the first await, or use a request token.

### H8 (High) — Workspace store: silent-mutation rehydrate
`workspace.ts:3875-3970` `onRehydrateStorage` mutates the hydrated state object in place (`Object.assign(state, …)`, `state.storeHydrated = true`) with no `set()`, so subscribers are never notified. Hydration is async, so components are already mounted; the active workspace's sources/artifacts/notes and the `storeHydrated` gate only reflect once some *unrelated* `set()` happens to fire. **Result**: intermittent empty workspace / eternal loading gate. Compounded by `workspace-list-slice.ts:1208` `reset: () => set(initialState)` leaving `storeHydrated:false`. **Fix**: apply hydrated data via `set()`.

### H11 (High) — Sticky persisted failure: one 404 kills folder sync forever
`folder.tsx`: any 404 (or a message merely containing "404", `:409`) sets `folderApiAvailable:false`; `partialize` persists it (`:843`); `refreshFromServer` hard-returns when false (`:314`); and the only reset to true (`:401`) is inside the now-skipped path. A transient 404 (server restart, proxy blip, older server) disables folder sync across every future session until localStorage is cleared. **Fix**: don't persist the availability flag, or add a reset path / retry.

### Medium/Low (state) — mostly systemic
- **Persist without `version`/`migrate`** (Medium, systemic — 8 of 9 stores): only `workspace.ts` has it. The day anyone adds `version:1` to reshape a store without a `migrate`, all users' persisted state is silently discarded; any field rename before then ships `undefined` into consumers. Stores: `playground-session`, `persona-buddy-shell`, `notes-dock`, `ui-mode`, `actor`, `quick-ingest-session`, `folder`, `feedback`, `acp-sessions`.
- `workspace.ts` split-key persistence is a non-atomic async read-modify-write with no serialization (`:1777,1500-1756`) → two tabs / overlapping writes can leave the index pointing at deleted keys → workspaces vanish or rehydrate empty.
- `workflow-editor.ts:827-861` `loadRunInvestigation` can deadlock its own loading flag (early returns don't reset it) → switching runs mid-fetch blocks all future loads.
- `folder.tsx:213-237` throttled localStorage adapter drops the final write on fast tab-close and serves stale reads during the 1s window.
- `refreshFromServer` (`folder.tsx:309`) has no in-flight guard → older response overwrites newer in Dexie + store.
- Unbounded persisted growth: `feedback.tsx` `entries` (no cap/eviction/clear), `workspace.ts` `savedWorkspaces`/`workspaceSnapshots` (snapshots embed full data-URLs, re-serialized every state change).
- `quick-ingest.tsx:346` module-scope cross-store `subscribe()` never unsubscribed (HMR leak). `workspace.ts:73-76 ⇄ slices` circular value imports (works only by init order; reorder → TDZ crash).

### Verified OK (state)
**No zustand v4-vs-v5 API divergence** despite the frontend(^5)/extension(^4) split — 36/59 stores use `createWithEqualityFn` (default `Object.is` in both majors), no removed-API usage, no lost equality functions. `timeline.ts`/`workflow-editor.ts`/`acp-sessions.ts` use correct request-token guards and bounded histories; workspace sources/studio slices are synchronous and stale-closure-free; quota handling (QuotaExceeded → LRU eviction) is coherent; storage adapters are SSR-guarded.

---

## 6b. Browser-API shims (web build only)

The web build fakes `chrome.storage`/`browser.*` and `react-router-dom` over `localStorage`/Next router. These shims are **live** and have real bugs:

- **H9 (High)** `wxt-browser.ts:189-214` — `clear()` for any area calls `backend.clear()`, wiping the entire origin's localStorage (all areas, theme/flags, plasmo-prefixed keys). Live caller `system-settings.tsx:62`. And `:108-215` — no per-area isolation: `local`/`sync`/`session` share unprefixed keys, so `sync.set` clobbers `local`, and `session` (should be memory-only) persists to disk.
- **H10 (High)** `plasmo-storage.ts:143-185` + `plasmo-storage-hook.tsx:37-59` — `watch()` callbacks are per-instance and `useStorage` never subscribes at all, so two components on the same key desync and settings written elsewhere don't apply until a full reload (e.g. sticky-chat toggle, ReviewPage config watch).
- **Medium** `react-router-dom.tsx:192-213` — `useSearchParams` setter rebuilds the URL from `router.pathname` (the `[bracket]` pattern on dynamic routes) so it **silently fails on every dynamic route** (`/sources/:id`, `/media-collections/:id`, `/knowledge/thread/:id`) — only a `console.error`. Also `useNavigate` returns a new identity each render (dep-array churn), and `useLocation` mixes `router.asPath` with live `window.location` (hydration mismatch + stale `search`).
- **Medium (runtime-unused, maintenance trap)** `tldw-frontend/extension/routes/*` (route-registry, app-route, all `option-*`) is **not rendered at runtime** in the web build — pages mount `packages/ui/src/routes/*` instead, so editing a component here silently no-ops. **Correction to an earlier draft:** it is **not safely deletable** — ~22 tests reference it (3 direct imports + ~19 `readFileSync` parity-guard tests that keep it in sync with `packages/ui/src/routes/*`). So it's a deliberately-maintained mirror, not disposable dead code; the trap is that its runtime-irrelevance isn't obvious. A `_RUNTIME_UNUSED.md` marker documents this; genuine removal requires retiring the parity tests first.

---

## 7. Error Handling & Resilience

### H2 (High) — MV3 worker suspension orphans background sessions
Chrome kills an idle MV3 service worker (~30s). `background.ts` keeps critical state only in `main()`-closure Maps with no `chrome.storage.session` rehydration: `ingestSessions` (`:444`), `pendingAuthReplay` (`:445`), `quickIngestModalSessions` (`:446`). Long polls run via detached `setTimeout` loops for up to 10 min (`:1716`). When the worker is reclaimed mid-poll: an ingest that the server completed never emits `media-ingest-ready` → sidepanel stuck on "Queued for processing", and `cancel`/`retry` return "Ingest session not found" (permanently unrecoverable, `:1490,1792`); the 401 "ingest will retry automatically" promise (`:695-706`) never fires because the replay set is empty on wake; quick-ingest batches are orphaned with a frozen progress UI. **Fix**: persist session state to `chrome.storage.session`/`local`, rehydrate on `onStartup`, and drive long polls from `chrome.alarms` (the model-warmup code already does this correctly — a good template).

### H5 (High) — Chat abort/stream lifecycle is broken in several ways
- **Stop doesn't abort the transport** in normal/RAG modes: `ChatTldw.stream()` gets the UI `signal` but calls `tldwChat.streamMessage` **without** it (`ChatTldw.ts:181-223`); the signal is only polled at loop top, so the fetch/port stays open (server keeps generating + persisting) until the next token or 30s idle. Character chat threads the real signal, so behavior diverges by mode.
- **Singleton controller collisions**: `tldwChat` is a module singleton with one `currentController`, and every `streamMessage()` starts with `cancelStream()` (`TldwChat.ts:443-446`) — so any two concurrent streams cancel each other. Compare mode (N parallel models) has N-1 die with "Request cancelled".
- **Shared-controller clobber** (`chatModePipeline.ts:808`): a finishing turn's `finally` unconditionally resets the shared streaming flag + controller, so an old turn re-enables the send button and nulls the controller of a newer in-flight turn (which then can't be stopped).
- **Stuck-streaming**: `onSubmit` sets `setStreaming(true)` then `await`s `buildChatModeParams` **outside** the try (`useChatActions.ts:2335-2394`); a throw leaves the spinner + disabled send button stuck until reload.
- **Fix bundle**: thread the UI `AbortSignal` through `ChatTldw.stream → streamMessage → bgStream`; replace the singleton with per-call controllers; make each `finally` reset flags only if it still owns the current controller; move `buildChatModeParams` inside the try.

### H6 (High) — 5s connection timeout replays a non-idempotent chat POST
`background-proxy.ts:1236-1243,1320-1324`: if no stream byte arrives within a hard-coded 5s, `bgStream` disconnects and **re-sends the whole request** via `bgStreamDirect`. `/api/v1/chat/completions` (with `save_to_db`) and `/complete-v2` are not idempotent, and TTFT > 5s is normal for large prompts, RAG, or a cold local model. **Result**: duplicate generation and duplicate persisted messages. (Independently flagged by three reviewers.) Related dead-code hazard: the `stream_transport_interrupted` handling in the pipeline can never match in normal modes (`chatModePipeline.ts:582-599`) because the token extractor drops the event, so an extension port loss mid-answer silently truncates and saves as complete. **Fix**: derive the timeout from config (not a 5s constant), and don't auto-replay non-idempotent POSTs; surface the interruption event through the token pipeline.

### H12 (High) — 10-second default timeouts abort normal LLM work
`request-core.ts:95-100` defaults `/chat/completions` to a 10s **total** timeout (vs 45s stream-idle); `background-proxy.ts:31-32,271-281` fails any extension-context write after 10s (`Number(undefined) → NaN → 10_000`) while the worker keeps running. Most `TldwApiClient` POST wrappers (including `synthesizeSpeech`) pass no `timeoutMs`. **Result**: unconfigured non-stream chat and TTS abort mid-generation as "Network error" / "Extension messaging timeout" while the server finishes and the result is lost. **Fix**: raise/annotate defaults for generation endpoints; decouple the messaging-ack timeout from the request timeout.

### Medium/Low (resilience)
- `research.tsx` control handlers (`handlePauseRun`/`Resume`/`Cancel`/`LoadArtifact`) are async `onClick` with **no try/catch** (`:1249-1319`) → a failed POST is an unhandled rejection that can trip the global handler and replace the whole app with the recovery screen. Plus a last-writer-wins SSE-vs-refetch race that strands a completed run as "running" (`:1108`), and a transient reconnect that wipes an in-progress checkpoint-editor draft (`:1184`).
- App-level `ErrorBoundary` never resets on route change → after one page throws, healthy routes still show the error screen until "Try again"/reload.
- Message index-space mixups: `deleteMessage`/`createEditMessage` address Dexie rows by UI-array index while the UI list contains never-persisted entries (character greeting) → wrong row deleted/overwritten (`useChatActions.ts:3285`, `messageHandlers.ts:172`).
- `createChatCompletion` sanitizer aside, `bulkUpdateMediaKeywords` fabricates per-ID success when the response lacks a `results` array (`TldwApiClient.ts:3006`) → UI claims "all updated" when nothing was.

---

## 9. Dependencies & Technical Debt

### Config "time bombs" (verified by direct read)
- `apps/tldw-frontend/tsconfig.json:11` and `apps/extension/tsconfig.json:9` — **`"strict": false`** in both. No null-safety on a ~1.2M-LOC shared surface; large refactors risk runtime crashes the compiler would otherwise catch.
- `apps/tldw-frontend/next.config.mjs:59` — **`typescript.ignoreBuildErrors: true`**. TS errors never fail the build or CI; combined with `strict:false` this is two safety nets removed at once.
- `apps/tldw-frontend/eslint.config.mjs:78-84` — the newer **react-compiler-era `react-hooks` rules are disabled** (`immutability`, `purity`, `preserve-manual-memoization`, `refs`, `set-state-in-effect`, `static-components`, `use-memo`). `set-state-in-effect` in particular would have flagged several of the effect-race bugs in this audit. **Correction to an earlier draft of this section:** the classic `react-hooks/rules-of-hooks` (which catches conditional/looped-hook crashes) is **not** globally disabled — the `off` at `:118` is scoped to `e2e/**` only, and the rule is active everywhere else via the `reactHooksRules` preset. `no-explicit-any` is only `warn`.
- **Dependency skew on shared code**: the frontend and extension pin *different majors* of libraries that both feed `packages/ui` — `zustand ^5` vs `^4`, `dexie-react-hooks ^4` vs `^1.1.7`, `marked 17` vs `15`, `d3-dsv 3` vs `2`, `react ^18.3` vs pinned `18.2`, TypeScript `5.6` vs `5.9`. A shared component that relies on one major's behavior can pass in one app and break in the other. (Zustand specifically was checked and is currently safe — see §6 — but it's a standing hazard.)

### Recommendation
Turn the safety nets back on incrementally: `noImplicitAny` first, then `strictNullChecks`; remove `ignoreBuildErrors` once `packages/ui` typechecks; re-enable `react-hooks/rules-of-hooks` and fix violations. Add a `version`/`migrate` to every persisted store (§6). Align the shared-dependency majors or hoist them to one workspace-level version.

---

## 10. Testing & Automation

Solid e2e footprint (140+ Playwright specs in the extension, tiered gates in the web app) and the manifest/permission posture is good. Gaps worth closing: no coverage threshold; TS errors don't gate (see §9); extension e2e isn't part of the required frontend job; the shims (§6b) — which are load-bearing for the whole web build — have thin unit coverage relative to their bug density.

---

## 12. Summary & Next Steps

- **Overall health**: **Yellow.** The architecture is reasonable (thin shells over a shared package, tight extension permissions, a properly sanitized main render path, good e2e coverage) and there was **no** repo-history disaster. But there is a cluster of real correctness/security bugs concentrated in four areas: the shared API client, the auth seam, the MV3 background worker, and the browser-API shims — plus disabled safety nets that let this class of bug ship.

- **Top 3 risks**
  1. **Silent data loss / credential exposure on normal use** — C1 (chat replies corrupted) and C2 (tokens in localStorage). Both confirmed, both happen without anything unusual.
  2. **Auth/streaming lifecycle** — no web refresh (H4), Stop doesn't stop (H5), 5s replay duplicates messages (H6), 10s timeouts abort generations (H12). These make the core chat feature unreliable.
  3. **State/storage foundations** — connection & workspace store races (H7/H8), sticky failure states (H11), and shims that wipe storage / don't propagate changes (H9/H10).

- **Top short-term fixes (1–2 weeks)**
  1. C1: stop routing successful completions through the error-string sanitizer (or narrow it to genuine error payloads).
  2. C2: redact auth headers/tokens from request-history and clear it on logout.
  3. H1: add `safeExternalUrl()` at the ~10 anchor/`window.open` sinks + ship a CSP.
  4. H3: add the URL allowlist to the extension upload/stream handlers.
  5. H6/H12: fix the streaming-replay and timeout defaults.

- **Longer-term**
  - Persist MV3 session state and move long polls to `chrome.alarms` (H2).
  - Re-enable TypeScript strict + `react-hooks` lint incrementally (§9).
  - Thread abort signals and use per-call controllers throughout the chat pipeline (H5).
  - Delete or quarantine the half-wired web auth stack and the dead `extension/routes` tree.
  - Add `version`/`migrate` to persisted stores.

Per-finding backlog tasks were created for the Critical and High items (`backlog/tasks/task-12091`…`task-12103`). Full reviewer notes and the verification log are archived with this audit.

---

# Round 2 (2026-07-02): Character Chat + TTS/STT

Focused follow-up audit of the two areas the maintainer was concerned about: **character chat** and **TTS/STT**. Five parallel reviewers over character-chat core, character card/data handling, TTS playback, mic capture, and real-time voice WebSockets. Findings verified against code identical between local `dev` and `origin/dev`.

## R2 findings summary (ranked)

| # | Sev | One-line | Where |
|---|-----|----------|-------|
| R1 | **High** | Character chat delete/edit hits the **wrong Dexie row** — the greeting sits at UI index 0 but is never in Dexie, so array-index deletes/edits are off by one | `useMessage.tsx:2929,2781`; `db/dexie/helpers.ts:406` |
| R2 | **High (privacy)** | Microphone can **stay live after a `MediaRecorder` error** (no `onerror` handler) or on a double-start; indicator stuck on, capture coordinator locked | `useAudioRecorder.ts:110-130` (+ `useServerDictation.tsx`, `SpeechPlaygroundPage.tsx`) |
| R3 | **High (security)** | Voice/STT WS **auth token in the URL query string** → server/proxy logs. Backend already supports subprotocol (persona) and `{type:"auth"}` first-message (audio STT); client uses only the URL token | `persona-stream.ts:20,26`; `voice-conversation.ts:361`; `background.ts:3332` |
| R4 | Medium | Greeting seed is **never persisted to Dexie** (root cause of R1) → same conversation shows a different message count depending on Dexie vs server rehydrate | `chat-helper/index.ts:438-491` |
| R5 | Medium | A **stalled character stream hangs indefinitely** — the live path has no inactivity watchdog; the copy that *does* (`useCharacterChatMode.ts`) is tested-but-unused. Three diverged copies of `characterChatMode` | `useChatActions.ts`/`useMessage.tsx` vs `useCharacterChatMode.ts` |
| R6 | Medium | **Steady TTS memory leaks** — MediaSource/blob URL leaks on stream-fallback and on cancel-during-playback | `useStreamingAudioPlayer.tsx:273`, `useTTS.tsx:283-330` |
| R7 | Medium | **Overlapping TTS playback** in the chat drawer — playing a second clip doesn't stop the first (two voices) | `TtsClipsDrawer.tsx:128-160` |
| R8 | Medium | Real-time voice: **barge-in doesn't stop TTS** (assistant talks over you), no `bufferedAmount` backpressure, no handshake timeout (UI wedges "connecting"), WS leaks if unmounted mid-connect | `useVoiceChatStream.tsx`, `usePersonaLiveSession.tsx`, `usePersonaLiveControl.tsx` |
| R9 | Medium | Character cards: **PNG export fetches an attacker-controlled `avatar_url`** (tracking/SSRF beacon), avatar/import have no size caps, two unsynchronized favorite systems | `character-export.ts:269`, `AvatarField.tsx:118`, `Characters/utils.ts:516`, `CharacterSelect.tsx` vs `useCharacterCrud.tsx` |
| R10 | Low-Med | Swiping to a not-yet-persisted variant keeps the prior `serverMessageId` → later edit/delete hits the wrong server row | `message-variants.ts:75` |

Lower-severity items (empty-completion bubbles, greeting double-render, undo version guess, soft-delete visible window, headerless-WebM long recordings, play/pause races) are in the reviewer notes.

## Cross-cutting themes (Round 2)
1. **Duplicated/dead-tested code drifts** — `characterChatMode` exists in triplicate, and the *safe* copy (with an inactivity watchdog + recovery) is the unused one; tests pass against code that never runs (same trap as Round 1's dead auth stack / `extension/routes`).
2. **UI-index vs storage-index mismatch** — R1/R4: messages are addressed by array position; a greeting at index 0 desyncs UI from Dexie. Root fix is to address by stable message id.
3. **Lifecycle cleanup only covers the happy path** — mic tracks, blob URLs, and WebSockets leak on error/race/unmount paths. `useMicStream.ts` is the correct template the leaky sites should follow.
4. **WS token in URL** is systemic across every voice/STT path.

## What's solid (verified OK)
No XSS in character/card rendering; character card image decode validates magic bytes + MIME allow-list; download filenames are traversal-safe; no ReDoS from lorebook regex; **no duplicate assistant persistence** (the server skips persistence while streaming and the client persists once, idempotently); cross-character session switching resets state correctly; TTS streaming chunk ordering is correct; `useMicStream.ts` teardown is complete on every path; no zero-delay reconnect loops.

## R2 remediation status
Backlog tasks `task-12107`…`task-12113` (renumbered off `task-12104`–`12106`, which teammates used for PR-2573 work).

**Fixed (with tests):**
- **R1 (task-12111)** — character delete/edit now address the Dexie row by **stable message id**, not array position, so a greeting at UI index 0 no longer corrupts the store. *Deferred (AC#3):* the greeting is still not persisted to Dexie, so a Dexie-sourced rehydrate shows one fewer message than a server-sourced one — cosmetic, not corruption.
- **R2 (task-12112)** — the three mic-capture sites now match `useMicStream` (synchronous re-entry guard, stream held in a `catch`-reachable ref, `MediaRecorder` `onerror` that stops tracks + releases the capture lock).
- **R6/R7 (task-12107)** — TTS blob-URL/MediaSource leaks freed on every path; overlapping playback fixed; audiobook cancel aborts the in-flight chapter.
- **R8 (task-12109)** — real-time voice hardening: barge-in stops TTS + single interrupt, `bufferedAmount` backpressure, handshake timeout, unmount-mid-connect WS-leak guards.
- **R9/R10 (task-12110)** — card handling: export SSRF guard (same-origin/allowlisted + timeout + 5 MB cap), avatar/import size caps, favorite reconciled to the server flag with the correct cache key, and a swiped-but-unpersisted variant no longer inherits a stale `serverMessageId`.

**Partially done / gated:**
- **R3 (task-12113)** — WS token **moved out of the URL** (persona subprotocol `["bearer",cred]`; audio/STT `{type:"auth"}` first message), with a charset fallback so a non-token-safe custom key can't crash `new WebSocket`. ⚠️ **Needs a live-server smoke before merge** (subprotocol handshake, single-user key charset, auth-before-config ordering, extension STT).
- **R5 (task-12108)** — the high-value **stream-inactivity watchdog** (60s) is now on both live character paths + recovery classification; the full 3-copy consolidation is intentionally deferred (a large refactor while dev is actively churning chat).

**Not done (deliberately deferred, see the running commentary):** TS-strict enablement (blocked on ~66 pre-existing type errors — a real migration, not a flag flip); dropping CSP `'unsafe-eval'` (needs WASM/OCR browser verification); deleting the `extension/routes` mirror (kept in sync by ~22 parity tests; negative-value churn during dev's route refactor).
