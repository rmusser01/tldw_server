# Chat Rails UX Rebaseline Audit - 2026-05-27

## Baseline

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`
- Branch: `codex/chat-rails-ux-rebaseline`
- Pre-artifact baseline/provenance capture commit: `69a80b4b5` (`git rev-parse --short HEAD` output captured before the audit artifact commit).
- Task 3 evidence capture HEAD: `122f3af64`.
- origin/dev at baseline: `efe42fe0c`
- origin/dev at manual browser capture: `70a230aad` (branch had advanced upstream; resync is deferred until a clean task boundary).
- Merge-base expectation: `git merge-base --is-ancestor origin/dev HEAD` produced no stdout and exited `0` during the pre-artifact baseline capture.
- Backend: `http://127.0.0.1:8000`; coordinator-confirmed healthy with approved localhost access.
- WebUI URL: `http://127.0.0.1:18014`
- Rail source files:

```text
apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundRailSection.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx
```

## Required Evidence

- Desktop cockpit screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-cockpit.png`
- Desktop focus screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-focus.png`
- Mobile focus screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-focus.png`
- Mobile cockpit screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit.png`
- Extension sidepanel screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png`
- Evidence JSON: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json`

## Task 3 Evidence Status

- Backend health: coordinator-confirmed OK via `curl -sf http://127.0.0.1:8000/api/v1/health`; response status was `ok`, `auth_mode` was `single_user`, and database, metrics, and ChaChaNotes checks were healthy.
- Sandboxed backend health attempt: `curl -sf http://127.0.0.1:8000/api/v1/health` exited `7` with empty stdout/stderr from this agent context. Treat this as sandbox-localhost access failure, not backend downtime.
- Live real-server Playwright run: first blocked inside the sandbox on `listen EPERM`; rerun with approved localhost access reached the real backend and WebUI. Result: 11 passed, 4 failed. The failures were existing/non-rail assertions: stale Web search status-strip expectation, character clear state, disposable plain chat creation returning 422, and missing model-list scope toggle.
- Manual browser capture: started `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -H 127.0.0.1 -p 18014`, seeded the single-user server config, opened `/chat`, captured required desktop/mobile cockpit/focus screenshots, and measured no horizontal overflow in all four states.
- Focused static/source Playwright verification: the exact requested command also failed before tests because Playwright config tried to autostart Next on `0.0.0.0:8080` and hit `listen EPERM`. Re-running the same grep with `TLDW_WEB_AUTOSTART=false` and the fake e2e API key passed: 2 tests, 595 ms.
- Source/e2e hardening: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts` now includes `assertNoHorizontalOverflow(page)` and calls it at stable desktop cockpit/focus states and mobile focus/cockpit panel states.
- Evidence JSON: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json` records `backendAvailable: true`, the sandbox curl failure, the escalated real-server run result, manual browser capture details, and captured viewport metrics.
- Screenshot artifacts: captured at `desktop-cockpit.png`, `desktop-focus.png`, `mobile-focus.png`, and `mobile-cockpit.png`.

## Task 4 Evidence Status

- Pre-fix focused regression: `bunx vitest run src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx` failed because `browser.runtime.getURL` was called with `/options.html#/` instead of `/options.html#/chat`.
- Fix: `apps/packages/ui/src/components/Sidepanel/Chat/SidepanelHeaderSimple.tsx` now routes the sidepanel full-screen action to `browser.runtime.getURL("/options.html#/chat")`.
- Verification: `bunx vitest run src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.tts-clips-lazy-mount.test.ts` passed with 2 test files and 2 tests.
- Scope note: dashboard route cleanup remains outside this Task 4 fix. The existing dashboard button still targets `/options.html#/flashcards` and needs separate audit evidence before any change.

## Prior Finding Reclassification

| ID | Prior finding | Current route/viewport | Classification | Evidence | Severity | First-plan eligible |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Mobile `/chat` horizontal overflow | `/chat`, `390x844` mobile focus and mobile cockpit | Not reproduced on the rail-enabled page. Manual browser capture measured `innerWidth=390`, `documentElement.scrollWidth=390`, and `body.scrollWidth=390` in both mobile focus and mobile cockpit. Real-server e2e now also guards these states. | `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-focus.png`; `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit.png`; `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json` | Prior severity reduced; no current overflow issue found | No |
| C2 | First-run connection/setup feedback | | | | | |
| C3 | First-run control overload | | | | | |
| C4 | Dense settings modal | | | | | |
| C5 | Prompt picker empty state | | | | | |
| C6 | Compare disabled without reason | | | | | |
| C7 | Character/persona timeline ambiguity | | | | | |
| C8 | Search & Context preview opacity | | | | | |
| C9 | Extension full-screen/dashboard handoff | Extension sidepanel full-screen button | Still reproduced before Task 4: the full-screen action generated `/options.html#/`, landing at the options root instead of `/chat`. Fixed by Task 4: focused component coverage now proves the action opens `/options.html#/chat`. Dashboard route cleanup is not part of this immediate fix and remains separately unaudited. | `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx`; `apps/packages/ui/src/components/Sidepanel/Chat/SidepanelHeaderSimple.tsx`; focused Vitest command in Task 4 Evidence Status | Fixed for full-screen handoff; dashboard portion remains unclassified pending separate audit | No for full-screen handoff; dashboard cleanup requires separate scope |
| C10 | Duplicate accessible sidebar labels | | | | | |

## Refreshed Findings

| ID | Severity | Journey | Route | Viewport | Evidence | UX issue | User impact | Recommended solution | Effort | Confidence | First-plan eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Notes

- Observed behavior: Source-level cockpit rail wiring was already covered by Task 2. Task 3 hardened the real-server spec so future successful runs assert no horizontal overflow at desktop initial cockpit, desktop focus, desktop return-to-cockpit, mobile initial focus, mobile cockpit context/runtime panels, mobile return-to-focus, and mobile return-to-cockpit. Manual browser capture confirmed no horizontal overflow in the four required audit states.
- First-pass cockpit rail classification: Cockpit rail presence is restored/available on the proper `origin/dev`-based `/chat` page. Desktop cockpit, desktop focus, mobile focus, and mobile cockpit screenshots now exist in the review asset directory.
- Limitations: The full real-server Playwright suite still has four non-rail baseline failures and should not be reported as fully passing. Agent-side sandboxed curl still fails without approved localhost access.
- Non-goals: No product UI changes, backend setup, dependency installation, or screenshot fabrication in the Task 3 slice. Task 4 does not change the `/flashcards` dashboard route; that cleanup remains separate unless later evidence confirms it belongs in a follow-up fix.
