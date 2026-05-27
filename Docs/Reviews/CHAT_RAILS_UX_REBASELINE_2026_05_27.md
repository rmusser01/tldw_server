# Chat Rails UX Rebaseline Audit - 2026-05-27

## Baseline

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`
- Branch: `codex/chat-rails-ux-rebaseline`
- Pre-artifact baseline/provenance capture commit: `69a80b4b5` (`git rev-parse --short HEAD` output captured before the audit artifact commit).
- Task 3 evidence capture HEAD: `65357cac1`.
- origin/dev: `efe42fe0c`
- Merge-base expectation: `git merge-base --is-ancestor origin/dev HEAD` produced no stdout and exited `0` during the pre-artifact baseline capture.
- Backend: `http://127.0.0.1:8000`; coordinator-confirmed healthy with approved localhost access.
- WebUI URL: Planned `http://localhost:18014`; Playwright-managed Next server could not start inside this sandbox because binding `127.0.0.1:18014` failed with `listen EPERM`.
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
- Live real-server Playwright run: attempted with `TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18014'`; it failed before tests because Next could not bind `127.0.0.1:18014` (`listen EPERM`).
- Focused static/source Playwright verification: the exact requested command also failed before tests because Playwright config tried to autostart Next on `0.0.0.0:8080` and hit `listen EPERM`. Re-running the same grep with `TLDW_WEB_AUTOSTART=false` and the fake e2e API key passed: 2 tests, 595 ms.
- Source/e2e hardening: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts` now includes `assertNoHorizontalOverflow(page)` and calls it at stable desktop cockpit/focus states and mobile focus/cockpit panel states.
- Evidence JSON: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json` records `backendAvailable: true` from coordinator-confirmed health, the sandbox curl failure, the Playwright bind failure, and viewport checks as blocked.
- Screenshot artifacts: not captured during Task 3 because the sandbox blocked the WebUI server bind. No screenshot paths in this report should be treated as live evidence until the real-server suite can run in an environment allowed to bind localhost.

## Prior Finding Reclassification

| ID | Prior finding | Current route/viewport | Classification | Evidence | Severity | First-plan eligible |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Mobile `/chat` horizontal overflow | `/chat`, `390x844` | Pending live recheck; e2e guard added in the real-server mobile cockpit/focus flow, but no live viewport result was captured because the sandbox blocked WebUI server binding. | `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`; `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json` | Unknown until live browser run | Yes, if the next live run finds overflow |
| C2 | First-run connection/setup feedback | | | | | |
| C3 | First-run control overload | | | | | |
| C4 | Dense settings modal | | | | | |
| C5 | Prompt picker empty state | | | | | |
| C6 | Compare disabled without reason | | | | | |
| C7 | Character/persona timeline ambiguity | | | | | |
| C8 | Search & Context preview opacity | | | | | |
| C9 | Extension full-screen/dashboard handoff | | | | | |
| C10 | Duplicate accessible sidebar labels | | | | | |

## Refreshed Findings

| ID | Severity | Journey | Route | Viewport | Evidence | UX issue | User impact | Recommended solution | Effort | Confidence | First-plan eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Notes

- Observed behavior: Source-level cockpit rail wiring was already covered by Task 2. Task 3 hardened the real-server spec so future successful runs assert no horizontal overflow at desktop initial cockpit, desktop focus, desktop return-to-cockpit, mobile initial focus, mobile cockpit context/runtime panels, mobile return-to-focus, and mobile return-to-cockpit.
- First-pass cockpit rail classification: Cockpit rail presence is guarded at source level and in the existing real-server spec assertions. This pass cannot classify the rendered live cockpit as healthy because the local sandbox blocked WebUI startup before browser navigation.
- Limitations: Backend health is coordinator-confirmed OK, but agent-side sandboxed curl and Playwright-managed Next startup are blocked. Live screenshots, live viewport overflow metrics, and full real-server pass/fail evidence were not produced.
- Non-goals: No product UI changes, backend setup, dependency installation, or screenshot fabrication in this Task 3 slice.
