# Chat Rails UX Rebaseline Audit - 2026-05-27

## Baseline

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`
- Branch: `codex/chat-rails-ux-rebaseline`
- Pre-artifact baseline/provenance capture commit: `69a80b4b5` (`git rev-parse --short HEAD` output captured before the audit artifact commit).
- Task 3 evidence capture HEAD: `122f3af64`.
- Task 5 audit refresh pre-commit HEAD: `c5e20d4dd`.
- origin/dev at baseline: `efe42fe0c`
- origin/dev at manual browser capture: `70a230aad` (branch had advanced upstream; resync is deferred until a clean task boundary).
- Local origin/dev at Task 5 review: `64c27d18b`; `git merge-base --is-ancestor origin/dev HEAD` exited `1`. Task 5 evidence is therefore time-scoped to the captured origin/dev-based branch state, not a claim about the latest moving `origin/dev` ref.
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

## Executive Summary

Top 5 UX risks after rebaselining the correct rail-enabled `/chat` page:

1. The missing-siderails observation was branch/page provenance, not the captured origin/dev-based `/chat` reality. Rails are present in this evidence set and covered by source guards, but the full real-server cockpit suite still has four non-rail workflow failures.
2. Character/persona continuity is the highest remaining workflow risk: clearing a character does not reliably update runtime state, and a character-control overlay path can fail plain-chat creation with a 422.
3. Advanced-state feedback is weak in places. The Web search toggle can become active without the status strip clearly reflecting it, and the model picker run missed the provider-scope toggle expected by the workflow.
4. First-time mobile `/chat` is usable without horizontal overflow, but the starter card, floating scroll affordance, and sticky composer compete for the same narrow viewport.
5. Extension full-screen handoff now targets `/chat`, but the directly connected sidepanel debug route requires explicit auth/config seeding for valid evidence, and the adjacent "Open dashboard" action still lands on `/flashcards`.

## Evidence Notes

- Tested live `/chat` through the WebUI dev server at `http://127.0.0.1:18014` against the healthy backend at `http://127.0.0.1:8000`.
- The upstream branch moved after the original capture. Treat this audit as evidence for the captured origin/dev-based branch state; Task 6 should rebase or refresh against latest `origin/dev` before final handoff.
- Inspected desktop cockpit, desktop focus, mobile focus, and mobile cockpit screenshots in `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/`.
- Captured the sidepanel chat debug route from a `390x844` viewport after seeding auth/config and forcing demo mode, matching the sidepanel smoke-test pattern. A plain screenshot command without seeding produced an all-background/setup-gated capture, so packaged-extension live validation remains unclaimed.
- Relied on source and test evidence where the live real-server run failed before completing a journey. Those failures are recorded in `evidence.json` and mapped below.
- Did not run a fresh first-install profile with no prior settings; first-run comments are based on the configured empty-chat state plus the setup fallback behavior observed when sidepanel auth/config was absent.

## Task 5 Evidence Status

- Sidepanel screenshot captured at `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png`.
- Sidepanel screenshot was taken from a `390x844` viewport as a full-page PNG; the file dimensions are `390x1139`.
- Sidepanel capture metrics: `innerWidth=390`, `documentElement.scrollWidth=390`, `body.scrollWidth=390`, horizontal overflow `false`.
- Visible sidepanel starter affordances: header sidebar toggle, shortcut/help/theme controls, TTS clips, Persona Garden, dashboard, full-screen, connected/demo status, starter prompts, save-chat toggle, image/voice/config/dictate/character controls, and send button.
- Source evidence confirms the full-screen sidepanel button now opens `/options.html#/chat`; the dashboard header button still opens `/options.html#/flashcards`.
- Evidence JSON now includes `extensionSidepanelCapture` and the sidepanel screenshot entry.

## First-Time User Walkthrough

| Step | Observation | Friction / opportunity | Evidence |
| --- | --- | --- | --- |
| Finds and opens `/chat` | The captured rail-enabled page opens with cockpit controls and an empty "Start a new chat" state. The earlier no-rails report came from stale/wrong branch/page evidence, not this rebaseline. | Keep rail source guards in place so the main branch cannot silently drop cockpit rails again. | `desktop-cockpit.png`; `Playground.cockpit-regression.guard.test.ts` |
| Understands purpose | Empty copy says users can experiment with models, prompts, and knowledge sources. Desktop cockpit also exposes context/runtime/character controls immediately. | The page is powerful but dense on first view; novice attention splits between empty card, model/status controls, and rails. | `desktop-cockpit.png`; `PlaygroundEmpty.tsx:160` |
| Setup requirements | Configured WebUI shows ready/connected state. Missing sidepanel auth/config fell back to setup/settings instead of sidepanel chat. | First-run setup still needs a clear route-specific recovery path: "configure server, then return to chat." | `evidence.json` `backendHealth`; sidepanel raw screenshot limitation |
| Starts first conversation | Composer remains reachable in focus and cockpit states. Mobile has no horizontal overflow. | On mobile, starter card and sticky composer compete vertically, and the floating scroll affordance can sit between the empty state and composer. | `mobile-focus.png`; `mobile-cockpit.png`; `Playground.tsx:3366` |
| Loading, streaming, errors | Full send/streaming was not newly revalidated in Task 5. Existing live cockpit run reached the backend but did not complete all workflow checks. | Do not claim end-to-end chat health until the four real-server failures are resolved and rerun. | `evidence.json` `realServerPlaywright` |
| History, save/resume, context/RAG/persona/tools | Rails expose context, runtime/tool, and character/persona affordances. Save/resume behavior is visible in sidepanel starter state. | Discoverability is present, but feedback after toggling or clearing advanced state is inconsistent. | `desktop-cockpit.png`; `extension-sidepanel.png`; e2e failures in `evidence.json` |

## Power-User Walkthrough

| Step | Observation | Friction / opportunity | Evidence |
| --- | --- | --- | --- |
| Starts/resumes quickly | Focus mode keeps chat/composer prominent; cockpit mode restores context/runtime rails. | Focus-mode copy says rails are hidden while the desktop character rail can remain visible outside the cockpit shell, making the mode boundary fuzzy. | `desktop-focus.png`; `PlaygroundCockpitShell.tsx:206`; `Playground.tsx:3437` |
| Switches models/providers | Model/provider workflows exist, but the real-server run missed the expected `model-list-scope-toggle`. | Provider scope needs either a visible toggle or a revised interaction contract; power users need confidence about configured vs catalog models. | `evidence.json` failure; `chat-cockpit.real-server.spec.ts` selector |
| Uses context/RAG/tools | Context and runtime rails are restored and visible. | Web search can be toggled without reliable status-strip confirmation. | `desktop-cockpit.png`; `evidence.json` failure |
| Uses personas/characters | Character rail is present on desktop and character controls are reachable in sidepanel. | Clearing character state and overlay-to-plain-chat continuity are unreliable in live workflow evidence. | `evidence.json` failures; `extension-sidepanel.png` |
| Long sessions and retries | The page has focus/cockpit mode switching and status surfaces, but Task 5 did not newly exercise long sessions, retries, export/share, or context limits. | Treat long-session management as not revalidated in this audit slice. | Scope limitation |
| Extension to WebUI | Full-screen handoff now targets `/chat`. Sidepanel starter state renders with direct chat affordances. | No state-preservation contract is proven for carrying a sidepanel draft/conversation into `/chat`; only route target is fixed. | `SidepanelHeaderSimple.tsx:96`; `extension-sidepanel.png` |

## Prior Finding Reclassification

| ID | Prior finding | Current route/viewport | Classification | Evidence | Severity | First-plan eligible |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Mobile `/chat` horizontal overflow | `/chat`, `390x844` mobile focus and mobile cockpit | Not reproduced on the captured rail-enabled page. Manual browser capture measured `innerWidth=390`, `documentElement.scrollWidth=390`, and `body.scrollWidth=390` in both mobile focus and mobile cockpit. Real-server e2e now also guards these states. | `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-focus.png`; `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit.png`; `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json` | Prior severity reduced; no overflow issue found in captured states | No |
| C2 | First-run connection/setup feedback | `/chat` configured empty state; sidepanel debug route without seeded config | Partially active. Configured WebUI has ready state, but the sidepanel debug route falls back to setup/settings without seeded auth/config. A clean first-install WebUI profile was not rerun in this slice. | `evidence.json` `backendHealth`; `extensionSidepanelCapture.rawScreenshotCommandResult` | P2 until clean first-run is revalidated | Yes, for route-specific setup return copy |
| C3 | First-run control overload | `/chat`, `1440x960` desktop cockpit; `390x844` mobile | Still active, reduced from "rails missing" risk to "dense first-view" risk. Rails are present, but novice empty state competes with context/runtime/character controls and composer controls. | `desktop-cockpit.png`; `mobile-focus.png`; `PlaygroundEmpty.tsx:166` | P2 | Yes |
| C4 | Dense settings modal | `/chat` model/settings flow | Not fully re-evaluated. The real-server run did exercise model/provider controls but failed on a missing scope toggle before a complete settings assessment. | `evidence.json` model-list scope-toggle failure | P3 pending targeted rerun | No |
| C5 | Prompt picker empty state | `/chat` prompt/picker workflow | Not re-evaluated in Task 5. No live prompt-picker state was captured. | Scope limitation | Unclassified | No |
| C6 | Compare disabled without reason | `/chat` compare/output iteration workflow | Not re-evaluated in Task 5. Compare/parallel-output workflow was not reached in the rail rebaseline. | Scope limitation | Unclassified | No |
| C7 | Character/persona timeline ambiguity | `/chat`, runtime rail and character control rail | Still active and higher confidence. Real-server workflow failed after clearing a selected character, and another character-control overlay workflow failed plain-chat creation with 422. | `evidence.json` failures; `chat-cockpit.real-server.spec.ts` clear-state and create-chat expectations | P1 | Yes |
| C8 | Search & Context preview opacity | `/chat`, context rail | Still active in a narrower form. Context rail exists, but Web search active state was not reliably reflected in the status strip after toggle. | `evidence.json` Web search status failure; `desktop-cockpit.png` | P2 | Yes |
| C9 | Extension full-screen/dashboard handoff | Extension sidepanel chat header | Full-screen handoff fixed for `/chat`: the button now generates `/options.html#/chat` and focused coverage proves it. Dashboard remains active as a separate confusing affordance because "Open dashboard" still opens `/flashcards`. | `SidepanelHeaderSimple.tsx:96`; `SidepanelHeaderSimple.tsx:122`; `SidepanelHeaderSimple.fullscreen-route.test.tsx`; `extension-sidepanel.png` | P2 for dashboard ambiguity; full-screen subfinding fixed | Yes, once target is confirmed |
| C10 | Duplicate accessible sidebar labels | `/chat` context/runtime rails | Partially active from source evidence. "No assistant selected" is used by runtime inspector and composition/context summaries, which is semantically accurate but repetitive for scan and screen-reader output. | `PlaygroundRuntimeInspector.tsx:201`; `Playground.tsx:1994`; `playground-composition-preview.ts:173` | P3 | Yes |

## Refreshed Findings

| ID | Severity | Journey | Route | Viewport | Observed behavior | Prior classification | Evidence | UX issue | User impact | Recommended solution | Effort | Confidence | First-plan eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | P1 | Power user; first-time users who try personas | `/chat` | Desktop | Runtime rail did not settle on the expected cleared assistant state after a live clear flow. | C7 still active | `evidence.json` character clear failure; `chat-cockpit.real-server.spec.ts` clear-state expectations | Unreliable / confusing: clearing a selected character does not reliably update runtime rail state. | Users cannot trust whether a persona is still influencing the conversation. | Make character clear a single state transition shared by runtime inspector and character rail, show a deterministic cleared state, and keep the real-server clear test as a gate. | M | Medium-high | Yes |
| F2 | P1 | Power user | `/chat` | Desktop/mobile workflow | Character-control overlay continuity hit a 422 while creating a disposable plain chat. | C7 still active | `evidence.json` plain-chat creation 422 failure; `chat-cockpit.real-server.spec.ts` create helper expected 201 | Unreliable / poor workflow fit: overlay-to-plain-chat continuity can fail at chat creation. | Iterating between character and plain chats can break mid-session and undermine trust in save/resume behavior. | Align the frontend create payload with the live API contract or relax the backend requirement if the UI is valid; add user-facing recovery when creation fails. | M | High | Yes |
| F3 | P2 | First-time and power user | `/chat` | Desktop | Context rail Web search toggle changed state, but the expected status-strip confirmation did not appear. | C8 still active | `evidence.json` Web search status failure; `desktop-cockpit.png` | Weak feedback: context rail Web search state can change without clear status-strip confirmation. | Users may send with Web search on/off unintentionally. | Reflect active context sources in the status strip and composer-adjacent summary, or remove stale status assertions if the intended surface has moved. | S | High | Yes |
| F4 | P2 | Power user | `/chat` | Desktop | Real-server model picker did not expose the expected configured/catalog scope toggle. | C4 partially evaluated | `evidence.json` model-list scope-toggle failure; model selector expected `model-list-scope-toggle` | Missing / hard to discover: model picker did not expose the configured/catalog scope control expected by the workflow. | Provider switching confidence drops; users cannot tell whether they are choosing configured models or broad catalog entries. | Restore a visible scope toggle or replace it with an equally explicit segmented control and update tests to the chosen contract. | S | High | Yes |
| F5 | P2 | First-time user | `/chat` | Mobile `390x844` | Empty starter state, sticky composer, and scroll-to-latest control are all present in the first-run mobile viewport. | C3 still active | `mobile-focus.png`; `mobile-cockpit.png`; `Playground.tsx:3366`; `PlaygroundEmpty.tsx:166` | Weak visual hierarchy / inefficient: starter card, sticky composer, and scroll-to-latest affordance crowd the first-run mobile viewport. | New users can miss the starter action or believe the page starts halfway down. | Suppress scroll-to-latest affordance on empty chats, reduce first-run card height on mobile, and keep the composer dock from visually covering the empty state. | S | Medium-high | Yes |
| F6 | P2 | Power user | `/chat` | Desktop focus | Focus mode was visible while the character rail remains outside the cockpit shell and can still be shown on desktop. | New rail-enabled finding | `desktop-focus.png`; `PlaygroundCockpitShell.tsx:206`; `Playground.tsx:3437` | Confusing: focus-mode copy says rails are hidden while the desktop character rail can remain visible outside the cockpit shell. | Users cannot predict what focus mode hides and what remains active. | Rename copy to "Context and runtime rails hidden" or include the character rail in focus-mode hiding semantics. | S | High | Yes |
| F7 | P2 | Extension handoff | Sidepanel chat to options UI | `390x844` sidepanel | Sidepanel header exposes both full-screen and dashboard actions; full-screen targets `/chat`, dashboard still targets `/flashcards`. | C9 partially fixed | `extension-sidepanel.png`; `SidepanelHeaderSimple.tsx:122`; `SidepanelHeaderSimple.tsx:257` | Confusing / poor workflow fit: "Open dashboard" from chat still lands on `/flashcards`. | Users leaving sidepanel chat can land in an unrelated workflow and lose chat intent. | Decide the intended dashboard target; likely route to `/chat` or a real dashboard. Keep `/flashcards` only if the label names flashcards. | S | High | Yes, after target decision |
| F8 | P3 | Extension handoff | Sidepanel debug route | `390x844` sidepanel | Unseeded debug-route screenshot produced an all-background/setup-gated capture; seeded debug-route capture rendered chat. | New evidence-process finding | `evidence.json` `extensionSidepanelCapture.rawScreenshotCommandResult`; `extension-sidepanel.png` | Missing evidence contract: raw debug-route screenshot needs auth/config seeding to render useful chat state. | Future audits can capture blank/setup images and mistake them for product state. | Add a small sidepanel evidence helper or Playwright audit fixture that seeds config and records whether it is debug-route or packaged-extension evidence. | S | High | Yes |
| F9 | P3 | First-time and screen-reader users | `/chat` cockpit | Desktop | Runtime inspector and composition/context summary paths reuse the same empty assistant string. | C10 partially active | `PlaygroundRuntimeInspector.tsx:201`; `Playground.tsx:1994`; `playground-composition-preview.ts:173` | Weak visual hierarchy / inaccessible: repeated "No assistant selected" text is reused across runtime and composition/context summaries. | Screen-reader and scan users hear/read the same empty state repeatedly without knowing which area matters. | Qualify empty labels by region, for example "No runtime assistant selected" and "No context assistant attached." | S | Medium | Yes |

## Quick Wins

- Change focus-mode summary copy to specify which rails are hidden, or hide the character rail consistently in focus mode.
- Suppress the scroll-to-latest floating button when the chat is empty.
- Add a visible status-strip chip for active Web search/context source state.
- Restore or replace the model-list scope toggle with an explicit configured/catalog control.
- Rename or retarget the sidepanel "Open dashboard" button so it does not silently open Flashcards.
- Add a seeded sidepanel screenshot helper to avoid blank/setup-gated audit captures.

## Larger Improvements

- Unify character/persona state across runtime inspector, character rail, overlay, and chat creation so clearing, switching, and returning to plain chat are atomic and recoverable.
- Define a route contract for extension handoff beyond the immediate `/chat` target if state preservation becomes a product requirement. Current evidence proves route target only, not draft/thread transfer.
- Create a dedicated first-run `/chat` onboarding state that prioritizes setup readiness, model choice, and the first send before exposing every advanced rail section.
- Add a real-server "chat green path" suite that must pass before future UX audits: configured empty state, first send/stream/stop/retry, Web search toggle, model switching, character select/clear, sidepanel full-screen.

## Suggested Ideal Workflow

First-time `/chat` user:

1. Open `/chat` and see a clear ready/not-ready state with the next required setup action.
2. If ready, pick or accept a default model in one compact control and type immediately.
3. Send the first message and see streaming, stop, retry, and save behavior in the composer/status area.
4. Discover rails progressively: context/RAG first, runtime/tools second, personas/characters as a distinct optional mode.
5. If coming from sidepanel, land on `/chat` with clear confirmation of whether the sidepanel thread/draft was carried over or not.

Power user:

1. Resume recent chats from `/chat` without leaving the keyboard.
2. Switch model/provider scope with an explicit configured/catalog control and visible readiness state.
3. Toggle context sources and tools with immediate status-strip and request-preview feedback.
4. Select, clear, or swap personas/characters without ambiguous leftover runtime state.
5. Use focus mode to protect the writing surface while keeping a predictable path back to cockpit rails.

## Open Questions, Assumptions, and Non-Goals

- Open question: Should the sidepanel "Open dashboard" action target `/chat`, a true dashboard route, or keep `/flashcards` with clearer labeling?
- Open question: Is extension-to-WebUI state preservation required for this release, or is route-target correctness sufficient?
- Assumption: The four real-server failures are captured baseline workflow issues, not caused by the new rail guard or sidepanel route fix.
- Assumption: The sidepanel debug route is acceptable fallback evidence when clearly labeled; packaged extension validation was not performed in Task 5.
- Non-goal: No broad WebUI redesign, no backend API changes, no prompt picker redesign, and no compare/export/share remediation in this audit slice.

## Task 6 Final Verification Status

- Rebased the branch onto `origin/dev` at `64c27d18b`; post-rebase branch HEAD was `13b8e5e61`, and `git merge-base --is-ancestor origin/dev HEAD` exited `0`.
- Focused rail/sidepanel verification passed after the rebase: 9 Vitest files, 104 tests.
- Backend health check passed against `http://127.0.0.1:8000/api/v1/health` with status `ok` and `auth_mode=single_user`.
- Real-server Playwright rerun reached the backend and WebUI after the rebase: 11 tests passed, 4 failed. The failures matched the audit findings: Web search status feedback, runtime character clear state, disposable plain-chat creation returning 422, and missing model-list scope toggle.
- Bandit is skipped for this final slice because the touched files are Markdown, JSON evidence, a screenshot artifact, and Backlog metadata.

## Notes and Limitations

- Observed behavior: Source-level cockpit rail wiring was already covered by Task 2. Task 3 hardened the real-server spec so future successful runs assert no horizontal overflow at desktop initial cockpit, desktop focus, desktop return-to-cockpit, mobile initial focus, mobile cockpit context/runtime panels, mobile return-to-focus, and mobile return-to-cockpit. Manual browser capture confirmed no horizontal overflow in the four required audit states.
- First-pass cockpit rail classification: Cockpit rail presence is restored/available on the captured origin/dev-based `/chat` page, and focused tests still pass after rebasing onto latest `origin/dev`. Desktop cockpit, desktop focus, mobile focus, and mobile cockpit screenshots now exist in the review asset directory.
- Why the earlier audit saw no siderails: this branch's source/evidence shows the cockpit rails were present on the captured origin/dev-based `/chat` page. The missing-rails observation came from stale/wrong branch or wrong page evidence rather than the rail-enabled development branch state captured here.
- Limitations: The full real-server Playwright suite still has four non-rail baseline failures and should not be reported as fully passing. Agent-side sandboxed curl still fails without approved localhost access. The sidepanel screenshot is debug-route evidence with seeded auth/config, not a packaged-extension browser session.
- Non-goals: No product UI changes, backend setup, dependency installation, or screenshot fabrication in the Task 3 slice. Task 4 does not change the `/flashcards` dashboard route; that cleanup remains separate unless later evidence confirms it belongs in a follow-up fix.
