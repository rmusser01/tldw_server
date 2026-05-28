# Chat Rails UX Rebaseline Audit - 2026-05-27

## Executive Summary

Post-rebase reconciliation for `TASK-535`: the rows below were valid findings from the corrected rail-enabled audit, and the current branch now has focused component plus live real-server proof for the main `/chat` cockpit path. The current top UX risks are narrower:

1. The configured `/chat` cockpit rails are restored and regression-covered on desktop and mobile. The focused real-server subset now passes provider first send, mobile send, model-provider confidence, and strict slow-stream stop/regenerate proof.
2. Mobile cockpit mode had a real regression where the expanded composer toolbar could push the draft textarea under the rail panel. TASK-535 fixes that by constraining the mobile cockpit composer and suppressing redundant toolbar layout while keeping rail-triggered selector controllers mounted.
3. TASK-536 removes the global assistant setup modal from the `/chat` first-run path. First-time users can now reach the chat surface, while assistant setup remains available through the inline chat nudge.
4. The sidepanel-to-WebUI handoff is deliberately route-only. Users now get explicit copy, but draft transfer, current-page context transfer, and thread resume remain product/architecture follow-ups.
5. Live third-party Web search provider calls, long-session behavior, compare/export/share, and context-limit recovery remain outside this focused proof.

## Evidence Notes

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`
- Branch: `codex/chat-rails-ux-rebaseline`
- Original audit pre-doc HEAD: `477744b47`
- Reconciliation pre-doc HEAD: `4e7a4266e`
- Post-rebase proof base HEAD: `8a577770c`
- Current local `origin/dev`: `fa8e549c8`
- Branch divergence after rebase: ahead `29`, behind `0`
- Backend used for live audit: `http://127.0.0.1:8000`
- WebUI used for live audit: `http://127.0.0.1:18015`
- Post-rebase live proof backend: `http://127.0.0.1:18001`
- Post-rebase mock OpenAI backend: `http://127.0.0.1:18088`
- Post-rebase WebUI proof ports: `18017`, `18019`, `18021`, `18022`
- TASK-535 focused proof backend: `http://127.0.0.1:18023`
- TASK-535 focused proof WebUI: `http://localhost:18024`
- TASK-535 focused proof mock OpenAI: `http://127.0.0.1:18088`
- TASK-537 first-time screenshot backend: `http://127.0.0.1:18041`
- TASK-537 first-time screenshot WebUI: `http://localhost:18042`
- Backend health was confirmed with approved localhost access before the browser pass: status `ok`, auth mode `single_user`.
- Post-rebase provider proof confirmed `/api/v1/llm/providers` returned configured providers; the final live Playwright proof used the configured `tldw:gpt-4o` route backed by the mock OpenAI server.
- Initial live browser evidence was captured from the dev WebUI and debug sidepanel route.
- TASK-534 added packaged-extension evidence using `chrome-extension://<dynamic-id>/sidepanel.html#/chat`, plus the focused role-play handoff unit contract.
- Earlier no-rails evidence was stale or from the wrong page/branch. Current `/chat` evidence shows context and runtime cockpit rails present.
- TASK-536 changed the route contract so `/chat` bypasses the global assistant setup overlay, while the inline chat nudge remains available.
- TASK-537 refreshed first-time evidence from an unseeded browser storage state after that route change. The current capture opens `/chat` to the chat cockpit empty state with context/runtime rails visible and no `Build Your Assistant` global setup overlay.
- TASK-537 backend startup note: the first sandboxed backend attempt on `127.0.0.1:18041` exited with `[Errno 1] operation not permitted` while binding; the same command succeeded with approved elevated localhost binding and health returned `ok` in `single_user` mode.
- Reconciliation verification:
  - `bunx vitest run ... Playground.cockpit-controls ... PlaygroundSendControl.accessibility ... PlaygroundStatusStrip ... Playground.cockpit-regression.guard ... PlaygroundRuntimeInspector ... PlaygroundCompositionPreview ... playground-composition-preview ... ControlRow.role-play-handoff --reporter=verbose` passed: 8 files, 92 tests.
  - `bunx vitest run ChatModelSelectorDropdown.character-usability.test.tsx SidepanelHeaderSimple.fullscreen-route.test.tsx --reporter=verbose` passed: 2 files, 6 tests.
  - `bunx vitest run Playground.cockpit-a11y.test.tsx Playground.cockpit-shell.test.tsx --reporter=verbose` passed: 2 files, 39 tests.
  - `bunx vitest run src/routes/__tests__/sidepanel-chat.narrow-layout.contract.test.ts --reporter=verbose` passed: 1 file, 2 tests.
  - `npx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line --workers=1 --grep 'uses the running server|model provider confidence|selects and clears'` passed: 4 tests. Covered running-server cockpit/focus controls, tracked character clear/plain WebUI return, persona select/clear, and configured model provider first send.
  - `env TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://localhost:18024 TLDW_E2E_SERVER_URL=http://127.0.0.1:18023 TLDW_E2E_API_KEY=<local-e2e-api-key> TLDW_E2E_EXPECT_STREAMING_CONTROLS=true npx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line --workers=1 --grep 'uses the running server|keeps mobile cockpit|sends a real mobile|model provider confidence|captures streaming stop'` passed: 5 tests. Covered configured desktop cockpit/focus, mobile cockpit draft preservation, mobile first send, model-provider confidence, and strict slow-stream stop/regenerate proof.
  - `bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts` passed: 6 tests. Covered restored rails, removed standalone character rail, and the mobile cockpit composer containment guard.
  - `npx playwright test tests/e2e/sidepanel-chat-smoke.spec.ts --project=chromium-extension --reporter=line --workers=1` passed: 3 tests. Covered packaged extension `/chat` at 390 px, route-only full-screen handoff to `/options.html#/chat`, no standalone `CharacterControlRail`, and send/reply.
  - `bunx vitest run src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx --reporter=verbose` passed: 2 files, 6 tests. Covered role-play `/chat?mode=character&characterId=...` handoff and route-only header copy.
  - TASK-537 refreshed `first-time-unseeded.png` with Playwright Chromium clean browser storage against `http://localhost:18042/chat`: `overlayCount=0`, `setupCopy=0`, `chatInputCount=1`, `startChatCount=1`.
  - `bunx vitest run __tests__/app/app-layout.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx` passed for TASK-537: 2 files, 17 tests.

## Captured Artifacts

- First-time localStorage-cleared `/chat` after TASK-536/TASK-537: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/first-time-unseeded.png`
- Desktop cockpit: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-cockpit.png`
- Desktop focus: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-focus.png`
- Desktop configured conversation: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-conversation.png`
- Desktop context-collapsed side restore state: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-context-collapsed-side-only.png`
- Desktop model-provider conversation: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-model-provider-conversation.png`
- Desktop streaming stop clicked: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-streaming-stop-clicked.png`
- Desktop regenerate ready: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-regenerate-ready.png`
- Desktop regenerated response: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-regenerated-response.png`
- Mobile focus: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-focus.png`
- Mobile cockpit context panel: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit.png`
- Mobile cockpit runtime panel: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit-runtime.png`
- Mobile cockpit active draft after composer containment fix: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit-active-draft.png`
- Mobile configured send state: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-send-state.png`
- Extension sidepanel chat debug route: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png`
- Packaged extension sidepanel `/chat` handoff: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel-packaged-handoff.png`
- Structured evidence: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json`

## What Was Actually Tested Or Inspected

- Opened `/chat` on desktop and mobile viewports against a running local backend.
- Toggled desktop cockpit/focus and captured both states.
- Toggled mobile focus/cockpit and captured context/runtime rail panels.
- Verified the removed standalone `CharacterControlRail` text/control surface did not appear in captured `/chat` screenshots.
- Inspected first-time localStorage-cleared state; this is not a full first-install machine profile, but after TASK-536/TASK-537 it opens to the chat cockpit empty state with context/runtime rails visible and no global assistant setup overlay.
- TASK-536 added focused app-shell coverage proving `/chat` bypasses the global first-run overlay, plus inline chat-nudge coverage proving assistant setup remains available from the chat surface. TASK-537 added live screenshot evidence for that current route behavior.
- Verified configured desktop and mobile first-send paths against the local backend and mock OpenAI route.
- Verified strict slow-streaming evidence: a stop control became observable/clickable, regenerate became enabled after the first turn, and regenerate sent a second provider request.
- Found and fixed a mobile cockpit regression where the expanded next-gen composer toolbar could push the draft textarea under the rail panel.
- Opened the sidepanel chat debug route at 390 x 844 and measured horizontal overflow.
- Opened packaged extension sidepanel `/chat` at 390 px, verified no horizontal overflow, captured screenshot evidence, clicked the full-screen handoff, and verified it opened `/options.html#/chat`.
- Reused focused component regression evidence for role-play `/chat?mode=character&characterId=...` handoff.
- TASK-531 added focused regression coverage and accessible copy for the sidepanel-to-WebUI handoff contract: it opens `/chat`, carries role-play route intent only where applicable, and keeps sidepanel draft/current-page/unsaved chat state in the sidepanel.
- TASK-532 inspected completed remediation tasks `TASK-522` through `TASK-531`, ran focused current regression tests, and updated this audit so stale findings no longer drive duplicate work.
- TASK-533 rebased the branch onto `origin/dev`, refreshed live backend proof, found and fixed an assistant-clear race where the UI could clear visible assistant state before detaching the tracked server chat, and hardened the real-server persona selector proof against transient catalog load retries.

## Post-Remediation Reconciliation

| Finding | Current status | Evidence now | Residual risk / next proof |
| --- | --- | --- | --- |
| F1 provider/model readiness contradiction | Addressed by `TASK-522`, `TASK-525`, and `TASK-526`. | Current focused tests pass for standard readiness, provider setup blocking, setup recovery focus, runtime rail, status strip, and composition preview. Post-rebase real-server proof passed configured provider selection and first send. | None beyond keeping the focused live provider proof in regression rotation. |
| F2 ambiguous send controls | Addressed by `TASK-524`. | Current `PlaygroundSendControl.accessibility.test.tsx` passes; primary action remains `Send message`, adjacent trigger is `Open message delivery options`. | None beyond normal a11y regression coverage. |
| F3 sidepanel 390 px overflow | Addressed by `TASK-523` and packaged-smoked by `TASK-534`. | Current source contract passes for sidepanel shell/composer/control-row containment; packaged extension `/chat` smoke passed at 390 px and captured `extension-sidepanel-packaged-handoff.png`. | None beyond keeping the packaged smoke in regression rotation. |
| F4 setup-blocked first-run overload | Addressed by `TASK-526`; TASK-536 changed `/chat` first-run gating; TASK-537 refreshed the live first-time screenshot. | Current cockpit shell/a11y tests pass; setup-blocked mode suppresses the starter deck and collapses secondary rail detail while preserving restored rails. TASK-536 focused tests prove `/chat` bypasses the global assistant setup overlay and keeps the inline setup nudge available. TASK-537 live screenshot proof shows the current first-time `/chat` cockpit with context/runtime rails visible. | None beyond keeping the first-run route contract and screenshot path in regression rotation. |
| F5 active context/Web search feedback | Addressed by `TASK-527`. | Current status-strip tests pass for active context source chips and inactive-context suppression. | Live Web search provider behavior remains a separate proof item if required before release. |
| F6 configured/catalog model scope discoverability | Addressed by `TASK-528`. | Current selector and cockpit regression tests pass; post-rebase real-server proof selected a configured model and sent a provider-qualified request. | None beyond keeping the focused live provider proof in regression rotation. |
| F7 assistant clear and plain-chat return | Addressed by `TASK-530` and hardened by `TASK-533`. | Current cockpit-control tests pass for canonical assistant clear, legacy mirror clear, server metadata clear, persisted overlay clear, and `serverChatId` detach. Post-rebase real-server proof passed for tracked character clear/plain WebUI return and persona clear. | None beyond normal regression coverage. |
| F8 sidepanel handoff ambiguity | Addressed by `TASK-531` and packaged-smoked by `TASK-534`. | Current `SidepanelHeaderSimple.fullscreen-route` and `ControlRow.role-play-handoff` tests pass; packaged extension smoke verified the header opens `/options.html#/chat`; copy states that sidepanel draft/page/unsaved state stays in the sidepanel. | Draft/page/thread transfer remains a larger product decision, not a bug in the route-only contract. |
| F9 repeated empty assistant labels | Addressed by `TASK-529`. | Current runtime, composition preview, and cockpit a11y tests pass with region-specific empty labels. | None beyond normal a11y regression coverage. |
| F10 mobile cockpit density | Addressed by `TASK-521.3` and hardened by `TASK-535`. | Current cockpit a11y/shell tests pass; TASK-535 focused mobile E2E passes for context/runtime tabs, draft preservation, no rail/composer overlap, and rail-triggered prompt/search/model/tool dialogs. | Keep the mobile cockpit real-server test in regression rotation. |

## First-Time User Walkthrough

| Step | Observation | Friction / opportunity | Evidence |
| --- | --- | --- | --- |
| Finds and opens `/chat` | Configured `/chat` opens to the chat cockpit, and context/runtime rails are visible in cockpit mode. TASK-536/TASK-537 now prove unseeded `/chat` bypasses the global assistant setup overlay in both unit contract and live screenshot evidence. | The earlier missing-siderails report should be treated as branch/page provenance failure, not current route behavior. | `first-time-unseeded.png`; `desktop-cockpit.png`; `Playground.cockpit-regression.guard.test.ts`; `app-layout.test.tsx` |
| Understands what the page is for | Empty state says `Start a new chat` and mentions models, prompts, and knowledge sources. Rails expose context, prompt, model, MCP, runtime, and assistant state. | Purpose is understandable, but the first viewport is dense for a new user. The page shows advanced rail controls before the first successful send. | `PlaygroundEmpty.test.tsx`; `desktop-cockpit.png` |
| Handles setup/model requirements | Original audit showed setup copy and runtime/model readiness disagreeing. The remediation now routes setup, rail, status, composition, and send-blocked state through the same readiness interpretation. Assistant/persona setup is no longer a global blocker for `/chat`; it remains an inline nudge. | Current first-time screenshot still exposes model/provider readiness as a send blocker, but no longer blocks chat entry behind global assistant setup. | `TASK-522`; `TASK-525`; `TASK-526`; `TASK-536`; `TASK-537`; focused Vitest |
| Starts first conversation | Composer is reachable in focus and cockpit states. On mobile cockpit, the redundant composer toolbar is suppressed so the draft textarea remains below the rail panel. Configured desktop and mobile first-send paths pass. | Primary send and delivery options are distinct to assistive tech and tests. | `TASK-524`; `TASK-535`; `mobile-cockpit-active-draft.png`; focused Playwright |
| Understands loading, streaming, errors, response actions | Provider error/recovery states remain visible and actionable. TASK-535 strict slow-stream proof clicked stop and then regenerated through a second provider request. | Long-session, context-limit, and broad retry recovery remain outside this focused proof. | `TASK-522`-`TASK-526`; `TASK-535`; `desktop-streaming-stop-clicked.png`; `desktop-regenerated-response.png` |
| Discovers history/save/resume/context/persona/tools | Cockpit rails expose context and runtime/tool state. Header shows saved-state affordances. Sidepanel starter state includes save-to-history and composer controls. | Deep history search/resume remains outside TASK-535. | `desktop-cockpit.png`; `extension-sidepanel.png`; sidepanel route tests |

## Power-User Walkthrough

| Step | Observation | Friction / opportunity | Evidence |
| --- | --- | --- | --- |
| Starts or resumes quickly | Focus mode gives a cleaner writing surface, and cockpit mode restores context/runtime rails. Configured first send now passes against the local backend/mock provider. | Resume/deep history workflows remain outside TASK-535. | `desktop-focus.png`; `desktop-cockpit.png`; `desktop-conversation.png` |
| Switches models/providers/settings | Runtime rail and model route show provider/model state, and model settings are reachable. The configured/catalog scope control is now wired into the model selector. | Post-rebase real-server proof selected a configured model and sent through the provider-qualified route. | `TASK-528`; `TASK-533`; focused Vitest; live Playwright |
| Uses personas/characters/prompts/RAG/context/tools | Context and runtime rails are back. Runtime panel uses region-specific empty assistant copy. Assistant clear now clears canonical state, legacy character mirror state, server metadata, persisted assistant overlay settings, and tracked `serverChatId`. | Post-rebase real-server proof passed tracked character clear/plain WebUI return and persona select/clear. | `TASK-529`; `TASK-530`; `TASK-533`; focused Vitest; live Playwright |
| Compares outputs or iterates across settings | Not reached in this corrected pass. | Compare/parallel-output and deep model-settings iteration should stay follow-up scope unless provider readiness becomes the same root blocker. | Scope limitation |
| Manages long sessions, failures, retries, context limits | Run controls and timeline sections are visible in the runtime rail. Strict slow-stream proof clicked stop and verified regenerate after the first turn. | Long-session behavior, context limits, and recovery after provider errors were not exercised. | `mobile-cockpit-runtime.png`; `desktop-streaming-stop-clicked.png`; `desktop-regenerate-ready.png` |
| Moves between extension and WebUI | Sidepanel full-screen handoff is covered by focused tests and packaged extension smoke. The sidepanel starter UI renders at 390 px, the header opens `/options.html#/chat`, and TASK-531 makes the route-only handoff explicit in the header and ControlRow affordances. | Draft/thread/state preservation remains out of scope for the route-only handoff. | `TASK-523`; `TASK-531`; `TASK-534`; sidepanel tests |

## Severity-Ranked Findings And Current Status

| ID | Severity: P0/P1/P2/P3 | Journey affected | Evidence | UX issue | User impact | Recommended solution | Effort: S/M/L | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | Addressed; was P1 | First-time; power user | `TASK-522`, `TASK-525`, `TASK-526`, `TASK-533`; focused readiness tests and configured-provider live proof passed | Provider readiness and selected model state originally contradicted each other. | The original user impact was severe first-send uncertainty. | Keep the shared readiness contract and focused configured-provider live proof. | M | High |
| F2 | Addressed; was P2 | First-time; accessibility; test automation | `TASK-524`; current `PlaygroundSendControl.accessibility` passed | Primary send and adjacent options originally shared fuzzy `Send` naming. | Screen-reader and automation targeting could be ambiguous. | Keep `Send message` for submit and `Open message delivery options` for the adjacent trigger. | S | High |
| F3 | Addressed; was P2 | Extension handoff | `TASK-523`, `TASK-534`; sidepanel narrow-layout contract and packaged extension smoke passed | Sidepanel debug route originally overflowed at 390 px. | Extension users could get sideways scroll or clipped composer/header controls. | Keep min-width/overflow containment and packaged extension smoke in regression rotation. | S | High |
| F4 | Addressed; was P2 | First-time | `TASK-526`, `TASK-536`; current cockpit a11y/shell tests and app-shell first-run tests passed | Setup-blocked first view originally competed with starter deck and advanced rail detail; global assistant setup also blocked first-time `/chat` entry. | New users had too many primary choices before setup and could be diverted before reaching chat. | Keep setup recovery focus for provider/model blockers, but bypass the global assistant setup overlay on `/chat`; keep assistant setup as an inline nudge. | M | High |
| F5 | Addressed; was P2 | Power user | `TASK-527`; current status-strip tests passed | Active context/Web search state lacked obvious status-strip confirmation. | Users could send with different context state than expected. | Keep active source chips; run live Web search provider proof only if release scope requires it. | S | Medium-high |
| F6 | Addressed; was P2 | Power user | `TASK-528`, `TASK-533`; selector/cockpit tests and live configured-model selection passed | Configured-vs-catalog model scope was not consistently exposed. | Users could not confidently switch model scope. | Keep configured/catalog controls inside the selector and the live model-provider proof. | S | High |
| F7 | Addressed; was P2 | Power user; persona/character users | `TASK-530`, `TASK-533`; cockpit-control tests and live character/persona clear proof passed | Assistant/persona clear and plain-chat return were not stable/proven. | Users could think an assistant was cleared while stale metadata survived. | Keep overlay/server/canonical clear behavior and tracked-server-chat detach. | M | High |
| F8 | Addressed; residual P3 product follow-up | Extension handoff | `TASK-531`, `TASK-534`; current sidepanel handoff tests and packaged `/chat` header handoff passed | Handoff transfer semantics were previously implicit. | Users could assume unsaved sidepanel state moved into WebUI. | Keep route-only copy/test guard; treat draft/current-page/thread transfer as a separate architecture decision. | S for current contract; L for transfer | High |
| F9 | Addressed; was P3 | First-time; screen-reader users | `TASK-529`; current runtime/composition/a11y tests passed | Empty assistant labels repeated generic copy across regions. | Users heard/read similar labels without region priority. | Keep region-specific empty assistant labels. | S | High |
| F10 | Addressed; was P3 | Mobile first-time | `TASK-521.3`, `TASK-535`; current cockpit a11y/shell tests and focused mobile real-server E2E passed | Mobile cockpit panels and an expanded composer toolbar could compete for the same viewport. | Chat draft input could be pushed under the rail panel on small screens. | Keep compact mobile panel cap, accessible summary, and mobile cockpit composer suppression/containment. | S | High |

## Quick Wins

- Done: provider-readiness alignment, send-control accessible names, sidepanel narrow containment, setup-blocked first-run focus, active context/status chips, configured/catalog model scope, assistant clear continuity, region-specific assistant labels, route-only handoff copy, and restored rail absence guards.
- Done in TASK-535: focused real-server `/chat` workflow suite passed against a live backend and mock OpenAI provider, including configured first send, mobile cockpit draft preservation, model-provider confidence, and strict slow-stream stop/regenerate.
- Done in TASK-535: mobile cockpit composer containment suppresses redundant toolbar layout while keeping prompt/model/tool controllers mounted for rail-triggered dialogs.
- Done in TASK-534: packaged extension sidepanel smoke proves F3/F8 in the actual extension shell for `/chat`, with role-play route intent covered by the focused `ControlRow.role-play-handoff` unit contract.
- Done in TASK-536: `/chat` bypasses the global assistant setup first-run overlay, and the inline chat assistant setup nudge remains available.
- Done in TASK-537: refreshed the first-time unseeded `/chat` screenshot so the current evidence shows the chat cockpit empty state instead of the old global setup overlay.

## Larger Improvements

- Promote the focused TASK-535 real-server `/chat` subset into regular regression rotation and extend it only where product risk justifies it: live Web search provider call, long-session/context-limit recovery, compare/export/share.
- Preserve the explicit route-only extension-to-WebUI handoff contract and decide later whether draft transfer, current-page context transfer, or conversation resume is worth the architecture work.
- Treat default mobile first-run focus behavior as a separate product decision rather than mixing it with rail restoration.
- Continue converging persona/assistant state ownership so runtime rail, context rail, overlays, and chat create payloads cannot disagree.

## Suggested Ideal `/chat` Workflow

First-time user:

1. Open `/chat` and see one clear readiness state.
2. If setup is blocked, see the exact missing requirement, a primary settings action, and a return-to-chat path.
3. If ready, accept the default model or change it in one compact control.
4. Send a first message and see streaming, stop, retry, save, and error feedback in or near the composer.
5. Discover context/RAG, tools, and personas progressively after the first successful send.

Power user:

1. Open `/chat` directly into the last-used focus/cockpit mode with keyboard-friendly resume.
2. Switch provider/model scope with a visible configured/catalog control and immediate readiness feedback.
3. Toggle context sources, RAG, tools, and Web search with status-strip confirmation before sending.
4. Select, clear, or swap assistants/personas with a deterministic runtime rail state.
5. Move from sidepanel to WebUI `/chat` with an explicit statement that the current release carries route intent only; sidepanel draft, page context, and unsaved state remain in the sidepanel.

## Open Questions, Assumptions, And Non-Goals

- Assumption: Sidepanel-to-WebUI handoff is route-only for this release. Draft/context/thread transfer would be a separate product and storage-contract decision.
- Open question: Should mobile `/chat` default to focus mode for first-time users even when cockpit rails are available?
- Assumption: The TASK-535 live proof used the local mock OpenAI provider route for deterministic configured-provider first send and slow-stream stop/regenerate evidence; live third-party provider behavior can still differ.
- Non-goal: This audit does not redesign app-wide navigation, history/sidebar architecture, prompt picker IA, compare/export/share, or backend provider setup.
