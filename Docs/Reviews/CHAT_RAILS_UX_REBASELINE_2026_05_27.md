# Chat Rails UX Rebaseline Audit - 2026-05-27

## Executive Summary

Post-remediation reconciliation for `TASK-532`: the rows below were valid findings from the corrected rail-enabled audit, but most have now been addressed by follow-up tasks on this branch. The current top UX risks are narrower:

1. The branch still needs a final real-server `/chat` green-path run after remediation, including configured provider first send, streaming, retry/stop, Web search status, model scope, and assistant clear/plain-chat return.
2. The sidepanel-to-WebUI handoff is deliberately route-only. Users now get explicit copy, but draft transfer, current-page context transfer, and thread resume remain product/architecture follow-ups.
3. Packaged-extension sidepanel validation should be refreshed before PR closeout. Source contracts and earlier 390 px checks pass, but this reconciliation did not rebuild and run the full packaged extension smoke.
4. The branch is ahead of `origin/dev` and also behind it; rebase/merge refresh is required before final PR verification.
5. The screenshots in this document are historical audit artifacts. They should not be read as current UI proof for rows now marked addressed; the current proof is in the focused tests and task records listed below.

## Evidence Notes

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`
- Branch: `codex/chat-rails-ux-rebaseline`
- Original audit pre-doc HEAD: `477744b47`
- Reconciliation pre-doc HEAD: `4e7a4266e`
- Current local `origin/dev`: `fa8e549c8`
- Branch divergence at reconciliation: ahead `28`, behind `3`
- Backend used for live audit: `http://127.0.0.1:8000`
- WebUI used for live audit: `http://127.0.0.1:18015`
- Backend health was confirmed with approved localhost access before the browser pass: status `ok`, auth mode `single_user`.
- Live browser evidence was captured from the dev WebUI, not from a packaged extension install.
- The sidepanel evidence uses `/__debug__/sidepanel-chat?nextgenComposer=1`, which is valid for directly connected sidepanel chat layout and handoff review but is not a full packaged-extension validation.
- Earlier no-rails evidence was stale or from the wrong page/branch. Current `/chat` evidence shows context and runtime cockpit rails present.
- Reconciliation verification:
  - `bunx vitest run ... Playground.cockpit-controls ... PlaygroundSendControl.accessibility ... PlaygroundStatusStrip ... ChatModelSelectorDropdown ... Playground.cockpit-regression.guard ... PlaygroundRuntimeInspector ... PlaygroundCompositionPreview ... playground-composition-preview ... SidepanelHeaderSimple.fullscreen-route ... ControlRow.role-play-handoff --reporter=verbose` passed: 10 files, 98 tests.
  - `bunx vitest run Playground.cockpit-a11y.test.tsx Playground.cockpit-shell.test.tsx --reporter=verbose` passed: 2 files, 39 tests.
  - `bunx vitest run src/routes/__tests__/sidepanel-chat.narrow-layout.contract.test.ts --reporter=verbose` passed: 1 file, 2 tests.

## Captured Artifacts

- First-time localStorage-cleared `/chat`: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/first-time-unseeded.png`
- Desktop cockpit: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-cockpit.png`
- Desktop focus: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-focus.png`
- Mobile focus: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-focus.png`
- Mobile cockpit context panel: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit.png`
- Mobile cockpit runtime panel: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit-runtime.png`
- Mobile send/recoverable blocked state: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-send-state.png`
- Extension sidepanel chat debug route: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png`
- Structured evidence: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json`

## What Was Actually Tested Or Inspected

- Opened `/chat` on desktop and mobile viewports against a running local backend.
- Toggled desktop cockpit/focus and captured both states.
- Toggled mobile focus/cockpit and captured context/runtime rail panels.
- Verified the removed standalone `CharacterControlRail` text/control surface did not appear in captured `/chat` screenshots.
- Inspected first-time localStorage-cleared state; this is not a full first-install machine profile because the local backend/config remained available.
- Attempted the mobile first-send path; the UI was blocked by `No LLM provider configured`, so streaming, stop, retry, and final response actions were not revalidated in this pass.
- Opened the sidepanel chat debug route at 390 x 844 and measured horizontal overflow.
- Reused existing regression-test evidence for rail wiring and sidepanel full-screen handoff to `/chat`.
- TASK-531 added focused regression coverage and accessible copy for the sidepanel-to-WebUI handoff contract: it opens `/chat`, carries role-play route intent only where applicable, and keeps sidepanel draft/current-page/unsaved chat state in the sidepanel.
- TASK-532 inspected completed remediation tasks `TASK-522` through `TASK-531`, ran focused current regression tests, and updated this audit so stale findings no longer drive duplicate work.

## Post-Remediation Reconciliation

| Finding | Current status | Evidence now | Residual risk / next proof |
| --- | --- | --- | --- |
| F1 provider/model readiness contradiction | Addressed by `TASK-522`, `TASK-525`, and `TASK-526`. | Current focused tests pass for standard readiness, provider setup blocking, setup recovery focus, runtime rail, status strip, and composition preview. | Re-run the real-server `/chat` green path with a configured provider before PR closeout. |
| F2 ambiguous send controls | Addressed by `TASK-524`. | Current `PlaygroundSendControl.accessibility.test.tsx` passes; primary action remains `Send message`, adjacent trigger is `Open message delivery options`. | None beyond normal a11y regression coverage. |
| F3 sidepanel 390 px overflow | Addressed by `TASK-523`. | Current source contract passes for sidepanel shell/composer/control-row containment; this session also ran a temporary rendered guard that passed at 390 px before deleting the duplicate exploratory test. | Re-run packaged extension sidepanel smoke after extension build refresh. |
| F4 setup-blocked first-run overload | Addressed by `TASK-526`. | Current cockpit shell/a11y tests pass; setup-blocked mode suppresses the starter deck and collapses secondary rail detail while preserving restored rails. | Browser screenshot refresh can confirm the improved first-run hierarchy after rebase. |
| F5 active context/Web search feedback | Addressed by `TASK-527`. | Current status-strip tests pass for active context source chips and inactive-context suppression. | Re-run the real-server Web search toggle assertion that originally failed. |
| F6 configured/catalog model scope discoverability | Addressed by `TASK-528`. | Current selector and cockpit regression tests pass; model selector renders existing configured/catalog controls. | Re-run the real-server selector interaction that originally could not find the scope toggle. |
| F7 assistant clear and plain-chat return | Addressed by `TASK-530`. | Current cockpit-control tests pass for canonical assistant clear, legacy mirror clear, server metadata clear, and persisted overlay clear. | Run the updated real-server character/persona clear journey against a live backend. |
| F8 sidepanel handoff ambiguity | Addressed by `TASK-531`. | Current `SidepanelHeaderSimple.fullscreen-route` and `ControlRow.role-play-handoff` tests pass; copy states that sidepanel draft/page/unsaved state stays in the sidepanel. | Draft/page/thread transfer remains a larger product decision, not a bug in the route-only contract. |
| F9 repeated empty assistant labels | Addressed by `TASK-529`. | Current runtime, composition preview, and cockpit a11y tests pass with region-specific empty labels. | None beyond normal a11y regression coverage. |
| F10 mobile cockpit density | Mitigated by `TASK-521` / commit `5f4d9d5b3`. | Current cockpit a11y/shell tests pass; mobile panel cap is compact and summary is accessible without occupying visible space. | Optional screenshot refresh after rebase if mobile hierarchy is part of final PR evidence. |

## First-Time User Walkthrough

| Step | Observation | Friction / opportunity | Evidence |
| --- | --- | --- | --- |
| Finds and opens `/chat` | `/chat` opens directly to the chat cockpit. Context and runtime rails are visible in cockpit mode. | The earlier missing-siderails report should be treated as branch/page provenance failure, not current route behavior. Keep the rail guards. | `first-time-unseeded.png`; `desktop-cockpit.png`; `Playground.cockpit-regression.guard.test.ts` |
| Understands what the page is for | Empty state says `Start a new chat` and mentions models, prompts, and knowledge sources. Rails expose context, prompt, model, MCP, runtime, and assistant state. | Purpose is understandable, but the first viewport is dense for a new user. The page shows advanced rail controls before the first successful send. | `first-time-unseeded.png` |
| Handles setup/model requirements | Original audit showed setup copy and runtime/model readiness disagreeing. The remediation now routes setup, rail, status, composition, and send-blocked state through the same readiness interpretation. | Current component proof is strong; live first-run screenshot should be refreshed after rebase to replace the historical stale capture. | `TASK-522`; `TASK-525`; `TASK-526`; focused Vitest |
| Starts first conversation | Composer is reachable in focus and cockpit states. On mobile, the composer remains below the rail panel and no WebUI `/chat` horizontal overflow was observed. | Primary send and delivery options are now distinct to assistive tech and tests. A configured-provider live first-send pass remains the missing end-to-end proof. | `TASK-524`; `mobile-focus.png`; `mobile-cockpit.png`; focused Vitest |
| Understands loading, streaming, errors, response actions | Provider error/recovery states remain visible and actionable. | Streaming, stop, retry, regenerate, and response actions still need a final real-server green-path rerun after the remediation stack. | `TASK-522`-`TASK-526`; scope limitation |
| Discovers history/save/resume/context/persona/tools | Cockpit rails expose context and runtime/tool state. Header shows saved-state affordances. Sidepanel starter state includes save-to-history and composer controls. | Setup-blocked state is now less competing, but final browser screenshots should prove the hierarchy on the rebased branch. | `desktop-cockpit.png`; `extension-sidepanel.png`; sidepanel route tests |

## Power-User Walkthrough

| Step | Observation | Friction / opportunity | Evidence |
| --- | --- | --- | --- |
| Starts or resumes quickly | Focus mode gives a cleaner writing surface, and cockpit mode restores context/runtime rails. | Fast switching works visually, but a blocked provider state prevents proving fast first-send/resume behavior. | `desktop-focus.png`; `desktop-cockpit.png` |
| Switches models/providers/settings | Runtime rail and model route show provider/model state, and model settings are reachable. The configured/catalog scope control is now wired into the model selector. | Re-run the real-server selector path before closeout because the original failure came from browser interaction evidence. | `TASK-528`; focused Vitest; `mobile-cockpit-runtime.png` |
| Uses personas/characters/prompts/RAG/context/tools | Context and runtime rails are back. Runtime panel uses region-specific empty assistant copy. Assistant clear now clears canonical state, legacy character mirror state, server metadata, and persisted assistant overlay settings. | The updated real-server clear/plain-chat journey should be executed against a live backend. | `TASK-529`; `TASK-530`; focused Vitest |
| Compares outputs or iterates across settings | Not reached in this corrected pass. | Compare/parallel-output and deep model-settings iteration should stay follow-up scope unless provider readiness becomes the same root blocker. | Scope limitation |
| Manages long sessions, failures, retries, context limits | Run controls and timeline sections are visible in the runtime rail. | Long-session behavior, context limits, retries, and recovery after provider errors were not exercised. | `mobile-cockpit-runtime.png`; scope limitation |
| Moves between extension and WebUI | Sidepanel full-screen handoff is covered by focused tests and targets `/chat`. The sidepanel starter UI renders. TASK-531 makes the route-only handoff explicit in the header and ControlRow affordances. | The 390 px overflow has source-contract and earlier rendered proof, but packaged extension smoke should be rerun after build refresh. Draft/thread/state preservation remains out of scope for the route-only handoff. | `TASK-523`; `TASK-531`; sidepanel tests |

## Severity-Ranked Findings And Current Status

| ID | Severity: P0/P1/P2/P3 | Journey affected | Evidence | UX issue | User impact | Recommended solution | Effort: S/M/L | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | Addressed; was P1 | First-time; power user | `TASK-522`, `TASK-525`, `TASK-526`; current focused readiness tests passed | Provider readiness and selected model state originally contradicted each other. | The original user impact was severe first-send uncertainty. | Keep the shared readiness contract; close with a configured-provider real-server green path. | M | High |
| F2 | Addressed; was P2 | First-time; accessibility; test automation | `TASK-524`; current `PlaygroundSendControl.accessibility` passed | Primary send and adjacent options originally shared fuzzy `Send` naming. | Screen-reader and automation targeting could be ambiguous. | Keep `Send message` for submit and `Open message delivery options` for the adjacent trigger. | S | High |
| F3 | Addressed; was P2 | Extension handoff | `TASK-523`; current sidepanel narrow-layout contract passed | Sidepanel debug route originally overflowed at 390 px. | Extension users could get sideways scroll or clipped composer/header controls. | Keep min-width/overflow containment and rerun packaged extension smoke before closeout. | S | High |
| F4 | Addressed; was P2 | First-time | `TASK-526`; current cockpit a11y/shell tests passed | Setup-blocked first view originally competed with starter deck and advanced rail detail. | New users had too many primary choices before setup. | Keep setup recovery focus: suppress starter deck, collapse secondary rail detail, preserve rail affordances. | M | Medium-high |
| F5 | Addressed; was P2 | Power user | `TASK-527`; current status-strip tests passed | Active context/Web search state lacked obvious status-strip confirmation. | Users could send with different context state than expected. | Keep active source chips and rerun the real-server Web search status assertion. | S | Medium-high |
| F6 | Addressed; was P2 | Power user | `TASK-528`; current selector/cockpit tests passed | Configured-vs-catalog model scope was not consistently exposed. | Users could not confidently switch model scope. | Keep configured/catalog controls inside the selector and rerun the real-server selector path. | S | Medium-high |
| F7 | Addressed; was P2 | Power user; persona/character users | `TASK-530`; current cockpit-control tests passed | Assistant/persona clear and plain-chat return were not stable/proven. | Users could think an assistant was cleared while stale metadata survived. | Keep overlay/server/canonical clear behavior and run the updated real-server clear/plain-chat journey. | M | Medium |
| F8 | Addressed; residual P3 product follow-up | Extension handoff | `TASK-531`; current sidepanel handoff tests passed | Handoff transfer semantics were previously implicit. | Users could assume unsaved sidepanel state moved into WebUI. | Keep route-only copy/test guard; treat draft/current-page/thread transfer as a separate architecture decision. | S for current contract; L for transfer | High |
| F9 | Addressed; was P3 | First-time; screen-reader users | `TASK-529`; current runtime/composition/a11y tests passed | Empty assistant labels repeated generic copy across regions. | Users heard/read similar labels without region priority. | Keep region-specific empty assistant labels. | S | High |
| F10 | Mitigated; was P3 | Mobile first-time | `TASK-521`; current cockpit a11y/shell tests passed | Mobile cockpit panels occupied too much viewport. | Chat content and composer context were visually compressed. | Keep compact mobile panel cap and accessible summary; refresh screenshot evidence if needed. | S | Medium-high |

## Quick Wins

- Done: provider-readiness alignment, send-control accessible names, sidepanel narrow containment, setup-blocked first-run focus, active context/status chips, configured/catalog model scope, assistant clear continuity, region-specific assistant labels, route-only handoff copy, and restored rail absence guards.
- Next small proof item: rerun the real-server `/chat` workflow suite against a live backend after rebasing, then update this document with current screenshots only if the browser evidence differs from component proof.
- Next extension proof item: rebuild/run packaged extension sidepanel smoke so F3/F8 are proven in the actual extension shell, not only source contracts and debug route evidence.

## Larger Improvements

- Build and keep one real-server green-path suite for `/chat`: configured provider, first send, streaming/stop/retry, Web search toggle, model switch, assistant select/clear, and sidepanel full-screen entry.
- Preserve the explicit route-only extension-to-WebUI handoff contract and decide later whether draft transfer, current-page context transfer, or conversation resume is worth the architecture work.
- If mobile cockpit still feels heavy after screenshot refresh, treat default mobile first-run focus behavior as a separate product decision rather than mixing it with rail restoration.
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
- Open question: Which configured provider/model fixture should the final real-server green-path proof use so first-send, streaming, retry, model switch, and assistant clear are deterministic?
- Assumption: Prior real-server failures are valid follow-up signals until the updated real-server suite is rerun against the current branch.
- Non-goal: This audit does not redesign app-wide navigation, history/sidebar architecture, prompt picker IA, compare/export/share, or backend provider setup.
