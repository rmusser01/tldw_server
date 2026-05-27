# Chat Rails UX Rebaseline Audit - 2026-05-27

## Executive Summary

Top 5 UX risks after re-auditing the corrected rail-enabled `/chat` page:

1. First send is still blocked in the observed local WebUI state: `/chat` shows `No LLM provider configured` while also showing a `tldw:gpt-4o` route as ready/active.
2. The extension sidepanel chat debug surface has horizontal overflow at a 390 px mobile-width viewport, so the directly connected chat entry surface is not as stable as the WebUI `/chat` page.
3. The main `/chat` cockpit rails are restored and the removed standalone character-control rail remains absent, but the first-run cockpit still presents setup, rail, empty-state, and composer controls all at once.
4. Send controls and hidden/secondary controls are not fully unambiguous to automation or assistive tech: exact `Send` role lookup returns two matches, while the primary send has a clearer `Send message` accessible name.
5. Advanced chat workflows still need separate remediation and green-path proof: provider/model readiness, Web search status feedback, character/persona clear state, and richer extension-to-WebUI transfer beyond the current route-only handoff are not fully validated by this slice.

## Evidence Notes

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`
- Branch: `codex/chat-rails-ux-rebaseline`
- Current audit pre-doc HEAD: `477744b47`
- Current local `origin/dev`: `7659953cf`
- Backend used for live audit: `http://127.0.0.1:8000`
- WebUI used for live audit: `http://127.0.0.1:18015`
- Backend health was confirmed with approved localhost access before the browser pass: status `ok`, auth mode `single_user`.
- Live browser evidence was captured from the dev WebUI, not from a packaged extension install.
- The sidepanel evidence uses `/__debug__/sidepanel-chat?nextgenComposer=1`, which is valid for directly connected sidepanel chat layout and handoff review but is not a full packaged-extension validation.
- Earlier no-rails evidence was stale or from the wrong page/branch. Current `/chat` evidence shows context and runtime cockpit rails present.

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

## First-Time User Walkthrough

| Step | Observation | Friction / opportunity | Evidence |
| --- | --- | --- | --- |
| Finds and opens `/chat` | `/chat` opens directly to the chat cockpit. Context and runtime rails are visible in cockpit mode. | The earlier missing-siderails report should be treated as branch/page provenance failure, not current route behavior. Keep the rail guards. | `first-time-unseeded.png`; `desktop-cockpit.png`; `Playground.cockpit-regression.guard.test.ts` |
| Understands what the page is for | Empty state says `Start a new chat` and mentions models, prompts, and knowledge sources. Rails expose context, prompt, model, MCP, runtime, and assistant state. | Purpose is understandable, but the first viewport is dense for a new user. The page shows advanced rail controls before the first successful send. | `first-time-unseeded.png` |
| Handles setup/model requirements | The page shows `No LLM provider configured` with `Open Settings` and `Refresh`, but the rails also show `Ready`, provider `tldw`, model `gpt-4o`, and route `tldw:gpt-4o`. | Confusing and unreliable feedback: the user cannot tell whether setup is incomplete or the selected route is usable. | `first-time-unseeded.png`; `mobile-send-state.png` |
| Starts first conversation | Composer is reachable in focus and cockpit states. On mobile, the composer remains below the rail panel and no WebUI `/chat` horizontal overflow was observed. | First send could not be completed in the current audit state because provider readiness blocked the path. Exact `Send` role lookup also produced two matches, which makes the action less clear for tests and assistive tech. | `mobile-focus.png`; `mobile-cockpit.png`; `mobile-send-state.png`; `evidence.json` |
| Understands loading, streaming, errors, response actions | The provider error card is visible and actionable. Streaming, stop, retry, regenerate, and response actions were not revalidated because no message was successfully sent. | The first-run green path remains unproven until provider readiness is made internally consistent or a known working local provider is seeded. | `mobile-send-state.png`; scope limitation |
| Discovers history/save/resume/context/persona/tools | Cockpit rails expose context and runtime/tool state. Header shows saved-state affordances. Sidepanel starter state includes save-to-history and composer controls. | Discovery exists, but setup failure competes with rail discovery. The sidepanel handoff route is fixed to `/chat` and now explicitly states that sidepanel draft/current-page/unsaved chat state stay in the sidepanel. | `desktop-cockpit.png`; `extension-sidepanel.png`; sidepanel route tests |

## Power-User Walkthrough

| Step | Observation | Friction / opportunity | Evidence |
| --- | --- | --- | --- |
| Starts or resumes quickly | Focus mode gives a cleaner writing surface, and cockpit mode restores context/runtime rails. | Fast switching works visually, but a blocked provider state prevents proving fast first-send/resume behavior. | `desktop-focus.png`; `desktop-cockpit.png` |
| Switches models/providers/settings | Runtime rail and model route show provider/model state, and model settings are reachable. | The route appears selected/ready while the global setup card says no provider is configured. Power users cannot trust the selected route. | `first-time-unseeded.png`; `mobile-cockpit-runtime.png` |
| Uses personas/characters/prompts/RAG/context/tools | Context and runtime rails are back. Runtime panel clearly says no persona or character will shape replies. | The corrected mobile runtime copy is better, but advanced state transitions still need live regression coverage for select/clear and Web search status feedback. | `mobile-cockpit-runtime.png`; prior real-server failures in `evidence.json` |
| Compares outputs or iterates across settings | Not reached in this corrected pass. | Compare/parallel-output and deep model-settings iteration should stay follow-up scope unless provider readiness becomes the same root blocker. | Scope limitation |
| Manages long sessions, failures, retries, context limits | Run controls and timeline sections are visible in the runtime rail. | Long-session behavior, context limits, retries, and recovery after provider errors were not exercised. | `mobile-cockpit-runtime.png`; scope limitation |
| Moves between extension and WebUI | Sidepanel full-screen handoff is covered by focused tests and targets `/chat`. The sidepanel starter UI renders. TASK-531 makes the route-only handoff explicit in the header and ControlRow affordances. | Sidepanel still has horizontal overflow at 390 px. Draft/thread/state preservation remains out of scope for the current handoff and should not be implied. | `extension-sidepanel.png`; `SidepanelHeaderSimple.fullscreen-route.test.tsx`; `ControlRow.role-play-handoff.test.tsx`; `evidence.json` |

## Severity-Ranked Findings

| ID | Severity: P0/P1/P2/P3 | Journey affected | Evidence | UX issue | User impact | Recommended solution | Effort: S/M/L | Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | P1 | First-time; power user | `first-time-unseeded.png`; `mobile-send-state.png`; runtime rail shows `Ready`/`tldw:gpt-4o` while setup card says `No LLM provider configured` | Confusing / unreliable: provider readiness and selected model state contradict each other. | Users cannot know whether to configure settings, change model, refresh, or send. First send is blocked. | Create a single chat readiness contract shared by the setup card, model rail, runtime rail, composer, and send button. If no provider is usable, show one blocking state and disable send with the exact reason. | M | High |
| F2 | P2 | First-time; accessibility; test automation | Playwright role check returned `Send` exact count 2 and `Send message` count 1 at mobile `/chat`; `mobile-send-state.png` | Inaccessible / confusing: send action naming is ambiguous when using exact role text. | Screen-reader users and automation can target the wrong send-related control; first-send tests become brittle. | Give the primary submit button one stable accessible name, for example `Send message`, and ensure the split-menu/dropdown control has a distinct name such as `Open send options`. | S | High |
| F3 | P2 | Extension handoff | Sidepanel debug route at 390 x 844 measured `innerWidth=390`, `documentElement.scrollWidth=420`, `body.scrollWidth=420`; `extension-sidepanel.png` | Inaccessible / poor workflow fit: directly connected sidepanel chat surface horizontally overflows on a common narrow viewport. | Extension users can get sideways scroll or clipped composer/header controls before handing off to WebUI chat. | Audit sidepanel header/composer fixed widths, icon rows, and bottom composer padding. Add a 390 px no-overflow Playwright guard for the sidepanel chat route. | S | High |
| F4 | P2 | First-time | `first-time-unseeded.png`; `desktop-cockpit.png` | Weak visual hierarchy: first view shows setup error, empty starter card, context rail, runtime rail, header controls, and composer controls at once. | New users must decide between setup, start chatting, quick ingest, rails, model settings, and MCP before the first successful message. | When setup is blocked, collapse advanced rails into summaries and focus the primary action on provider setup/refresh. Once ready, show the empty starter card and progressive rail discovery. | M | Medium-high |
| F5 | P2 | Power user | Prior real-server evidence retained in `evidence.json`: Web search status assertion failed after toggle | Weak feedback: context source state can change without an obvious status-strip confirmation. | Users can send with Web search or context state different from what they believe is active. | Add a composer/status-strip chip for active context sources and keep the real-server Web search status test as a gate. | S | Medium-high |
| F6 | P2 | Power user | Prior real-server evidence retained in `evidence.json`: model-list scope toggle was not visible | Hard to discover / missing: configured-vs-catalog model scope is not consistently exposed. | Users cannot confidently switch between configured models and broader provider/catalog entries. | Restore the scope toggle or replace it with an explicit segmented control, then update tests to the intended contract. | S | Medium-high |
| F7 | P2 | Power user; persona/character users | Prior real-server evidence retained in `evidence.json`: character clear state did not settle as expected; plain-chat create returned 422 | Unreliable: assistant/persona state transitions and plain-chat continuity are not proven stable. | Users may believe a persona is cleared when it is not, or hit a failed transition when returning to plain chat. | Unify assistant clear/select state across runtime rail, chat creation payloads, and overlay flows. Add one real-server green path for select, clear, and plain-chat return. | M | Medium |
| F8 | P3 | Extension handoff | `SidepanelHeaderSimple.fullscreen-route.test.tsx`; `ControlRow.role-play-handoff.test.tsx`; `extension-sidepanel.png` | Weak feedback, addressed in TASK-531: the handoff is route-only, with role-play route intent preserved when applicable; sidepanel draft/current-page/unsaved chat state stay in the sidepanel. | Users now get the contract from the handoff affordance, but richer transfer remains unavailable. | Keep the route-only copy/test guard for this release. Treat draft transfer, current-page context transfer, or thread resume as a larger product/architecture follow-up. | S | High |
| F9 | P3 | First-time; screen-reader users | Desktop cockpit still contains `No assistant selected` in more than one rail surface; mobile runtime copy now clarifies `No persona or character will shape replies.` | Weak visual hierarchy / inaccessible: repeated empty assistant labels are better than before but still scan-heavy on desktop. | Users hear/read similar empty states without clear priority. | Qualify repeated labels by region, for example `No runtime assistant selected` vs `No assistant attached to context`. | S | Medium |
| F10 | P3 | Mobile first-time | `mobile-cockpit.png`; `mobile-cockpit-runtime.png`; mobile rail panel and composer do not overlap, but rail panels occupy most of the viewport | Inefficient: mobile cockpit is usable but still dense. | Users can inspect rails, but chat content and composer context are visually compressed. | Keep cockpit available, but default mobile first-run to focus mode with a compact rail summary and explicit `Open cockpit` affordance. | S | Medium |

## Quick Wins

- Fix the provider-readiness copy so the setup card, runtime rail, model route, and composer all report the same state.
- Rename the send dropdown/split control so exact `Send` role lookup only targets the primary submit action.
- Add a sidepanel `390x844` no-horizontal-overflow test.
- Hide or collapse advanced rail detail when the page is in a setup-blocked state.
- Add a visible status chip for active Web search/context source state.
- Keep the current guard that proves the removed standalone character-control rail is absent from `/chat`.

## Larger Improvements

- Define a first-run `/chat` readiness model: no provider, provider configured, model selected, ready to send, streaming, errored, recoverable.
- Build one real-server green-path suite for `/chat`: configured provider, first send, streaming/stop/retry, Web search toggle, model switch, assistant select/clear, and sidepanel full-screen entry.
- Preserve the explicit route-only extension-to-WebUI handoff contract and decide later whether draft transfer, current-page context transfer, or conversation resume is worth the architecture work.
- Simplify mobile cockpit information architecture so focus mode is the default working mode and cockpit mode is the inspection/configuration mode.
- Unify persona/assistant state ownership so runtime rail, context rail, overlays, and chat create payloads cannot disagree.

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
- Open question: Which provider/model should the local default use when `tldw:gpt-4o` appears selected but no provider is configured?
- Assumption: The current provider-readiness mismatch is a frontend state/contract problem until proven otherwise by backend config evidence.
- Assumption: Prior real-server failures are still valid follow-up signals, but this corrected audit only revalidated the visible page and directly connected sidepanel route.
- Non-goal: This audit does not redesign app-wide navigation, history/sidebar architecture, prompt picker IA, compare/export/share, or backend provider setup.
