Method: dual-agent (A: /root/chat_reaudit_design · B: /root/chat_reaudit_evidence), with complementary authenticated walkthroughs by the coordinator.

# Chat UX reassessment — TASK-13709 — 2026-10-15

Reviewed `origin/dev` **2e48dc91848887dcb74ebe39d6ff877fed4d64bd**, after PR #3229 merged. Remote head was checked again before writing and remained identical. Scope: WebUI and extension full-page Chat, compact extension Chat, conversation setup/discovery/loading, and character roleplay setup/resumption. This is an assessment, not an implementation or release certification.

**Result: one P1 issue and five P2 issues.** No P0 issue was established. The main remaining weaknesses are connection-first guidance, compact-client draft control, and roleplay setup hierarchy. Existing conversation ownership/recovery safeguards should be preserved.

## Design specificity and overall impression

The interface fits tldw's research and character workflows: persistent context, identity, scenes, saved setups and distinct history locations are purposeful. The collapsed standard Chat is substantially easier to approach. The biggest opportunity is to make the next action follow the actual task prerequisite. A disconnected person needs connection setup before model, account-recovery or scene controls.

## Design health

Scores are expert judgments using Nielsen Norman Group's ten usability heuristics, not measured participant outcomes. Higher is better; 4 means excellent.

| # | Heuristic | Score /4 | Main observation |
|---|---|---:|---|
| 1 | Visibility of system status | 2 | Offline status conflicts with roleplay's model guidance. |
| 2 | Match with the real world | 2 | Connection, model, identity and scene prerequisites are mixed. |
| 3 | User control and freedom | 3 | Cancel, Return to chat, focus mode and draft safeguards help. |
| 4 | Consistency and standards | 3 | Shared UI is coherent; recovery destinations differ. |
| 5 | Error prevention | 3 | Account/ownership and unsupported-persona gates protect work. |
| 6 | Recognition rather than recall | 2 | History search helps; initial setup actions and roleplay identity can fall below the visible area. |
| 7 | Flexibility and efficiency | 3 | Shortcuts, focus mode, history search and reusable setups help experts. |
| 8 | Aesthetic and minimalist design | 2 | Collapsed Chat is quiet; expanded customization and roleplay setup are dense. |
| 9 | Error diagnosis and recovery | 2 | Helpful preservation copy, but a reproduced recovery path reaches another error. |
| 10 | Help and documentation | 3 | Tour, shortcuts and contextual hints exist. |
| | **Total** | **25/40** | **Acceptable; significant usability improvements remain.** |

Framework: [NN/G, 10 Usability Heuristics](https://www.nngroup.com/articles/ten-usability-heuristics/).

## What works

- A saved server conversation loaded its synthetic question/reply and remained selected after reload in the authenticated WebUI. Discovery showed All, Server and On this device views, search, type filtering, and 25-row pagination over 310 existing fixture conversations.
- Normal Server settings Save and Test Connection worked in the isolated authenticated WebUI profile. Inline setup preserves the current chat, and history recovery has distinct loading, changed-history and conflict actions in source.
- Both full-page clients share the calmer collapsed layout. Compact extension Character controls clearly disables unsupported tracked persona chat and explains overlay/character alternatives.

## Priority issues

### 1. [P1] Disconnected character chat sends the user into unavailable model settings

**Evidence:** Independently reproduced by both reviewers on fresh origins. `/chat` → Explore chat modes → Character chat / role-play. The picker reports account verification failure, while readiness asks for a model. Dismiss it and choose Open model settings: Current Chat Model Settings shows a scene/account-load error with Close as its only action. The separate Open settings link remains a workaround, so this is P1 rather than P0.

**Root:** `apps/packages/ui/src/components/Option/Playground/Playground.tsx:420` initializes health readiness to null. At `:2824`, anything except `blocked` is treated as connected. At `:3352`, the server-settings callback itself opens model settings; readiness actions dispatch through `:3375`. The genuine connection hook already exists on the page.

**Fix:** Derive readiness from current connection/account state plus the health snapshot. Recover the first missing prerequisite. Route server configuration to the same working inline Server settings dialog; only open model settings when that prerequisite is satisfied. Keep all ownership and Send guards. **Heuristics 1,2,4,9. Suggested workflow: impeccable harden.**

### 2. [P2] Extension starter prompts appear inserted, then disappear

**Evidence:** Freshly rebuilt production extension, isolated headless profile, connected compact Chat. Click “Summarize this page”: textarea value becomes `Summarize this page`. Open Character controls and dismiss it: textarea becomes empty. Reproduced again after waiting for asynchronous session loading. No messages were sent.

**Root:** `apps/packages/ui/src/components/Sidepanel/Chat/empty.tsx:178` assigns `input.value` and dispatches an input event. The textarea at `form.tsx:3793` is controlled by form state. The shortcut does not reliably update that state; a subsequent render restores its previous value.

**Fix:** Pass a draft/message setter to the empty state and use the existing form setter, as other insertions do. Verify both starter prompts remain through opening controls and can be submitted exactly once in a dedicated regression. **Heuristics 1,3,5,7. Suggested workflow: impeccable harden.**

### 3. [P2] First-run Chat emphasizes account failure before server setup

**Evidence:** At 1280×720, the unconfigured page combines “Choose a model,” a large account-verification error/retry block, draft-recovery controls, and a separate offline notice. Welcome actions fall below the visible transcript area. This is a hierarchy problem with a scroll workaround, not a claim of an inaccessible hidden control.

**Source:** `components/Chat/composer/ChatDraftRecovery.tsx:156`, `hooks/useChatDraftOwner.ts:41`, `components/Option/Playground/PlaygroundEmpty.tsx:134`, and `PlaygroundComposerNotices.tsx:733` under `apps/packages/ui/src/`.

**Fix:** Distinguish unconfigured from failed verification. Show one primary “Connect server” action and brief prerequisite copy until configured. Keep detailed account failure/retry for configured accounts. Reduce repeated notices and keep the next action visible at laptop height. **Heuristics 1,2,6,8,9. Suggested workflow: impeccable onboard.**

### 4. [P2] Roleplay setup puts empty preview and saving ahead of character choice

**Evidence:** Confirmed in the connected WebUI and shared full-page source. At 390×844, five empty preview rows plus Saved role-play setups consume most of the drawer before Character and Behavior. The footer offers Apply while lower scene controls require scrolling. Experienced users also repeat this scroll for a frequent identity change.

**Source:** `apps/packages/ui/src/components/Option/Playground/RolePlaySetupDrawer.tsx:1245` renders preview/reuse before character controls at `:1267`; generation and scene controls follow.

**Fix:** Lead with Character, show optional Behavior/Scene next, and collapse Preview changes and Saved setups until useful. Surface reuse compactly for returning users. Keep staged Apply/Cancel behavior and dirty-state protection. **Heuristics 6,7,8. Suggested workflow: impeccable distill.**

### 5. [P2] Roleplay preview lacks accessible before/after labels

**Evidence:** Source-confirmed semantic gap; physical screen-reader execution was not performed. Changed preview rows render two values separated only by an `aria-hidden` arrow. Their accessible text does not identify which is current and which will be applied. Long values also truncate.

**Source:** `apps/packages/ui/src/components/Option/Playground/RolePlaySetupPreview.tsx:59–72`.

**Fix:** Add “Current” and “After applying” labels, visually hidden where appropriate, and allow long values to wrap or expand. Validate accessible names and then VoiceOver/NVDA reading order. **Heuristics 1,3,5. Suggested workflow: impeccable harden.**

## Additional issue and improvements

**6. [P2, source-confirmed] Compact character history samples ordinary history before filtering.** `CharacterControlsSheet.tsx:103` requests the default first 25 chats with `filterMode: all`, then `:127` filters tracked sessions and takes six. Older tracked sessions disappear from this sheet when newer ordinary chats fill that page; there is no See all or pagination. It also discards query loading/error state, briefly reporting “No tracked … sessions yet” while loading (observed before the six real rows appeared). Request character-filtered history, show loading/error/stale states, and add See all or pagination. The general history browser is the workaround. Heuristics 1,6,7,9. The >25 ordinary-chat threshold is established from source, not a newly seeded server experiment.

Potential enhancements, not new confirmed functional failures:

- **Task language:** rename “Context sidechannel”/“Runtime rail” affordances to task terms such as Context and Chat settings; align compact first-run “Companion Home” copy with the actual Chat destination.
- **Customization:** retain the existing categories but reduce the roughly 18 visible actions after Customize chat opens. Offer a few common actions and a clearly named advanced section. Preserve direct shortcuts for experts.
- **History disambiguation:** extend character-session row metadata to the compact six-row list: updated time, selected state, and enough character/title information to distinguish duplicate titles.
- **Further validation:** the legacy sidepanel character picker drops list-query error state into an empty array (`Sidepanel/Chat/CharacterSelect.tsx:275,1454`). Its exact active legacy popover path was not fault-injected during this run; verify it before promoting it to a confirmed primary-path finding. The newer common picker already has error handling.

## Persona walkthrough and cognitive load

**Jordan, first timer:** the standard page looks calm but gives three competing next steps: choose model, retry account, connect server. Character mode compounds this by opening a scene error. The connection-first fix addresses the main emotional valley.

**Alex, power user:** search and pagination work; the tested saved chat survived reload. Friction remains in character changes buried below preview/reuse, and a compact resume list that omits older tracked chats.

**Sam, accessibility-dependent user:** labeled controls, polite status regions and semantic generation choices help. Before/after changes need explicit labels. Small recovery/header targets and zoom/physical assistive-technology behavior still require validation.

Cognitive load is moderate in collapsed standard Chat and high in the roleplay drawer. Expanded customization offers about 18 actions; the five chat-mode alternatives are acceptable as secondary discovery, but connection prerequisites should not compete with them. Roleplay currently fails single focus, one-decision-at-a-time, minimal choices and progressive disclosure.

## Evidence and boundaries

A and B ran independently; B's detector findings were withheld until A completed. Fresh hidden IAB tabs used separate origins after initial shared storage was noticed. Both tabs were closed. No visible Chrome windows were opened. Coordinator walkthroughs used separate temporary headless profiles.

The WebUI preview reused the previously qualified production artifact `djqgo1P99_kCc9XFkLCg9`: 101 tracked production deltas matched current source, 969 emitted JSON manifests referenced 1134 existing JS/CSS assets. It was not rebuilt in this assessment. The extension was freshly built from the reviewed source (`build:chrome:prod`, exit0,118s). Its temporary installed copy had fixture-only loopback host permission and seeded private test authentication; this does not certify the interactive browser-permission grant flow.

CLI detector ran once on the narrowed Playground/Sidepanel targets: **3 warnings, no errors**. `border-accent-on-rounded` at `PlaygroundForm.tsx:6267` and `side-tab` at `Sidepanel/Chat/RagSearchBar.tsx:937` are intentional state/context styling, not established defects. `broken-image` at `__tests__/AttachmentsSummary.integration.test.tsx:16` is a mocked-test false positive. No functional issue was inferred from those warnings. Read-only IAB evaluation cannot inject the live overlay; no overlay was claimed and no detector server started. The CLI browser path was unavailable; hidden IAB and installed Playwright were used.

The first coordinator authentication seed overwrote itself on navigation; that harness error was removed. Only the corrected normal Save/Test Connection and connected walkthrough count as evidence. Auth/fixture artifact files are private and contain no credential values in this report.

Original preservation baseline: **9 conversations / 28 messages, deeply equal before and after; baseline never rewritten**. Temporary preview PID46069 stopped and its process session exited0. Review-owned browser contexts closed. Existing backend, unrelated previews, user tabs and the primary checkout were untouched.

Not executed: generation/streaming reruns, fresh character-card import/export, physical touch/native zoom, VoiceOver/NVDA, real participant studies, full browser/OS matrix, Firefox extension, full CI or unit suite. Existing safety and automated evidence is not represented as new execution. Bandit is N/A: this task changes only review/tracking artifacts, with no Python or product-code changes. Review scripts/logs/screenshots remain private diagnostic evidence under `/private/tmp/chat-dev-reaudit-*`; no report-body temporary file is retained after archival.

ADR assessment: no new ADR is required; this report proposes local UX fixes and changes no durable architecture rule. Source ownership and persistence decisions remain in force. No product fixes, commit, PR or merge were made in this review task.
