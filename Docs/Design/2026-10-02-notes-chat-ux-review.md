# UX Review: /notes and /chat — WebUI & Browser Extension

**dev @ 86e287fee7 · 2026-10-02 · Sr. Design / HCI expert review**

**Tracking:** epic #3101 · defect groups #3102–#3124 · enhancement groups #3125–#3132 · plan: `Docs/superpowers/plans/2026-10-03-notes-chat-ux-remediation-plan.md`

## Table of contents

- [1. Executive summary](#1-executive-summary)
- [2. Scope, method & limitations](#2-scope-method--limitations)
  - [2.1 Personas](#21-personas)
  - [2.2 Surfaces](#22-surfaces)
  - [2.3 Live stack](#23-live-stack)
  - [2.4 Methods](#24-methods)
  - [2.5 Verification](#25-verification)
  - [2.6 Severity, priority and effort](#26-severity-priority-and-effort)
  - [2.7 Limitations and coverage gaps](#27-limitations-and-coverage-gaps)
- [3. Persona journeys](#3-persona-journeys)
  - [3.1 First-time user · Notes](#31-first-time-user--notes)
  - [3.2 First-time user · Chat](#32-first-time-user--chat)
  - [3.3 First-time user · Extension](#33-first-time-user--extension)
  - [3.4 Power user · Notes](#34-power-user--notes)
  - [3.5 Power user · Chat](#35-power-user--chat)
  - [3.6 Power user · Cross-page (chat ↔ notes, navigation, responsive)](#36-power-user--cross-page-chat--notes-navigation-responsive)
  - [3.7 Power user · Extension](#37-power-user--extension)
- [4. Prioritized issue list](#4-prioritized-issue-list)
- [5. Notes page — issues and solutions](#5-notes-page--issues-and-solutions)
  - [5.1 List, search & organization (NL)](#51-list-search--organization-nl)
  - [5.2 Editor (NE)](#52-editor-ne)
  - [5.3 Saving & data safety (NS)](#53-saving--data-safety-ns)
  - [5.4 Onboarding, layout & visual design (NO)](#54-onboarding-layout--visual-design-no)
- [6. Chat page — issues and solutions](#6-chat-page--issues-and-solutions)
  - [6.1 Sessions & history (CS)](#61-sessions--history-cs)
  - [6.2 Messages & generation (CM)](#62-messages--generation-cm)
  - [6.3 Composer, models & context (CC)](#63-composer-models--context-cc)
  - [6.4 Onboarding, layout & visual design (CO)](#64-onboarding-layout--visual-design-co)
- [7. Extension, cross-page workflows & accessibility — issues and solutions](#7-extension-cross-page-workflows--accessibility--issues-and-solutions)
  - [7.1 Extension sidepanel (XS)](#71-extension-sidepanel-xs)
  - [7.2 Cross-page workflows & extension parity (XP)](#72-cross-page-workflows--extension-parity-xp)
  - [7.3 Accessibility (AX)](#73-accessibility-ax)
- [8. Improvement opportunities (beyond defects)](#8-improvement-opportunities-beyond-defects)
  - [8.0 North star: one knowledge workflow (capture → organize → converse → distill)](#80-north-star-one-knowledge-workflow-capture--organize--converse--distill)
  - [8.1 Quick wins (days)](#81-quick-wins-days)
  - [8.2 Next (weeks)](#82-next-weeks)
  - [8.3 Strategic bets (larger redesigns)](#83-strategic-bets-larger-redesigns)
  - [8.4 Sequencing and dependencies](#84-sequencing-and-dependencies)
- [9. Strengths to preserve](#9-strengths-to-preserve)
  - [9.1 Notes data safety and recovery](#91-notes-data-safety-and-recovery)
  - [9.2 Performance and robustness](#92-performance-and-robustness)
  - [9.3 Finding and organizing notes](#93-finding-and-organizing-notes)
  - [9.4 Accessibility groundwork](#94-accessibility-groundwork)
  - [9.5 Chat answer quality and composer craft](#95-chat-answer-quality-and-composer-craft)
  - [9.6 Cross-surface plumbing that already works](#96-cross-surface-plumbing-that-already-works)
- [10. Terminology and UX-writing guide](#10-terminology-and-ux-writing-guide)
  - [10.1 Principles](#101-principles)
  - [10.2 Core objects and states (the glossary)](#102-core-objects-and-states-the-glossary)
  - [10.3 Chat controls and features](#103-chat-controls-and-features)
  - [10.4 Errors, status and recovery copy](#104-errors-status-and-recovery-copy)
  - [10.5 Notes-specific copy](#105-notes-specific-copy)
  - [10.6 Side panel and onboarding copy](#106-side-panel-and-onboarding-copy)
  - [10.7 Identifiers, numbers and dates](#107-identifiers-numbers-and-dates)
  - [10.8 Copy patterns and the lint list](#108-copy-patterns-and-the-lint-list)
- [11. Suggested remediation roadmap](#11-suggested-remediation-roadmap)
- [Appendix A. Findings rejected during verification](#appendix-a-findings-rejected-during-verification)
- [Appendix B. Evidence index](#appendix-b-evidence-index)

---

## 1. Executive summary

**First-time users.** Sam can find the basics on **/notes**: create a note, tag it, and find it again. But the first things a newcomer does can quietly destroy work. In WYSIWYG mode, text is typed backwards and then autosaved (NE-01). Leaving a note within five seconds of the last keystroke throws the edit away (NS-01). The tour hangs after step 6 of 8, and the empty state sits on top of a fully live editor. **/chat** is harder to learn. The default model is an unreachable provider marked "Healthy" (CC-01). The composer shows about twenty jargon controls. Every failure ends in the same unexplained "Turn needs review" panel (CC-02). Sam got an answer only after switching models, and could not regenerate, edit and resend, start a clean new chat, or find the chat again. Meanwhile the header said "Saved · Locally + Server" for a chat that was never saved (CS-03). The **extension side panel** opens to a dashboard of "Setup required" cards and has no default model. Its only model control is an unlabelled brain icon (XS-17, XS-02).

**Power users.** On **/notes**, Riley's search, import and rapid editing work well. The foundations (versioned autosave, undo plus Trash, real full-text search) are among the strongest parts of the product. They break at scale, though. The browse list, pager, total, Timeline and Graph only ever see the 100 most recent notes (NL-01). "Export matching notes" writes a file of duplicates that still misses notes, and reports success (NL-02). Bulk tagging replaces existing tags (NL-03). `[[Wikilinks]]` don't resolve (NE-02). On **/chat**, iterating is broken: Regenerate, Continue and Edit → "Save & Send" fail with raw error codes (CM-01). "New chat" and "Clear conversation" leak old context into the next request (CS-01, CS-05). The server prompt library is invisible (CC-05). History is hard to reach and search (CS-02). Restoring a chat from Trash brings it back empty (CS-N2). In the **side panel**, opening a past chat overwrites the current tab (XS-01), and continuing a chat on two surfaces silently forks it (XP-08). Notes is closer to ready: a few high-severity bugs with a small number of root causes undermine good foundations. Chat's problems are more systemic. Most of them trace back to one half-integrated history-selection controller and to three unreconciled copies of every conversation.

### Scorecard

Ratings reflect the confirmed issues and the strengths recorded in §9. IDs name the main drivers.

| Dimension | Notes | Chat | Extension sidepanel |
|---|---|---|---|
| Learnability | **Fair** (NO-02, NO-04, NL-12) | **Poor** (CO-03, CC-02, CC-04) | **Poor** (XS-17, XS-02, XS-12) |
| Efficiency | **Poor** (NL-01, NS-04, AX-03) | **Poor** (CS-02, CM-01, CC-05) | **Fair** (XS-06, XS-15) |
| Feedback & status | **Fair** (NS-06, NS-03) | **Poor** (CS-03, CC-01, CS-05) | **Poor** (XS-05, XS-07, XS-10) |
| Error prevention & recovery | **Fair** (NS-03, NS-N2; Undo + Trash are good) | **Poor** (CC-02, CM-02, CS-06) | **Poor** (XS-01, XS-11) |
| Data safety | **Poor** (NE-01, NS-01, NS-N1, NL-03) | **Poor** (CS-04, CS-N2, CS-01) | **Poor** (XS-01, XP-08, XS-07) |
| Consistency | **Fair** (XP-13, XP-17) | **Poor** (CC-08, CC-06, XP-13) | **Poor** (XP-12, XS-16) |
| Accessibility | **Poor** (AX-03, AX-05, AX-15) | **Fair** (AX-01, AX-04, AX-11) | **Fair** (AX-07, AX-11, AX-16) |
| Responsiveness (layout across widths and zoom) | **Fair** (NO-01, NO-03) | **Poor** (CO-01, CO-06, AX-04) | **Poor** (XS-03, XS-04, XS-15) |

### Top 10 issues to fix first

Chosen by priority and severity, by breadth (how many personas and surfaces are affected), and by effort: when two issues were similar in impact, the one with a shared root cause or a small fix ranked higher.

| # | ID(s) | Why first |
|---|---|---|
| 1 | **NE-01** | WYSIWYG typing produces reversed text and autosaves it. It corrupts data on the first keystroke for anyone who picks that mode. |
| 2 | **NS-01** | Edits made in the last five seconds are silently discarded on any in-app navigation, which is the most common way to leave a note. |
| 3 | **NL-01 + NL-02** | One list-API contract mismatch (`page`/`results_per_page` vs `limit`/`offset`) caps browsing at 100 notes and turns Export into a 1,000-request loop of duplicates. Fixing the contract once fixes both (S + M). |
| 4 | **CS-01 + CS-05** | "New chat" and "Clear conversation" don't reset the history-selection controller, so old messages are sent to the model or the send is destroyed. Users' natural workaround depends on this fix. |
| 5 | **CS-03** (with XS-05) | "Saved · Locally + Server" is shown for chats that are never written to the server. False reassurance about where data lives. |
| 6 | **CM-01** | Regenerate, Continue and Edit → "Save & Send" are visible on every chat and fail with raw error codes. Edits are silently lost. These are the core iteration actions. |
| 7 | **CS-02** | Users can't reliably find or return to past chats: history is buried, resets, searches titles only, and has no URL or ⌘K entry. |
| 8 | **XS-01** | Opening a past chat from side-panel search overwrites the current tab's conversation. Silent data loss on the extension's main history path. |
| 9 | **NS-N1** | The conflict toast's "Reload notes" doesn't reload, and the next autosave overwrites another tab's or device's changes (S). |
| 10 | **CC-02 + CM-N1** | The shared "Turn needs review" panel hides every failure cause and offers no Retry, and slow models hit it after 30 s on all three surfaces. One fix to the panel and the timeout repairs recovery everywhere (S + M). |

Next in line: XP-02 (Save to Notes missing on most server chats, S), NL-03 (bulk tagging replaces tags), CS-N2 (restore returns an empty chat), CS-04 (reload mid-reply erases the turn), XS-07 (fake "Delete") and NE-04 (Print always fails, S). The remaining P0s complete Stages 1-2 of the roadmap (§11): CC-04 (one click blanks the context), CC-05 (server prompts invisible), NE-02 (wikilinks broken) and XP-08 (silent forks between side panel and full page). CC-02 is the only P1 in the top 10; it ranks there because CM-N1 (P0) can't be fixed without it.

### Key themes

1. **Status copy reports intent, not acknowledged state.** "Saved · Locally + Server" (CS-03, XS-05), "Healthy" (CC-01, CC-N2), "Conversation cleared" (CS-05), "Chat restored." (CS-N2), "NOTES 100 TOTAL" (NL-01), a successful export of duplicates (NL-02), and "Delete — cannot be undone" that only closes a tab (XS-07). Users can't trust what the UI tells them, which makes every other defect worse. Adopt one rule: success and status copy is rendered only from an acknowledged state change.
2. **Chat runs through a half-integrated history-selection controller.** The controller that owns saved-chat turns doesn't handle new chat, clear, regenerate, continue, edit, stop, timeouts or interruptions. It leaks its internal model into the UI. This one integration gap explains CS-01, CS-03, CS-05, CS-N3, CM-01, CM-02, CM-N1, CM-N2, CM-06, CC-02, CC-04, CM-14 and CS-04. Temporary chats bypass the controller and work, which shows this is a regression in the wiring, not in the features.
3. **Client and server have drifted apart, and mocks hid it.** The notes list sends parameters the server ignores (NL-01, NL-02, NS-04, NL-11). Wikilinks use `[[Title]]` on the client and `[[id:UUID]]` on the server (NE-02, NL-11, NE-10). Loaded messages drop `serverMessageId` (XP-02). The prompt picker ignores the server library (CC-05). The UI ignores the server's default provider (CC-01). Model metadata is never stored (CM-04). In each case a unit-test mock agreed with the client. Contract tests against the real FastAPI app would have caught all of them.
4. **No object has one identity or one address across surfaces.** A conversation exists as a side-panel tab snapshot, a local Dexie record and a server chat, and nothing reconciles the three (XS-01, XS-05, XS-06, XS-07, XP-08). Notes and chats have no URL and can't be found from ⌘K (XP-05, XP-06, CS-02). Pins, recents and UI mode live in different stores per surface (XP-18), and the side panel feels like a different product (XP-12).
5. **The interface exposes internals instead of intent.** Engineering vocabulary ("turn", "history path", "cockpit rail", "overlay", "checkpoint"), raw IDs and URLs, about twenty composer controls, and colour tokens that fail contrast all come from shared building blocks (XP-13, XP-14, XP-15, XP-20, CO-03, CO-04, CC-08, AX-07, AX-08). Fixing the tokens, the glossary (§10) and a handful of shared components clears dozens of issues at once.

---

## 2. Scope, method & limitations

### 2.1 Personas

- **Sam, first-time user.** Starts on a fresh install with empty databases, has never seen tldw, knows ChatGPT-style chat and a basic notes app, and does not read documentation. Sam's goals are to write and find a first note, get a first useful answer, save it, and come back to it.
- **Riley, power user.** Works daily in a seeded library of 158 notes (it grew to about 176 as reviewers added items), 49 server chats, 11 server prompts, 3 characters and 4 media documents. Riley's goals are browsing, bulk organizing, exporting, linking notes, iterating on answers (regenerate, edit, branch, compare), moving between chat and notes, and switching between the full page and the side panel. Riley relies on the keyboard.

### 2.2 Surfaces

| Surface | How it was reached |
|---|---|
| **WebUI** | `/notes` and `/chat` on the Next.js dev frontend at `127.0.0.1:8080` |
| **Extension options page** | `options.html#/notes` and `#/chat` from the built Chrome MV3 extension. These share the WebUI's components |
| **Extension side panel** | `sidepanel.html` opened as a tab at 360-420 px, because the real Chrome side-panel container can't be driven headless |

### 2.3 Live stack

- Dev frontend plus FastAPI backend in single-user mode, at dev @ 86e287fee7.
- The first-time journeys ran against **fresh databases**. The power-user journeys ran against the **seeded library** above.
- The only LLM was a **mock OpenAI-compatible provider** (`custom-openai-api`: mock-gpt-4o, mock-claude-sonnet, mock-llama-3-8b). It returns canned Markdown in one burst about 1.5 s after send. `Ollama / gemma3:1b` was listed by the server but not running.
- **What the mock limits:**
  - Answer quality was not judged.
  - Token-by-token streaming, auto-scroll during long streams, partial-output retention on Stop, and slow-model timeouts could not be observed live. CM-N1 and parts of CM-02 and CS-04 rest on code reading plus simulated truncated SSE streams.
  - With only one working model, real multi-model switching and Compare output could not be tested.
  - Error UX for real providers (rate limits, auth failures, context-length errors) is inferred from the unreachable Ollama case and from code.

### 2.4 Methods

- **Cognitive walkthroughs.** Seven persona journeys (§3), 124 steps in total. Each step was rated none, minor, major or blocker against the four walkthrough questions.
- **Heuristic evaluation** against Nielsen's 10 heuristics. Separate lenses covered visual design, states (loading, empty, error, offline, conflict, performance) and accessibility.
- **WCAG 2.2 AA audit.** axe-core scans across states, plus keyboard-only Tab-order and focus probes, reflow at emulated 200-400% zoom (640×400 and 320×256 CSS px), forced-colours emulation, and target-size and contrast measurement (§7.3.6).
- **State, data-safety and performance probes.** Two-tab conflicts, navigation during the debounce window, offline toggling, truncated SSE streams, mid-reply reload, export request counting, payload sizes, polling cadence, and render timing on 25k-character notes and 80-message threads.
- **Evidence.** About 800 screenshots from the ten reviewer passes, and about 500 more from verification, all captured with scripted Playwright (headless Chromium). Every finding cites screenshots, code locations or both.

### 2.5 Verification

- Every candidate issue was **re-checked adversarially**, live in the running stack, in code, or both. Claims that didn't hold were narrowed or dropped. Where verification changed a severity or a claim, the section entry says so.
- Every issue at **severity 3 or higher** was re-checked by an **independent skeptic** who tried to refute it. Disagreements were settled by a tiebreak review.
- No finding was rejected outright (Appendix A). Several were narrowed. For example, the "unnamed in-bubble Stop" claim (CM-02, AX-13) and the "failures announced as status" claim (AX-13) were refuted and dropped from the write-ups, and seven Notes severities were adjusted (shown, for example, as "2 (from 3)" in §5).

### 2.6 Severity, priority and effort

**Severity** follows Nielsen's 0-4 scale:

| Severity | Meaning |
|---|---|
| 4 | Catastrophe: data loss or corruption, or a core task can't be completed |
| 3 | Major: a core task is seriously impaired, or the UI misleads in a way users act on |
| 2 | Minor: friction, confusion or inefficiency that has a workaround |
| 1 | Cosmetic: polish |
| 0 | Not a usability problem |

**Priority:**

| Priority | Rule | Count |
|---|---|---|
| **P0** | Severity 4, or a severity-3 data-safety or functional bug | 22 |
| **P1** | Any other severity-3 issue (usability or accessibility) | 9 |
| **P2** | Severity 2 | 97 |
| **P3** | Severity 0-1 | 23 |

**Effort** is a rough estimate:
- **S:** one component or hook, about a day.
- **M:** several files or a hook plus its tests, a few days to two weeks.
- **L:** cross-cutting, or needing both backend and frontend changes, multi-week.

**Area codes:**
- Notes: **NL** list, search & organization; **NE** editor; **NS** saving & data safety; **NO** onboarding & layout.
- Chat: **CS** sessions & history; **CM** messages & generation; **CC** composer, models & context; **CO** onboarding & layout.
- Extension and cross-cutting: **XS** extension side panel; **XP** cross-page & parity; **AX** accessibility.

In all, **151 issues** were kept: 6 at severity 4, 25 at severity 3, 97 at severity 2 and 23 at severity 1. That is 42 for Notes, 52 for Chat, and 57 for the extension, cross-page and accessibility.

### 2.7 Limitations and coverage gaps

Each reviewer recorded coverage gaps (see the `coverage_gaps` arrays per reviewer in `wf1/reviews.json` in the local evidence bundle described in Appendix B). These are the material ones:

- **Real LLM providers were not exercised.** Only the mock (and an unreachable Ollama) was available. Streaming behaviour, slow-model timeouts, provider-specific errors, multi-model switching, Compare output (forced on with `ff_compareMode`) and answer quality are inferred from code or simulation.
- **No screen-reader run.** VoiceOver, NVDA, JAWS and TalkBack were not used. Live-region and announcement findings (AX-06, AX-11, CM-06) come from DOM mutation logs and ARIA inspection. Speech input (Dragon, Voice Control) was not tested.
- **Zoom and contrast were emulated.** Zoom was emulated by viewport size, so true browser zoom, text-only resize (1.4.4) and text-spacing overrides (1.4.12) were not tested. Forced colours were emulated in Chromium only. Light theme was mostly checked by eye; nearly all screenshots use the default dark theme.
- **Chromium only.** Firefox (the extension's memory router and private-mode temporary chats), Safari and Edge were not tested.
- **No real touch device.** Mobile findings come from 390×844 emulation with `hasTouch`, so virtual-keyboard behaviour is inferred from code. macOS Option-key behaviour for Alt shortcuts (XP-10) is also inferred from code, because headless Chromium delivers US-layout keys.
- **Extension limits.**
  - The real Chrome side-panel container and browser context menus can't be driven headless. The right-click "Save to Notes" flow was simulated by sending the background script's runtime message.
  - Side-panel bulk delete was deliberately not exercised, to protect seeded chats.
- **Single-user mode only.** Multi-user (JWT) mode, PostgreSQL AuthNZ, cross-user sharing and permissions were not covered.
- **Build and data realism.**
  - Timings come from the Next.js dev server (React StrictMode), so absolute latencies may be inflated. Relative findings (request counts, payload sizes, polling cadence) still hold.
  - Seeded notes and chats all share one creation day, so date sorting, grouping and recency could not be judged realistically.
  - The backend was shared by ten concurrent reviewers, so counts drifted during the session.
- **Features not exercised.** Voice, dictation and TTS; attachments, OCR and image generation; MCP tools; web search (no provider keys); Notes Studio and AI assist actions; flashcard generation; note version history; notes import with the overwrite strategy; offline-queue reconnect after the 30 s health poll; chat trash purge (a 500 per the seed report); end-to-end character-chat forks. The full Export run was stopped after 132 requests to protect the shared backend, so NL-02's file size and duplicate count are extrapolated from code (`MAX_EXPORT_PAGES = 1000`).
- **Webui server-chat flows were only partly testable.** Because new webui chats are never promoted to the server (CS-03), server-backed behaviour was tested on seeded chats.

---

## 3. Persona journeys

**Method.** Each journey is a cognitive walkthrough, run live against dev @ 86e287fee7 with scripted Playwright (headless Chromium, default dark theme unless stated). At every step the reviewer asked the four walkthrough questions: will the user try to reach this goal, will they notice the right control, will they connect that control to the goal, and will they understand the feedback? Each step was then rated:

- **none**: the goal was met as expected.
- **minor**: met, with hesitation or a one-to-two-click detour.
- **major**: met only through a non-obvious workaround, or the result was misleading.
- **blocker**: the goal can't be met, or data is lost or corrupted.

Issue IDs refer to the prioritized issue list in §4.

**Personas.**
- **Sam** is a first-time user on an empty account.
- **Riley** is a power user working in the seeded library. The library held about 160-176 notes (the count grew as reviewers added items), 49 server chats, 11 server prompts, 3 characters and 4 media documents.

**Environment limits that shape what could be observed:**
- The mock LLM returns each answer in one burst about 1.5 s after send. Token streaming, slow-model timeouts and mid-stream Stop could not be observed live.
- `Ollama / gemma3:1b` is listed by the server but not running.
- The Chrome side panel was emulated by opening `sidepanel.html` in a tab at 360-420 px.
- Browser context menus can't be driven headless. The extension's right-click 'Save to Notes' was simulated with the runtime message the background script sends.

Screenshot paths are relative to the session's `shots/<reviewer-tag>/` folder.

### At a glance

| Journey | Reviewer | Steps | Blocker | Major | Minor | None | Did the persona reach the core goal? |
|---|---|---|---|---|---|---|---|
| 3.1 First-time · Notes | ft-notes | 15 | 2 | 6 | 7 | 0 | Partly. A note was created, tagged and found again, but WYSIWYG text was saved reversed and the last edit was lost on navigation |
| 3.2 First-time · Chat | ft-chat | 19 | 5 | 6 | 7 | 1 | Partly. Sam got an answer after switching model, but couldn't regenerate, edit and resend, save to Notes, start a clean new chat or find the chat again |
| 3.3 First-time · Extension | ft-ext | 20 | 1 | 10 | 8 | 1 | Partly. Sam got an answer once the model icon was found. There is no in-chat Save to Notes, and Pro mode is unusable at 420 px |
| 3.4 Power user · Notes | pu-notes | 15 | 2 | 8 | 4 | 1 | No for browsing, sorting and exporting at scale. Yes for search, import and rapid editing |
| 3.5 Power user · Chat | pu-chat | 21 | 5 | 7 | 8 | 1 | No for iterating (regenerate, continue, edit, new chat, compare, saved prompts). Yes, slowly, for resuming, renaming, trash and character chat |
| 3.6 Power user · Cross-page | pu-cross | 17 | 2 | 7 | 8 | 0 | Partly. One chat-to-note round trip worked, but it can't be repeated on 4 of 5 chats and there is no notes-to-chat path |
| 3.7 Power user · Extension | pu-ext | 17 | 2 | 8 | 6 | 1 | Partly. History search and export work, but opening a past chat overwrites the current tab, and sync and hand-offs each lose context |
| **Total** | | **124** | **19** | **52** | **48** | **5** | |

**Coverage map: persona × page × surface.** Every combination in the brief was walked by at least one journey. The extension options page runs the same Notes and Playground code as the WebUI, so its rows list only the steps that were re-walked there; all WebUI issues for that page apply to it too (see the parity table after §3.7). The side panel has no Notes page: Notes reach it only as a context-menu capture. Fixes are in §§5-7 under each ID; the improvement ideas (§8) are listed for each row.

| Persona · surface · page | Walked in | Main issues | Improvement ideas (§8) |
|---|---|---|---|
| First-time · WebUI · /notes | 3.1 (all 15 steps) | NE-01, NS-01, NO-02, NO-N1, NO-04, NE-05, NE-06, NL-12 | Q10, Q11, Q4, S5 |
| First-time · WebUI · /chat | 3.2 (all 19 steps) | CC-01, CC-02, CS-01, CS-03, CM-01, CC-04, CO-03, CO-02 | Q1, Q2, N12, N14 |
| First-time · Ext options · #/notes | 3.3 steps 12-13 | XS-13, NO-02, plus all WebUI /notes issues | Q11, N1 |
| First-time · Ext options · #/chat | 3.3 steps 14-17 | XP-09, XP-12, CC-01, CC-02, CC-04, XP-01 | Q2, N6 |
| First-time · Ext side panel · chat and capture | 3.3 steps 1-12 and 18-20 | XS-02, XS-17, XS-10, XS-09, XS-03, XS-04, XS-05, XS-12, XP-01 | Q12, Q2, Q5, N3 |
| Power · WebUI · /notes | 3.4 (all 15 steps); 3.6 steps 3-6, 10, 16 | NL-01, NL-02, NL-03, NE-02, NE-04, NS-03, NS-04, NE-03, AX-03 | N15, N16, N17, N18, N19, N20, Q8, Q9 |
| Power · WebUI · /chat | 3.5 (all 21 steps); 3.6 steps 1-2, 5, 7-9, 11 | CS-01, CS-02, CM-01, CC-05, XP-02, CM-05, CM-12, CC-10, AX-01 | N5, N7, N8, N9, N10, S1, S3 |
| Power · Ext options · #/notes | 3.7 steps 15-16; 3.6 step 17 | NL-01, NE-04, XP-19, XP-05, plus all WebUI /notes issues | N4, N16 |
| Power · Ext options · #/chat | 3.7 steps 11-13 and 17; 3.6 step 17 | XP-08, XP-09, CM-13, CM-N2, CS-02, XP-02 | N6, S2 |
| Power · Ext side panel · chat and capture | 3.7 steps 1-10 and 14 | XS-01, XS-07, XP-08, XS-06, XS-08, XS-15, XS-16, XS-13 | S2, S1, N3, N21, Q13 |
| Both · phone width (390 px) | 3.2 step 19; 3.6 steps 13-14 | CO-01, CO-N1, NO-03, AX-05, AX-16, XP-11 | N13, S6 |
| Both · tablet, laptop and zoom (768-1024 px; 200-400%) | 3.1 step 14; 3.2 step 18; 3.4 step 15; 3.5 step 20; 3.6 step 13; §7.3.2 | NO-01, NO-03, CO-04, CO-05, CO-06, AX-04, AX-05 | N12, N20, S6 |
| Both · narrow side panel (360-420 px) | 3.3 steps 9, 18-19; 3.7 step 7 | XS-03, XS-04, XS-14, XS-15, XS-18 | N12 |

**Patterns that recur across all seven journeys:**

1. **The UI claims state that isn't true.** Examples: 'Saved · Locally + Server' (CS-03, XS-05), 'NOTES 100 TOTAL' (NL-01), 'Delete — cannot be undone' that only closes a tab (XS-07), and 'Healthy' on an unreachable model (CC-01).
2. **Actions are offered and then fail with raw error codes.** Regenerate, Continue and Edit → 'Save & Send' (CM-01); 'New chat' (CS-01); Print (NE-04); Compare (CM-05).
3. **Notes and chats have no address.** They have no URL, the ⌘K palette can't find them, and links between them don't click (XP-05, XP-06, XP-03, NE-02). Returning to a specific chat or note is a major or blocker step in six of the seven journeys.

---

### 3.1 First-time user · Notes

*Reviewer `ft-notes`. WebUI `/notes` at 1440x900 and 1024x900, empty account, first run. Evidence in `shots/ft-notes/`.*

Sam lands on /notes and immediately loses the page header, because the auto-started tour scrolls the window. The tour then clips its popovers and quietly ends after step 6 of 8. Writing the first note is where trust breaks. In WYSIWYG mode every keystroke lands at the start of the document, so text comes out backwards and is autosaved that way. Markdown mode works but assumes syntax knowledge, and its preview drops list markers. Tagging, search and Trash are solid. However, leaving the page within 5 s of typing silently discards the edit, and a working [[wikilink]] looks exactly like a broken one.

| # | Goal | What the user does | What happens | Friction | Issue IDs |
|---|---|---|---|---|---|
| 1 | Understand the page within 5 s | Opens /notes cold (first run, empty account) | The tour auto-starts and scrolls the window down 315 px, so the app header and 'NOTES 0 TOTAL' are off-screen and an empty void sits under the sidebar. 'Select or create a note' sits on top of a fully live editor. About 12 equal-weight controls compete (7 view buttons, ORGANIZE, FILTERS, Import/Sync/Export, ASSIST, 'Create study pack') | major | NO-02, NO-01, NO-04, NL-12 |
| 2 | Follow the onboarding tour | Clicks 'Next (Step n of 8)' | Step 2 is clipped at the top and its targets are scrolled off-screen. Step 3 spotlights the sort dropdown while its copy talks about tag filters. After step 6 the tour vanishes: no step 7 or 8, no 'done' state, no message | major | NO-02, NO-N1 |
| 3 | Start the first note | Clicks 'Create note' in the editor's empty state | Focus moves to Title, which is good feedback. It is one of four create affordances on screen, and nothing exists on the server yet | minor | NO-04 |
| 4 | Write paragraphs, a heading, bullets and a checklist without knowing Markdown | Switches to WYSIWYG, types, uses the Heading and List buttons | Every keystroke is inserted at offset 0: 'Hello' becomes 'olleH' and paragraphs stack in reverse order (reproduced at human typing speed). Heading and List have no visible effect. Autosave persists the garbled text, which also shows in the list snippet | **blocker** | NE-01 |
| 5 | Recover in Markdown mode | Clears the note and retypes using the H1 and List toolbar buttons | The toolbar inserts pre-selected placeholders ('# Heading', '- List item'), which is nice. Enter doesn't continue a list, and there is no checklist button. Preview drops bullets and numbers. A duplicate checklist strip appears with 'Portable markdown with best-effort task continuity' | major | NE-06, NE-05, NE-09 |
| 6 | Know whether and when it saved | Watches the status while typing, while idle, and after Ctrl+S | Autosave runs after about 5 s idle ('Saved just now', 'Version 1 · Last saved …'), but the status is shown three times. Save stays the primary button when nothing is unsaved. Where the note is stored is never stated ('No server save status yet') | minor | NS-06 |
| 7 | Add two tags | Types tags and presses Enter, then tries 'Suggest tags' | Chips appear with usage counts ('thesis (1)') and autosave runs. Suggest tags pre-checks 5 junk suggestions (first, idea, ideas, key, paragraph) | minor | NL-17 |
| 8 | Link note 2 to note 1 | Types '[[My fi' and picks the suggestion with Enter | The suggestion appears detached below the textarea. The inserted link renders as raw text in Preview, identical to an unresolved link. No backlink and no graph edge result, because the bracketed '[ft-notes]' title never resolves | major | NE-02 |
| 9 | Link via Connections instead | Expands CONNECTIONS and clicks 'Add link' | A toast confirms the link. The same note is then listed twice, under three unexplained relation concepts, and the panel re-collapses when the other note opens | minor | NE-10 |
| 10 | Find the first note again | Uses full-text search and Search tips, types 'thes' + Enter in the tag filter, opens Browse tags | Search matches body words, and the no-results state is excellent. The tag filter applies a non-existent 'thes' tag instead of the highlighted 'thesis (1)', and 'thes' then pollutes Browse tags | minor | NL-08 |
| 11 | Leave for /chat and come back | Types an edit, clicks Chat within 5 s, returns with Back | No prompt and no flush: the edit is gone. The editor shows 'Select or create a note' instead of reopening the note. A full reload *is* protected by the beforeunload prompt | **blocker** | NS-01, XP-05 |
| 12 | Delete a note, then recover it | ••• → Delete → confirm; Trash → Restore | Works well: Undo toast, Trash with 'Deleted · time', and Restore reopens the note. Polish issues: the confirm copy doesn't mention Trash, a stale Undo toast remains after restore, and Trash still shows Import/Export and an empty editor | minor | NL-15 |
| 13 | Make sense of the views | Tries Timeline, Inbox, Collection and Graph | Timeline files Oct 2 notes under 'SEPTEMBER 2026'. Inbox silently duplicates the 'Captured' filter. Collection is a dead end. Graph scrolls the page, focuses an arbitrary note, leaves nodes unlabeled and uses jargon. Modes and views share one 7-button grid | major | NL-09, NL-12, NL-13, NL-14 |
| 14 | Work at 1440 and 1024 px | Re-runs the tour at 1024 and reviews the layout | The page overflows (1215 px document in a 900 px viewport). There is an 80 px empty band, and FILTERS is clipped under RESULTS. At 1024 px the header actions wrap onto extra rows | major | NO-01, NO-03 |
| 15 | Explore the header actions and shortcuts | Opens Keyboard shortcuts, presses Cmd+K, clicks 'Create study pack' | The shortcuts dialog is clear and includes 'Restart tutorial'. Cmd+K is documented as 'Focus the search input' but opens the chat-oriented global palette. 'Create study pack' (primary styling) leaves Notes for /flashcards and shows a raw note UUID | minor | XP-07, NE-08, XP-14 |

Key evidence: `04g-wysiwyg-slow.png` and `04d-list.png` (step 4), `09a-unsaved-before-leave.png` and `09d-back-to-notes.png` (step 11), `02-tour-1440-step2.png` (step 2), `07e-preview-link.png` (step 8).

**Moments that matter**
- **Pain 1. Text typed backwards in WYSIWYG, then autosaved (NE-01).** Newcomers choose WYSIWYG precisely because they don't know Markdown, and the first thing they write is corrupted on the server.
- **Pain 2. Silent loss of the last edit on in-app navigation (NS-01).** Nothing warns Sam, and the loss is found only after returning.
- **Pain 3. Onboarding works against itself.** The tour scrolls its own targets away and ends silently (NO-02, NO-N1), and wikilinks give no sign of whether they worked (NE-02).
- **Delight. The recovery loop.** 'Note deleted · Undo', a clear Trash with timestamps, and 'Note restored'. The no-results state ('No notes match "…"' with 'Clear search & filters') is also a model.
- **Delight. Small craft touches.** Toolbar placeholders arrive pre-selected so typing replaces them, and tag chips show usage counts.

---

### 3.2 First-time user · Chat

*Reviewer `ft-chat`. WebUI `/chat` at 1440x900, plus 1024x768 and 390x844, first run. Evidence in `shots/ft-chat/`.*

Sam knows to type, but the page defaults to an Ollama model that isn't running and labels it 'Healthy'. The first question therefore fails into a jargon 'Turn needs review' panel that gives no cause, Retry or model switch. After switching to the working model the answers render beautifully. Then almost every second-order action breaks:
- Regenerate and Edit → 'Save & Send' fail with raw error codes.
- There is no Save to Notes.
- 'New chat' sends the old conversation along with the next question.
- The chat can't be found again, because nothing was ever saved to the server despite the 'Saved' label.

| # | Goal | What the user does | What happens | Friction | Issue IDs |
|---|---|---|---|---|---|
| 1 | Understand the page within 5 s | Opens /chat cold | The 'Start a new chat' card and composer are clear. Around them sit about 40 controls, many in jargon (MCP None, OpenUI, Context/Runtime rail, '0 + ~0 = 0 tokens'). There are two theme toggles, two shortcut buttons, a 'Saved' pill on an empty chat, and a scroll chevron over the card | minor | CO-03, CO-07, CC-08 |
| 2 | Learn what the unclear controls mean | Hovers each one | MCP has no tooltip. OpenUI's tooltip is circular. 'Saved' says 'Locally + Server'. The rails' tooltip is 'Restore context sidechannel'. Three pills ('Standard chat', 'Conversation timeline', 'Composer') are decorative | minor | CO-03, CC-09, AX-17, CS-03 |
| 3 | Pick a model that works | Reads the model chip and opens the picker | The default is 'Ollama / gemma3:1b · Healthy', although Ollama isn't running and the server's default is custom-openai-api. Nothing indicates which model actually works | major | CC-01, CC-06 |
| 4 | Ask a first question | Clicks 'Start chatting', types, presses Enter | The send returns 502. An unstyled panel reads 'Turn needs review / The original send outcome is retained separately from history…'. It gives no cause, no Retry and no model switch, and the chip still says 'Healthy' | **blocker** | CC-02, CC-01 |
| 5 | Recover from the failure | Picks mock-gpt-4o | The switch works. The chip truncates to 'Custom OpenAI API / …', and the bubble header shows the raw id 'custom_openai_api:mock-gpt-4o' | minor | CC-06, XP-14 |
| 6 | Get an answer | Sends the question | A 'Generating response…' banner and cursor appear (good). Send turns into 'QUEUE', and two differently labelled Stop buttons appear (composer and in-bubble). Markdown, tables and code render very well | minor | CM-02, AX-13 |
| 7 | Cancel a generation | Clicks Stop at about 600 ms and again at about 270 ms | At 600 ms the turn is shown as the same 'Turn needs review… response not saved' failure, with no 'Stopped' state. At 270 ms the user's message vanished and a raw toast read 'request_config_scope_changed' | major | CM-02 |
| 8 | Reuse the answer | Hovers, clicks Copy, looks for a copy button on the code block | Copy gives a green check and copies the markdown. Message actions appear only on hover. Code blocks have no copy button or language label, and the '•••' chip hides itself on hover | minor | CM-07, CM-09 |
| 9 | Ask a follow-up | Sends a second question | Works, and the view auto-scrolls to the answer | none | — |
| 10 | Get a better answer | Clicks Regenerate | No request is sent. A toast reads 'Message action unavailable — unsupported_history_regeneration' (every attempt) | **blocker** | CM-01 |
| 11 | Fix a typo and re-ask | Edit → 'Save & Send' | The editor closes, the edit is discarded, nothing is sent and no toast appears. Plain 'Save' keeps the edit, but the old answer stays with no stale marker | **blocker** | CM-01 |
| 12 | Keep the answer in Notes | Searches the hover toolbar, overflow menu, text selection, More tools and ⌘K | There is no Save to Notes anywhere, and the overflow offers role-play steering instead. Workaround: Copy, open Notes Dock, paste, add a title, Save. The resulting note has no link back to the chat | major | XP-01, CM-09 |
| 13 | Start an unrelated conversation | Clicks 'New saved chat' (or the sidebar '+'), sends Q2 | The view clears, but the 'Review conversation history' bar persists. After sending, Q1 and A1 reappear, and the request payload was [Q1, A1, Q2] | **blocker** | CS-01, CC-04 |
| 14 | Return to the earlier chat | Tries the sidebar, Recent conversations, Server/Folders, search and ⌘K | The Chats panel opens on 14 navigation shortcuts, with Recent collapsed to about 30 px. It says 'No server chats yet' while the open chat reads 'Saved'. The backend held 0 chats after about 15 conversations (every completion went out with save_to_db=false). ⌘K finds only Flashcards settings | **blocker** | CS-02, CS-03, XP-06 |
| 15 | Get oriented | Clicks 'Take a quick tour' | Nothing happens, even after waiting 12 s | major | CO-02 |
| 16 | Understand modes, Buddy and the rails | Opens Explore chat modes, Modes, Buddy & Persona and both rails | The 'Explore chat modes' cards are the best newcomer copy on the page. Modes greys out Compare and Voice with no reason. Buddy and the rails are jargon, and the model appears as 'tldw:gemma3:1b' in one place and 'Ollama / gemma3:1b' in another | minor | CM-05, CO-04, CC-06, XP-13 |
| 17 | Judge the error messages | Sends empty or whitespace input, toggles 'All known models', sends to the unreachable model | An empty send is a silent no-op while Send looks enabled. The scope toggle doesn't switch the list. 'Inspect' on the failure panel shows only Sam's own input | major | CC-02, CC-06 |
| 18 | Use a 1024 px laptop | Opens /chat at 1024x768 and sends | No overflow. The header wraps to two rows, and the composer plus chips take about 240 px. The chevron covers the empty-state text | minor | CO-06, CO-07 |
| 19 | Use a phone (390 px) | Opens /chat at 390x844, picks a model, sends | The page opens in Focus mode with no header, sidebar or new-chat control. The composer takes about 46% of the height, with duplicate unlabeled controls and the model shown twice | major | CO-01 |

Key evidence: `03c-after-send-1000ms.png` (step 4), `12regen-b-regen-500ms.png` (step 10), `21b-user-edit-saved.png` (step 11), `13header-b-after-q2.png` (step 13), `09c-sidebar-search.png` (step 14).

**Moments that matter**
- **Pain 1. The very first send fails, with no explanation (CC-01, CC-02).** A green 'Healthy' badge on a dead provider, followed by a recovery panel written for engineers, teaches Sam that the product is broken before it has answered once.
- **Pain 2. The conversation model is broken in both directions.** 'New chat' isn't new and leaks prior context into the next request (CS-01). The old chat can't be found, because 'Saved · Locally + Server' was never true (CS-02, CS-03).
- **Pain 3. Core message actions throw raw codes (CM-01).** Regenerate and Edit → 'Save & Send' are visible on every answer and fail every time. Save & Send also throws away the user's edit.
- **Delight. Answer rendering.** Clean headings, lists and syntax highlighting, and tables with View, Copy as CSV and Download CSV.
- **Delight. 'Explore chat modes'.** Plain-language mode cards are the one place the page speaks to a newcomer, and Copy's inline green check is clear feedback.

---

### 3.3 First-time user · Extension

*Reviewer `ft-ext`. Side panel (`sidepanel.html#/chat`) at 420x900 and 360x800, options page (`options.html#/chat`, `#/notes`), fresh profile. Evidence in `shots/ft-ext/`.*

The side panel invites a question, but the first send is rejected with 'Please select a model'. The only model control is an unlabeled 16 px brain icon, and it disappears once a model is picked. Answers render well even in a narrow panel, and the slash menu is a genuine delight. However, the composer, the Chats drawer and Pro mode all fight the 420 px width. There is no Save to Notes in chat, and 'Save chat to history' claims a server save that never happens. Handing off to the full page carries the conversation, but it lands in a differently worded product whose own default model is unreachable yet marked 'Healthy'.

| # | Goal | What the user does | What happens | Friction | Issue IDs |
|---|---|---|---|---|---|
| 1 | Understand the panel within 5 s | Opens the side panel cold | A clear 'Start a conversation below' with two 'Try asking' chips. The model picker is an unlabeled brain glyph. Five header icons are unlabeled, and a scroll chevron already floats with nothing to scroll | minor | XS-02, XS-12 |
| 2 | Ask a first question | Types and presses Enter | The draft autosaves, then a red 'Please select a model' appears. Nothing points to the brain icon | major | XS-02 |
| 3 | Pick a model | Clicks the brain icon | The entry reads 'Custom OpenAI API - m…', so the model name is truncated away. After picking, the model control disappears from the composer | major | XS-02 |
| 4 | Send and watch the reply | Sends again | While waiting, the primary button reads 'QUEUE' and there is no Stop. The reply renders well. The byline 'tldw:mock-gpt-4o' is the only place the model appears | minor | XS-10 |
| 5 | Read the answer and continue | Lets the reply finish, scrolls | Auto-scroll stops 56 px short, so SEND is clipped and the scroll pill covers the input. When scrolled up, the composer scrolls away with the messages | minor | XS-09 |
| 6 | Save the answer to Notes | Opens More actions, selects text | The menu offers New Branch, single-letter S/T/L/E actions, a disabled 'Read aloud' and Delete. There is no Save to Notes, and selecting text shows nothing | major | XP-01, CM-09 |
| 7 | Understand the composer buttons | Clicks Config, Character, Voice and Improve prompt | 'Config' is actually voice settings. Voice is disabled with no reason. Character offers 'Overlay' and 'Tracked character/persona chat'. The Improve-prompt menu is transparent, so its text collides with the composer | major | XS-12, CC-12 |
| 8 | Discover commands | Types '/' | A clear slash menu appears (/search, /web, /vision, /model), each with a description | none | — |
| 9 | Find earlier chats | Opens the 'Chats' drawer | The drawer pushes the chat into a 132 px column, so answers render one letter per line. 'TODAY' lists only the current tab, although 9 chats are stored | major | XS-03, XS-06 |
| 10 | Search for an older chat | Types 'forgetting' | A HISTORY section appears with a 'Local' badge. Stored history is reachable only through search | minor | XS-06 |
| 11 | Understand 'Save chat to history' | Hovers the switch, then toggles it off | The tooltip says 'Locally + Server', but the server never receives the chat (/chats total = 0). Turning it off gives excellent feedback: the label changes, the header tints and a toast explains | major | XS-05 |
| 12 | Quick-save to Notes | Right-click 'Save to Notes' (simulated) | A clean modal with the title prefilled, editable content and source URL. It has no tags field, and the 'Saved to Notes' toast has no link to the note | minor | XS-13 |
| 13 | Check the note in options #/notes | Opens Notes cold | The note shows a 'captured' tag and a Source line, but the footer says 'Origin: Typed manually'. The first-run tour scrolls the page 315 px and leaves it there after Skip | minor | XS-13, NO-02 |
| 14 | Hand off to the full page | Types a draft, clicks 'Expand in full page' | Conversation and model carry over (good). The draft is lost, and the page opens in Focus mode with no navigation | minor | XP-09 |
| 15 | Compare with options #/chat | Fresh profile, sends a question | Vocabulary and controls differ from the panel. The page auto-selects 'Ollama / gemma3:1b · Healthy' and the send fails with 'Turn needs review', although the server returned a readable provider_unavailable message | major | XP-12, CC-01, CC-02 |
| 16 | Save the answer from the full page | Opens More actions | Still no Save to Notes, and role-play items are added (Continue as user, Impersonate user, Force narrate) | major | XP-01, CM-09 |
| 17 | Try the chips above the conversation | Clicks 'Use no prior messages', then 'Review conversation history' | The whole thread vanishes with no confirm or undo, and stays gone after reload. The review panel uses 'Complete source / included path' and 'Omitted alternative', shows raw markdown and has overlapping text | major | CC-04 |
| 18 | Try Pro mode | Switches Mode to Pro at 420 px, then tries to close the docked list | A 288 px docked list leaves a 132 px chat with vertical placeholder text. Collapse, Esc, the backdrop and the header toggle all fail to close it | **blocker** | XS-04 |
| 19 | Use a narrow 360 px panel | Resizes and sends | The header title wraps, the table clips its last column, and the scroll pill covers the table's View button | minor | XS-14 |
| 20 | Recover from a failing model | Picks the unreachable Ollama model and sends | The same opaque 'Turn needs review' box offers only Copy and Dismiss. There is no reason, no Retry and no picker on screen | major | CC-02, XS-02 |

Key evidence: `02f-after-enter-2s.png` (step 2), `09g-sidebar.png` (step 9), `23a-save-toggle-tooltip.png` (step 11), `25a-review-history-panel.png` (step 17), `20c-pro-answer-bottom.png` (step 18).

**Moments that matter**
- **Pain 1. The first send is blocked by an invisible model control (XS-02).** No default model, an unlabeled icon, and a picker that hides itself after use.
- **Pain 2. Capture, the extension's core job, is half-built (XP-01, XS-13).** There is no Save to Notes on any chat answer. The context-menu quick-save drops tags and gives no way to open the note.
- **Pain 3. The panel's own layouts destroy the transcript at panel widths (XS-03, XS-04, XS-09).** The Chats drawer and the Pro dock reduce answers to one letter per line, and SEND is clipped after every reply.
- **Delight. The slash menu and the Temporary-chat switch.** Both explain themselves: one-line command descriptions, and a label, tint and toast on toggle.
- **Delight. Rich rendering in 420 px.** Code, lists and tables with View, Copy CSV and Download, plus a clean quick-save modal that shows the source URL.

---

### 3.4 Power user · Notes

*Reviewer `pu-notes`. WebUI `/notes` at 1440x900 and 1024x768, seeded library of about 163-173 notes. Evidence in `shots/pu-notes/`.*

Riley's library is bigger than the list can show. The header says '100 TOTAL', every pager page renders the same 100 rows, and 'Export all' loops over those rows without end. The parts Riley leans on daily are excellent: search, import, rapid note switching and long-note performance. But the keyboard flow, wikilinks, bulk tagging, conflict recovery and printing each fail at exactly the moment a heavy user depends on them.

| # | Goal | What the user does | What happens | Friction | Issue IDs |
|---|---|---|---|---|---|
| 1 | Land as a returning user | Opens /notes (not first run) | The first row renders in 1.7 s. The list says 'NOTES 100 TOTAL' while the API reports 163. The results area is about 276 px, so only 2 notes are visible. Linked-chat rows print two raw UUIDs, and browse rows show no tag chips | major | NL-01, NL-04, XP-14 |
| 2 | Page through the library and sort by title | Clicks pages 2 and 5, then Title (A-Z) | Every page renders the identical 100 rows; only the label changes. A-Z omits 63 notes. The oldest notes can't be reached without searching | **blocker** | NL-01 |
| 3 | Find a note by search and tag | Searches, filters by 'research', then combines the two | Search reaches beyond the first 100 and pages correctly, and tag counts show ('research (32)'). Results are re-sorted by date, so the exact title match comes last. Body-only hits show the first line rather than the match | minor | NL-07 |
| 4 | Work keyboard-only | Uses Tab, the arrow keys, End, Enter, '/', Esc, Alt+N, '?' and Cmd+K | It takes 35 Tabs to reach the first row. End does nothing. Enter opens the note but focus stays on the row, and 61 more Tabs never reach the editor. '?' opens a page navigator and Cmd+K the global palette | major | AX-03, XP-07 |
| 5 | Quick-capture a thought from /chat | Tries Cmd+K 'new note', then Alt+5 → Alt+N → title → Tab → Cmd+S, then Notes Dock | The palette has no 'New note' command and can't find notes. Enter in the title does nothing, and the body is 12 Tabs away. Notes Dock works, but it is labelled 'ARCHIVE' | minor | XP-06 |
| 6 | Edit deep inside a 4,238-word note | Uses the table of contents, types mid-document, switches to Split | It opens in 80 ms and typing costs about 19 ms per key. A 24-entry TOC that can't be collapsed pushes the textarea to y = 1170. A TOC jump scrolls the whole page 974 px. Split panes don't scroll together | major | NE-03 |
| 7 | Link notes with [[wikilinks]] and follow them | Types [[…]] with the sidebar filtered and unfiltered, opens Preview, clicks a link, opens Connections | Autocomplete offers only notes in the current sidebar list (nothing when filtered, nothing beyond the first 100). Resolved links have href='' and do nothing when clicked. With a filtered sidebar they render as plain text. There are no backlinks and no outgoing links | major | NE-02 |
| 8 | Use Graph view at this scale | Focuses a note, tries All notes, picks another note from the list | 'All notes' is disabled ('up to 100 … this library has 173') while the sidebar says 100. Only the focused node is labelled. Picking a note keeps the editor hidden | major | NL-11, NL-14 |
| 9 | Bulk-tag, export, delete and restore 6 notes | Shift-click range select → Assign tags → Export → Delete → Trash → Restore | Range select works. Assign tags *replaced* the existing tags. There is no progress indicator, so the deletes interleaved with tag updates still running. Trash has no search, checkboxes, bulk restore or permanent delete, and each Restore leaves Trash | major | NL-03, NL-05, NL-06 |
| 10 | Switch notes rapidly with unsaved edits | Edits note A and switches to B within 250 ms, repeatedly | Every edit was saved to the right note with no cross-contamination. Alt+N inside the editor is ignored | none | — |
| 11 | Keep a place in the list while autosave runs | Scrolls the list to 3000 px and types | About 5 s later, the autosave refetch resets the list scroll to 0 | major | NS-04 |
| 12 | Handle a concurrent edit from another device | Edits the note through the API, waits, then edits locally and presses Cmd+S | After the 30 s poll a clear banner appears (good). On save, the conflict produces 5 stacked messages, including 'check your connection'. Retry repeats the conflict, 'Save anyway' fails, and the only exit discards local edits. There is no diff | major | NS-03, NS-N1, NS-N2 |
| 13 | Export everything; print a note to PDF | Export → Markdown (all matching); More actions → Print / Save as PDF | Export fired 132 identical requests in 9 s and showed 'Exporting MD 13100 notes exported so far…' for a 163-note library, with no Cancel (aborted by the reviewer). Print opened about:blank while toasting 'Please allow pop-ups' | **blocker** | NL-02, NE-04 |
| 14 | Use the alternative views; import | Opens Timeline, Inbox, Collection and saved filters; imports 2 .md files | Timeline shows the same 100 notes, with no pager, under 'SEPTEMBER 2026'. Collection cards are crammed into a 340 px sidebar. Saved-filter counts count keywords, not notes. Import is excellent: multi-file, preview, duplicate strategy, and YAML tags | minor | NL-01, NL-09, NL-10 |
| 15 | Trust the shortcut sheets; work on a 1024 px laptop | Opens both shortcut sheets, presses Cmd+B, resizes | The sheets contradict each other (Cmd+B is Bold in one, Ctrl+B toggles the sidebar in the other), and Cmd+B does nothing in the editor. At 1024 px the header wraps to 3 rows and the list shows about 2.5 rows | minor | XP-07, NO-03 |

Key evidence: `01-land-1440.png` and `02c-sort-az.png` (steps 1-2), `10b-export-progress.png` (step 13), `09b-conflict-save.png` (step 12), `07g-trash.png` (step 9).

**Moments that matter**
- **Pain 1. 40% of the library is invisible to browsing (NL-01).** Older notes look deleted, and the count and pager are placebo. Wikilink autocomplete, Graph, Timeline and Export inherit the cap.
- **Pain 2. 'Export all' never finishes and reports success-in-progress (NL-02).** It made thousands of duplicate rows and hundreds of requests against the server, with no Cancel.
- **Pain 3. Surprise data changes with no undo or exit.** Bulk 'Assign tags' wipes existing tags (NL-03). Conflict recovery offers only 'discard your edits' (NS-03).
- **Delight. Note switching is bulletproof.** Five sub-250 ms switches saved every edit to the right note, and a 25k-character note opens in 80 ms.
- **Delight. Import.** Multi-file, preview, duplicate strategy, and front-matter tags and titles honoured.

---

### 3.5 Power user · Chat

*Reviewer `pu-chat`. WebUI `/chat` at 1440x900 and 1024x768, 49 seeded server chats, 11 server prompts, 3 characters. Evidence in `shots/pu-chat/`.*

Riley wants to get back to work fast, but /chat offers no fast path:
- The palette contains no chats.
- History must be re-expanded on every load, and its results render behind the sidebar footer.
- The URL never changes.

Once inside a saved chat, iteration is broken. Regenerate, Continue and Edit & Send fail with raw codes. 'New chat' destroys the next message. Save to Notes and the saved-prompt library are missing, and Compare is hidden behind a flag. Character chat, rename, trash and thread search are the bright spots.

| # | Goal | What the user does | What happens | Friction | Issue IDs |
|---|---|---|---|---|---|
| 1 | Land as a returning user | Reloads /chat | The sidebar is collapsed and the empty-state card is shown despite 49 server chats. The model chip still shows 'Ollama / gemma3:1b · Healthy'. Both rails are hidden behind edge tabs | minor | CS-02, CC-01 |
| 2 | Resume 'Evaluate embedding models…' by keyboard | Cmd+K, types 'embedding' | The palette has no chats or recents and returns only a settings toggle. 0 of 49 chats are reachable this way | major | CS-02, XP-06 |
| 3 | Resume it from the sidebar | Ctrl+B → Recent conversations → types 'embedding' | The Shortcuts section takes about 550 px, so both matches render behind the Settings/Mode footer and none is visible at 900 px. Recent conversations must be re-opened after every reload | major | CS-02 |
| 4 | Scan the history list | Collapses Shortcuts | About 4 rows fit per screen. Titles are cut at about 22 characters, the title repeats as the topic, and 'Server' appears on every row | minor | CS-08 |
| 5 | Open the chat and check which model answered | Clicks the row | It opens in about 1 s, scrolled to the bottom. The URL stays /chat, so the chat can't be bookmarked. Old replies are labelled only 'Assistant' | minor | XP-05, CM-04 |
| 6 | Switch model by keyboard | Shift+Esc → Tab to the chip → Enter, then the arrow keys | It takes 17 Tabs to reach the chip. Enter opens the list, but focus never reaches search or the options. Palette 'Switch Model' and /model open the settings modal instead | major | AX-01 |
| 7 | Check model attribution | Picks mock-gpt-4o with the mouse, sends 2 turns | Live attribution is good ('mock-gpt-4o'). After reload the replies revert to 'Assistant', because the server stores no model metadata | minor | CM-04 |
| 8 | Apply a saved system prompt | Clicks Prompt, searches 'socratic' | The picker says 'No saved prompts' while the server holds 11 | **blocker** | CC-05 |
| 9 | Iterate on a saved chat | Regenerate, then Continue, then Edit message 3 → Save & Send | Regenerate shows 'unsupported_history_regeneration' (the toast is gone in 5 s). Continue shows 'unsupported_history_action_context'. Save & Send loses the edit. No request is sent, and only the hidden runtime rail explains why | **blocker** | CM-01 |
| 10 | Branch from an earlier answer | More actions → New Branch | Opens 'Forked conversation' with no breadcrumb or toast. The server records no parent link, and the fork can't be found by its source's prefix | major | CM-12 |
| 11 | Save a useful answer to Notes | Opens the overflow menu in a server-backed chat | No Save to Notes, Save to Flashcards or feedback. serverMessageId is undefined on all 6 loaded messages. The menu shows role-play C/I/N letter icons and a disabled 'Read aloud' | **blocker** | XP-02, CM-09 |
| 12 | Organize chats | Uses Rename, Pin, folders, Select chats, Delete, the Trash filter, Restore and Delete permanently | Rename works via a modal and inline in the header. Pin is browser-local. Folders exist only in bulk mode. Trash and permanent delete give no toast or Undo. There is no export anywhere on /chat | minor | CS-09, CS-10, XP-18, CS-N2† |
| 13 | Start a fresh chat (a high-frequency action) | 'New saved chat' or Ctrl+Shift+U, types, presses Enter | A 'request_config_scope_changed' toast appears, the typed message is erased and nothing is sent. In another state, the 'new' chat's first message was appended to the previously open server chat | **blocker** | CS-01 |
| 14 | Compare 2 models | Alt+Shift+C; Modes → Compare responses; More tools → Compare models | All three do nothing ('Compare responses — Off' is greyed with no reason). With the flag forced on, the composer grows to about 470 px with overlapping notices, and 'Add models' opens the wrong dialog | **blocker** | CM-05 |
| 15 | Attach knowledge (RAG) context | Search & Context, searches 'embedding model evaluation' | Search works: a generated answer plus 2 chunks with Copy/Insert/Preview/Pin. The panel takes the whole viewport, hiding the transcript and pushing SEND to y ≈ 1456 | major | CC-10 |
| 16 | Use slash commands and @mentions | Types '/' and '@' | '/' lists 5 commands with descriptions. '@' does nothing, though the placeholder promises '@ mentions' | minor | CC-11 |
| 17 | Work in an 80-message thread | Cmd+F 'Part 17', scrolls to the middle, sends | No jank. Search shows '1 / 2', but the match sits at the bottom edge and highlights persist after closing. After sending, the view stays 11,690 px above the new message | minor | CS-11, CM-11 |
| 18 | Learn the shortcuts | Opens the Shortcuts panel, presses '?', opens the header shortcuts modal | Three surfaces disagree. '?' opens 'Search pages…'. The listed Alt+1-7, Ctrl+E and Alt+W do nothing | major | XP-07, XP-10 |
| 19 | Decide whether the rails earn their space | Opens the Context and Runtime rails | They take about 40% of the width, mostly with 'Empty'/'Idle' chips, and restate the model 4 times in jargon. The one useful line ('Regeneration is unavailable…') isn't next to the Regenerate button | minor | CO-04 |
| 20 | Work on a 1024 px laptop | 1024x768, with the sidebar and then both rails open | The header takes 2 rows, the toolbar 3 rows, and the composer about 300 px. With the rails open, the textarea is about 90 px wide and the transcript about 130 px tall | major | CO-06, CO-04 |
| 21 | Start a character chat | Header 'Character' → Ada | Works well: greeting picker with Reroll/Select, and character and context chips. The composer entry is a 16x16 icon | none | AX-16 |

† CS-N2 ('Restore' from Trash brings back an empty chat) was found in a later verification pass, not on this walk. It belongs to this step's flow.

Key evidence: `04b-palette-embedding.png` and `02d-search-embedding.png` (steps 2-3), `15a-prompt-menu.png` (step 8), `27a-regen.png` (step 9), `07c-overflow.png` (step 11), `19z-after-enter-1.png` (step 13), `13a-compare-on.png` (step 14).

**Moments that matter**
- **Pain 1. 'New chat' destroys the next message, or sends it into the previous chat (CS-01).** This is Riley's most frequent action, and it fails with a raw code.
- **Pain 2. Iteration on saved work is impossible (CM-01).** Regenerate, Continue and Edit & Send are visible on every message and fail on every saved chat. Save to Notes is also missing there (XP-02).
- **Pain 3. There is no fast way back to a chat (CS-02).** No palette entry, no URL, history collapsed on every load, and results hidden behind the footer.
- **Delight. Character chat entry.** Character list, greeting picker with Reroll, and an 'Include greeting in context' toggle.
- **Delight. Lightweight organizing.** Inline rename from the header title, plus fast thread search over 80 messages with no lag.

---

### 3.6 Power user · Cross-page (chat ↔ notes, navigation, responsive)

*Reviewer `pu-cross`. WebUI `/chat` and `/notes` at 390, 768, 1024, 1440 and 1920 px, light and dark themes, plus an extension options-page round trip. Evidence in `shots/pu-cross/`.*

Chat-to-note works in one chat out of five. Where it does, it produces a note titled 'Snippet: &lt;chat title&gt;' with no tags, after a silent toast, and the note shows link-blue UUIDs that don't click. 'Open conversation' reaches the chat but not the saved message, and browser Back returns to an empty /notes. Going the other way, the only notes-to-chat path is buried RAG search. ⌘K can't find either kind of object, and neither page keeps its state in the URL. Responsive layouts never overflow horizontally, but at tablet and phone widths the pages spend most of the screen on chrome.

| # | Goal | What the user does | What happens | Friction | Issue IDs |
|---|---|---|---|---|---|
| 1 | Reopen a past server chat | Sidebar → Recent conversations → search 'LoRA' | Search across 49+ chats is fast, and the tab title follows the chat. The URL stays /chat. Thirteen shortcut links leave room for about one result row | minor | CS-02, XP-05 |
| 2 | Save an answer to Notes | More actions → Save to Notes | Offered in this chat. The toast says only 'Saved to Notes': no link, no title choice, no tags, and no saved state on the message | minor | XP-03 |
| 3 | Find the saved note | Goes to /notes and searches 'Snippet' (the title was learned from the API) | The note is titled 'Snippet: Explain LoRA fine-tuning' with no tags. The row shows 'Linked to conversation: d64f6a52-… · msg 00a5fd30-…' in link blue, and it isn't clickable. 'Origin: Saved from Chat' is correct | major | XP-03, XP-14 |
| 4 | Edit the note | Types, presses Cmd+S | The save works. The header link line still shows a raw message UUID and isn't clickable | minor | XP-03, XP-14 |
| 5 | Go back to the source chat | More actions → Open conversation | The full conversation loads in /chat, but it doesn't scroll to or highlight the saved message | minor | XP-03 |
| 6 | Return to the note | Browser Back | /notes loads with the selection and search gone, showing the empty 'Select or create a note' editor | major | XP-05 |
| 7 | Save another answer from a chat reached via a note | Opens the overflow in that chat and in 4 other seeded chats | Save to Notes, Save to Flashcards and Pin are missing, along with timestamps and feedback. Only 1 of 5 server chats offers Save to Notes, and the extension options page behaves the same | **blocker** | XP-02 |
| 8 | Jump between a note and a chat with ⌘K | Types LoRA, asyncio, 'Explain LoRA' and Snippet | 'No results found', or navigation and settings commands only. There are no notes, chats or recents | major | XP-06 |
| 9 | Switch areas with Alt shortcuts | Alt+1 on /notes, Alt+5 on /chat | Alt+1 works. Alt+5 is swallowed because /chat autofocuses the composer | major | XP-10 |
| 10 | Use browser history and deep links | Opens two notes, Back/Forward, reload, `/notes?source_ref_id=…` | Back skips both note selections and lands on /chat, and Forward shows an empty editor. The deep link works, but the URL goes stale after switching notes, so reload opens the wrong note. Reloading /chat does restore the last chat | major | XP-05 |
| 11 | Use a note as context for a question | Looks for 'Chat about this note', then uses Search & Context → Notes | There is no such action. A phrase query returns nothing, while a single word works. Insert pastes raw text with no link back, the first Insert belongs to the generated answer, and the panel pushes the composer below the fold | major | XP-04, CC-10 |
| 12 | Use the Notes Dock from chat | Clicks the rail's 'Open Notes Dock' | The dock floats over the transcript, and its list is labelled 'Archive'. It has no insert-into-chat action, and its icon is identical to the Notes navigation icon | minor | XP-11 |
| 13 | Check responsive layouts | Loads both pages at 390, 768, 1024 and 1920 px | No horizontal overflow anywhere. Tablet /chat shows a 2x2 grid of rail buttons. Tablet /notes truncates search to 'Se…' and wraps the header to 3 rows. At 1920 px, note lines are about 1,390 px long. On a phone, note text starts at y ≈ 740 of 844, help icons are 11x11, and there is no header Search | major | CO-05, NO-03, AX-16 |
| 14 | Navigate on mobile | Taps Browse notes and 'Expand sidebar' at 390 px | The notes drawer closes only via its backdrop. Global navigation opens as a nested sidebar inside a drawer, with two 'Chats' headers and two close controls | minor | AX-05, XP-11 |
| 15 | Compare light and dark themes | Switches theme and runs axe color-contrast | Dark /chat is clean. Dark /notes primary buttons measure 4.02:1. Light theme fails on the tag placeholder (1.64:1), the 'Healthy' badge (2.94:1) and subtle text. /chat has two theme toggles | minor | AX-07, CC-08 |
| 16 | Browse all ~170 notes | Clicks pages 2 and 5 | The same 100 rows on every page, while the API reports 173 | **blocker** | NL-01 |
| 17 | Repeat the round trip in the extension options page | Deep link → Open conversation → Back | All three work, and the hash router restores the note on Back (better than the WebUI). The tab title keeps the chat's name while Notes is showing. Save to Notes is missing on the reopened chat | minor | XP-19, XP-02 |

Key evidence: `s03-c-note-open.png` (step 3), `s03-g-after-back.png` (step 6), `s14-a-overflow-after-openconv.png` and `s15-a-overflow-sidebar-path.png` (step 7), `s04-b-palette-LoRA.png` (step 8), `s10-768x1024-notes.png` (step 13).

**Moments that matter**
- **Pain 1. The loop breaks on the second lap (XP-02).** Save to Notes is missing on most server chats, including the very chat a note's 'Open conversation' leads to.
- **Pain 2. Provenance exists but can't be used (XP-03).** Every saved note gets an identical 'Snippet:' title and no tags, and its links are dead UUIDs. The back-link doesn't land on the message.
- **Pain 3. Notes and chats have no address (XP-05, XP-06).** No URL state, no palette search and Back skips selections, so 'go back to that thing' always starts from scratch.
- **Delight. Origin is recorded and the back-link loads reliably.** 'Origin: Saved from Chat', the resolved conversation title and 'Open conversation' work in both the WebUI and the options page. The options page's hash router even restores the note on Back.
- **Delight. Zero horizontal overflow at 390-1920 px on both pages.** Dark /chat also passes axe color-contrast.

---

### 3.7 Power user · Extension

*Reviewer `pu-ext`. Side panel (`sidepanel.html`) at 400-420 px with side-panel storage emulated, options page (`options.html#/chat`, `#/notes`), seeded server data. Evidence in `shots/pu-ext/`.*

The panel opens on a Companion dashboard of 'Setup required' cards rather than on chat. History search is fast, but:
- Opening a result overwrites the current tab's conversation, which is data loss.
- The panel never refreshes from the server, so continuing after full-page work silently forks the chat.
- The right-click 'Delete — cannot be undone' and Rename only touch the local tab.

Hand-offs each keep half the context. The Notes side of the extension is the WebUI's Notes, with the same 100-note cap and the same broken Print.

| # | Goal | What the user does | What happens | Friction | Issue IDs |
|---|---|---|---|---|---|
| 1 | Open the panel as a returning user | Opens the side panel at 420 px | It opens on Companion Home, where every card says 'Setup required'. 'Open Chat' is about 3,060 px down | major | XS-17 |
| 2 | Find an old chat | Ctrl+B, then searches 'LoRA', 'embedding' and 'Morrow' | Server results arrive in about 3 s with topic tags. The default list shows open tabs only. Focus stays in the composer, and the title-only search misses a character match | minor | XS-06 |
| 3 | Open a search result | Clicks two results | Each opens in a new tab, but the drawer stays open over a transcript one letter wide. The previously active tab is left holding a detached copy, relabelled from its first message | major | XS-01, XS-03 |
| 4 | Keep tab A while looking something up | Sends in tab A, then searches 'Spanish' and opens that chat | Tab A, still labelled 'bloom filters', now shows the Spanish chat under a red 'Selected history unavailable' card. The bloom-filter conversation is gone from the UI, and searching 'bloom' finds only the corrupted tab | **blocker** | XS-01 |
| 5 | Resume after closing the panel | Reopens the side panel | It resumes straight into the last tab, and per-tab drafts persist ('Draft saved') | none | — |
| 6 | Switch model | 'Select a Model', search, pick | Search and the provider filter work, but entries truncate to 'Custom OpenAI API - m…'. In Casual mode the picker disappears after the pick. Palette 'Switch Model' opens the settings modal | minor | XS-02 |
| 7 | Use the Pro controls in a narrow panel | Pro at 400 px; expands Model Parameters and Provider & API | The composer grows to 473-570 px of a 900 px panel. The parameters show Temperature 0.7 and Top P 0.9, while 'Current Chat Model Settings' shows them unset | major | XS-15, XS-16 |
| 8 | Pull knowledge into the prompt | Knowledge Search with Balanced, then Fast | Balanced times out after about 45 s (environmental), with Retry. Fast returns in about 5 s, and Insert adds a titled snippet. The panel pushes the textarea about 1,200 px down | minor | XS-15 |
| 9 | Work efficiently by keyboard | Uses Ctrl+K, '/', Shift+Esc, Tab and Ctrl+E | The palette lists recent tabs and actions, and Shift+Esc focuses the composer. Ctrl+E (described as 'Toggle Search & Context') silently ticks 'Chat with current page'. Typing 'notes' in the palette finds nothing | major | XS-08, XS-13 |
| 10 | Use Character, TTS clips and artifacts | Opens each at 420 px | Character lists 2 of 4 tracked chats, in overlay/tracked jargon. TTS clips shows only an empty state (no TTS provider is configured). In the artifact footer, 'Run (N/A)' sits outside the panel | minor | XS-12, XS-18 |
| 11 | Hand off to the full page | 'Expand in full page' | The same server conversation opens with its title and a 'Saved' badge (good). The UI mode is shared, so it opens in Pro, where the 'Copy', 'Edit' and 'Redo' labels overlap inside 32 px pills | minor | CM-13, XP-18 |
| 12 | Continue in the full page, then come back | Sends in the options page, reopens the panel, sends again | The panel shows its stale local snapshot. Sending created a second child of the same parent on the server, silently forking the chat. The full page now shows only the panel's branch | major | XP-08 |
| 13 | Carry an unsent draft to the full page | Pro → More tools (past 150 MCP toggles) → 'Continue in WebUI' | It opens the extension options page, not the WebUI. The draft lands in that page's last session, a different chat, so Send would post into the wrong conversation | major | XP-09 |
| 14 | Export, rename and delete a chat | Right-click → Export → Markdown; right-click → Delete → confirm | Export downloads clean Markdown. Delete warns 'cannot be undone' but only closes the tab: the chat is still on the server and still searchable. Rename only relabels the tab (from code) | major | XS-07 |
| 15 | Browse ~176 notes in options #/notes | Reads the count, pages to page 5 | 'NOTES 100 TOTAL', the same 100 rows on every page, and Next is disabled. The WebUI behaves identically | **blocker** | NL-01 |
| 16 | Deep-link, export and print a note | `#/notes?source_ref_id=…`, Export → .md, Print / Save as PDF | The deep link opens the note but doesn't highlight it in the list. The .md export works. Print opens a blank tab *and* toasts 'Please allow pop-ups'. The tab title is 'tldw Assistant — Options' | major | NE-04, XP-19 |
| 17 | Check chat history parity in options | Opens Recent conversations → Load → searches 'LoRA' | Same as the WebUI: collapsed, server-only and loaded on demand. A stored regenerated answer renders as a trailing standalone message with no variant pager, in the side panel as well | minor | CS-02, CM-N2 |

Key evidence: `01a-sidepanel-root.png` (step 1), `11b-tabA-top.png` (step 4), `23a-full-fork.png` (step 12), `33b-continued.png` (step 13), `27b-delete-confirm.png` (step 14), `24a-opt-notes.png` (step 15), `26-ext-print.png` (step 16).

**Moments that matter**
- **Pain 1. Opening a past chat destroys the current one (XS-01).** It happens on the panel's primary retrieval path, with no warning, and leaves a corrupted, mislabelled tab behind.
- **Pain 2. Two surfaces, one chat, silent forks (XP-08).** The panel never re-reads the server, so working in both places hides turns on both. Hand-offs keep only half the context (XP-09).
- **Pain 3. A destructive confirmation that doesn't do what it says (XS-07).** 'Delete — cannot be undone' leaves the chat on the server and in search, and Rename is tab-only.
- **Delight. Fast cross-chat search with topic tags, and resume-on-reopen.** The panel reopens into the last tab with per-tab drafts.
- **Delight. Per-chat Markdown/JSON export from the right-click menu.** It is the only per-chat export anywhere in the product. 'Expand in full page' also reliably carries the server conversation and its title.

#### Capability parity: WebUI vs extension options vs extension side panel

This table was reconstructed from the `pu-ext` parity notes and the `ft-ext` journey. **Bold** marks a defect or gap. LIVE means observed live, CODE means inferred from code, and 'not walked' means not exercised. The options page runs the same shared UI code as the WebUI, so it inherits the WebUI's behaviour and defects. The side panel is a separately composed experience with its own model rules, history model, persistence semantics and destructive actions.

| Capability | WebUI (`/chat`, `/notes`) | Ext options (`options.html`) | Ext side panel | Issues |
|---|---|---|---|---|
| ***Getting started*** | | | | |
| Model on first send | **Auto-selects unreachable 'Ollama / gemma3:1b · Healthy'; server default ignored** (LIVE) | **Same** (LIVE) | **No default: first send rejected with 'Please select a model'** (LIVE) | CC-01, XS-02 |
| Model control | Toolbar chip; **mouse-only; name truncated** | Same | **Unlabeled brain icon; hidden after the first pick in Casual**; Pro control row; palette 'Switch Model' opens the settings modal | AX-01, CC-06, XS-02 |
| Send-failure recovery | **'Turn needs review': no cause, Retry or switch** | **Same** (LIVE) | **Same, and no picker on screen** (LIVE) | CC-02 |
| Pending / Stop | Two differently labelled Stop buttons (composer and in-bubble); **Stop presented as a failure** | Shared code (not walked) | **'QUEUE', no Stop** (LIVE) | CM-02, XS-10 |
| Guided onboarding | Notes tour auto-starts (**scrolls, clips, ends at step 6**); **/chat 'Take a quick tour' does nothing** | Notes tour (**scrolls 315 px, stays scrolled after Skip**) | None; **opens on a Companion 'Setup required' dashboard** | NO-02, CO-02, XS-17 |
| Slash / @ | 5 slash commands with descriptions; **'@' promised, does nothing** | Shared code (not walked) | /search, /web, /vision, /model with descriptions | CC-11 |
| ***Chat history & persistence*** | | | | |
| What 'Saved' means | **'Saved · Locally + Server', but new chats never reach the server** | **Same** | **'Save chat to history: Locally + Server', but never written** (LIVE) | CS-03, XS-05 |
| Browse past chats | **Collapsed, server-only list below 13-14 nav shortcuts** | **Same, plus a 'Load conversations' step** | **Open tabs only; history appears only while searching** | CS-02, XS-06 |
| Resume last chat | Restores last session | Restores last session (LIVE) | Restores tabs when tab state exists; **otherwise Companion Home** (LIVE) | XS-17 |
| Open a past chat | In place | In place | **New tab, and the previously active tab is overwritten (data loss)** (LIVE) | XS-01 |
| Find a chat from ⌘K | **No chats or recents** | **Same** | Recent tabs + 'Search chat history' | XP-06 |
| Cross-surface sync | n/a | **Shows only the panel's branch after a fork** (LIVE) | **Never refreshes from the server; sending forks the chat** (LIVE) | XP-08 |
| Hand-off | n/a | Target of 'Expand' (**history, no draft, Focus mode**) and 'Continue in WebUI' (**draft, wrong chat**) | Source of both; **'Continue in WebUI' opens the extension, not the WebUI** | XP-09 |
| Rename chat | Server (modal + inline header) | Server | **Tab label only** (CODE) | XS-07 |
| Delete chat | Soft delete to Trash, restorable; **no toast or Undo**; **restore returns an empty chat** | Same | **'Cannot be undone', but only closes the tab; chat stays on the server** (LIVE) | CS-10, CS-N2, XS-07 |
| Per-chat export (MD/JSON) | **None (Chatbooks only)** | **None** | Right-click Export works (LIVE) | CS-09 |
| Pin chat | localStorage per origin | Extension localStorage | Tab-store pin; **none shared across surfaces** | XP-18 |
| UI mode (Casual/Pro) | Own copy | Shared with side panel (LIVE) | Shared with options | XP-18 |
| ***Working with answers*** | | | | |
| Regenerate / Continue / Edit & Send | **Fail with raw codes** (LIVE) | **Same** (shared code) | Not walked | CM-01 |
| Stored alternative replies | **Flattened onto the end of the thread** | **Same** (LIVE) | **Same** (LIVE) | CM-N2 |
| Save answer to Notes | **Missing on new chats and on most server chats** | **Missing** (LIVE) | **Not in the message menu**; only the browser context-menu quick-save (**no tags, no 'Open note'**) | XP-01, XP-02, XS-13 |
| Compare models | **Advertised in 4 places, disabled by a hidden flag** | **Same** | Not offered | CM-05 |
| Knowledge / RAG | Search & Context panel (**takes over the viewport**), /search | Same | Pro Knowledge Search works (Fast preset); **Ctrl+E toggles a different, hidden 'Chat with current page' mode** | CC-10, XS-08 |
| Model parameters | Chat Settings modal + runtime rail | Same | **Pro inline panel shows invented 0.7/0.9; the modal shows them unset** | XS-16 |
| History-path chips | **'Use no prior messages' always visible** | **Same; one click blanks the thread** (LIVE) | **Same** | CC-04 |
| Composer footprint | About 240-300 px at 1024 px (**3-row toolbar**) | **Pro: about 59% of the page** | **Pro: about 60-65% of the panel** | CO-06, XS-15 |
| Artifacts | Right panel, auto-opens | Same | Drawer; **footer overflows at 420 px** | XS-18 |
| ***Navigation & keyboard*** | | | | |
| Shortcuts | Alt+1…0 (**swallowed by the composer; likely broken on macOS**), '?' (**opens a page navigator**), ⌘K | Same | ⌘K, Ctrl+E, Alt+W, Ctrl+B; **no Notes navigation** | XP-07, XP-10, XS-08 |
| Document title | Follows the page ('Notes \| tldw', chat title) | **Notes: 'tldw Assistant — Options' or the last chat's title** | n/a | XP-19 |
| ***Notes*** | | | | |
| Browse 170+ notes | **Capped at 100, inert pager** (LIVE) | **Identical** (LIVE) | No Notes UI; capture via context menu only | NL-01, XS-13 |
| Editing (WYSIWYG, autosave, wikilinks) | See 3.1 and 3.4 | Same NotesManagerPage code (not re-walked) | n/a | NE-01, NS-01, NE-02 |
| Note .md export | Works | Works (LIVE) | n/a | — |
| Print / Save as PDF | **Broken: blank tab + pop-up blame** (LIVE) | **Broken** (LIVE) | n/a | NE-04 |
| Deep link `?source_ref_id` | Works; **URL goes stale after switching notes** | Works; hash router restores the note on Back | n/a | XP-05 |

**Reading the table.**

- **Two different products.** Parity between the WebUI and the options page is near-total, so their bugs are shared, though the options page's hash-router Back is better. The side panel, by contrast, diverges on most chat rows. Model selection, history browsing, opening a past chat, sync, delete and rename semantics differ in ways that make it feel like a separate product (XP-12).
- **Capabilities stranded on one surface.** Per-chat export exists only in the side panel. Compare and the Search & Context panel exist only on the full page. Notes exist in the side panel only as a write-only context-menu capture.
- **No cross-surface story.** No organizing state (pins, recents, UI mode) is shared consistently across surfaces (XP-18).

---

## 4. Prioritized issue list

All 151 kept issues, ordered by priority, then severity, then ID. Surfaces: **W** = WebUI, **O** = extension options page, **S** = extension side panel. Full write-ups, evidence and recommendations are in §5 (Notes), §6 (Chat) and §7 (Extension, cross-page and accessibility).

| Priority | Sev 4 | Sev 3 | Sev 2 | Sev 1 | Total |
|---|---|---|---|---|---|
| P0 | 6 | 16 | — | — | 22 |
| P1 | — | 9 | — | — | 9 |
| P2 | — | — | 97 | — | 97 |
| P3 | — | — | — | 23 | 23 |
| **Total** | 6 | 25 | 97 | 23 | **151** |

| ID | Priority | Sev | Effort | Area | Issue | Surfaces |
|---|---|---|---|---|---|---|
| **CS-01** | P0 | 4 | M | Chat · Sessions & history | 'New chat' does not start a clean conversation: old messages leak into the next request, the next send can be destroyed with a raw 'request_config_scope_changed' error, or is appended to the previous server chat | W · O |
| **CS-02** | P0 | 4 | M | Chat · Sessions & history | Users can't find or return to previous chats: history is buried under 14 navigation shortcuts, shows only server chats, resets on every load, searches titles only, and isn't in ⌘K or the URL | W · O |
| **NE-01** | P0 | 4 | M | Notes · Editor | WYSIWYG editor inserts every keystroke at the start of the document (text typed backwards) and autosaves the corrupted text | W · O |
| **NL-01** | P0 | 4 | M | Notes · List & organization | Notes list is capped at the 100 most recent notes: pager, total, sort, Timeline and anything built on the browse list work on the same 100 rows | W · O |
| **NL-02** | P0 | 4 | S | Notes · List & organization | 'Export matching notes' (MD/CSV/JSON) loops over the same 100 notes: up to 1,000 requests and a 163 MB file of duplicates that still misses notes, reported as success, with no Cancel | W · O |
| **NS-01** | P0 | 4 | M | Notes · Saving | Unsaved note edits are silently discarded on in-app navigation (5s debounce, no flush on unmount, no prompt) | W · O |
| **CC-04** | P0 | 3 | M | Chat · Composer & context | Always-visible 'Review conversation history / Use no prior messages' controls expose an internal history-path model; one click silently blanks the context (and in the sidepanel the visible thread) | W · O · S |
| **CC-05** | P0 | 3 | M | Chat · Composer & context | The chat Prompt picker ignores the server prompt library ('No saved prompts' with 11 server prompts) | W · O |
| **CM-01** | P0 | 3 | L | Chat · Messages | Regenerate, Continue and Edit → 'Save & Send' fail on every chat with raw error codes (the edit is silently discarded) while the buttons stay visible | W · O |
| **CM-N1** | P0 | 3 | S | Chat · Messages | Slow models time out after 30s with no first token, even though the startup timeout is 120s, and the turn ends in an unexplained 'Turn needs review' | W · O · S |
| **CS-03** | P0 | 3 | M | Chat · Sessions & history | Chats are never saved to the server, yet the UI says 'Saved · Locally + Server' | W · O |
| **CS-04** | P0 | 3 | L | Chat · Sessions & history | Reloading or leaving /chat mid-reply erases the question and partial answer, replaced by a jargon recovery card, with no warning | W · O · S |
| **CS-05** | P0 | 3 | S | Chat · Sessions & history | 'Clear conversation' reports 'Conversation cleared' but removes nothing, and old messages are still sent to the model | W · O |
| **CS-N2** | P0 | 3 | M | Chat · Sessions & history | 'Restore' from Trash brings back an empty chat: its messages stay deleted although the UI says 'Chat restored.' | W · O |
| **NE-02** | P0 | 3 | M | Notes · Editor | [[Wikilinks]] are broken end to end: can't be followed, render as raw text, bracketed titles never resolve, resolution/autocomplete only use the current sidebar page, and no backlinks or graph edges result | W · O |
| **NE-04** | P0 | 3 | S | Notes · Editor | 'Print / Save as PDF' always fails: opens a blank tab and blames pop-up blocking (webui and extension) | W · O |
| **NL-03** | P0 | 3 | M | Notes · List & organization | Bulk 'Assign tags' silently replaces every selected note's existing tags, with no undo | W · O |
| **NS-N1** | P0 | 3 | S | Notes · Saving | The 409 toast's 'Reload notes' does not reload: it silently moves the base version forward, and the next autosave overwrites the other tab's/device's changes | W · O |
| **XP-02** | P0 | 3 | S | Cross-page & parity | Save to Notes, Save to Flashcards and feedback are missing on most server-backed chats (incl. the chat opened from a note) because loaded messages lose serverMessageId | W · O |
| **XP-08** | P0 | 3 | L | Cross-page & parity | Sidepanel tabs never refresh from the server; continuing a chat after working on it in the full page silently forks it and hides the other branch on both surfaces | O · S |
| **XS-01** | P0 | 3 | M | Ext. side panel | Opening a past chat from sidepanel search overwrites the current tab's conversation (silent data loss and duplicate tabs) | S |
| **XS-07** | P0 | 3 | S | Ext. side panel | Sidepanel right-click 'Delete — cannot be undone' only closes the tab (chat stays on server and in search); 'Rename' only relabels the local tab | S |
| **AX-03** | P1 | 3 | M | Accessibility | Keyboard-only use of Notes is impractical: 35 Tabs to the list, 3 tab stops per row, and no way from the list into the editor or back | W · O |
| **AX-04** | P1 | 3 | M | Accessibility | At 200-400% zoom the chat composer fills the viewport: transcript is 0px at 320x256 CSS px and the 'Exit focus' pill covers the input | W · O |
| **AX-05** | P1 | 3 | S | Accessibility | Notes off-canvas list (mobile or desktop at 200% zoom) keeps ~285 invisible controls in the Tab order when closed; when open it takes no focus and Esc doesn't close it | W · O |
| **CC-01** | P1 | 3 | M | Chat · Composer & context | First-run default model is an unreachable provider labelled 'Healthy'; the server's default provider is ignored | W · O |
| **CC-02** | P1 | 3 | M | Chat · Composer & context | Send failures show a jargon 'Turn needs review' recovery panel with no cause, no Retry and no model switch (webui, options and sidepanel) | W · O · S |
| **CC-N1** | P1 | 3 | S | Chat · Composer & context | Typing a slash command runs it on every keystroke (/web and /search flip settings while you type and Enter cancels them; /model opens settings mid-word and takes focus) | W · O |
| **CM-02** | P1 | 3 | M | Chat · Messages | Stop is presented as a failure ('Turn needs review'), early Stop can destroy the user's message with a raw error, and the in-bubble Stop is unnamed | W · O |
| **CS-N1** | P1 | 3 | S | Chat · Sessions & history | Choosing the Trash (or Character) filter can lock users out of their chat list: when that view is empty the filter control disappears, and the choice is saved across reloads | W · O |
| **XS-08** | P1 | 3 | S | Ext. side panel | Ctrl+E silently toggles a hidden 'Chat with current page' mode (labelled as knowledge search in the palette) and hijacks macOS end-of-line | S |
| **AX-01** | P2 | 2 | M | Accessibility | Composer model picker can't be operated from the keyboard; palette 'Switch Model' and /model open a settings modal instead | W · O |
| **AX-02** | P2 | 2 | S | Accessibility | Message 'More actions' popover can't be reached by keyboard (Branch, Continue, Delete, Save to Notes), and generation Info opens on hover only | W · O · S |
| **AX-06** | P2 | 2 | M | Accessibility | Toasts (save, delete, Undo, errors) are silent to screen readers, and deleting a note drops focus to `<body>` with Undo available for only 10s | W · O · S |
| **AX-07** | P2 | 2 | M | Accessibility | Text contrast fails at the colour-token level in both themes (primary buttons incl. SEND, success badges, menu group titles, placeholders, code syntax, subtle text) | W · O · S |
| **AX-08** | P2 | 2 | S | Accessibility | Focus indicators are too faint on every Ant Design button (1.35-1.57:1) and absent on two sidepanel header links | W · O · S |
| **AX-09** | P2 | 2 | S | Accessibility | Focus is lost to `<body>` after closing the Notes shortcuts modal, Esc on chat thread search, and deleting a note; the Notes modal also lets Tab escape | W · O |
| **AX-10** | P2 | 2 | S | Accessibility | Markdown tables in chat answers and the notes preview render without a `<table>` element, so screen readers lose table semantics | W · O · S |
| **AX-11** | P2 | 2 | M | Accessibility | Chat transcript live region is too broad: interface chrome and the post-stream re-render are announced with the reply; the sidepanel nests two logs | W · O · S |
| **AX-12** | P2 | 2 | M | Accessibility | Slash-command menu and [[wikilink]] suggestions aren't exposed as comboboxes, so screen readers stay silent while arrowing (axe critical) | W · O · S |
| **AX-14** | P2 | 2 | S | Accessibility | Landmarks and headings are malformed: nested `<main>` on /chat, chat history sidebar not a landmark, no h1 and only an h5 on /notes | W · O · S |
| **AX-15** | P2 | 2 | S | Accessibility | Selected view/mode is shown by colour only (Notes/Trash, List/Timeline/…, Markdown/WYSIWYG): no aria-pressed, invisible in forced colours; sidebar toggles lack aria-expanded and share a name | W · O |
| **AX-16** | P2 | 2 | S | Accessibility | Undersized touch/pointer targets across Notes, Chat and the sidepanel (11x11 help icons, 16x16 checkboxes/persona icon, 22px search, 28x16 switch, 22px-tall '•••' chip) | W · O · S |
| **AX-18** | P2 | 2 | S | Accessibility | Global keyboard-shortcuts modal has a scroll area keyboard users can't scroll | W · O |
| **CC-03** | P2 | 2 | S | Chat · Composer & context | Chat starts in a false 'You're offline' state, and Enter pressed during it is silently dropped despite the QUEUE button | W · O |
| **CC-06** | P2 | 2 | S | Chat · Composer & context | Model identity is shown three different ways and truncated (model name cut, raw provider ids), and the picker's scope toggle doesn't visibly switch | W · O |
| **CC-08** | P2 | 2 | M | Chat · Composer & context | The same capability has different names and multiple homes across composer menus (Compare, Web search, Knowledge, Saved/Temporary, Shortcuts, theme) | W · O · S |
| **CC-10** | P2 | 2 | M | Chat · Composer & context | Search & Context takes over the whole viewport: transcript hidden, Send pushed off-screen, source chunks below an LLM-generated answer | W · O |
| **CC-11** | P2 | 2 | S | Chat · Composer & context | Composer placeholder promises '@ mentions' but '@' does nothing in the webui; slash commands are minimal | W · O |
| **CC-12** | P2 | 2 | S | Chat · Composer & context | Improve-prompt (wand) menu renders with a transparent background so its items collide with the composer | W · O · S |
| **CC-N2** | P2 | 2 | S | Chat · Composer & context | Provider failures are reported as 'Something went wrong while talking to your tldw server' and the model chip stays 'Healthy' | W · O |
| **CM-03** | P2 | 2 | M | Chat · Messages | Cut-off or length-truncated answers look complete: no 'incomplete' marker, Continue or Retry (webui and sidepanel) | W · O · S |
| **CM-04** | P2 | 2 | M | Chat · Messages | Which model answered (and sources/generation info) is lost after reload; the server never stores model metadata | W · O |
| **CM-05** | P2 | 2 | S | Chat · Messages | Compare mode is advertised in four places but disabled by a hidden feature flag, and when forced on its 'Add models' opens the wrong dialog | W · O |
| **CM-06** | P2 | 2 | S | Chat · Messages | Long waits show no elapsed time or expectation, then silently time out into 'Turn needs review'; screen readers hear 'checkpoint N' every 5s | W · O |
| **CM-07** | P2 | 2 | S | Chat · Messages | Code blocks in chat answers have no copy button or language label | W · O |
| **CM-09** | P2 | 2 | S | Chat · Messages | Message actions are hover-only, the visible '•••' chip vanishes on approach, and the overflow mixes role-play steering, unexplained single-letter icons and a reasonless disabled 'Read aloud' into normal chat | W · O |
| **CM-11** | P2 | 2 | S | Chat · Messages | Sending while scrolled up shows neither your message nor the reply; second answers land below the fold | W · O |
| **CM-12** | P2 | 2 | M | Chat · Messages | New Branch creates an orphan 'Forked conversation' with no parent link, breadcrumb, meaningful title or carried-over character | W · O |
| **CM-13** | P2 | 2 | S | Chat · Messages | Pro-mode message action labels ('Copy', 'Edit', 'Redo') overflow fixed 32px pills and overlap | W · O |
| **CM-N2** | P2 | 2 | M | Chat · Messages | Alternative replies stored as branches (parent_message_id siblings) are flattened onto the end of the thread and sent to the model as context | W · O |
| **CO-01** | P2 | 2 | M | Chat · Onboarding & layout | Mobile /chat (390px) opens in Focus mode with no navigation, and the composer takes ~46% of the screen with duplicate unlabeled controls | W · O |
| **CO-02** | P2 | 2 | S | Chat · Onboarding & layout | 'Take a quick tour' on /chat does nothing | W |
| **CO-03** | P2 | 2 | M | Chat · Onboarding & layout | Chat composer and chrome overload first-timers with ~20 jargon controls, duplicate buttons and decorative pseudo-buttons | W · O |
| **CO-04** | P2 | 2 | M | Chat · Onboarding & layout | Cockpit rails mostly restate state in jargon and break the layout at 1024px | W · O |
| **CO-05** | P2 | 2 | S | Chat · Onboarding & layout | Tablet /chat (<1024px) always shows a 2x2 grid of jargon rail buttons that duplicate each other and cut transcript height | W · O |
| **CO-06** | P2 | 2 | M | Chat · Onboarding & layout | At 1024px the chat chrome takes over: two-row header and a three-row composer toolbar leave ~340px for the transcript | W · O |
| **CS-06** | P2 | 2 | S | Chat · Sessions & history | A chat that fails to load shows a raw request URL that overflows its box, duplicated in a toast, with no Retry while the header still says 'Saved' | W · O |
| **CS-07** | P2 | 2 | S | Chat · Sessions & history | Chat history sidebar error has no Retry and is clipped out of view when Shortcuts is expanded | W · O |
| **CS-08** | P2 | 2 | S | Chat · Sessions & history | Chat history rows are low-density: ~4 rows per screen, short truncated titles, 'Server' on every row, grey lowercase state enums, topic repeats title, no date grouping | W · O |
| **CS-09** | P2 | 2 | S | Chat · Sessions & history | No per-chat export or 'Move to folder' on the full-page chat; export exists only behind the sidepanel's right-click menu | W · O · S |
| **CS-10** | P2 | 2 | S | Chat · Sessions & history | Moving a chat to Trash or deleting permanently gives no toast or Undo, and the Trash filter hides when search is empty | W · O |
| **CS-N3** | P2 | 2 | M | Chat · Sessions & history | After a reply is interrupted by leaving /chat, returning creates an empty server chat (title only, 0 messages) that shows up in history | W · O |
| **NE-03** | P2 | 2 | M | Notes · Editor | Long notes: a non-collapsible Table of Contents pushes the editor below the fold, TOC jumps scroll the whole page away, and Split panes don't stay in sync | W · O |
| **NE-05** | P2 | 2 | S | Notes · Editor | Markdown preview drops bullet and number markers (ordered lists lose numbering) | W · O |
| **NE-06** | P2 | 2 | M | Notes · Editor | Lists and checklists require Markdown knowledge: no checklist/numbered-list buttons, H1 only, no list continuation on Enter | W · O |
| **NE-07** | P2 | 2 | S | Notes · Editor | Code blocks in Notes Preview (and sidepanel Artifacts) render as a 'staircase': line numbers and code drift right on each line | W · O · S |
| **NE-N1** | P2 | 2 | S | Notes · Editor | Notes preview code blocks: 'View code' does nothing on /notes, and clicking the centre of its label downloads the snippet as a file | W · O |
| **NL-04** | P2 | 2 | M | Notes · List & organization | Notes results list shows only 2-3 low-density rows padded with raw UUIDs and raw markdown; search sits third and truncated | W · O |
| **NL-05** | P2 | 2 | M | Notes · List & organization | Bulk actions and Trash don't scale: no select-all, no progress or locking, no bulk undo/restore, and each restore exits Trash | W · O |
| **NL-06** | P2 | 2 | M | Notes · List & organization | Notes can never be permanently deleted and Trash can't be emptied (UI and API are soft-delete only) | W · O |
| **NL-07** | P2 | 2 | M | Notes · List & organization | Notes search discards server relevance ranking and never shows where a note matched | W · O |
| **NL-09** | P2 | 2 | S | Notes · List & organization | Timeline groups notes under the wrong month for users west of UTC (Oct 2 notes under 'SEPTEMBER 2026') | W · O |
| **NL-10** | P2 | 2 | L | Notes · List & organization | Organizing at scale: server folders have no UI, saved-filter counts mislead, Collection cards unreadable in the sidebar, pins are per-device | W · O |
| **NL-11** | P2 | 2 | L | Notes · List & organization | Graph view caps out at 100 notes, links by [[Title]] create no edges, and picking a note from the list keeps the editor hidden | W · O |
| **NL-12** | P2 | 2 | M | Notes · List & organization | Views panel mixes modes and views as seven equal buttons with no descriptions, and duplicates the 'Captured' filter as 'Inbox' | W · O |
| **NL-13** | P2 | 2 | S | Notes · List & organization | Collection view is a dead end: 'Create or select a collection to start' but the controls are hidden in collapsed ORGANIZE | W · O |
| **NL-N1** | P2 | 2 | M | Notes · List & organization | Tag filter matches by substring and ORs multiple tags: picking 'eval (0)' shows 6 'evaluation' notes, adding a second tag widens results | W · O |
| **NO-01** | P2 | 2 | M | Notes · Onboarding & layout | Notes page overflows at desktop widths: header scrolls away, 80px empty band and void under the sidebar, filters clipped under RESULTS | W · O |
| **NO-02** | P2 | 2 | M | Notes · Onboarding & layout | First-run Notes tour scrolls the page so the header and its targets are off-screen, clips popovers, leaves the page scrolled after Skip, and silently ends after step 6 of 8 | W · O |
| **NO-03** | P2 | 2 | M | Notes · Onboarding & layout | /notes at tablet, landscape-laptop, wide and phone widths: truncated search, three-row header actions, ~2 visible list rows, unbounded line length, chrome-first phone layout | W · O |
| **NO-04** | P2 | 2 | S | Notes · Onboarding & layout | Notes empty state is stacked on top of a fully live editor ('Select or create a note' plus active Title/Tags/editor and four create actions) | W · O |
| **NO-N1** | P2 | 2 | S | Notes · Onboarding & layout | Tours hang invisibly when a step's target is missing: TutorialRunner's retry remount stops Joyride from re-checking, so the skip-to-next-step fallback never runs and the tour is never marked complete | W · O |
| **NS-02** | P2 | 2 | S | Notes · Saving | A note typed without a title can never be saved, and the error blames the network | W · O |
| **NS-03** | P2 | 2 | M | Notes · Saving | Save-conflict recovery is a dead end: five contradictory messages, 'Save anyway' resends the stale version and fails again, and the only exit discards your edits | W · O |
| **NS-04** | P2 | 2 | M | Notes · Saving | Every autosave refetches the whole ~700 KB notes list, blanks it with a spinner and resets list scroll to the top | W · O |
| **NS-05** | P2 | 2 | M | Notes · Saving | A connection drop isn't noticed for up to 30s; the offline save queue doesn't engage and edits exist only in memory | W · O |
| **NS-06** | P2 | 2 | S | Notes · Saving | Save status is shown (and announced) three times, Save stays primary when clean, and where notes are stored is never stated | W · O |
| **NS-N2** | P2 | 2 | S | Notes · Saving | Failed autosave retries every ~5s forever (including non-retryable 400/409) and, during a conflict, opens the 'Remote changes detected' modal unprompted and takes focus | W · O |
| **XP-01** | P2 | 2 | M | Cross-page & parity | 'Save to Notes' never appears on answers in unsynced chats (webui new chats and all extension chats); the only path is a manual copy-paste that loses provenance | W · O · S |
| **XP-03** | P2 | 2 | M | Cross-page & parity | Chat-to-note provenance is one-way and opaque: silent toast, identical 'Snippet: &lt;chat title&gt;' titles, no tags, link-styled raw UUIDs that don't click, and 'Open conversation' doesn't land on the message | W · O |
| **XP-04** | P2 | 2 | M | Cross-page & parity | No 'Chat about this note': the only notes-to-chat path is buried in the composer's Search & Context panel and is brittle | W · O |
| **XP-05** | P2 | 2 | M | Cross-page & parity | Notes and chats have no URL state: Back/Forward skip selections, returning to /notes doesn't reopen the last note, deep links go stale, chats can't be linked or opened side by side | W · O |
| **XP-06** | P2 | 2 | M | Cross-page & parity | Header 'Search ⌘K' can't find or create any notes or chats — it is only a command palette; note capture by keyboard is slow | W · O |
| **XP-07** | P2 | 2 | M | Cross-page & parity | Keyboard shortcut docs contradict behaviour across Notes and Chat: '?' opens a page navigator, Ctrl/Cmd+B toggles the sidebar instead of bolding, Cmd+K is documented as notes search, and listed keys do nothing | W · O |
| **XP-09** | P2 | 2 | M | Cross-page & parity | Sidepanel → full-page hand-offs each keep half the context: 'Expand in full page' drops the draft and lands in Focus mode with no navigation; 'Continue in WebUI' keeps the draft but drops it into whichever chat the full page last had open (and opens the extension, not the WebUI) | O · S |
| **XP-10** | P2 | 2 | S | Cross-page & parity | Alt+1…0 area shortcuts don't fire from chat (composer autofocus) and likely never on macOS (event.key matching) | W · O |
| **XP-11** | P2 | 2 | M | Cross-page & parity | Global navigation is chat-centric and ambiguous: a 'Chats' sidebar with a New-Chat '+' on every page (beside Notes' own '+'), identical Notes and Notes Dock icons, and a nested mobile drawer | W · O |
| **XP-12** | P2 | 2 | L | Cross-page & parity | Sidepanel and full-page chat feel like two different products: model selection rules, vocabulary, placeholders, controls and menus all differ | W · O · S |
| **XP-13** | P2 | 2 | M | Cross-page & parity | Inconsistent vocabulary for core objects across pages (chat/conversation/session/thread; Temporary/Temp/Private; 'Saved' as mode vs status; tags vs keywords; Character/Persona/Buddy/Assistant; date idioms) | W · O · S |
| **XP-14** | P2 | 2 | S | Cross-page & parity | Raw IDs and machine strings leak into user-facing copy across Notes and Chat | W · O |
| **XP-15** | P2 | 2 | S | Cross-page & parity | All 'secondary' helper/meta text renders as 14px primary-colour text app-wide, so hints are louder than section headings | W · O · S |
| **XP-16** | P2 | 2 | S | Cross-page & parity | Constant background polling and duplicate fetches on every page (buddies every 5s, persona/profile/providers fetched 2-4x on load) | W · O · S |
| **XP-17** | P2 | 2 | M | Cross-page & parity | Destructive actions use inconsistent safety patterns and toast systems across Notes and Chat | W · O · S |
| **XP-18** | P2 | 2 | M | Cross-page & parity | Pins, UI mode and recents live in different stores per surface, so organising in one place doesn't carry over | W · O · S |
| **XS-02** | P2 | 2 | M | Ext. side panel | Sidepanel model control: no default model so the first send fails, the only control is an unlabeled brain icon, and in Casual mode it disappears once a model is chosen | S |
| **XS-03** | P2 | 2 | S | Ext. side panel | Sidepanel 'Chats' drawer pushes the conversation into a 72-132px column instead of overlaying it | S |
| **XS-04** | P2 | 2 | S | Ext. side panel | Pro mode at panel widths above 400px docks a 288px chat list that can't be closed, crushing the chat to ~130px | S |
| **XS-05** | P2 | 2 | M | Ext. side panel | Sidepanel 'Save chat to history' says 'Locally + Server', but sidepanel chats are never written to the server | O · S |
| **XS-06** | P2 | 2 | M | Ext. side panel | Sidepanel chat list shows only open tabs; past chats appear only when searching and never in the full page's history | O · S |
| **XS-09** | P2 | 2 | S | Ext. side panel | Sidepanel composer scrolls with the messages: after each reply SEND is clipped and the 'scroll to latest' pill sits on the input | S |
| **XS-10** | P2 | 2 | S | Ext. side panel | While a sidepanel request is pending the primary button becomes 'QUEUE' and no Stop is offered | S |
| **XS-12** | P2 | 2 | S | Ext. side panel | Sidepanel labels and header icons use jargon or mislead: 'Config' is voice settings, 'Open dashboard' opens Flashcards, Character offers 'Apply overlay / tracked persona chat' | S |
| **XS-13** | P2 | 2 | M | Ext. side panel | Quick-save to Notes from the sidepanel is a dead end: no tags, no 'Open note', no Notes entry in the panel, and origin reads 'Typed manually' | O · S |
| **XS-14** | P2 | 2 | S | Ext. side panel | Narrow sidepanel (360px): header title wraps, markdown tables clip their last column, and the scroll pill covers table tools | S |
| **XS-15** | P2 | 2 | M | Ext. side panel | Pro-mode composer consumes ~60-65% of the side panel (and ~59% of the extension full page), leaving ~260px for the conversation | O · S |
| **XS-16** | P2 | 2 | S | Ext. side panel | Two parameter editors disagree: the sidepanel shows invented defaults (Temperature 0.7 / Top P 0.9) while 'Current Chat Model Settings' shows them unset | S |
| **XS-17** | P2 | 2 | S | Ext. side panel | Fresh side panel opens to a Companion dashboard of 'Setup required' cards; Chat is ~3,000px down | S |
| **AX-13** | P3 | 1 | S | Accessibility | Chat composer accessibility defects: invalid aria-expanded on the textarea, unnamed in-bubble Stop button, send failures announced as status not alert | W · O |
| **AX-17** | P3 | 1 | S | Accessibility | Accessible names don't match visible labels ('Context rail' is named 'Restore context sidechannel'; 'Save & new' is 'Save and start another note') | W · O |
| **AX-19** | P3 | 1 | S | Accessibility | Chat ignores prefers-reduced-motion (shake, pulse, smooth scrolling), and the app's animation setting doesn't default from the OS preference | W · O · S |
| **CC-07** | P3 | 1 | S | Chat · Composer & context | Send is the visually weakest control in the composer, and the toolbar has no consistent size/style grammar | W · O · S |
| **CC-09** | P3 | 1 | S | Chat · Composer & context | 'Advanced controls' reveals a connection dot styled like a switch and four unlabeled response-style preset icons | W · O |
| **CM-08** | P3 | 1 | S | Chat · Messages | Inline code in chat answers shows literal backticks and no code styling | W · O · S |
| **CM-10** | P3 | 1 | S | Chat · Messages | User and assistant messages look almost identical, headers say 'Assistant' not the model, and each answer carries duplicate overflow buttons plus a feedback prompt | W · O · S |
| **CM-14** | P3 | 1 | M | Chat · Messages | First send in a new chat flashes a 'Loading selected history' panel and the user's message appears late | W · O |
| **CO-07** | P3 | 1 | S | Chat · Onboarding & layout | Floating 'scroll to latest' chevron covers content, even on an empty chat | W · O |
| **CO-N1** | P3 | 1 | S | Chat · Onboarding & layout | Mobile focus mode: the floating 'Exit focus' pill covers the top-right of message text while reading | W · O |
| **CS-11** | P3 | 1 | S | Chat · Sessions & history | In-thread search works but is hidden (Cmd/Ctrl+F only), the current match isn't scrolled into view, and highlights remain after closing | W · O |
| **NE-08** | P3 | 1 | S | Notes · Editor | 'Create study pack' is a primary header button that navigates away from Notes and exposes raw IDs | W · O |
| **NE-09** | P3 | 1 | S | Notes · Editor | Notes with tasks show a duplicate checklist strip above the preview plus jargon 'Portable markdown with best-effort task continuity' | W · O |
| **NE-10** | P3 | 1 | S | Notes · Editor | Connections panel lists the same manual link twice and uses three unexplained relation concepts | W · O |
| **NL-08** | P3 | 1 | S | Notes · List & organization | Tag filter accepts free text: Enter applies a partial, non-existent tag and pollutes 'Browse tags' | W · O |
| **NL-14** | P3 | 1 | M | Notes · List & organization | Graph view is opaque for newcomers: unlabeled nodes, jargon controls, arbitrary focus note, and the page scrolls | W · O |
| **NL-15** | P3 | 1 | S | Notes · List & organization | Delete/Trash polish: generic confirm copy, stale 'Note deleted · Undo' toast after restore, Trash shows irrelevant controls | W · O |
| **NL-16** | P3 | 1 | S | Notes · List & organization | Notes loading/error states: spinner-only list, '0 TOTAL' while loading, raw request URL in the error card, editor stays live while the API is down | W · O |
| **NL-17** | P3 | 1 | S | Notes · List & organization | 'Suggest tags' pre-selects all low-value heuristic suggestions (stop-words, plural duplicates) | W · O |
| **XP-19** | P3 | 1 | S | Cross-page & parity | Extension options page title doesn't follow the route ('tldw Assistant — Options' for Notes and Chat) and keeps the last chat's name after returning to Notes | O |
| **XP-20** | P3 | 1 | S | Cross-page & parity | Two primary blues and a teal accent compete for 'interactive/active' meaning; green is used for both modes and statuses | W · O · S |
| **XS-11** | P3 | 1 | S | Ext. side panel | Sidepanel offline state: the 'Can't reach server' card has no Retry and the read-only composer still offers a primary QUEUE button | W · S |
| **XS-18** | P3 | 1 | S | Ext. side panel | Sidepanel Artifact drawer footer overflows: 'Run (N/A)' is off-screen and permanently disabled | S |

*Notes on the list.*

- *Titles are the canonical index titles.* Where verification narrowed a claim, §§5-7 hold the corrected detail. Notably, the "unnamed in-bubble Stop" part of CM-02 and AX-13 and the "announced as status" part of AX-13 were refuted (the Stop button has the screen-reader name "Stop streaming response").
- *Effort* is S (about a day, one component), M (a few days to two weeks) or L (multi-week, cross-cutting or backend + frontend).

---


## 5. Notes page — issues and solutions

This section covers 42 confirmed Notes issues in four clusters: List, search & organization (NL, 18), Editor (NE, 11), Saving & data safety (NS, 8), and Onboarding, layout & visual design (NO, 5). Eight of them are **P0** (severity 3–4) and get a full write-up below. Severity-2 items are P2 and severity-1 items are P3; these appear in compact tables. Where verification changed a severity, the table shows the original value in brackets, for example "2 (from 3)".

*Conventions.* UI paths are relative to `apps/packages/ui/src/`. Notes files are under `components/Notes/`, and hooks under `components/Notes/hooks/`. Backend paths are relative to `tldw_Server_API/app/`; `notes.py` means `api/v1/endpoints/notes.py`. Screenshot paths are relative to the review evidence folder (`scratchpad/shots/`). "Extension options" means the built extension's options page, which shares the same Notes components. Unless noted, line numbers refer to dev @ 86e287fee7.

### Notes at a glance

| ID | P0 issue | Sev | Effort |
|---|---|---|---|
| NE-01 | WYSIWYG mode types text backwards and autosaves the corrupted text | 4 | M |
| NL-01 | The browse list is capped at the 100 most recent notes; the pager, total, sort and Timeline all work on those same 100 rows | 4 | M |
| NL-02 | "Export matching notes" repeats the same 100 notes up to 1,000 times, misses the rest, and reports success | 4 | S |
| NS-01 | Unsaved edits are silently discarded on in-app navigation | 4 | M |
| NE-02 | `[[Wikilinks]]` are broken end to end | 3 | M |
| NE-04 | "Print / Save as PDF" always fails and blames the pop-up blocker | 3 | S |
| NL-03 | Bulk "Assign tags" silently replaces existing tags | 3 | M |
| NS-N1 | The 409 toast's "Reload notes" silently overwrites the other tab's or device's edits | 3 | S |

Most of the 42 issues trace back to four root causes. Fixing these causes clears more issues than working through the list one by one:

1. **One list-API contract mismatch.** The browse and export code send `page`/`results_per_page` and read `pagination.total_items`. The server accepts only `limit`/`offset` and returns `pagination.total`. This single mismatch causes NL-01 and NL-02. It also feeds NS-04 (100 full notes downloaded on every save), NL-04 (no tag chips), NL-09 and NL-11 (Timeline and Graph inherit the 100-row cap), and NE-02 (autocomplete can't see older notes).
2. **The save pipeline has no single state model.** Autosave timing, retry policy, conflict handling, offline queueing, status copy and the navigation guard are each implemented separately and contradict each other (NS-01, NS-N1, NS-02, NS-03, NS-N2, NS-05, NS-06).
3. **Client and server use different wikilink syntax.** The client inserts `[[Title]]`; the server only indexes `[[id:UUID]]`. As a result there are no backlinks and no graph edges (NE-02, NL-11, NE-10).
4. **The page frame isn't constrained to the viewport.** `/notes` is a 135vh scrolling document (NO-01), which drives NO-02, NE-03, NL-14 and NO-03.

Unit-test mocks hid several of these bugs. A mock server honours `results_per_page` (NL-01). A mocked `window.open` returns a window even with `noopener` (NE-04). `react-joyride` is mocked out of the tour tests (NO-N1). The jsdom WYSIWYG tests never check the caret (NE-01). We recommend contract tests against the real FastAPI app, plus one Playwright smoke test per editing mode.

---

### 5.1 List, search & organization (NL)

**Browse list and export integrity**

#### NL-01 · Notes list is capped at the 100 most recent notes: pager, total, sort, Timeline and anything built on the browse list work on the same 100 rows
**Priority** P0 · **Severity** 4/4 · **Effort** M · **Surfaces** WebUI, Extension options · **Persona** Power users · **Heuristic** Nielsen #1 Visibility of system status; #5 Error prevention (data appears lost) · **Verification** Confirmed (live + code); upheld by independent skeptic

- **What happens:** The header reads "NOTES 100 TOTAL" and the footer "Showing 1-20 of 100", while the API reported 163 to 229 notes during the review (the count grew as reviewers added notes). Pages 1, 2, 3 and 5 render the same 100 rows; only the footer label changes. The "20 / page" and "50 / page" settings both render 100 rows. On extension options, Next is `aria-disabled`. Title (A-Z) sorts only the fetched 100, so notes that should sort first globally are missing. The Graph banner says "This library has 228" next to a sidebar that says 100. Browse rows show no tag chips. Search still finds the hidden notes, which hides the bug. The cause is a pure client/server contract mismatch:
  - `hooks/useNotesListManagement.tsx:394-405` requests `/api/v1/notes/?page=&results_per_page=&sort_by=&sort_order=` and calls `setTotal(pagination?.total_items || items.length)`.
  - `notes.py:2309-2316` (`list_notes`) accepts only `limit` (default 100), `offset` and `include_keywords`, and returns `pagination.total`, with no `total_items` and no sort.
  - The trash branch (`:330-351`) and the search branch (`:251-257`) of the same hook already use `limit`/`offset` correctly.
  - The mock in `NotesManagerPage.stage30.export-progress.test.tsx:202` honours `results_per_page`, which hides the drift.

  Evidence: `shots/skeptic-NL/s01-page3.png`, `shots/verify-NL/w01-p2.png`.
- **Why it matters:** First-time users with fewer than 100 notes never notice. The page then fails abruptly as the library grows: older notes look deleted, and nothing explains why. Power users cannot browse, sort or see a timeline of about half their library. The count and pager are meaningless. Everything built on the browse list (Timeline, wikilink autocomplete, Export, Graph) inherits the truncation. Every load also downloads 100 full notes. Search is the only workaround, and it only helps for notes whose terms the user remembers. This damages trust in the notes store more than any other issue in this section.
- **Recommendation:**
  1. In the `fetchNotes` browse branch (`useNotesListManagement.tsx:394-405`), build the request like the trash branch: `limit=pageSize`, `offset=(page-1)*pageSize`, `include_keywords=true`. Read the total from `Number(res?.total ?? res?.pagination?.total ?? items.length)`.
  2. Add whitelisted `sort_by` (`last_modified|created_at|title`) and `sort_order` (`asc|desc`) parameters to `list_notes` and `list_deleted_notes` (`notes.py:2309`, `:2370`) and to the `note_store` list queries. Then remove the client-only `sortNoteRows` (`notes-manager-utils.ts:1136`) from browse and trash, or keep it only as a stable tiebreak.
  3. Fix the mocks at `NotesManagerPage.stage30.export-progress.test.tsx:202` and `:253` so they honour `limit`/`offset` the way the real API does.
  4. Add a FastAPI integration test with 120 notes. It should assert that page-2 ids are disjoint from page 1, that `total == count_notes()`, and that page 1 of `title_asc` starts with the global minimum title.

  With these changes the footer (`NotesListPanel.tsx:184-185`) becomes correct automatically.

#### NL-02 · 'Export matching notes' (MD/CSV/JSON) loops over the same 100 notes: up to 1,000 requests and a 163 MB file of duplicates that still misses notes, reported as success, with no Cancel
**Priority** P0 · **Severity** 4/4 · **Effort** S · **Surfaces** WebUI, Extension options · **Persona** Power users · **Heuristic** Nielsen #1 Visibility of system status; #3 User control (no cancel); #9 Error recovery; data integrity · **Verification** Confirmed (live + code); upheld by independent skeptic

- **What happens:** The unfiltered branch of `gatherAllMatching` (`hooks/useNotesExport.tsx:136-171`) pages with `page`/`results_per_page=100`. The server ignores both parameters and always returns the same 100 rows. The loop's stop condition is `totalPages = pagination?.total_pages || (items.length < ps ? p : p+1)` (line 166). The server never sends `total_pages` and always returns a full page of 100, so the condition never ends the loop. The loop stops only at `MAX_EXPORT_PAGES = 1000` (line 78) or when a request fails. To limit server load, the review served pages after the first few from a cache. With that setup, a JSON export made 1,000 requests and downloaded `notes-export.json` at 163 MB: 100,000 items but only 100 unique ids. The UI showed two toasts: "Export limited to 100000 notes. Some notes may be excluded." and "Exported 100000 notes as JSON (163.11 MB)". When the skeptic aborted request 5, the export still downloaded 400 items (100 unique, so 56% of notes were missing) and toasted "Exported 400 notes as JSON". The progress row ("Exporting JSON 100 notes exported so far...", `NotesListPanel.tsx:363-395`) has no Cancel button. Filtered exports (`limit`/`offset`) and libraries under 100 notes export correctly. Evidence: `shots/lens-states/n03-export-done.png`, `shots/skeptic-NL/s02-done.png`.
- **Why it matters:** This is the Notes page's own backup and migration path. A user who exports before a migration or bulk delete gets a corrupt, oversized file that omits most notes, and the UI reports success. They find out only when the data is gone. The export can freeze the tab, and it sends hundreds of full-library requests to the server for every client of that user. Power users are the people most likely to rely on it.
- **Recommendation:** In `gatherAllMatching`'s unfiltered branch:
  - Page with `limit=100&offset=…`. Alternatively, use the server export endpoints, which already exist: `GET /api/v1/notes/export?limit=1000&offset=…&include_keywords=true` (`notes.py:2676-2684`) and `/export.csv` (`:2747-2761`).
  - Stop when `has_more === false`, when `next_offset == null`, or when `offset >= total`. Dedupe by id as a safety guard.
  - Read the total from the first response and show determinate progress ("300 of 1,734").
  - Add an `AbortController` and a **Cancel** button to the progress row (`NotesListPanel.tsx:363`).
  - If `arr.length !== total` at the end, show a warning ("Exported N of TOTAL notes") instead of a plain success message.
  - Lower `EXPORT_PREFLIGHT_NOTE_THRESHOLD` (line 79) from 100,000 to about 1,000.

**Bulk tagging**

#### NL-03 · Bulk 'Assign tags' silently replaces every selected note's existing tags, with no undo
**Priority** P0 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, Extension options · **Persona** Power users · **Heuristic** Nielsen #5 Error prevention; #2 Match with the real world (the copy implies "add") · **Verification** Confirmed (live + code); upheld by independent skeptic

- **What happens:** Bulk bar → **Assign tags** opens a free-text prompt, "Assign tags to selected notes / Enter tag names separated by commas.", which is empty unless a tag filter is active. The confirm dialog ("Apply skeptic-added to 2 selected notes?") doesn't say anything about replacing tags. Each note then receives `PATCH /api/v1/notes/{id} {"keywords":["skeptic-added"]}` (`NotesManagerPage.tsx:1385-1440`, PATCH at `:1418-1427`). The server's `_sync_note_keywords` (`notes.py:1677-1741`) unlinks every keyword that isn't in the payload, so every curated tag is wiped. The toast says "Updated tags on 2 selected notes" with no Undo. There is no version history (`GET /{id}/versions` returns 404), so the old tags can't be recovered. Additive per-note keyword endpoints exist (`notes.py:7455-7588`), but the bulk action doesn't use them. The prompt is an antd static `Modal`, which renders off-theme and logs "Static function can not consume context like dynamic theme". Evidence: `shots/skeptic-NL/s03-confirm.png`, `shots/skeptic-NL/s03-after.png`.
- **Why it matters:** "Assign" reads as "add". A routine re-tag across dozens of notes silently and irreversibly destroys the user's taxonomy. Saved filters, rule-based collections and Inbox then lose members without any visible sign. Power users who organise by tags are hit hardest. It deletes metadata, not note content, which is why it is rated 3 rather than 4.
- **Recommendation:**
  - In `assignKeywordsToSelectedBulk`, make the action additive by default. Union the new tags with each note's current keywords before the PATCH. List rows carry keywords once `include_keywords=true` is sent (the NL-01 fix); otherwise GET the note detail.
  - Replace the `promptModal` with a themed `App.useApp()` modal. It should offer a tag `Select` with autocomplete and a Segmented control: **Add / Remove / Replace all**.
  - State the effect in the confirm dialog: "Adds 1 tag to 6 notes; existing tags are kept."
  - Snapshot the previous keyword sets and show an **Undo** toast that PATCHes them back.

**Scanning, search and tag filtering**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NL-04 | The results list is very low density. At 1440×900 it gets about 275 px, which fits 2–3 rows, because Views, Organize and Filters sit above it. "Saved from chat" rows print the conversation and message UUIDs in link blue (not clickable). Previews show raw Markdown (`**Mood:**`, ```` ```sql ````). Search sits third in the sidebar and is truncated: 118 px wide at 1440, 78 px at 1024. Browse rows show no tag chips. | 2 | M | Move search above Views at full width and make sort an icon dropdown (`NotesSidebar.tsx` ≈742-804). In `derivePreviewText` (`NotesListPanel.tsx:20-35`), strip Markdown with a shared `toPlainPreview()` (headings, emphasis, fences, `[[ ]]` → title) and clamp to one line. Replace the conversation line (`:650-668`) with a "From chat" link and drop the message UUID. Batch-resolve chat titles for visible rows: today UUID-like ids are excluded at `NotesManagerPage.tsx:114-115`. Tag chips appear once browse sends `include_keywords=true` (NL-01). Optional density toggle. |
| NL-07 | Search discards the server's relevance order. The client re-sorts each page of 20 by the current sort, Date modified by default (`hooks/useNotesListManagement.tsx:297`). For "sourdough", the title match "Sourdough country loaf" therefore renders last. Body-only matches (e.g. "FAISS") show no snippet or highlight. With more than 20 hits, the order is wrong across pages under both relevance and date sorting. | 2 | M | Add a "Best match" sort and make it the default while a query is active; skip `sortNoteRows` for it. For date and title sorts with a query, ORDER BY on the server: the `sort_by`/`sort_order` parameters are already accepted at `notes.py:2600-2607` but ignored. Return an FTS `snippet` field (`snippet()` on SQLite, `ts_headline` on Postgres) and render it in place of the first line. On open, scroll to and highlight the first hit. |
| NL-N1 | The tag filter matches by substring and ORs multiple tags. Choosing "eval (0)" returns 6 notes tagged only "evaluation". `rag` + `baking` returns the union (19 notes), so adding a tag widens the results instead of narrowing them. Counts in the dropdown disagree with the results. Inbox inherits the same behaviour. Found during verification. | 2 | M | Match tokens exactly and case-insensitively (`LOWER(k.keyword) = ?`) in `search_notes_with_keywords` and `count_notes_matching_keywords` (`core/DB_Management/chacha/note_store.py:2529`, `:2620`; SQLite and Postgres branches). Add a `match` parameter (`all` by default, or `any`) implemented with `GROUP BY n.id HAVING COUNT(DISTINCT …) = n`. Add a "Match all / any" toggle beside the filter (`NotesSidebar.tsx` ≈860). Hide zero-count tags. Add API tests for both cases. |
| NL-08 | The tag filter accepts free text: typing "thes" and pressing Enter applies the partial token "thes", not the "thesis" tag. The token then appears under "Recently used" in Browse tags. Verification showed this is transient and nothing is persisted, and the results are not empty because of NL-N1's substring matching. | 1 (from 2) | S | Use `mode="multiple"` with `showSearch` so Enter picks the first matching existing tag (`NotesSidebar.tsx:860-879`). Set `notFoundContent` to 'No tag named "thes"'. Stop merging filter-only tokens into `availableKeywords` (`hooks/useNotesKeywords.tsx:103-105`). |
| NL-17 | "Suggest tags" pre-selects all five word-frequency suggestions, including stop-words and plural duplicates ("first, idea, ideas, key, paragraph"). One click on "Apply selected" adds noise to the tag vocabulary. | 1 | S | Default the selection to none, or only to suggestions that already exist as tags (`hooks/useNotesEditorState.tsx:1872-1873`). In `suggestKeywordsDraft` (`notes-manager-utils.ts:1226-1250`), extend the stop-word list, merge singular and plural forms, boost existing tags and require a count of at least 2. Label the source: "Based on word frequency in this note". |
| NL-16 | Loading and error states. While loading, the header shows "NOTES 0 TOTAL" and the list shows only a centred spinner, with no skeleton rows. The load-error card prints the raw request URL. The editor stays fully live, with "Create note", while the list API is failing. The Retry and "Health & diagnostics" actions on the error card work well. | 1 | S | Show skeleton rows and "—" for the count until the first response arrives (`isFetching && data === undefined`). Move the request path behind a "Details" disclosure. When the list query errors, show an editor banner tied to the offline draft queue: "Notes server unavailable — drafts are kept on this device". |

**Bulk actions, Trash and deletion**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NL-05 | Bulk actions and Trash don't scale. There is no Select all, either for the page or for all matching notes. Bulk runs show no progress and don't lock the bar: Export and Delete can start while the Assign tags PATCHes are still running. Bulk delete offers no Undo. Trash has no search, checkboxes or bulk restore, and every Restore switches back to Notes and opens the restored note, so restoring 6 notes takes 12 or more clicks. | 2 | M | Add a header "Select all on page" checkbox and a "Select all N matching" link (`NotesListPanel.tsx`). Hold a `bulkRunning` state in `NotesManagerPage.tsx:1335-1440`: show n/N progress, disable the bar buttons while running, and list failures with Retry. Add Undo for bulk delete by calling the restore endpoint per id. In Trash, enable checkboxes, search and "Restore selected" (`NotesSidebar.tsx:739`, `:1338`; `NotesListPanel.tsx:490`, `:683`). In `restoreNote` (`:1300-1333`), stay in Trash, refetch, and show a toast with an "Open note" action. |
| NL-06 | Notes can never be permanently deleted, and Trash can't be emptied. `DELETE /api/v1/notes/{id}` only soft-deletes (`notes.py:6502-6540`), and neither the API nor the DB layer has a purge path. Chat, Prompts, Skills, Items and Reading list all offer "Delete permanently". | 2 | M | Add `DELETE /api/v1/notes/{id}/permanent` (only for notes already in Trash, with `expected_version`) and `POST /api/v1/notes/trash/empty`. Implement a `note_store` hard delete that removes the note row, its keyword links, the FTS row, graph edges and the attachments directory, and writes a sync tombstone. In the UI, add "Delete forever" per Trash row and "Empty trash" in the Trash header. Each needs a confirm that names the item(s) and says the action can't be undone. Optionally, auto-purge after N days and say so in Trash. |
| NL-15 | Delete and Trash copy. The single-note confirm says only "Delete this note?", with no mention of Trash; the bulk confirm does mention it. If a note is restored within 10 s, the "Note deleted · Undo" toast is never dismissed and stacks with "Note restored". Trash keeps showing disabled Import/Sync folder/Export/Create study pack and a live "New note" editor. | 1 | S | Change the confirm to 'Move "&lt;title&gt;" to Trash? You can restore it from Trash.' with the button "Move to Trash" (`NotesManagerPage.tsx:1251-1256`), or drop the confirm since Undo exists. In `restoreNote`, call ``message.destroy(`notes-delete-${id}`)``. In Trash mode, title the panel "Trash (n)", hide rather than disable the irrelevant actions (`NotesListPanel.tsx:188-205`), and replace the editor with guidance about Trash. |

**Alternative views and organization**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NL-09 | For users west of UTC, Timeline puts notes under the wrong month: notes from October 2–3 appear under "SEPTEMBER 2026". The bucket key is built in UTC (`NotesManagerPage.tsx:2099`), but the label is a UTC-midnight date formatted in local time (`:2113-2115`). | 2 | S | In the `timelineSections` memo (`:2087-2117`), key buckets by local year and month (`getFullYear()`, `getMonth()+1`) and build the label from `new Date(year, month-1, 1)`. Add a Vitest test under `TZ=America/Los_Angeles` that asserts an Oct 2 note lands under "October 2026". |
| NL-10 | Organizing at scale. The server holds 17 folder paths, but `/notes` has no folder tree, filter or "Move to" (the only UI caller of the folders API is the extension Web Clipper). Saved-filter labels such as "Engineering (4)" count tags, not notes (`NotesSidebar.tsx:691`, `:727`). The Collection view crams a 3-column card grid into the 340 px sidebar. Pins are stored only on the device (`tldw:notesPinnedIds`, `ui-settings.ts:967-973`) and sort first only within the loaded rows. | 2 | L | Add a Folders section to Organize using `listNoteFolders`: a tree with counts, filtering by folder, bulk "Move to…", and a breadcrumb in the editor header. Relabel the saved-filter suffix "(4 tags)" or compute real note counts. Render Collections in the main pane, or as a single-column list, with full titles; replace the "RULE-MATCHED" badge with "Auto (tag: rag)". Store pins as a server-side note flag and sort them first on the server. |
| NL-11 | Graph limits. The "All notes" scope is disabled above 100 notes; that cap is a deliberate performance guard (`NotesGraphWorkspace.tsx:262-271`), but its banner ("This library has 228") contradicts the sidebar's "100 TOTAL". `[[Title]]` links create no graph edges. Picking a note in the list while in Graph mode leaves the editor hidden. | 2 | L | Resolve the wikilink syntax mismatch (NE-02, items 4–5). In Graph mode, re-centre the graph on a note picked from the list and open a side panel with "Open in editor" (`NotesManagerPage.tsx` ≈2494-2509). Keep the 100-node cap, but add degree and tag filters for larger libraries, and label nodes on hover or zoom. |
| NL-12 | The Views panel puts the mode (Notes/Trash) and the view (List/Timeline/Inbox/Collection/Graph) in one 7-button grid. Two selections are filled in primary blue, so they look like primary buttons, and none of the buttons has a tooltip. Inbox and the "Captured" filter chip send the identical request, so Inbox duplicates the chip. | 2 | M | Split the two axes (`NotesSidebar.tsx` ≈438-540). Use a Segmented "Notes / Trash" control (or a Trash link with a count) and a "View" select whose options carry one-line descriptions, such as "Inbox: notes captured from the browser". Remove either the Captured chip (≈815-835) or Inbox. Use a tinted selected style, and name the active view in the list header. |
| NL-13 | The Collection view is a dead end for users with no collections. It shows "Create or select a collection to start." with no button, and the New/Rename/Delete controls are hidden inside the collapsed Organize section (`NotesSidebar.tsx:558`, `:608`, `:1089-1097`). Users who do have collections get the first one auto-selected, and its name is shown nowhere. | 2 | S | Add a Collection header row with the collection Select, the active collection's name and a "New collection" button (`NotesSidebar.tsx:1086-1120`). Put a primary "New collection" button in the empty state, plus one line on how to add notes to it. Auto-expand Organize when entering Collection view. |
| NL-14 | Graph is hard for newcomers to read. Tag nodes are unlabeled squares. The controls use jargon (Radius, Max nodes, Layout "Dagre"). The window scrolls and the app header disappears. The focus note itself is reasonable: it follows the open note, then the most recent note, then the first visible note. | 1 | M | Label tag nodes and add a legend (circle = note, square = tag). Label every node when 30 or fewer are shown, otherwise on hover or zoom. Move Radius, Max nodes and Layout into a "Graph settings" popover with plain labels ("Layout: Tree / Circle / Grid"). Fit the graph inside the page frame (NO-01). When no note is open, say "Showing your most recent note — pick a note to focus". |

---

### 5.2 Editor (NE)

**Editing integrity**

#### NE-01 · WYSIWYG editor inserts every keystroke at the start of the document (text typed backwards) and autosaves the corrupted text
**Priority** P0 · **Severity** 4/4 · **Effort** M · **Surfaces** WebUI, Extension options · **Persona** First-time and power users · **Heuristic** Nielsen #5 Error prevention; #1 Visibility of system status; basic editing integrity · **Verification** Confirmed (live + code); upheld by independent skeptic

- **What happens:** After switching a note to WYSIWYG, the first keystroke lands in place. From then on, the caret collapses to offset 0 of the editor after every keystroke, so typing "Hello world" stores "dlrow olleH". In an existing note, typing " XYZ" at the end of "Alpha beta gamma delta." produces "QQZYXAlpha beta…" (the "QQ" came from a separate IME-style insertion, which also landed at offset 0). The user's existing text is altered, not just the new text. Autosave persists the result within about 5 s (POST 201 or PUT, confirmed by a GET read-back). This was reproduced with real CDP key events in the WebUI and in the production extension build, so it is not a dev-mode or headless artifact. The mechanism:
  - `handleWysiwygInput` copies `event.currentTarget.innerHTML` into React state on every input (`NotesManagerPage.tsx:1719-1727`).
  - Both contentEditables render that same state through `dangerouslySetInnerHTML` (`NotesEditorPane.tsx:1600`, `:1739`).
  - React 18.3 re-assigns `innerHTML` whenever `__html` changes. That replaces the text nodes and resets the selection to offset 0.

  The Heading and List buttons do change the DOM, but nothing changes on screen, because the editor has no `prose` styles and Tailwind preflight strips heading sizes and list markers. They also produce invalid `<h2><ul>…</ul></h2>`. Evidence: `shots/verify-NE/r2-01a-wysiwyg-typed.png`, `shots/skeptic-NE/s01b-after-autosave.png`.
- **Why it matters:** WYSIWYG is the obvious choice for people who don't know Markdown, which is largely first-time users. They cannot write a single sentence, and the garbled text is saved before they understand what happened. Power users who switch to WYSIWYG for a quick edit corrupt an existing note. The Notes UI has no version restore, and native undo can't help after `innerHTML` replacement. The mode isn't persisted (`hooks/useNotesEditorState.tsx:206`), so only users who choose WYSIWYG are exposed, but every one of them is.
- **Recommendation:**
  - **Now:** hide the WYSIWYG toggle, or label it "Beta" and keep it off by default, until the fix ships.
  - **Fix:** make the contentEditable uncontrolled.
    - Remove `dangerouslySetInnerHTML` from both branches in `NotesEditorPane`.
    - Add a `useLayoutEffect` that writes `richEditorRef.current.innerHTML` only when an explicit "external revision" counter changes.
    - Bump that counter in `enterWysiwygMode`, on `selectedId` change, after an attachment insert (`NotesManagerPage` ≈1647), on remote refresh (≈1892), and after the wysiwyg-sync effect (`useNotesEditorState.tsx:2056-2060`).
    - `handleWysiwygInput` should keep the live HTML in a ref and only call `setContentDirty(wysiwygHtmlToMarkdown(html))`. It must never update the state that drives `innerHTML`.
  - Give the editor `prose prose-sm` so headings and lists are visible, and stop `formatBlock` from nesting `<ul>` inside `<h2>`.
  - **Test:** add a Playwright e2e test (`apps/tldw-frontend/e2e`, `NotesPage` page object, testid `notes-wysiwyg-editor`) that types "Hello world", waits for autosave, and asserts the stored content through the API.

**Wikilinks and output**

#### NE-02 · [[Wikilinks]] are broken end to end: can't be followed, render as raw text, bracketed titles never resolve, resolution/autocomplete only use the current sidebar page, and no backlinks or graph edges result
**Priority** P0 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, Extension options · **Persona** First-time and power users · **Heuristic** Nielsen #1 Visibility of system status; #4 Consistency; #6 Recognition over recall · **Verification** Confirmed (live + code); upheld by independent skeptic

- **What happens:** Five separate breaks combine:
  1. *Links can't be followed in place.* Resolved links are emitted as `[token.raw](note://id)` (`wikilinks.ts:111`), so the link text keeps its brackets. The Markdown sanitizer blanks the `note:` scheme (`Common/Markdown.tsx:37`, `:59-75`), and the anchor gets `target=_blank` (`:512-514`). Clicking opens a duplicate `/notes` tab and leaves the original on the source note. `handlePreviewLinkClick` (`NotesManagerPage.tsx:1781-1795`) requires `note://`, so it never fires.
  2. *Bracketed titles never resolve.* `WIKILINK_PATTERN` (`wikilinks.ts:19`) excludes `[` and `]`, yet autocomplete inserts such titles as-is (for example `[[[ft-notes] My first note]]`). Those links, like any unresolved link, render as raw text.
  3. *Resolution depends on the loaded list.* Candidates come only from the list data, relations and the selected note (`hooks/useNotesWikilinks.tsx:69-107`). The same note renders links when the sidebar is unfiltered and plain text when it is filtered, and autocomplete can't suggest notes beyond the 100-row cap (NL-01).
  4. *No backlinks or graph edges.* The server parses only `[[id:UUID]]` (`core/Notes_Graph/wikilink_parser.py:22-30`, `core/Notes/wikilinks.py:12`; title links are "deferred to Phase 2"). The UI never inserts that form and renders it as raw text. The Connections help text (`NotesEditorPane.tsx:978-980`) still promises that links are "created … by [[ ]] note links".
  5. *Duplicate titles and suggestion placement.* For duplicate titles the picker shows "Title (uuid)", but the inserted link drops the id, so it may resolve to a different note than the one picked. The suggestion list sits below the textarea, not at the caret.

  Evidence: `shots/verify-NE/r2-02b-after-link-click.png`, `shots/ft-notes/07g-first-connections.png`.
- **Why it matters:** Linking is advertised in the empty state and in the tour. First-time users try it and see nothing happen: no navigation and "No backlinks yet". Power users and Zettelkasten users can't navigate links or build a graph, and the same note renders differently depending on an unrelated sidebar filter. Manual links in Connections and search still work, which is why this stays at 3, but a headline feature fails at every step.
- **Recommendation:**
  - *Quick wins (S):*
    1. Emit `[title](#note-<id>)` from `renderContentWithResolvedWikilinks`, or let `transformMarkdownUrl` accept `note:` only for the Notes preview via a prop. Render these anchors without `target=_blank`, and match that href in `handlePreviewLinkClick`.
    2. Use `token.title`, not `token.raw`, as the link text.
    3. Render unresolved links as a styled "missing link" with a 'Create note "X"' action.
  - *Structural (M):*
    4. Resolve titles and autocomplete through a debounced server title lookup (for example a title-only `/api/v1/notes/search` flag or a new `/notes/titles?q=`), independent of the sidebar list.
    5. Adopt one syntax shared by client and server: insert `[[id:<uuid>|Title]]`, render it as "Title", and extend `core/Notes/wikilinks.py` to parse it. Alternatively, ship server-side title parsing ("Phase 2") so backlinks and graph edges appear.
    6. Keep the chosen candidate's id on insert so duplicate titles resolve correctly.
    7. Allow brackets in titles, by matching up to the first `]]` not followed by `]` or by escaping on insert.
    8. Anchor the suggestion popover at the caret.

#### NE-04 · 'Print / Save as PDF' always fails: opens a blank tab and blames pop-up blocking (webui and extension)
**Priority** P0 · **Severity** 3/4 · **Effort** S · **Surfaces** WebUI, Extension options · **Persona** First-time and power users · **Heuristic** Nielsen #9 Help users recognize, diagnose and recover from errors · **Verification** Confirmed (live + code); upheld by independent skeptic

- **What happens:** More actions → Export → **Print / Save as PDF** opens an empty `about:blank` tab and shows the toast "Unable to open print view. Please allow pop-ups and try again." The pop-up was not blocked, since the tab opened. `hooks/useNotesExport.tsx:357` calls `window.open('', '_blank', 'noopener,noreferrer,width=1024,height=768')`. Under the HTML spec, `window.open` returns `null` in every browser whenever `noopener` is set. The error branch (`:358-364`) therefore always runs, and the document write and print (`:423-427`) are unreachable. This is the only print/PDF path in Notes. The Notes Studio printable output (`:368-405`) goes through the same broken function. The unit tests (`NotesManagerPage.stage45.notes-studio-export.test.tsx:372`, `:406`, `:484`) mock `window.open` to return a window, which hides the bug. Evidence: `shots/verify-NE/r2-04a-print.png`, `shots/pu-ext/26-ext-print.png`.
- **Why it matters:** The feature is 100% broken on both surfaces. Its error message sends users to change browser settings that can never fix it, and each attempt leaves a stray blank tab. Non-technical users have no realistic workaround; exporting `.md` and converting it with another tool is beyond most of them. Power users lose Notes Studio's paper-sized print output. Printing is occasional, which keeps this at 3.
- **Recommendation:**
  - In `printSelected`, render the printable HTML into a hidden `iframe` (`iframe.srcdoc = html`). On load, call `iframe.contentWindow.focus(); iframe.contentWindow.print()`, and remove the iframe on `afterprint`. This needs no pop-up and works on the extension options page.
  - Minimal alternative: open the window without `noopener`, write the document, then set `printWindow.opener = null`.
  - Fix the stage45 tests so a mock returns a window only when `noopener` is absent, and assert that the print document contains the note title.

**Long documents and Markdown rendering**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NE-03 | Long notes start below the fold. A Table of Contents with no collapse control (716 px and 24 entries on the seeded thesis note, including the note's own title and a literal "Table of contents" heading) pushes the textarea to y≈1170 at 1440×900. On a 9-heading note, about 2 lines are visible. A TOC jump scrolls the whole window, which pushes Save and the header off-screen. In Split view, the preview isn't height-bounded and doesn't scroll in sync with the 376 px editor. Workarounds exist: scrolling, autosave and Ctrl/Cmd+S. | 2 (from 3) | M | Make the TOC (`NotesEditorPane.tsx:1500-1528`) an "Outline (n)" disclosure: collapsed when it has more than 6 entries, open state remembered via `useStorage`, `max-h-48 overflow-auto`; or move it to a right-hand rail. Filter out entries that equal the note title or match `/^table of contents$/i` (`hooks/useNotesWikilinks.tsx:155-156`). In `handleTocJump` (`NotesManagerPage.tsx:1742-1777`), use `focus({preventScroll:true})` and set the textarea's `scrollTop` instead. Make the editor header and toolbar sticky. In Split, give both panes the same bounded height, with proportional or heading-anchored scroll sync. |
| NE-05 | The Markdown preview drops bullets and numbers (computed `list-style-type: none` on UL and OL), so numbered steps lose their meaning. `MarkdownPreview` maps size `sm` to `prose-sm` without the base `prose` class (`Common/MarkdownPreview.tsx:17-18`). This affects all four `size="sm"` call sites: Notes, Research workspace, Quick notes and Writing playground. | 2 | S | Use `prose prose-sm` and `prose prose-xs`. Add `[&_.contains-task-list]:list-none [&_.contains-task-list]:pl-0` so task lists don't get bullets. Visually check the other call sites, because the base `prose` class also restyles blockquotes, headings and links. Add a test that asserts disc and decimal markers in the Notes preview. |
| NE-07 | Code blocks render as a "staircase": line numbers and code start at a different x on each line. Python indentation becomes unreadable ("if x:" sits right of "def f(x):"). The cause is that each line is its own `<div class="table w-full">` (`Common/CodeBlock.tsx:526-531`); `Sidepanel/Chat/ArtifactsPanel.tsx:335` has the same markup. Chat's compact and github variants are unaffected. | 2 | S | Render all lines in one table or grid (`grid-cols-[auto_1fr]`) with a fixed `min-w-[3ch]` right-aligned gutter. Alternatively, pass `codeBlockVariant` through `MarkdownPreview` and reuse the github variant. Add a test with uneven line lengths that asserts equal code-cell x offsets. |
| NE-N1 | In the Notes preview, "View code" does nothing, because the artifacts panel is only mounted in Chat (`handleOpenArtifact`, `CodeBlock.tsx:355-368`). Its visible centre is also covered by the absolutely positioned Download and Copy icons (`:441-447` vs `:473-495`), so clicking it downloads the snippet as a file. Found during verification. | 2 | S | Render "View code" only when an artifacts host is mounted: use a store flag set by `ArtifactsPanel` on mount, or pass `showArtifactButton={false}` from `MarkdownPreview`/Notes. Move Download and Copy into the same flex header row, and drop the chat-tuned `top-9 md:top-[5.75rem]` sticky offsets. Add a test that the header controls' bounding boxes don't overlap. |

**Authoring affordances**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NE-06 | Lists and checklists require Markdown knowledge. The toolbar has no checklist or numbered-list button, Heading always inserts H1, and Enter doesn't continue bullets or `- [ ]` items. The key handler (`hooks/useNotesWikilinks.tsx:183`) only manages wikilinks. | 2 | M | Add checklist (`- [ ] `) and numbered (`1. `) actions to the toolbar (testids `notes-toolbar-checklist` and `notes-toolbar-numbered`). Make Heading an H1–H3 dropdown. In `handleEditorKeyDown`, continue list prefixes on Enter: increment numbers, reset `[x]` to `[ ]`, and remove the prefix when Enter is pressed on an empty item. Link a Markdown cheatsheet from the "Markdown + LaTeX supported" hint. |
| NE-09 | Notes with tasks show a second checklist strip above the preview, plus the jargon notice "Portable markdown with best-effort task continuity". The strip is the interactive control (the checkboxes in the preview are disabled), so the duplication has a purpose, but it reads as noise. | 1 | S | Replace the notice and the strip with a compact "Tasks 1/2" chip whose tooltip reads "Checking a box updates the note text and your task list" (`NotesEditorPane.tsx:513-524`, `:944-954`). Fuller fix: make the preview checkboxes interactive, mapped to their source lines, and drop the strip. |

**Header and Connections panel**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NE-08 | "Create study pack" is a second primary button next to Save. It navigates to `/flashcards`, where a drawer shows a raw note UUID and the manual-ID helper text. Verification found it is disabled while the note has unsaved edits, so it carries no data-loss risk. | 1 (from 2) | S | Make it a default or text button, or move it into the ••• menu under a "Study" group (`NotesEditorHeader.tsx:637-650`), so Save is the only primary action. In the study-pack drawer, show sources as a type icon plus title, and put the manual-ID help under "Advanced". |
| NE-10 | Connections lists a manual link twice: once under "Manual links" with Remove, then again under "Related" without Remove. It also separates Related, Manual links and Backlinks without explaining them, and collapses again on every note switch. | 1 | S | De-duplicate the related list against manual links (`NotesEditorPane.tsx:1099-1148` vs `:1192-1219`), or build one list grouped as "Links from this note" / "Links to this note" with source tags. Show Remove only on manual rows. Persist the expanded state via `useStorage`. |

---

### 5.3 Saving & data safety (NS)

The saving issues share one root cause: there is no single save state model. Today the autosave timer, retry loop, conflict handling, offline queue, three status displays and the beforeunload guard each act on their own. We recommend treating NS-01, NS-N1, NS-02, NS-03, NS-N2, NS-05 and NS-06 as one workstream. It would define a single state machine (*clean → dirty → saving → saved*, plus *needs attention (validation)*, *conflict* and *saved on this device*) that owns the status display, retry policy, conflict panel and navigation guard. The two P0 items below are its most urgent parts.

#### NS-01 · Unsaved note edits are silently discarded on in-app navigation (5s debounce, no flush on unmount, no prompt)
**Priority** P0 · **Severity** 4/4 (raised from 3 in verification) · **Effort** M · **Surfaces** WebUI, Extension options · **Persona** First-time and power users · **Heuristic** Nielsen #5 Error prevention; #1 Visibility of system status · **Verification** Confirmed (live + code); upheld by independent skeptic

- **What happens:** Autosave is a trailing 5-second debounce: `NOTE_AUTOSAVE_DELAY_MS` (`notes-manager-utils.ts:339`), with the effect at `hooks/useNotesEditorState.tsx:2017-2033`. The timer restarts on every keystroke. The effect cleanup and the unmount cleanups (`useNotesEditorState.tsx:2257-2262`, `NotesManagerPage.tsx:2336-2342`) only clear the timer; they never save. Only `beforeunload` is guarded (`:2036-2044`). Notes never mounts the existing `RouteLeavePrompt`, and the offline-draft effect returns early while online. Verifiers observed:
  - With "Unsaved changes" showing, clicking a left-rail item (Chat, Prompts, Characters, Media or Knowledge QA) 0.8–3.5 s after typing navigated with no dialog and no PUT. On return, the edit was gone.
  - 22 s of continuous typing with pauses under 5 s produced zero PUTs, and leaving lost all 180 characters.
  - A brand-new note (title and two lines), left 3 s after typing, disappeared entirely; no note was ever created.

  The extension options page behaves the same way. Slow route compiles in Next dev mode masked the bug in some runs, so a production build loses edits more often. Evidence: `shots/verify-NS/v01-A-1s-before-leave.png`, `shots/skeptic-NS/s03-before-leave.png`.
- **Why it matters:** Switching sections is the most ordinary action in the app. The loss is silent, can't be recovered, and covers everything typed since the last 5 s pause, not just the last few seconds. First-time users find out that autosave can't be trusted only after losing work. Power users who type continuously are the most exposed, because the debounce never fires while they are in flow.
- **Recommendation:**
  1. **Save on leave.** Keep a ref holding the latest `{isDirty, saveNote}`. In the unmount cleanup (`useNotesEditorState.tsx:2257-2262`), call `void saveNoteRef.current({showSuccessMessage:false})` when the note is dirty and has a title or content. SPA navigation doesn't unload the page, so the request completes.
  2. **Backstop.** On unmount, on `visibilitychange=hidden` and on `pagehide`, write the draft synchronously to the existing offline draft queue (`persistOfflineDraft`, `syncState: 'queued'`). That queue already auto-syncs and restores drafts (≈1581-1690).
  3. **Guard when the save can't succeed.** Mount `<RouteLeavePrompt when={isDirty && saveIndicator==='error'}>`. It already works on web through the shim's `routeChangeStart` and on the extension through `useBlocker`.
  4. **Debounce with a max wait.** Save after about 1.5 s idle, and at most every 10 s during continuous typing.

  Add a Playwright test: type, click a rail link within 1 s, return, and assert the edit persisted.

#### NS-N1 · The 409 toast's 'Reload notes' does not reload: it silently moves the base version forward, and the next autosave overwrites the other tab's/device's changes
**Priority** P0 · **Severity** 3/4 · **Effort** S · **Surfaces** WebUI, Extension options · **Persona** First-time and power users · **Heuristic** Nielsen #5 Error prevention; #2 Match between system and real world (label vs behaviour); #1 Visibility of system status · **Verification** Found and confirmed during verification (single verifier)

- **What happens:** A remote edit lands while the note has local edits, so the next save returns 409. A toast then shows "This note changed on the server. **Reload notes**". The button calls `reloadNotes()` (`hooks/useNotesEditorState.tsx:866-879`, wired at `:905-933`). That function refetches the list, sets `selectedVersion` to the server version and clears `remoteVersionInfo`, but it never touches the editor content. The stale banner disappears and the editor keeps the local text; the remote text is never shown. Within about 5 s, with no further user action, autosave PUTs with the new expected version, gets 200 and overwrites the remote edit. The status then reads "Saved just now / All changes saved". This was live-tested on the WebUI; the extension options page runs the same shared code and is affected by inference. Evidence: `shots/verify-NS/v06-b-after-toast-reload.png`, `shots/verify-NS/v08-a-autosave-after-toast-reload.png`.
- **Why it matters:** The button that looks safest silently destroys the other tab's or device's edits and then reports success. The user never sees the server version and gets no warning. Anyone working in two tabs, on two devices, or in the WebUI and the extension at once can hit this. It is rare, but the loss is invisible.
- **Recommendation:** Never advance `selectedVersion` without also replacing the editor content, unless the user explicitly chose to overwrite. Short term, point the toast action at `reloadSelectedNoteAfterConflict`, which has a discard confirm. Better, remove the toast action and open the single conflict panel proposed in NS-03, with "Keep my version" as an explicit, labelled overwrite, "Use server version", and "Save mine as a new note". Add a unit test that calling `reloadNotes` during a dirty conflict doesn't let autosave PUT with the newer version.

**Save errors and conflicts**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NS-02 | A note with a body but no title can't be saved, and the error blames the network. The POST returns 400 "Title is required unless auto_title=true", because the client never sends `auto_title` (`hooks/useNotesEditorState.tsx:1222-1227`). The status still says "Could not save — check your connection and try again", and the note-switch dialog says "Auto-save could not reach the server". Only clicking Save reveals the real reason, as a raw message ending in "(POST /api/v1/notes/)". "+ New note" focuses the Title field and "Generate title" sits next to it, which reduces how often this happens. | 2 (from 3) | S | Send `auto_title = true` when the title is blank; the server already derives a heuristic title, and `loadDetail` fills it in after create. Use the placeholder "Untitled — we'll name it from the first line". Branch the error copy on status in the catch block (≈1364-1395) and in `confirmDiscardIfDirty` (`:704-709`): 400/422 gets a field-level message ("Not saved — needs attention"), and network errors or 5xx get the connection message. Strip the "(METHOD /api/…)" suffix from user-facing toasts. |
| NS-03 | Conflict recovery is confusing. Five messages appear at once, with three different "reload" labels and red text blaming the network. "Save anyway" re-sends the stale version and gets 409 again (`useNotesEditorState.tsx:1140-1158`). The stale banner's "Reload note" can stack three modals. There is no diff, no "keep mine" and no "save as copy". Local edits do survive every path except the confirmed "Reload server version". | 2 (from 3) | M | Replace the toast, recovery notice and stale banner with one conflict panel in `NotesEditorPane`, and pause autosave while it is open. Actions: "Keep my version" (an explicit overwrite that adopts `remoteVersionInfo.version`, which also fixes "Save anyway"), "Use server version", "Save mine as a new note", and optionally "Compare" with a side-by-side diff. The copy should name the cause ("Changed in another tab or device at 12:25"), without version jargon or connection wording. Make the stale banner open this panel instead of calling `handleSelectNote`. |
| NS-N2 | A failed autosave retries every ~5 s forever, even on non-retryable 400/409 responses (4 identical POSTs in 21 s while idle). During a conflict, autosave opens the "Remote changes detected… Continue anyway?" modal unprompted and moves keyboard focus to its Cancel button. The assertive recovery alert re-fires on each cycle. Found during verification. | 2 | S | Track the failure class. On 4xx, stop auto-retrying until the title or content changes or the user clicks Retry. On network errors or 5xx, back off (5 s, 15 s, 60 s) and queue the offline draft (NS-05). Autosave must never open a modal: while `remoteVersionInfo` is set, skip autosave and show the NS-03 panel. Keep the recovery notice mounted so the alert fires once per new error. |

**Performance, offline and status**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NS-04 | Every autosave awaits a full list refetch. A single save makes 7 requests, and the list response is 460–700 KB because the same 100 full notes appear in three arrays (`notes`, `items`, `results`; `notes.py:2342-2357`). The list briefly swaps to a "Loading notes..." spinner and resets its scroll to the top. While the save is still running, the status pill shows a stale "Saved … ago". With the default newest-first sort, the edited row moves to the top anyway, which lessens the impact. | 2 (from 3) | M | Keep the rows on screen during a refetch, and show the full spinner only when there are no rows yet (`NotesListPanel.tsx:433`). Instead of `await refetch()` (`hooks/useNotesEditorState.tsx:1257`, `:1335`), patch the edited row with `queryClient.setQueryData` and run a debounced background refetch. Set `lastSavedAt` before switching the status to "saved". The NL-01 paging fix cuts the payload about 5×. On the server, return previews by default and put the duplicate arrays behind a compatibility flag. |
| NS-05 | A dropped connection isn't detected for up to ~30 s, the health-poll interval. The offline draft queue only engages once `isOnline` is false (`useNotesEditorState.tsx:1110`), so until then edits exist only in memory. After the draft is queued, the pill still says "Save failed" next to "stored locally", which contradicts it. After recovery, the global "Can't reach your tldw server" dialog stays open. | 2 | M | In the `saveNote` catch, treat a missing HTTP status or "Failed to fetch" as a disconnect. Call `persistOfflineDraft({syncState:'queued'})`, show "Saved on this device — will sync when the server is reachable", and trigger a one-shot connection check. Add window `online`/`offline` listeners in `store/connection.tsx`, and auto-close the global dialog when the connection recovers. |
| NS-06 | Save status is shown three times: the header pill, a line under Tags, and the footer ("Version 2 · Last saved …"). Screen readers hear up to 8 polite announcements per autosave cycle (`NotesSaveStatus.tsx:60-66` and `NotesEditorPane.tsx:810-827` announce in lockstep, plus the list-loading status). Save stays a primary button even when there is nothing to save, and the UI never says where notes are stored. | 2 (from 1) | S | Keep `NotesSaveStatus` as the only status. Remove `notes-save-feedback`, or show it only for errors with `aria-live="off"`. Announce only end states ("Saved" and errors). Mark the list `aria-busy` during refetch instead of using a live region. Put "Version N · Last saved" and "Saved to &lt;host&gt;" in a tooltip on the pill. Render Save as a default button, or disable it, when the note is clean. |

---

### 5.4 Onboarding, layout & visual design (NO)

All NO issues are severity 2 (P2). Most of them stem from NO-01: `/notes` is laid out as a scrolling document, while `/chat` and `/media` are fixed-height application frames. Fixing NO-01 first removes the window scrolling behind NO-02, NE-03 and NL-14.

**Page frame and responsive layout**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NO-01 | `/notes` overflows the viewport at every desktop size. The document is always 135vh tall (scrollHeight 1215 at a 900 px viewport), because `/notes` is missing from `VIEWPORT_CONSTRAINED_PATHS`. There is an 80 px empty band under the header (leftover `mt-16`) and a 288 px void under the sidebar, which is pinned to innerHeight−120 px. One mouse-wheel tick over the editor hides the app header. "Filter by tag" and "Browse tags" are silently clipped inside a nested scroller capped at 50% height. Graph view also scrolls the window. The same sizes appear in the production extension build. | 2 | M | Add `"/notes"` to `VIEWPORT_CONSTRAINED_PATHS` (`routes/route-paths.ts:36`), so WebLayout and OptionLayout give it the same `h-screen` frame as `/chat`. Change the root at `NotesManagerPage.tsx:2363` to `flex h-full min-h-0` and drop `mt-16`. Delete `calculateSidebarHeight` and its resize listener, and use `h-full` instead. Keep the reserved results area, but collapse Organize and Filters by default with active-filter badges, or add a bottom fade with a "More filters" cue. Use `focus({preventScroll:true})` in `NotesGraphWorkspace.tsx:110`. Add a Playwright assertion that scrollHeight equals innerHeight at 1440×900 and 1024×768. |
| NO-03 | Responsive layout across widths. At 768×1024, the search input is 38 px wide and the header actions wrap onto three rows. From 1024 to 1440 px, only 2–3 list rows are visible. At 1920 px, the Markdown editor is about 1,386 px wide (around 200 characters per line). On a 390×844 phone, the note body starts at y≈655, behind chrome that includes a "Keyboard shortcuts" button and Ctrl/Cmd+F tips, neither of which is gated by pointer type. | 2 | M | Below `lg`, give the search its own full-width row and make sort icon-only (`NotesSidebar.tsx` ≈760-780). Below `xl`, move "Save & new" and "Create study pack" into the ⋯ menu, keeping Save and Edit/Split/Preview on one row (`NotesEditorHeader.tsx`). Cap the editor column at `mx-auto max-w-[80ch]` (`NotesEditorPane.tsx`). Collapse the TOC by default (NE-03). Hide keyboard-only affordances on `(pointer: coarse)`. On phones, group title generation, tags, Connections and the toolbar into a collapsed "Details" section so the body is in the first viewport. |
| NO-04 | With no note selected, the "Select or create a note" empty state (with a Create note button) sits above a fully live Title field, Tags field and textarea. The header meanwhile says "New note", so there are three create entry points. The editor really is a working draft. On the first keystroke the empty-state block unmounts and the textarea jumps about 247 px under the caret. | 2 | S | Treat the blank editor as an implicit draft, which is the intent of commit 92736e2799. When notes exist, drop the `NotesEditorEmptyState` block (`NotesEditorPane.tsx:629-636`) in favour of placeholders: "Untitled note" in the title, and "Start typing — Ctrl/Cmd+S saves to your server. Type [[ to link a note." in the body. Hide the pre-save status for a pristine draft. Render the full empty state only when the user has zero notes, and instead of the editor rather than above it, as `NotesGraphWorkspace` already does. Add a test that the empty state and the title input never render together. |

**First-run tour**

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| NO-02 | The first-run tour auto-starts about 1 s after landing and scrolls the window to its maximum (scrollY 315 at 900 px), so the header and "+ New note" are off-screen. Step 2's popover is clipped and its target is off-screen. Step 3's copy describes tag and saved filters but spotlights only the sort control. "Skip tour" leaves the page scrolled. After "Step 6 of 8" the tour hangs invisibly: step 7 targets Connections, which only renders once a note is selected, so it is always missing on a first visit. The same happens in extension options. | 2 | M | Make "Step x of N" accurate: add an `optional`/`requiresSelection` flag to `TutorialStep` and drop such steps when their target isn't ready, or retarget step 7 to the editor body with the copy "Type [[ to link notes" (`tutorials/definitions/notes.ts:74-91`). Fix the runner fallback (NO-N1). After NO-01, pass `scrollOffset` and viewport-bounded `floaterProps`, and restore `scrollY` on skip or finish. Split step 3 into sort and tag-filter steps. End the tour with a "Create your first note" action. Add an e2e test that clicks through to completion and asserts `notes-basics` is recorded as completed. |
| NO-N1 | `TutorialRunner` retries a missing target by remounting Joyride, because its key includes `retryNonce` (`Common/TutorialRunner.tsx:253-266`, `:311`). A freshly mounted react-joyride 2.9.3 never re-emits `TARGET_NOT_FOUND`, so the skip/end fallback (`:276-291`) can never run. The tour is never marked complete, and since the auto-start flag is already set, it is never offered again. Any of the 19 tours with a conditionally rendered target is exposed. `TutorialRunner.retry.test.tsx:48` mocks react-joyride, so the suite can't catch this. Found during verification. | 2 | S | Remove `retryNonce` from the Joyride key. On `TARGET_NOT_FOUND`, poll `isStepTargetReady` up to `MAX_TARGET_RETRY_ATTEMPTS`. If the target appears, re-render with `setStepIndex(index)`; otherwise advance to the next step, or on the last step call `markComplete` and `endTutorial`. Better still, filter out non-ready or optional steps when building `joyrideSteps`. Add a test with the real react-joyride and a missing middle target that asserts the next tooltip renders and completion is recorded. |

---

## 6. Chat page — issues and solutions

**Scope.** This section covers `/chat` in the WebUI and the extension options page, which use the same Playground component. Sidepanel behaviour is included where shared components behave the same way. There are 52 issues: 2 at severity 4, 13 at severity 3, 29 at severity 2 and 8 at severity 1. Ten are P0 and five are P1. Severity 3–4 issues get a full write-up; the rest are in compact tables for each sub-area. Titles and evidence reflect the verification pass. Where verification refuted part of a finding, the write-up says so and drops that part. Code paths are relative to `apps/packages/ui/src` unless they start with `tldw_Server_API/` or `apps/`. Screenshot paths are relative to the review's `shots/` directory.

**One root cause behind most P0s.** Saved chats on `/chat` send every turn through the native history-selection controller (`hooks/useHistorySelection.ts`, `services/chat-history-selection.ts`, `components/Common/Playground/HistorySelectionReview.tsx`). That controller is only partly wired into the Playground, and this accounts for most of the severe findings:

- New chat and Clear conversation don't reset the controller, so old context leaks into the next request or the send fails (CS-01, CS-05).
- The controller owns every draft, so auto-promotion to the server never runs, while the UI says "Saved · Locally + Server" (CS-03, CS-N3).
- It has no implementation for regenerate, continue or edit-and-send (CM-01). Its default path also includes stored alternative replies (CM-N2).
- Every failed, stopped, timed-out or interrupted turn ends in the same "Turn needs review" panel, which drops the error and the partial answer (CC-02, CM-02, CM-N1, CM-06, CS-04).
- Its internal states and controls appear at the top of the transcript (CC-04, CM-14).

Temporary chats bypass the controller. In a Temporary chat, Regenerate works and provider errors get the product's full ErrorBubble recovery UI, so the regression lies in this integration. The existing unit tests (e.g. `usePlaygroundPersistence.test.tsx`) run without a `HistorySelectionProvider`, so none of these failures are caught.

We recommend handling these issues as one workstream with a shared integration-test harness that mounts the Playground with a `HistorySelectionProvider`. Fix CS-01 first: CS-05's fallback and the user's natural workaround (New chat) both depend on it. The sidepanel already resets the selection on new chat (`routes/sidepanel-chat.tsx:1220`) and can serve as the reference.

**Status copy that isn't true.** Four status messages state an outcome that did not happen: "Saved · Locally + Server" (CS-03), a model chip reading "Healthy" (CC-01), "Chat restored." (CS-N2) and "Conversation cleared" (CS-05). Set a product-wide rule: success and status copy must come from an acknowledged state change, never from intent or connectivity.

---

### 6.1 Sessions & history (CS)

**Starting fresh and clearing context**

#### CS-01 · 'New chat' does not start a clean conversation: old messages leak into the next request, the next send can be destroyed with a raw 'request_config_scope_changed' error, or is appended to the previous server chat
**Priority** P0 · **Severity** 4/4 · **Effort** M · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #5 Error prevention (privacy/context integrity); #1 Visibility of system status; #9 Error recovery · **Verification** Confirmed (live + code); upheld by independent skeptic
- **What happens:** In every case below, the transcript visibly empties after New chat (header button, sidebar "+" or Ctrl+Shift+U), and then one of three things goes wrong.
  1. *Fresh session:* send "Q1", click New chat, send "Q2". The `/chat/completions` payload is `[user Q1, assistant A1, user Q2]`, and Q1/A1 reappear in the transcript ("Chat 4 messages"). The same leak occurs in the production extension build, so it is not a dev or StrictMode artefact.
  2. *After a reload restores a local chat:* New chat → type → Enter clears the composer and shows the toast "Error / request_config_scope_changed". No request is sent, and retrying fails the same way.
  3. *With a saved server chat open:* New chat removes the header title, but the next turn and all prior turns are POSTed to `/api/v1/chats/<old id>/messages` (201). In verification this grew the old thread from 6 to 8 messages.

  The stale "Review conversation history / Use no prior messages" bar also survives New chat.

  Code: `hooks/chat/useClearChat.ts:93-148` resets messages, history, historyId and serverChatId, but never resets the HistorySelection controller. The only reset is in `clearPersistedSession` (`hooks/usePlaygroundSessionPersistence.tsx:851-859`), which New chat never calls. On the next send, `hooks/chat/useChatActions.ts:3450-3496` builds the turn from the stale owned view. In the restored-chat case, `hooks/chat-modes/chatModePipeline.ts:777-804` instead throws the raw code after the composer has already been cleared. Evidence: `shots/ft-chat/13header-b-after-q2.png`, `shots/verify-CS/v02-c-after-send.png`.
- **Why it matters:** "Start fresh" is the most basic chat action, and it fails without any visible sign. A first-time user who changes topic sends the whole previous conversation, which may be confidential, to the model. If they also switched models, it goes to a different provider. The transcript looks empty, so they cannot tell. Power users who keep saved server threads get new replies written into an unrelated thread, and every device sees that corrupted history. In the restored-chat case, the typed prompt is lost behind a developer error code.
- **Recommendation:** Make New chat atomic in the Playground.
  - `useClearChat` already dispatches `CHAT_ROUTE_REPLACEMENT_EVENT` (`useClearChat.ts:97`), and Playground already listens for it (`Playground.tsx:~990-1022`). In that handler, call `historySelection.reset()` and `clearPersistedSession()`. Alternatively, expose `reset()` through `HistorySelectionContext` so `useClearChat` can call it. Either way the reset should release the owner lease, clear the view, capture and recoveries, and remove the sessionStorage reference.
  - Keep Send disabled until the selection is back to `idle`.
  - As a second safeguard in `useChatActions` (~3450), treat the selection as idle when its `view.conversation_id` matches neither the current `historyId` nor `serverChatId`.
  - On any failure before streaming starts, put the typed text back in the composer. Map `request_config_scope_changed` to plain copy: "This chat changed while sending. Your message is back in the box — press Send again."
  - Add Playground coordinator integration tests: New chat, then send, from fresh, local-restored and server-loaded chats, via the header button, the sidebar "+" and Ctrl+Shift+U. Each should assert `payload.messages.length === 1` and no POST to the previous chat id.

#### CS-05 · 'Clear conversation' reports 'Conversation cleared' but removes nothing, and old messages are still sent to the model
**Priority** P0 · **Severity** 3/4 · **Effort** S · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #1 Visibility of system status (false success); privacy expectation · **Verification** Confirmed (live + code); upheld by independent skeptic
- **What happens:** More tools → Clear conversation shows the confirmation "This will remove all messages from the current conversation. This action cannot be undone." After Confirm, the toast says "Conversation cleared". The transcript still shows the exchange ("Chat 2 messages"), and the next payload still carries the old user and assistant turns.

  `handleClearContext` (`components/Option/Playground/PlaygroundForm.tsx:3468-3497`) only calls `setHistory([])` (~3487). In the default saved mode the send path rebuilds context from the history selection (`useChatActions.ts:3450-3496`), so even that call has no effect on what the model receives. In Temporary mode the model context is dropped, but the visible transcript still isn't cleared. The menu item's `title` is "Clear Context" (`PlaygroundToolsPopover.tsx:444`), which shows the intent was a context clear. Evidence: `shots/verify-CS/v08-after-clear.png`, `shots/lens-states/c06-after-clear.png`.
- **Why it matters:** Users clear a conversation to change topic or to stop the model seeing sensitive content. Here they get a success message, and a stern "cannot be undone" warning, for an action that does nothing. The obvious fallback, New chat, is broken too (CS-01). Once CS-01 is fixed this drops to severity 2.
- **Recommendation:**
  - Connect Clear conversation to the existing empty-context primitive, `historySelection.choose({ kind: 'empty' })`, as `HistorySelectionReview.tsx:330` already does, and insert a visible "Context cleared" divider in the transcript. If the label is meant to wipe the chat, call the fixed New-chat reset from CS-01 instead.
  - Replace the "cannot be undone" confirmation with an Undo toast that restores the previous cursor.
  - If earlier messages stay visible, rename the action "Start fresh context".
  - Add a regression test: after Clear, the next `/chat/completions` payload contains only the new user turn.

**Saving, interruption and recovery**

#### CS-03 · Chats are never saved to the server, yet the UI says 'Saved · Locally + Server'
**Priority** P0 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #1 Visibility of system status (truthful status); #5 Error prevention · **Verification** Confirmed (live + code); upheld by independent skeptic
- **What happens:** Across about 15 reviewer conversations and repeated verification sends, every completion went out with `save_to_db=false` and no `conversation_id`. No `POST /api/v1/chats` ever happened, and none of the new chats exist on the server. The production extension options build behaves the same.

  The UI says the opposite. The header shows a green "Saved" badge, even on an empty chat that has never sent anything. The composer pill's tooltip reads "Locally + Server / Saved Locally+Server".

  Code:
  - `hooks/playground/usePersistenceMode.tsx:16-21` and `:59-64` return the server label whenever `isConnectionReady` is true.
  - Auto-promotion is intended (`components/Option/Playground/hooks/usePlaygroundPersistence.tsx:556-571`), but it does nothing. `handleSaveChatToServer` returns early when `!isUnownedDraft()` (:250-255).
  - Every saved normal send goes through the history selection (`useChatActions.ts:3492-3496`, `3957-3964`), so the draft is always owned.
  - The unit test "saves plain chats without requiring a default character" passes only because it runs without a HistorySelection.

  Sending into a chat that was already loaded from the server does persist. Evidence: `shots/ft-chat/02-hover-header-saved.png`, `shots/verify-CS/v07-after-send-hover-pill.png`.
- **Why it matters:** Users believe their conversations are backed up on their tldw server. They live only in this browser's IndexedDB and are lost when site data is cleared, in a private window, or on another device. Features that need a server chat also miss them without saying so: the history list (see CS-02), Save to Notes/Flashcards and sharing. Severity is 3 rather than 4 because the data survives locally and session restore works.
- **Recommendation:**
  - **Truthful status** (`usePersistenceMode.tsx`): derive the label from `serverChatId` plus the last acknowledged write, not from connectivity. Show "Saved on server" only when the chat has a server id and its last write was acknowledged. Otherwise show an amber "Saved in this browser", "Saving…", or "Couldn't save to server — Retry". Hide the header badge until the chat has content.
  - **Promotion:**
    - In `handleSaveChatToServer` (:250-255), allow promotion when `owner.kind === 'local'` and the owner's `conversation_id === historyId`.
    - Alternatively, create the server chat on the first send with `ensureWorkspaceServerChatForTurn`, which `useChatActions.ts:3957` currently skips whenever `normalHistorySelection` is set.
    - Log every early return. If promotion is abandoned, show the existing "Chat saving is incomplete — Retry" toast.
  - **Test:** add an integration test that mounts the Playground with `HistorySelectionProvider`, sends one message, and asserts `POST /api/v1/chats` plus the message writes.

#### CS-04 · Reloading or leaving /chat mid-reply erases the question and partial answer, replaced by a jargon recovery card, with no warning
**Priority** P0 · **Severity** 3/4 · **Effort** L · **Surfaces** WebUI, extension options, extension sidepanel · **Persona** First-time and power users · **Heuristic** Nielsen #3 User control; #5 Error prevention; #2 Match with real world · **Verification** Confirmed (live + code); upheld by independent skeptic (sidepanel part from the earlier pass only)
- **What happens:**
  - *Reload mid-reply:* reloading about 1 s after Enter raises no "leave page?" warning. Afterwards the turn is missing from the transcript. In its place is a card: "Turn needs review — The original send outcome is retained separately from history… User input accepted; response not saved", with only Inspect, "Copy recovered text" and "Dismiss recovery".
  - *Switching pages mid-reply* is worse. If the user clicks Notes in the rail mid-stream and comes back, the completion has finished on the network, yet neither the question nor the answer is in the transcript. The header still shows the chat title and "Saved". A "Generated response needs review" card sits above the fold because the transcript auto-scrolls past it, and a full reload does not restore the turn.
  - *After the reply completes:* navigating away and back is safe.
  - *Sidepanel:* in the earlier pass, the recovery card and the "Start a conversation below" empty state appeared together. The sidepanel does have a beforeunload guard.

  Code: `components/Option/Playground` has no beforeunload or route-leave guard. Only `routes/sidepanel-chat.tsx` and `option-quick-chat-popout.tsx` have one. When unmounting invalidates a turn's history-selection fence, the turn is stored as a `HistoryTurnRecovery` instead of being committed. `HistorySelectionReview.tsx:197-240` offers only copy and dismiss. Evidence: `shots/verify-CS/v06-reload-b-after.png`, `shots/skeptic-CS/s04-nav-400-t1-back.png`.
- **Why it matters:** Switching to Notes while an answer streams is ordinary multitasking. The mock model answered in about 1.5 s; real models take 10–60 s, which leaves a much larger window. First-time users conclude the message was never sent or the chat was lost. Power users lose answers they have already paid compute for and have to dig the text out of a recovery card. The text can still be copied, so this is not a total loss.
- **Recommendation:**
  1. When a reply completes after the Playground has unmounted, commit it to the owned (Dexie) history instead of parking it as a recovery. If it cannot be committed, put both bubbles back in the transcript with an "Interrupted" badge and the actions [Resend] [Keep partial answer].
  2. Add a beforeunload guard in the Playground while `isStreaming`/`isProcessing` is true, as `routes/sidepanel-chat.tsx` does. Optionally add a react-router blocker for in-app navigation, or keep streaming in a store that outlives the route.
  3. Rewrite the copy in `HistorySelectionReview.tsx:197-240` for the `generated_unsaved` and `accepted_unsaved` states: "Your last message didn't finish saving.", with a primary [Restore to chat] and secondary [Copy] and [Dismiss]. Pin the card near the composer or scroll it into view.

#### CS-N2 · 'Restore' from Trash brings back an empty chat: its messages stay deleted although the UI says 'Chat restored.'
**Priority** P0 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #5 Error prevention; #1 Visibility of system status (false success); user control (undo) · **Verification** Found and confirmed during verification (single verifier)
- **What happens:** We tested with a dedicated two-message chat: row ⋯ → Delete → Move to trash, then the Trash filter → ⋯ → Restore. The toast says "Chat restored.", but `GET /chats/{id}/messages` returns 0 messages (2 with `include_deleted=true`), and opening the chat shows an empty transcript. A second trashed test chat behaved the same way.

  Backend:
  - `delete_chat_session` (`tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` ~7990-8037) soft-deletes every message, then the conversation.
  - `restore_chat_session` (~8059-8115) calls `db.restore_conversation`, which (`tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py:1481+`) only sets `conversations.deleted = 0`. The messages stay deleted.
  - The only restore tests cover error mapping (`tests/Character_Chat_NEW/unit/test_chat_session_error_mapping.py:2371-2407`).

  Evidence: `shots/verify-CS/v12-restored-chat-opened.png`.
- **Why it matters:** Trash is presented as the safe, reversible way to delete. Anyone who trashes a chat by mistake and restores it gets an empty shell with a success message, and nothing in the UI brings the messages back. In practice this is data loss, and it affects both personas.
- **Recommendation:**
  - Fix it in the backend, with either approach:
    - (a) Stop soft-deleting messages when a conversation is trashed; the conversation's deleted flag already hides them.
    - (b) Record a deletion batch or timestamp, and have `restore_conversation` (or the endpoint) undelete the messages from that batch, without bringing back messages the user had deleted individually earlier.
  - Add an integration test: create a chat with messages → DELETE → POST /restore → GET messages returns the original count.
  - Run a one-off data repair for conversations that are not deleted but whose messages were deleted in the same transaction as the conversation.
  - Until the fix ships, don't show a success toast on restore.

**Finding and returning to chats**

#### CS-02 · Users can't find or return to previous chats: history is buried under navigation shortcuts, shows only server chats, resets on every load, searches titles only, and isn't in ⌘K or the URL
**Priority** P0 · **Severity** 4/4 · **Effort** M · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #6 Recognition rather than recall; #7 Flexibility and efficiency; information scent · **Verification** Confirmed (live + code); upheld by independent skeptic, with a calibration note
- **What happens:** Each limitation below, with its cause:

  | What the user experiences | Cause |
  | --- | --- |
  | The sidebar starts collapsed. When opened, a Shortcuts block of 13–14 navigation links is expanded and Recent conversations is collapsed. With both open, the list gets about 30 px (0 px in one skeptic run). | `ChatSidebar.tsx:75` `useState(true)`; `:217-243` reapplies the tools-first layout on every open |
  | All of this resets after every reload. | `store/layout-ui.ts:9` keeps `chatSidebarCollapsed` in memory only |
  | Searching the title of the chat open in this browser returns "No server chats yet", while the tab reads "Server (60)". | The list is server-only (`ServerChatList`); `components/Common/ChatSidebar/LocalChatList.tsx` is never mounted, and the legacy local drawer renders only when the chatSidebar flag is off (`Layout.tsx:479`) |
  | ⌘K returns no chats on `/chat`. | `CommandPalette.tsx:491-525` limits chat history to the sidepanel |
  | Server search finds only titles: a word that appears only in message bodies returns 0 results. | Search matches title, topic and state only |
  | The URL stays `/chat`, so a chat can't be linked to. | No chat id in the route |

  Evidence: `shots/ft-chat/06b-sidebar-expanded.png`, `shots/skeptic-CS/s03-a-sidebar-search.png`.
- **Why it matters:** Because of CS-03, every chat created in the WebUI is local-only, and no list shows local chats. So once a first-time user clicks New chat, they cannot get back to anything they wrote. Power users spend 4–5 actions per session (Ctrl+B → expand Recent → collapse Shortcuts → type → click) and cannot find a chat by what was said in it.

  Calibration: the tools-first default is a deliberate, approved design (TASK-401, `Docs/superpowers/specs/2026-05-17-chat-sidebar-tools-first-expansion-design.md`). On its own it is friction (severity 2–3). Severity 4 comes from the combination with CS-03, and fixing either side (listing local chats, or saving to the server) removes the blocker.
- **Recommendation:**
  1. **Unblock first:** list local chats. Either mount `LocalChatList` as a "This browser" section or tab, or merge the Dexie histories into the list with a "Local" badge. When a search has no hits, say "No chats match “X”" instead of "No server chats yet".
  2. **Revisit the tools-first contract for `/chat`.** When any history exists, open Recent conversations expanded with a minimum height (e.g. 50% of the sidebar) and keep Shortcuts collapsed. Persist `recentCollapsed` (`ChatSidebar.tsx:75`) and `chatSidebarCollapsed` (`store/layout-ui.ts:9`) through `useSetting`, the way `shortcutsCollapsed` already is.
  3. **Add a "Chats" group to the CommandPalette** for the web and options scope: the 10 most recent chats plus a debounced `searchConversationsWithMeta`.
  4. **Follow-up:** search message content in `/api/v1/chats/conversations` (backend full-text search over messages), and add deep links of the form `/chat?chat=<id>` (XP-05/06).

#### CS-N1 · Choosing the Trash (or Character) filter can lock users out of their chat list: when that view is empty the filter control disappears, and the choice is saved across reloads
**Priority** P1 · **Severity** 3/4 · **Effort** S · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #3 User control and freedom; #9 Error recovery · **Verification** Found and confirmed during verification (single verifier)
- **What happens:** We set the sidebar filter to Trash and simulated an empty Trash, because other reviewers' items meant it couldn't really be emptied. The sidebar then shows only "Trash is empty." with no filter control, while the tab reads "Server (61)". The value persists in localStorage (`tldw:sidebar:serverChatFilter="trash"`), so the state survives reloads.

  `ServerChatList.tsx:1105-1120` returns the empty state before the filter Select is rendered (~1195-1215). The filter is a persisted setting (`services/settings/ui-settings.ts:397-410`), and only that Select changes it. The same trap applies to "Character Chats" for a user with no character chats, who sees "No server chats yet". Evidence: `shots/verify-CS/v10-b-empty-trash-view.png`.
- **Why it matters:** A user who empties Trash, by restoring or permanently deleting its last item, loses the whole history list on every load. So does a user who checks the Character filter before having any character chats. The only way out is clearing site data. It hits exactly the users who use Trash because they are trying to be careful.
- **Recommendation:**
  - In `ServerChatList`, always render the filter bar above the empty, error and loading states.
  - In a filtered empty state, show copy that names the filter and a [Show all chats] button that resets the filter to "all" and clears the search.
  - Reset the filter to "all" automatically when a restore or permanent delete empties Trash.
  - Don't persist "trash" across sessions; reset it on mount.
  - Add a unit test: an empty Trash view still renders the filter and a way back.

**Lower-severity issues — persistence and load errors**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CS-N3 | If a reply is interrupted by leaving `/chat`, returning creates an empty server chat (title only, 0 messages) that appears in history and on other devices | 2 | M | Make promotion transactional. Either create the chat and its messages in one call, or track a partial promotion: resume it, or delete the empty chat, and show the existing "Chat saving is incomplete — Retry" toast. Don't let the new `serverChatId` trigger a selection load or claim until every message is acknowledged (`usePlaygroundPersistence.tsx` ~389-480). Test: an interrupted turn plus remount never leaves a server chat shorter than the local transcript. |
| CS-06 | A chat that fails to load shows the raw request URL (overflowing its box) with no Retry, while the header still says "Saved" | 2 | S | Replace the plain div at `PlaygroundChat.tsx:1207-1212` with a state panel: "Couldn't load this conversation. Your messages are safe on the server." Add [Retry] (re-runs `useServerChatLoader`), [Open another chat], and collapsed technical details (`break-all`). Show the header badge as "Not loaded" while `serverChatLoadState==='failed'`, and suppress any duplicate toast. |
| CS-07 | The chat-history error in the sidebar has no Retry, tells users to check server logs, and is clipped out of view when Shortcuts is expanded (the default on every sidebar open) | 2 | S | In `ServerChatList.tsx:1079-1102`, show "Couldn't load your chats." with [Retry] (refetch via `useServerChatHistory`), and link to Diagnostics instead of "check your server logs". Add Retry to the stale-data banner (~1228). Give Recent conversations a minimum height (same fix as CS-02). |

**Lower-severity issues — history list and organisation**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CS-08 | Chat history rows are low-density: about 4–5 per screen, truncated titles, "Server" on every row, grey lowercase state chips, the topic repeating the title, and no date groups | 2 | S | Two-line `ServerChatRow`: the title, then state dot · relative time · topic (only when it differs from the title). Remove the unconditional "Server" label (`ServerChatRow.tsx:211-216`). Use Title Case states with semantic colours in `ChatStateBadge.tsx:12`. Show pin and ⋯ only on hover or focus. Group rows as Today / This week / Older. |
| CS-09 | The full-page chat has no per-chat export or "Move to folder"; export exists only in the sidepanel's right-click menu | 2 | S | Add Export ▸ Markdown/JSON and Move to folder… to the row menu (`ServerChatRow.tsx:109-182`) and the chat-header overflow. Move the Markdown formatter from `Sidepanel/Chat/ConversationContextMenu.tsx:115` into a shared util, and reuse the bulk folder picker. Rename "Open current chat settings" (it also switches chats) to "Chat settings". Add a visible ⋯ on sidepanel tabs. |
| CS-10 | Moving a chat to Trash or deleting it permanently shows no toast or Undo. With a search active, the filter disappears behind a misleading empty state ("No server chats yet") | 2 | S | After moving to Trash, show "Moved to Trash" with [Undo] (`restoreChat`) and [View Trash]. After a permanent delete, show "Chat permanently deleted." Render the filter and the search summary before the early returns (`ServerChatList.tsx:1105-1120`). Use the empty-state copy "No chats match “X”" with [Clear search]. |
| CS-11 | In-thread search opens only with Cmd/Ctrl+F, highlights each search term separately, and after Esc the highlights stay and focus drops to `<body>` (the claim that the active match isn't scrolled into view was not reproduced) | 1 | S | Add a search button to the header strip, with the shortcut in its tooltip. Pass `searchQuery` only while the bar is open (`Playground.tsx:4377`), or clear it on Esc/Close (2602-2625), and return focus to the composer. Highlight the whole phrase in `highlightText`. |

---

### 6.2 Messages & generation (CM)

**Revising answers**

#### CM-01 · Regenerate, Continue and Edit → 'Save & Send' fail on saved chats with raw error codes (the edit is discarded) while the buttons stay visible
**Priority** P0 · **Severity** 3/4 · **Effort** L · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #9 Error recovery; #3 User control; #5 Error prevention · **Verification** Confirmed (live + code); upheld by independent skeptic
- **What happens:** In a new saved chat (the default), each revision action fails:
  - **Regenerate** sends no request. It shows "Message action unavailable / unsupported_history_regeneration", which disappears within about 5 s, and it fails the same way after a reload.
  - **Continue Response** (in the overflow menu) shows "Error unsupported_history_action_context".
  - **Edit → "Save & Send"** on a user message closes the editor, reverts the text to the original, sends nothing, and shows "unsupported_history_edit_and_send:pa_…".

  Saved server chats behave the same. The only explanation is in the hidden Runtime rail ("Regeneration is unavailable for selected history."). Regenerate does work in an opt-in Temporary chat, but Save & Send fails there too.

  Code:
  - `hooks/handlers/messageHandlers.ts:88-95` throws whenever a history selection is passed in.
  - `hooks/chat/useChatActions.ts:4766-4771` passes the selection unless its status is `idle`. The status becomes `ready` after the first exchange (`hooks/useHistorySelection.ts:271-275`).
  - `messageHandlers.ts:201-209` always throws for `isHuman && isSend`.
  - `useChatActions.ts:3466-3478` throws for Continue.
  - `components/Common/Playground/EditMessageForm.tsx:41-44` calls `onClose()` before `onSumbit`, which throws away the edit.
  - `Playground.tsx` already computes `selectedHistoryBlocksRegeneration` (2984-2988, 3831-3839), but `MessageActionsBar` never receives it.

  Evidence: `shots/verify-CM/v01a-regen.png`, `shots/verify-CM/v01e-after-savesend.png`.
- **Why it matters:** Regenerate, continue, and "fix the question and ask again" are the standard ways to improve an answer, and none of them work in the default chat type. First-time users see developer error codes and lose their edit. Power users lose their main way of iterating. The only workaround, New Branch, loses the link to the original chat (CM-12). It stays at severity 3 rather than 4 because sending a new message still works.
- **Recommendation:**
  - **Short term (S):**
    - Pass `selectedHistoryBlocksRegeneration`, plus matching continue and edit-send flags, through `PlaygroundChat` → `Message` → `MessageActionsBar`.
    - Disable Regenerate and Continue with a tooltip that reuses the rail copy ("Regeneration is unavailable for this saved chat"), and offer "Branch from here instead".
    - In `EditMessageForm`, either hide "Save & Send" for user messages while `createEditMessage` cannot send, or make `onSumbit` return a promise and close only on success, so a failure keeps the typed text and shows an inline error.
    - Map all `unsupported_history_*` codes to plain copy in one helper used by every `notification.error` call.
  - **Long term (L):**
    - Implement regenerate, continue and edit-and-resend natively, as sibling rows linked by `parent_message_id` via `settleAcceptedAssistant`/`appendSelectedUser`, with a ‹ 1/2 › variant pager (also needed for CM-N2).
    - Mark answers below an edited question "Answer is for a previous version".
    - Order the editor buttons Cancel · Save · Save & Send.

**Waiting and stopping**

#### CM-N1 · Slow models time out after 30s with no first token, even though the startup timeout is 120s, and the turn ends in an unexplained 'Turn needs review'
**Priority** P0 · **Severity** 3/4 · **Effort** S · **Surfaces** WebUI, extension options, extension sidepanel · **Persona** First-time and power users · **Heuristic** Nielsen #9 Error recovery; #1 Visibility of system status · **Verification** Found and confirmed during verification (single verifier)
- **What happens:** We pointed the completions stream at a local SSE server that opens the stream but sends no tokens for 70 s. At 25 s the UI shows "Still generating response (checkpoint 4)". By 40 s it shows "Turn needs review … User input accepted; response not saved", with no reason given.

  Code:
  - `services/tldw/TldwChat.ts:568-581` allows 120 s for the first visible token (`startupTimeoutMs`, `chat-timeouts.ts:2`). But it also passes `streamIdleTimeoutMs = 30_000` to `tldwClient.streamChatCompletion`.
  - `services/background-proxy.ts:1716` starts that idle timer before the fetch, and only incoming bytes reset it (1795-1806).
  - Before the first token, the real backend sends only two events, `tldw_metadata` and `stream_start`. Heartbeats are off by default (`tldw_Server_API/Config_Files/config.txt:228` sets `streaming_heartbeat_interval_seconds = 0`; `streaming_utils.py:2391`).

  So any request whose first token takes more than about 30 s is aborted. The only override is `chatStreamIdleTimeoutMs` in Settings > tldw. Evidence: `shots/verify-CM/v03-hang-25s.png`, `shots/verify-CM/v03-hang-40s.png`.
- **Why it matters:** Self-hosted local models are a core use case for this product. Any model that takes more than about 30 s to produce its first token is killed on every long request, and the user sees only an unexplained panel. This is common with CPU inference, long prompts or large contexts. Users retry or reload, which wastes server compute, and nothing tells them a timeout setting exists.
- **Recommendation:**
  - In `TldwChat.ts`, don't start the idle timer until the first data chunk arrives. Alternatively, pass `max(startupTimeoutMs, streamIdleTimeoutMs)` until `hasVisibleAssistantProgress`, then tighten to 30 s.
  - Turn backend heartbeats on by default (e.g. `streaming_heartbeat_interval_seconds = 15`).
  - When a timeout fires, show the reason with Retry in the recovery panel, e.g. "No response from &lt;model&gt; after 30s — increase timeout in Settings" (see CC-02 and CM-06).
  - Add a test: a stream whose first token arrives after 40 s completes when the startup timeout is 120 s.

#### CM-02 · Stop is presented as a failure ('Turn needs review') and an early Stop can destroy the user's message with a raw error
**Priority** P1 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #3 User control and freedom; #1 Visibility of system status · **Verification** Confirmed (live + code); upheld by independent skeptic; one sub-claim refuted
- **What happens:** The result of Stop depends on when it is pressed:
  - **Before the first token:** the user bubble stays, and the panel shows "Turn needs review … User input accepted; response not saved" with only Copy and Dismiss. There is no "Stopped" label, Continue or Regenerate.
  - **Mid-stream:** the partial answer is removed from the thread ("Generated response needs review") and can only be seen under Inspect. The skeptic also found that after the next normal send, the stopped turn's user message disappears as well, and stays gone after a reload.
  - **Within the ~300–500 ms first-send preparation window:** the user message is removed, the composer is emptied, and nothing is sent. Sometimes a raw "Error request_config_scope_changed" toast also appears.

  During generation, Send becomes "QUEUE ▾" and two icon-only Stop buttons appear.

  Refuted: the original report said the in-bubble Stop had no accessible name. It does: `Message.tsx:2092-2103` gives it a `title` and screen-reader-only text, "Stop streaming response".

  Code:
  - `hooks/chat-modes/chatModePipeline.ts:888-891` throws on any abort that has a history turn.
  - Its catch block (1108-1127) calls `recover()` and filters out the assistant row.
  - `normalChatMode.ts:772-786` maps the result to `generated_unsaved`/`accepted_unsent`.
  - `chatModePipeline.ts:776-783` throws `request_config_scope_changed` when the abort happens before dispatch.

  Evidence: `shots/verify-CM/v03-slow-after-stop.png`, `shots/skeptic-CM/s3-real-after.png`.
- **Why it matters:** Users press Stop often with real LLMs, whenever an answer runs on or goes off track. Treating Stop as an error teaches them that Stop is dangerous, and throwing away the partial answer removes exactly the text they meant to keep. First-time users also meet an unexplained "QUEUE" just when they want to stop. The text can still be recovered through Copy or Inspect, so this is severity 3, not 4.
- **Recommendation:** Treat a user abort as a normal outcome.
  - In `chatModePipeline.ts` (~888), when `signal.aborted` and some text has arrived, save the partial answer via `settleAcceptedAssistant` with a "stopped" marker (this needs the metadata support from CM-04). Render it in place with a "Stopped" chip, Continue and Regenerate.
  - When nothing has arrived, dismiss the recovery record and show "Stopped before reply" under the user bubble.
  - For aborts before admission (776-783), return a cancel result instead of `request_config_scope_changed`, put the typed text back in the composer, and show no error toast.
  - Give the composer button a visible "Stop" (Esc) label, and rename QUEUE to "Send next" with a tooltip.

**Lower-severity issues — generation status and answer integrity**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CM-03 | Answers cut off by `finish_reason="length"`, or by a stream that ends without `[DONE]`, look complete: no marker, Continue or Retry | 2 | M | Expose the final `finish_reason` through `generationInfo` (today it exists only as a type field, `TldwChat.ts:408`). In `background-proxy.ts:1818-1828`, emit `stream_transport_interrupted` when the stream ends without `[DONE]`. Show both cases with the existing partial-save marker (`Message.tsx:499-520`) as "Response incomplete (length limit)" with Continue and Regenerate, and persist the flag (CM-04). |
| CM-06 | Long waits show no elapsed time. Screen readers hear "Still generating response (checkpoint N)" every 5 s, and timeouts end in an unexplained "Turn needs review" | 2 | S | In `ActionInfo.tsx:114-133`, replace the 5 s checkpoint with plain announcements at about 10 s and 30 s ("Still waiting for &lt;model&gt;"). Add a visible elapsed timer to the generating pill. Carry the timeout reason into the recovery record and offer Retry, Switch model and a link to Settings > tldw timeouts (pairs with CM-N1). |
| CM-N2 | Alternative replies stored as `parent_message_id` siblings are moved to the end of the thread, where they read as answers to the wrong question, and are included in the model's context | 2 | M | Find the active path by walking `parent_message_id` back from the latest message (in `useServerChatLoader` and the native capture in `services/chat-history-selection.ts`) instead of sorting by time. Group siblings as variants with a ‹ 1/2 › pager (shared with CM-01), leave inactive siblings out of the default path, and note "This chat has alternative replies" in `HistorySelectionReview`. |
| CM-04 | After a reload, answers are labelled "Assistant" instead of the model that wrote them, because the server stores no message metadata | 2 | M | Backend: add an optional, allow-listed `metadata_extra` field to `MessageCreate` (`tldw_Server_API/app/api/v1/schemas/chat_session_schemas.py:418`) for `model_id`, provider, `finish_reason`, interrupted and usage. Frontend: allow these keys in `nativeHistoryMessagePayload` (`services/chat-history-selection.ts:473-496`) and send them from `settleAcceptedAssistant`; `useServerChatLoader.ts:474-476` already reads `meta.model_id`. Show a compact model chip in the assistant header. |
| CM-14 | The first send in a new chat clears the composer, flashes a "Loading selected history" panel, and shows the user's bubble only after about 450–510 ms, versus about 170 ms on later sends (measured on a dev build) | 1 | M | Add the user's bubble immediately on Enter, before `captureNormalHistoryTurn`/`historySelection.open` run. For chats with no prior messages, suppress or delay (~300 ms) the loading panel (`HistorySelectionReview.tsx:243-249`). Keep the composer text until the bubble renders, so an early failure or Stop can restore it. |

**Lower-severity issues — branching and compare**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CM-12 | New Branch creates an unlinked "Forked conversation": no parent link, no breadcrumb, a generic title, and the character is dropped (a regression from the legacy path) | 2 | M | In `commitNativeFork` (`services/chat-history-selection.ts:807-815`), pass a title such as "&lt;parent&gt; — branch at msg N", `parent_conversation_id`, and the parent's character, assistant, state and topic, as the legacy path did (`useChatActions.ts:4672-4688`). Add a success toast with "Back to parent", a breadcrumb in the header, and a "Branches (n)" marker via `/api/v1/chat/conversations/{id}/tree`. |
| CM-05 | `/chat` advertises Compare in Modes, the Alt+Shift+C shortcut and the Shortcuts panel, but a hidden flag disables it and no reason is shown. Only More tools → "Compare models" works, by linking to Model Playground | 2 | S | When `!compareFeatureEnabled`, give Modes → "Compare responses" a reason line linking to `/model-playground` (`PlaygroundModeLauncher.tsx:90`). Make `toggleCompareMode` (`useModelComparison.ts:172-174`) navigate there, or show a toast with an "Open Model Playground" action. Hide Alt+Shift+C from the `/chat` Shortcuts panel (`Playground.tsx:4222-4225`). Before enabling compare in `/chat`, fix "Add models" (`PlaygroundComposerNotices.tsx:629-643`) so it opens the compare model picker, not single-model settings. |

**Lower-severity issues — message rendering and actions**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CM-09 | In casual mode, message actions appear only on hover, and the "•••" chip has no click handler. The overflow menu mixes role-play steering, unexplained single-letter icons and a disabled "Read aloud" with no reason into normal chats (Delete is guarded by a confirm) | 2 | S | Always show the compact action row on the latest assistant message (`Message.tsx:1712-1717`). Give the chip an `onClick`, or mark it `aria-hidden` (`MessageActionsBar.tsx:513-521`). Show steering items only when a character or scene is active (:258-264). Group the overflow as Reuse · Transform · Danger with dividers, and replace the letter spans with icons. Add a `disabledReason` tooltip to `OverflowMenuItem` (:51-75). |
| CM-07 | Code blocks in answers have no copy button or language label | 2 | S | In `components/Common/Markdown.tsx` (~423-480), give the `compact`/`github` variants a slim header with the language label and a Copy button (shared with `CodeBlock.tsx`). Show it on hover or focus, and always on touch. Add a unit test that a python code fence renders "Copy code". |
| CM-13 | In Pro mode the labelled action pills ("Copy", "Edit", "Redo") overflow their fixed 32 px width and overlap | 2 | S | Apply `sm:w-8` only in casual mode (`MessageActionsBar.tsx:33-34`, 204-206). In Pro mode use `sm:w-auto sm:min-w-8 sm:px-2 gap-1`. Add a visual or Vitest check that `scrollWidth ≤ clientWidth`. |
| CM-11 | Sending while scrolled up shows neither your message nor the reply. The layout shrinks on send, which is misread as the user scrolling up, so later answers can land below the fold | 2 | S | Scroll to the latest message on local send (today only `ArtifactsPanel.tsx:216` triggers this), and keep the new user message near the top. In `useSmartScroll.tsx:69-93`, treat a `scrollTop` decrease as user intent only when `scrollHeight` didn't shrink, or listen for wheel, touch and key events instead. While auto-scroll is paused, turn the chevron into a "Reply in progress ↓" pill. |
| CM-08 | Inline code shows literal backticks and bold weight, with no pill background | 1 | S | In the `code()` renderer at `Markdown.tsx:504-510`, use `rounded bg-surface2 px-1 font-mono font-normal before:content-none after:content-none`, or add `prose-code` overrides to `MARKDOWN_BASE_CLASSES` (`Message.tsx:1967`, `MessageContent.tsx:21`). Apply the same to user messages and the Notes preview. |
| CM-10 | User and assistant cards are hard to tell apart (low-contrast fill difference). Each answer has two ellipsis buttons plus a "Was this helpful?" prompt that appears only in some server chats (the "Assistant" header label is tracked under CM-04) | 1 | S | At `Message.tsx:1973-1977`, tint user cards and right-align them (`max-w-[75%]`) in casual mode. Fold `FeedbackButtons` into the action row or the main overflow so each message has one ellipsis. Show the model chip once CM-04 lands. |

---

### 6.3 Composer, models & context (CC)

**Controlling what the model sees**

#### CC-04 · 'Review conversation history / Use no prior messages' controls expose an internal history-path model; one click silently blanks the visible transcript and the model's context
**Priority** P0 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, extension options, extension sidepanel · **Persona** First-time and power users · **Heuristic** Nielsen #2 Match with real world; #3 User control; #5 Error prevention; #8 Minimalist design · **Verification** Confirmed (live + code); upheld by independent skeptic
- **What happens:** A "Review conversation history" link and a "Use no prior messages" button sit at the top of the transcript, including on a "new" chat after New chat. They are prominent in new or short chats and scroll out of view in longer ones. The button also appears on its own while a reply streams.

  One click on "Use no prior messages" does all of this:
  - Blanks the entire visible thread, with no confirmation, toast or undo.
  - Drops the token counter to 0.
  - Shrinks the next request from 3 messages to 1.
  - Keeps the transcript blank after a reload.

  Nothing is actually deleted; only the view cursor moves. But the only way back is the review panel, which has three problems:
  - It lists the assistant reply as "Path position 1", ahead of the user message it answers.
  - It shows raw markdown.
  - It uses copy such as "Complete source / included path: 4 / 4" and "Confirm creates a saved interpretation of this complete source."

  In the sidepanel, the same click replaced the thread with "Start a conversation below".

  Code: `components/Common/Playground/HistorySelectionReview.tsx:316-345` renders both controls whenever `selection.status === 'ready'`. The button's `onClick` immediately calls `selection.choose({kind:'empty'})` (`hooks/useHistorySelection.ts:465-500`). Evidence: `shots/verify-CC/v05-after-noprior.png`, `shots/verify-CC/v10-review-open.png`.
- **Why it matters:** First-time users read the pair as "see my chats" and "clear", and then the conversation vanishes, which looks the same as data loss. Power users who do want to control context only get a jargon-heavy, misordered panel, and nothing persistent shows what the model currently sees.
- **Recommendation:**
  1. Remove the top-level "Use no prior messages" button from `HistorySelectionReview.tsx:316-345`. Keep the choice inside the expanded review, or behind a message overflow item "Start fresh from here…", with an undo toast: "Earlier messages hidden from the AI · Undo".
  2. Never blank the visible transcript for an empty or partial selection. Keep earlier messages visible but dimmed behind a divider, and show a persistent chip such as "AI sees 0 of 4 messages · Restore all".
  3. Rename "Review conversation history" to "Choose what the AI remembers", and show it only once the conversation has at least 2 turns or contains alternative replies.
  4. In the review list, sort messages in chronological order (each user message before its reply), show plain-text previews, and replace the "saved interpretation" copy with plain language.

#### CC-05 · The chat Prompt picker ignores the server prompt library ('No saved prompts' with 11 server prompts)
**Priority** P0 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, extension options · **Persona** Power users · **Heuristic** Nielsen #4 Consistency; #6 Recognition rather than recall · **Verification** Confirmed (live + code); upheld by independent skeptic
- **What happens:** The server holds 11 prompts, including "Weekly review" and "Socratic tutor", and a "Daily drivers" collection. Yet Composer → Prompt shows "No saved prompts", and typing "Weekly" gives "No matching prompts". The only prompt request `/chat` makes is `GET /api/v1/prompts/capabilities`, and ⌘K finds nothing either. On `/prompts`, a search does reach the server (`POST /api/v1/prompts/search`). But the page shows "Showing synced local matches only — 1 result(s) from this page are not saved locally yet" above "No custom prompts yet", with no import action.

  Code:
  - `components/Common/PromptSelect.tsx:9`, `:328-334` and `Option/Prompt/index.tsx:310` read only the local Dexie store (`getAllPrompts`).
  - `custom-prompts-utils.ts:55-80` drops server results that have no local match.
  - `services/prompt-sync.ts` syncs only Prompt Studio projects. The palette's "Server prompt" action pulls from Prompt Studio, which is a different table.
  - `Common/PromptInsertModal.tsx` and `PromptSearch.tsx` already read from the server, but nothing mounts them.

  Evidence: `shots/verify-CC/v07-prompt-menu.png`, `shots/skeptic-CC/s05-prompts-search.png`.
- **Why it matters:** Curated prompts created through the API, on another device, in the extension, or by a Chatbooks import can't be used in chat. (Chatbooks reads and writes `/api/v1/prompts`, `ChatbooksPlaygroundPage.tsx:538-545`.) The empty state also tells users their prompts don't exist. The only workaround is copy-paste. This mainly affects power users; first-time users have no server prompts yet.
- **Recommendation:**
  1. In `PromptSelect.tsx`, add a second query that calls `tldwClient.getPrompts()` when `capabilities.hasPrompts` is true, reusing `normalizeServerPrompt` from `PromptInsertModal.tsx`.
  2. Merge the server prompts into the menu as a "Server library" group with a badge, de-duplicated by `serverId` or title.
  3. Change the empty state to "No prompts on this device — N prompts on your server", with an Open Prompts / Import action. Apply the same merge to the Prompts page (`Option/Prompt/index.tsx:310`).
  4. Later: add a `/prompt <name>` slash command (see CC-11).

**Model selection and failure recovery**

#### CC-01 · First-run default model is an unreachable provider labelled 'Healthy'; the server's default provider is ignored
**Priority** P1 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, extension options · **Persona** First-time users · **Heuristic** Nielsen #1 Visibility of system status; #2 Match with real world · **Verification** Confirmed (live + code); upheld by independent skeptic; previously reported (June 2026 chat UAT review, #3)
- **What happens:** On a genuine first run the model chip reads "Ollama / gemma3:1b · Healthy", although Ollama isn't running. The first send returns 502 `provider_unavailable`. Fifteen seconds later the chip still says "Healthy" and no toast has appeared.

  The server knows better. `GET /api/v1/llm/providers` returns `default_provider = custom-openai-api`. It also reports every provider as "healthy" with `success_count 0`, meaning none has ever been checked. The picker's "Current" group lists "gemma3:1b" without its provider.

  This is not a test-environment artefact. The shipped `tldw_Server_API/Config_Files/config.txt:1136-1137` sets `ollama_api_IP` and `ollama_model=gemma3:1b`, so Ollama counts as configured on stock installs.

  Code:
  - `utils/model-startup-selection.ts:19-61` picks the first favourite or the first configured model, and takes no server-default input.
  - `PlaygroundForm.tsx:1789-1800` sets the label to "Healthy" whenever the tldw server connection is ready.
  - `ChatModelSelectorDropdown.tsx:56-60` renders that label inside the model chip.

  Two things soften this. Users with a commercial API key get a working default, because commercial providers sort first. The setup wizard does hand off a verified model, but it can be skipped, and it is reduced to a banner when the server is already connected (`option-index.tsx:270-290`).

  Evidence: `shots/verify-CC/v01-idle.png`, `shots/verify-CC/v04-after-send-15000.png`.
- **Why it matters:** The first message is when users are most likely to give up. Local-only and custom-endpoint users, the core self-hosting audience, see their first send fail while the UI says the broken choice is healthy, and CC-02 then hides the cause. The "Healthy" label also stays after any later provider failure, so it misleads power users too.
- **Recommendation:**
  1. Pick a reachable default. In `resolveStartupSelectedModel`, add `serverDefaultProvider` and `providerHealth` inputs, taken from the `/llm/providers` query that `PlaygroundForm` already makes. Choose in this order: favourites, then the server default provider's model, then a provider with `success_count > 0`, then the first configured model. Normalise provider keys (`custom-openai-api` vs `custom_openai_api`). Note that the stock `default_api` is "openai", so honouring the server default alone won't help local-only installs. The reachability signal is required.
  2. Stop showing server connectivity in the model chip (`PlaygroundForm.tsx:1789-1800`); show it only in the header. Give the chip a per-model state: Not checked / Working / Unreachable. Set it to Unreachable after a `provider_unavailable` or 5xx response, and offer a one-click "Switch to &lt;server default model&gt;".
  3. In the backend `/llm/providers`, report "unknown" instead of "healthy" when `success_count + failure_count == 0`.
  4. Show the provider next to the model in the picker's "Current" group.

#### CC-02 · Send failures show a jargon 'Turn needs review' recovery panel with no cause, no Retry and no model switch (WebUI, options and sidepanel)
**Priority** P1 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, extension options, extension sidepanel · **Persona** First-time and power users · **Heuristic** Nielsen #9 Help users recognize, diagnose, and recover from errors; #2 Match with real world · **Verification** Confirmed (live + code); upheld by independent skeptic
- **What happens:** In a saved chat (the default), a 502 `provider_unavailable` produces only an unstyled box: "Turn needs review / The original send outcome is retained separately from history. It will not be sent again automatically. / User input accepted; response not saved / Inspect original input and result / Copy recovered text / Dismiss recovery". A stray "Use no prior messages" sits under it.

  What the box lacks:
  - Any error text, status code, provider or model. Inspect shows only the user's own input.
  - Any alert or toast; the box is `role=status`.
  - Recovery actions. The user message's own menu offers only Copy, Edit and Delete.

  The same failure in a Temporary chat shows UI the product already has. An ErrorBubble offers Retry same model / Switch model / Try provider fallback / Show technical details, and a composer banner offers Retry chat / Edit provider / Switch provider / View in Health & Diagnostics. Extension options and the sidepanel behave like the WebUI.

  Code:
  - `hooks/chat-modes/chatModePipeline.ts:1108-1132` sends every failed history turn, including user aborts, to `historyTurn.recover()` and removes the assistant row.
  - `HistoryTurnRecovery` (`db/dexie/types.ts:279`) has no fields for the error.
  - The composer error banner (`usePlaygroundChatErrorBanner`, `PlaygroundChatErrorBanner.tsx:109`) only appears when there is an error bubble.
  - `HistorySelectionReview.tsx:197-240` renders only the input and result text.

  Send also stays enabled when the input is empty (minor). Evidence: `shots/verify-CC/v04-after-send-15000.png` compared with `shots/verify-CC/v16-temp-error.png`.
- **Why it matters:** Provider failures are common with self-hosted models. Users can't tell which model failed or why, and can't recover in one step. They also have to read internal state-machine vocabulary, and many will conclude the app is broken. Refusing to retry automatically is reasonable when the outcome is genuinely unknown. A 502, though, is a definite failure, and saved chats lose a recovery UI the product already has.
- **Recommendation:**
  1. In the `chatModePipeline.ts` catch block, separate definite failures from unknown outcomes. A definite failure means an HTTP error (502, 4xx), no streamed content, and no user abort. For those, keep the assistant row as an ErrorBubble (`buildAssistantErrorContent`), exactly as the non-history path does, so the existing ErrorBubble actions and composer banner work for saved chats. Keep the recovery panel only for genuinely unknown outcomes, such as a network drop after the request was accepted.
  2. Add `error_code`, `status`, `message`, provider and model to `HistoryTurnRecovery`. Render them in plain language, e.g. "Couldn't get a reply from Ollama · gemma3:1b (provider unavailable)". Add Retry and Switch model, use error styling and `role=alert`, and put Inspect behind Details.
  3. Treat a user Stop as "Stopped", not a failure (see CM-02).
  4. Mark Send `aria-disabled` when the input is empty.

**Composer input**

#### CC-N1 · Typing a slash command runs it on every keystroke (/web and /search flip settings while you type and Enter cancels them; /model opens settings mid-word and takes focus)
**Priority** P1 · **Severity** 3/4 · **Effort** S · **Surfaces** WebUI, extension options · **Persona** First-time and power users · **Heuristic** Nielsen #1 Visibility of system status, #3 User control and freedom; WCAG 2.2 SC 3.2.2 On Input, SC 2.4.3 Focus Order · **Verification** Found and confirmed during verification (single verifier)
- **What happens:** Slash commands run while the user is still typing:
  - Typing "/web latest news" one key at a time flips the "Web search" chip on and off at almost every keystroke from "b" onward. Whatever state it ends in stays after the text is deleted.
  - Typing "/web" switches web search on before Enter. Enter then runs the command again, so "/web" + Enter ends where it started.
  - Typing the final "l" of "/model" opens "Current Chat Model Settings" and moves focus to its Close button. The user's Enter then presses Close, so the modal disappears, "/model" stays in the composer, and focus falls to `<body>`.

  This is not a dev-mode or StrictMode artefact.

  Code: `components/Option/Playground/hooks/usePlaygroundRawPreview.ts:380-386` recomputes `rolePlayCompatibility` on every change to the composer text via `useMemo(resolveSubmissionIntent(formMessage))`. `resolveSubmissionIntent` calls `applySlashCommand` (`hooks/playground/useSlashCommands.ts:204-217`), which runs the command's `action()`. The actions are `setWebSearch(!webSearch)`, the `/search` chat-mode toggle, `handleImageUpload` for `/vision`, and `setOpenModelSettings(true)` for `/model`. These side effects therefore run during rendering. Evidence: `shots/verify-AX/22-web-args.png`, `shots/verify-AX/17-slash-model.png`.
- **Why it matters:** Slash commands are the keyboard-first route for power users, and they misfire without any sign. Web search or RAG mode ends up on or off depending on how many characters were typed, which changes the next request's behaviour and cost. "/model" steals focus mid-word, so keyboard and screen-reader users lose their place, and "/vision" can open a file picker unprompted. First-time users who follow the placeholder hint ("/ commands") hit the same flicker.
- **Recommendation:**
  - In `hooks/playground/useSlashCommands.ts`, separate parsing from running:
    - Make `resolveSubmissionIntent` pure: it returns `{ handled, message, command }` without calling `action()`.
    - Add `executeSlashCommand(intent)`, called only from `usePlaygroundSubmit.ts:230` and `handleSlashCommandSelect`.
  - Switch `usePlaygroundRawPreview.ts:380-386` to the pure resolver.
  - Add RTL tests:
    - Type "/web abc" one character at a time and assert `webSearch` doesn't change before submit.
    - Type "/model" and assert the modal stays closed until Enter.
  - Check the sidepanel's `hooks/useSlashCommands.tsx` (actions at 440/461) for the same pattern.

**Lower-severity issues — connection and model status**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CC-03 | Returning users briefly see a false "You're offline" state (about 0.4–0.6 s on a warm server, longer when the health check is slow). Pressing Enter during it is silently ignored, although the QUEUE button would queue the message. Genuine first runs don't show this flash | 2 | S | At `PlaygroundForm.tsx:4067-4071`, make Enter use the same queue handler as the Queue button while `!isConnectionReady` (keeping the IME and slash-menu guards). In `PlaygroundComposerNotices.tsx:731`, show the offline alert only after a failed check, and show "Connecting…" while the connection is still being checked (phase `SEARCHING`, `store/connection.tsx:458`). Reserve the notice row's height so the layout doesn't jump. |
| CC-N2 | A provider failure (502 `provider_unavailable`) is reported as "Something went wrong while talking to your tldw server". It is shown twice, as an ErrorBubble and a composer banner with different actions, and the chip stays "Healthy" | 2 | S | In `utils/chat-error-message.ts`, add a branch for `provider_unavailable` and 502/503 (the generic fallback is at 310-318): "Couldn't reach &lt;provider&gt; (&lt;model&gt;)" with the hint "The tldw server is fine…", and make Switch model the main action. Feed the failure into the chip state (CC-01). Show the banner only when the ErrorBubble is out of view, or merge their action sets. |
| CC-06 | The model is named in several formats: the chip truncates the model name, and raw `provider_key:model` ids appear while streaming and in the rails. The picker's scope toggle labels the opposite state and doesn't add any models | 2 | S | In `ChatModelSelectorDropdown.tsx:185`, put the model name first and truncate only the provider. Use one formatter for the chip, the streaming header, the Improve menu, the offline chip and the picker's "Current" group. Replace the toggle in `PlaygroundModelCatalogControls.tsx:34-61` with a two-option segmented control, "Ready to use" / "All models". Either feed the full catalog into the catalog scope (`modelSelectorUtils.ts:231-239`) or remove the toggle. |

**Lower-severity issues — capability naming and placement**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CC-08 | Compare, Web search, Knowledge, Saved/Temporary, Shortcuts and theme each have two or three homes, under different names | 2 | M | Agree a table of one home per capability and apply it across `PlaygroundModeLauncher`, `PlaygroundToolsPopover` and `ComposerToolbar`: one name per capability; Modes holds conversation modes only; Web search becomes one composer toggle, with its sub-settings in More tools. Move Generate image and Use OCR out of ATTACHMENTS. Remove the page-strip theme toggle (`Playground.tsx` ~4034). Make the header Saved/Temporary badge the only control for that setting. |
| CC-10 | Search & Context takes over the viewport. The transcript collapses, Send is pushed off-screen, source chunks sit below an automatically generated LLM answer, and relevance is shown as a raw score | 2 | M | Render `PlaygroundKnowledgeSection` (`PlaygroundForm.tsx:5709`) in the context rail or a drawer no wider than 40%. List source chunks first, with Insert on Enter and multi-select. Make "Generate answer" an explicit button. Show relevance as a percentage or a bar. Cap the panel at 50vh so Send stays visible. |
| CC-11 | The placeholder promises "@ mentions", but "@" does nothing in the WebUI (it is extension-only and off by default). Only five slash commands exist | 2 | S | At `PlaygroundForm.tsx:5851-5854`, show "@ mentions" only when `isExtension && tabMentionsEnabled`; otherwise use "Type a message… (/ for commands)". Follow-up (L): make "@" attach Notes, Media, Chats and Prompts as context using the existing search services, and add `/prompt`, `/note` and `/new`. |

**Lower-severity issues — composer visual design**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CC-12 | The Improve-prompt (wand) menu has a transparent background, so its items run into the composer and thread behind it | 2 | S | In `PromptAssistMenu.tsx:331`, replace `bg-popover`, `text-popover-foreground` and `text-muted-foreground` with tokens that exist (`bg-surface`, `text-text`, `border-border`, `text-text-muted`), or add the missing tokens to both Tailwind configs. Add a lint rule that flags colour classes the theme doesn't define. |
| CC-07 | Send is the weakest control in the composer (75×24 px, 11 px uppercase, not styled as primary), and the toolbar mixes sizes, corner radii and borders | 1 | S | In `PlaygroundSendControl.tsx:315-337` and 365-369, make Send a primary button about 36 px tall with a 13 px sentence-case "Send" and icon, dimmed when the input is empty (as in the sidepanel). Shrink the 44 px Improve button to a 32 px ghost icon. Define toolbar tokens: 32 px height, 12–13 px labels, one radius. |
| CC-09 | "Advanced controls" shows a connection pill that looks like a switch and four unlabelled preset icons | 1 | S | In `ComposerToolbar.tsx:1003-1006`, replace the icons with a labelled segmented control, "Response style: Creative · Balanced · Precise · Custom" (non-compact `ParameterPresets`). Remove `ConnectionStatus` from the row or show its label. Size "System prompts" like the controls beside it. |

---

### 6.4 Onboarding, layout & visual design (CO)

No CO issue stayed at severity 3 after verification, but together they leave too little room for the conversation at common sizes:
- At 1024×768 with the rails open, the composer textarea is about 78 px wide (CO-04).
- On a portrait tablet the transcript gets 45% of the height (CO-05).
- At 1024 px with the sidebar open the transcript gets about 325 px (CO-06).
- On a phone the focused composer takes 56% of the screen (CO-01).

For a first-time user, the default screen at 1440 px shows about 20 controls before the first message (CO-03).

**First run and onboarding**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CO-02 | "Take a quick tour" does nothing in the WebUI's default cockpit layout. It works in Focus mode and in the extension, and the "Chat Basics" tutorial exists | 2 | S | Mount `<PageHelpModalHost/>` unconditionally at `apps/tldw-frontend/components/layout/WebLayout.tsx:709`: drop the `hideHeader` guard, as the extension's `Layout.tsx:604` already does. Better still, have `PlaygroundEmpty.tsx:246-252` start the `playground-basics` tutorial directly. Add a Vitest check that the link opens help or a tour step. |
| CO-03 | The composer and surrounding chrome show about 20 jargon-labelled controls, duplicate theme toggles, three different "shortcuts" controls, and pill-styled labels that aren't buttons | 2 | M | Cut the casual toolbar (`ComposerToolbar.tsx:953-990`) to Attach, "Add sources", the model chip and Send. Move MCP, Prompt, persona, Role-play, Buddy & Persona and OpenUI into More tools, each with a one-line description; keep the full row in Pro mode. Remove the second theme toggle (`Playground.tsx:4034-4047`). Fold the three "shortcuts" controls into the global "?" modal plus "Quick links". Render the region labels as plain text (`Playground.tsx:4005-4031`, 4413-4420). Give the disabled Compare option a reason. |

**Responsive layout**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CO-01 | On a phone (390 px) `/chat` opens in Focus mode with only an "Exit focus" control. Once focused, the composer takes about 56% of the screen and stays expanded after Send. Other problems: a duplicate model pill, identical icons for different menus, and a 16 px persona target. The layout is fine before the composer is tapped, and a compact mode exists but is hard to find | 2 | M | Default `composerOptionsExpanded` to false on mobile and add one labelled "Options" button that opens a bottom sheet. Move focus off Send after sending so the panel collapses. In mobile Focus mode, add a compact top bar (☰ history, title, + new chat) next to the layout-mode trigger in `Playground.tsx`. Remove the duplicate "Model …" pill, give More options and More tools different icons, show health as a dot in the chip, and make the persona target at least 24 px. |
| CO-04 | The cockpit rails mostly restate state in jargon: the model is shown up to 6 times, and most cards read "Empty" or "Idle". At 1024 px with the rails open, the textarea shrinks to about 78 px | 2 | M | In `PlaygroundCockpitShell.tsx:228-234`, render the rails inline only at 1280 px and wider. Between lg and xl, show a rail as an overlay drawer, one at a time, without changing the saved preference (`Playground.tsx:531-534`). Collapse empty cards into one summary line. Show the model once, with the same provider label as the composer. Rename Composition, Scope, Provider route and sidechannel. |
| CO-05 | Below 1024 px, a 2×2 grid of duplicate jargon rail buttons ("Restore context sidechannel", "Show context rail", …) takes about 115 px, and the transcript gets 45% of a portrait tablet screen | 2 | S | Replace the grid at `PlaygroundCockpitShell.tsx:266-345` with one "Panels" button (or two icon toggles) that opens the rails in a bottom drawer with tabs "Context" and "Model & tools". Delete the duplicate buttons. At lg, pad the transcript so the edge tabs don't overlap it. Aim for a transcript of at least 60% of the height at 768×1024. |
| CO-06 | At 1024 px with the sidebar open, the header wraps to two rows and the toolbar to three, leaving about 325 px for the transcript | 2 | M | Below about 1200 px of available width, collapse the secondary controls (Role-play, Buddy & Persona, OpenUI, MCP, Prompt) into More tools. Measure width with a ResizeObserver on the toolbar (`ComposerToolbar.tsx:953-990`), not the viewport. At the same width, move Temp, Character, Share, Settings and Notifications into a header ⋯ menu (`ChatHeader.tsx:378`). Use the same type size for Buddy & Persona as its neighbours. |

**Floating controls**

| ID | Issue | Sev | Effort | Recommendation |
| --- | --- | --- | --- | --- |
| CO-07 | The "Scroll to latest" chevron appears even on empty chats that can't scroll, and can cover calls to action and text | 1 | S | In `useSmartScroll.tsx:39-47` and 150-154, hold the at-bottom value in state, updated on scroll and by a ResizeObserver, and treat a missing or non-scrollable container as at-bottom. Render the chevron only when `messages.length > 0` (`Playground.tsx:4422`), and place it at the right edge above the composer. Apply the same fix in ResearchWorkspace/ChatPane and ModelPlayground. |
| CO-N1 | In mobile Focus mode, the floating "Exit focus" pill covers the top-right of message text | 1 | S | Reserve space for the pill: put it in the compact top bar proposed in CO-01, or add top padding to the transcript equal to its height while `focusModeActive && isMobileViewport` (`Playground.tsx`). |

---

## 7. Extension, cross-page workflows & accessibility — issues and solutions

This section covers 57 confirmed issues: 18 in the extension side panel (XS), 20 in cross-page workflows and extension parity (XP), and 19 in accessibility (AX). Eight are severity 3: four P0s (XS-01, XS-07, XP-02, XP-08) and four P1s (XS-08, AX-03, AX-04, AX-05). They get full write-ups. The remaining 49 are listed in compact tables, where severity 2 means P2 and severity 1 means P3. Severities are the final values after live verification and an independent skeptic pass. Where verification changed a severity or narrowed a claim, the entry says so.

Three themes run through this section:

1. **The extension keeps three copies of a conversation that never reconcile.** A conversation can exist as a side-panel tab snapshot in `chrome.storage`, a local Dexie history, and a server chat. No code keeps the three in sync. This one gap causes tab corruption (XS-01), silent forks (XP-08), a fake delete (XS-07), a false "Server" save label (XS-05) and invisible history (XS-06).
2. **The chat-to-notes loop is broken on most chats.** "Save to Notes" is hidden on unsynced chats (XP-01) and on server chats that were loaded through a history capture (XP-02). When a save does work, the provenance is one-way and hard to follow (XP-03). There is also no way to start a chat from a note (XP-04).
3. **Many accessibility defects share one root cause each.** They come from a handful of shared parts: the antd theme tokens (AX-07, AX-08, XP-20), antd `message` (AX-06), Popover used where a Dropdown menu belongs (AX-02), and the shared `TableBlock` and `SlashCommandMenu` components (AX-10, AX-12). Fixing each part once fixes the issue on all three surfaces.

*Conventions.* Code paths are relative to `apps/packages/ui/src/` unless they start with `tldw_Server_API/`. Line numbers refer to dev @ 86e287fee7. Screenshot paths are relative to the review's `scratchpad/shots/` directory.

---

### 7.1 Extension sidepanel (XS)

#### 7.1.1 Conversation integrity and data safety

#### XS-01 · Opening a past chat from sidepanel search overwrites the current tab's conversation (silent data loss and duplicate tabs)
**Priority** P0 · **Severity** 3/4 · **Effort** M · **Surfaces** Extension side panel · **Persona** Power user · **Heuristic** Nielsen #5 Error prevention; #1 Visibility of system status · **Verification** confirmed (live+code); upheld by independent skeptic. Lowered from 4 to 3 because the original messages survive in local history and can be recovered.
- **What happens:**
  - **Repro.** Have a side-panel tab with a live exchange (tab A, 2 messages). Open the chat list (Ctrl+B), search, and open any *server* chat.
  - **Result.** Tab A keeps its label, but its snapshot is replaced by the opened chat's messages with `serverChatId` null. A second tab holding the same chat is also created.
  - **Tab A afterwards.** Re-selecting tab A shows the foreign transcript under a red card: "Unavailable — Selected history unavailable. … A supported, verified history owner is required to continue from a selection." Sending a follow-up in that tab adds nothing, clears the typed text, and throws an uncaught `owner_conversation_mismatch` with no toast.
  - **Reproduction.** Four reviewers reproduced this on seeded and self-created chats, at 360px and at 420px, including from a blank first tab. Opening a *local* history does not trigger it.
  - **Cause.** `openServerChat` awaits `historySelection.loadConversation` (`routes/sidepanel-chat.tsx:1455`) while the old tab is still active. The `onCapture` handler (445-450) pushes the captured messages into the shared store. The effect at 1165-1168 then re-runs `saveActiveTabSnapshot` (935-944) and writes them into tab A. The guard `isSwitchingTabRef` is set only later, in `openSnapshotTab` (~1293-1305).
  - **Evidence:** `pu-ext/11b-tabA-top.png`, `verify-XS/02a-tabA-top.png`, `verify-XS/05a-tabA-continued.png`.
- **Why it matters:** Search is the only way to reach past chats in the panel (XS-06), so this fires on the panel's main navigation path. Power users who keep several tabs lose the working state of the current tab every time they look something up. The original can be recovered only by closing the corrupted tab and searching again, which nobody would guess. First-time users see an "Unavailable" error on a chat they were just using, plus a duplicate tab, and conclude the extension is unreliable.
- **Recommendation:** Changes in `openServerChat` (`routes/sidepanel-chat.tsx` ~1436-1520):
  1. Generate the destination tab id first and send the capture there: call `historySelection.activate(newTabId)` before `loadConversation`.
  2. If that is not feasible, set a `pendingOpenTargetRef`/`isSwitchingTabRef` before the await and clear it in `finally`. While it is set, `saveActiveTabSnapshot` and the label effect (1102-1135) must exit without writing.
  3. If the active tab is an empty "New chat", reuse it as the destination instead of creating another tab.
  4. In the send path, catch `owner_conversation_mismatch`, restore the draft and show an error toast.
  5. Regression test: tab A has 2 messages, call `openServerChat(X)`, then assert that tab A's snapshot and label are unchanged and exactly one new tab exists.

#### XS-07 · Sidepanel right-click 'Delete — cannot be undone' only closes the tab (chat stays on server and in search); 'Rename' only relabels the local tab
**Priority** P0 · **Severity** 3/4 · **Effort** S · **Surfaces** Extension side panel · **Persona** Power user · **Heuristic** Nielsen #2 Match with real world; #5 Error prevention · **Verification** confirmed (live+code); upheld by independent skeptic
- **What happens:**
  - **Menu and dialog.** The tab context menu offers Rename / Pin / Add to folder… / Status / Export / Delete. Delete opens a modal: "Delete conversation — Are you sure you want to delete this conversation? This action cannot be undone."
  - **Result.** After confirming, the tab disappears. `GET /api/v1/chats/{id}` still returns the chat (200, title intact, state in-progress), and sidebar search finds it again immediately. Three reviewers reproduced this with three separate chats.
  - **Cause.** `Sidebar.tsx:621` wires Delete to `onDelete={onCloseTab}`, and `handleCloseTab` (`routes/sidepanel-chat.tsx:1271-1290`) only removes the tab. `handleRename` (`Sidebar.tsx:326-331`) changes the local tab label only.
  - **Inconsistency.** The multi-select bulk delete in the same drawer (`Sidebar.tsx:834-856`) does delete server conversations. The full-page row menu renames on the server and soft-deletes to Trash (`Common/ChatSidebar/ServerChatRow.tsx:122-180`).
  - **Evidence:** `pu-ext/27b-delete-confirm.png`, `verify-XS/18a-delete-confirm.png`.
- **Why it matters:** The product shows an irreversibility warning and then deletes nothing. A user removing a sensitive conversation believes it is gone, but it stays on the server and in search on every surface, which is a trust and privacy failure. Renames also drift silently between the panel and the full page. Because the same object behaves differently depending on where the action starts, power users cannot build a reliable picture of what an action does.
- **Recommendation:**
  1. Give `ConversationContextMenu` a real delete handler. For tabs with a `serverChatId`, reuse the soft-delete from `handleBulkDelete`/`applyBulkDelete`: close the tab, invalidate `['serverChatHistory']`, and show "Moved to Trash · Undo", the same behaviour as `ServerChatRow`.
  2. For local-only tabs, label the item "Close tab" with no confirmation. Add a separate "Delete from this device" that removes the Dexie history.
  3. Use "cannot be undone" only for deletes that really are permanent.
  4. When `serverChatId` is set, make Rename PATCH the server title and update the Dexie title as well as the tab label.

##### Persistence and history (severity ≤ 2)

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XS-05 | The "Save chat to history" tooltip says "Locally + Server", but plain side-panel chats are never written to the server. The label reflects connection state rather than persistence (`Sidepanel/Chat/form.tsx:1165-1192`), and `createChat` runs only on the character/persona path (`hooks/useMessage.tsx:1431-1455`). Verification lowered this from 3: the false claim appears only on hover, and actual loss needs a reinstall, a data wipe or a device switch. | 2 | M | Short term: show the server wording only when `serverChatId` is set. Otherwise show "Saved on this device only" with a "Sync to server" action. Long term (shared with CS-03): when connected and not temporary, create the server chat on the first side-panel send. |
| XS-06 | The drawer lists only open tabs. Closed or cleared chats appear only after a search is typed (`Sidebar.tsx:344-380`). The full page never shows side-panel chats: `LocalChatList` exists but is mounted nowhere, and `showLocalChats = false`. Tabs do accumulate under Today/Yesterday until they are closed or cleared in place, which limits the impact. | 2 | M | With an empty query, show "Open" (current tabs) and then "Recent" (Dexie and server histories, newest first, Local/Server badges, 20 per page). On the full page, mount `LocalChatList` as a "This device" tab, or sync chats to the server as described in XS-05. |

#### 7.1.2 Hidden modes, model and status feedback

#### XS-08 · Ctrl+E silently toggles a hidden 'Chat with current page' mode (labelled as knowledge search in the palette) and hijacks macOS end-of-line
**Priority** P1 · **Severity** 3/4 · **Effort** S · **Surfaces** Extension side panel · **Persona** Power user · **Heuristic** Nielsen #1 Visibility of system status; #4 Consistency · **Verification** confirmed (live+code); upheld by independent skeptic
- **What happens:**
  - **The toggle.** With focus in the composer, Ctrl+E switches the active tab's `chatMode` from `normal` to `rag`. The only visible change on screen is "Draft saved".
  - **Effect on sends.** The next send is built from the page-context RAG prompt; the mock echoed "Use the following pieces of context to answer the question at the end". The mode is stored in the tab snapshot, so it persists.
  - **Indicators.** The only indicator is a checked "Chat with current page" box inside the Pro send split-button menu (`form.tsx:3711-3722`). Casual mode has none. The command palette describes the same shortcut as "Toggle Search & Context — Search your knowledge base and context" (`Common/CommandPalette.tsx:391-405`).
  - **macOS conflict.** The binding (`hooks/keyboard/useShortcutConfig.ts:43-48`) calls `preventDefault` without `ignoreIfEditable`, so on macOS it also suppresses the system Ctrl+E end-of-line in the textarea. In a caret test, Ctrl+A moved to line start but Ctrl+E did nothing.
  - **Evidence:** `pu-ext/19b-after-ctrl-e.png`, `pu-ext/19c-delivery-menu.png`.
- **Why it matters:** A Mac power user presses a text-editing habit key and silently changes what every later message sends, likely including page content sent to the model. In Casual mode there is no way to see or undo this. First-time users who trigger it by accident get unexpected answers with no explanation, and the palette label points them to the wrong concept.
- **Recommendation:**
  1. Add `ignoreIfEditable: true` to `toggleChatMode`, or move it to a chord that is not used for text editing, such as Alt+E or Mod+Shift+E.
  2. In `form.tsx`, show a persistent "Using this page ✕" chip whenever `chatMode === 'rag'`, in both Casual and Pro, and show a toast when the mode is toggled.
  3. Rename the palette entry (`common.json:460`, `toggleKnowledgeSearch`) to "Toggle chat with current page", so the palette, the help sheet and the menu checkbox all use the same term.

##### Model, status and offline feedback (severity ≤ 2)

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XS-02 | The side panel has no default model, so a user who starts there fails their first send with "Please select a model". The picker is an unlabelled brain icon. It files Ollama under CUSTOM and truncates "Custom OpenAI API - m…". In Casual it disappears once a model is chosen (`form.tsx:3312-3315`). Verification lowered this from 3: users who open the full page first inherit its auto-selected model, and Pro shows the model chip. | 2 | M | Reuse `resolveStartupSelectedModel` as `PlaygroundForm.tsx:1528-1546` does. Always show a compact model chip in Casual. When a send is blocked by a missing or unavailable model, or a provider returns an error, open the picker with search focused. List the model name first and the provider second. |
| XS-10 | While waiting for the first token, the primary button reads "QUEUE" and there is no Stop (`form.tsx:2889`; Stop is gated on `streaming` at 3967-3990). Verified with a simulated 6s delay. The abort controller already exists, so only the UI is missing. | 2 | S | Show Stop while the request is sending or streaming, with a "Waiting for &lt;model&gt;…" status line. Keep the primary label as Send, and move "Queue after current" into the Send ▾ menu. |
| XS-11 | The offline empty state offers only "Review settings" (no Retry, and Diagnostics is mentioned but not linked), next to a blue QUEUE button that does nothing beside a read-only textarea. Partly mitigated: the panel retries every 5s and recovered in about 4s, and once a chat has messages, `ConnectionBanner` offers Retry. | 1 | S | In `Sidepanel/Chat/empty.tsx:200-235`, add "Retry now" with a "Retrying automatically…" line and a Diagnostics link. While offline, render the primary button as a disabled "Send". |
| XS-16 | Pro "Model Parameters" shows Temperature 0.7 and Top P 0.9 for unset values (`Sidepanel/Chat/ModelParamsPanel.tsx:281-288, 309-316`). "Current Chat Model Settings" shows the same fields as empty. The request omits them, so the provider default actually applies. | 2 | S | Show "Default" with a muted slider until the user moves it, and add a per-field reset. Better: reuse the settings-modal parameter component with an Inherited/Override badge. |

#### 7.1.3 Layout at side-panel widths (severity ≤ 2)

Chrome's default side panel is about 320-360px wide. Several reports were captured at 420px, so the table states which widths are affected.

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XS-03 | In Casual, the "Chats" drawer computes `position: relative` because Tailwind emits `.relative` after `.fixed` (`Sidebar.tsx:870-877`). The drawer therefore pushes the chat into 132px at 420px width and 72px at 360px, and answers render one letter per line. Esc does not close it. Lowered from 3 because the layout recovers as soon as the drawer closes. | 2 | S | Apply `relative` only in the docked variant. For the overlay, use `fixed inset-y-0 left-0 z-40 w-[min(18rem,85vw)]`. Close on Esc, focus search on open and return focus to the hamburger on close. Add a test that the computed position is fixed. |
| XS-04 | Pro docks a 288px chat list at any width above 400px, and that list cannot be collapsed (`routes/sidepanel-chat.tsx:2200-2201`). Between 401px and 640px this crushes the chat to 113-352px, and the header toggle reports the wrong state. At the default 320-360px Pro uses the overlay, so only users who widen the panel are affected. | 2 | S | Dock only at 640px or wider. Add a persisted `proSidebarCollapsed` flag wired to "Collapse sidebar" and Ctrl+B. Pass the actual visibility to the header toggle. |
| XS-09 | The composer scrolls with the messages (`stickyChatInput: false`, `types/chat-settings.ts:97`). After a reply, Send is below the fold. The "scroll to latest" pill is pinned at a fixed 8rem, overlaps the input, and appears even on the empty state. | 2 | S | Make the input sticky by default in the panel. Position the pill from the measured composer height. Hide it when there are no messages or the view is within 48px of the bottom. Re-pin to the bottom with a ResizeObserver during auto-scroll. |
| XS-14 | At 360px the header title wraps to "tldw / Assistant", and markdown tables clip their last column with no visible scroll affordance. The column can be reached by horizontal scroll, and "View" opens the full table. The pill overlaps the table tools. | 2 | S | When `scrollWidth > clientWidth`, add an edge fade and a "Scroll →" hint. Truncate the title, or show only the logo below 380px. Keep the pill clear of content (see XS-09). |
| XS-15 | The Pro composer takes about 64% of a 400x900 panel: two accordions, a Knowledge Search toggle, a 160px textarea, a FeatureHint card, two icon rows and Send. On the extension full page it takes about 59%. | 2 | M | Collapse ModelParamsPanel into a status chip that opens a popover. Open Knowledge Search as an overlay sheet. Remember when the hint has been dismissed. Cap the Pro textarea at about 96px with auto-grow. Move "Startup templates" under Advanced. |
| XS-18 | The Artifact drawer footer does not wrap. A permanently disabled "Run (N/A)" button sits beyond the 420px viewport (`Sidepanel/Chat/ArtifactsPanel.tsx:389, 440-447`). | 1 | S | Add `flex-wrap`, or below 480px move secondary actions into a "⋯" menu. Remove Run unless the artifact can actually run. |

#### 7.1.4 Information architecture, copy and capture (severity ≤ 2)

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XS-12 | Labels are jargon or point to the wrong place. "Config" opens voice settings. The header "Open dashboard" opens Flashcards (`SidepanelHeaderSimple.tsx:145-146`). The Character sheet leads with "Apply overlay … without changing conversation ownership" and "Start tracked persona chat". The disabled Voice button gives no reason. | 2 | S | Rename to "Voice settings". Show "Flashcards" with a cards icon. Lead with "Chat as a character" and put the overlay and tracked options under an Advanced disclosure. Give disabled Voice a reason and a setup link. Expand tooltip: "Open this chat in a full tab". |
| XS-13 | Quick-saving a web selection to Notes is a dead end. The modal has no tags field (keywords are fixed to `captured`), and the toast has no "Open note" link. The panel has no Notes destination. In Notes, the footer says "Origin: Typed manually" (`components/Notes/hooks/useNotesEditorState.tsx:2356-2363`). | 2 | M | Add a Tags select prefilled with "captured". Add an "Open in Notes" toast action (`options.html#/notes?note=<id>`). Add a Notes entry or a "Recent notes" list to the panel. Show "Origin: Captured from &lt;domain&gt;" for captured notes. |
| XS-17 | A fresh panel opens to Companion Home with several "Setup required" cards, and "Open Chat" sits at y≈3061 in a 900px panel. `routes/sidepanel-home-resolver.tsx:31-43` picks Companion whenever there is no chat to resume, without checking whether personalization is available. | 2 | S | Resolve to Chat when personalization is unavailable or not enabled. Pin a Chat action, or add a Chat/Companion switch. Collapse the setup cards into one notice. Add a setting: "Side panel opens to: Chat / Companion / Last used". |

---

### 7.2 Cross-page workflows & extension parity (XP)

#### 7.2.1 Chat ↔ Notes round trip

#### XP-02 · Save to Notes, Save to Flashcards and feedback are missing on most server-backed chats (incl. the chat opened from a note) because loaded messages lose serverMessageId
**Priority** P0 · **Severity** 3/4 · **Effort** S · **Surfaces** WebUI, Extension options page (also reproduced in the Extension side panel) · **Persona** First-time and power users · **Heuristic** Nielsen #4 Consistency; #3 User control (round trip broken); #7 Flexibility · **Verification** confirmed (live+code); upheld by independent skeptic
- **What happens:**
  - **Symptom.** Opening an ordinary linear server chat from the sidebar renders its messages without timestamps or "Was this helpful?". The message overflow lacks Save to Notes, Save to Flashcards and Pin. "Generate document" still appears, which shows that `serverChatId` is set.
  - **Probe result.** A React-fiber probe shows every message with `serverMessageId` undefined and `createdAt` 0.
  - **Scope.** Of five seeded chats, only "Explain LoRA fine-tuning" works. The skeptic reproduced the failure on a fresh linear chat created through the public API, which rules out seed shape. It also occurs after Notes → "Open conversation" (WebUI and extension options), persists after reload, and happens in the side panel.
  - **Root cause.** When a history selection is captured, `hooks/chat/useServerChatLoader.ts:1133-1158` skips `setMessages(mappedMessages)`. The display then comes from `formatSelectedHistory` (`db/dexie/helpers.ts:377-393`), which builds rows from `local_history` metadata written at `db/dexie/history-selection.ts:620-640`. That metadata never stores the server message id, and `formatSelectedHistory` forces `createdAt: 0`. `Common/Playground/Message.tsx:1014-1021` requires `serverMessageId` before it offers knowledge saving.
  - **Evidence:** `pu-cross/s15-a-overflow-sidebar-path.png`, `pu-cross/s14-a-overflow-after-openconv.png`, `skeptic-XP/s03-mine-linear.png`.
- **Why it matters:** Saving a chat answer to Notes is the product's core knowledge-worker workflow, and today it works only when a chat happens to skip history capture. A user who jumps from a note back to its source chat finds the save option gone with no explanation. Flashcards, Pin, feedback and timestamps disappear with it, and only Save to Notes has a workaround (copy and paste). Together with XP-01, this makes Save to Notes unavailable on most chats.
- **Recommendation:**
  1. In `formatSelectedHistory`, when the capture belongs to a native server conversation, set `serverMessageId = node.id` and take `serverMessageVersion` from the node revision.
  2. Carry `created_at` in `HistorySelectedContentV1`, or look it up from the server rows, so `createdAt` holds the real time.
  3. Alternatively, have `useServerChatLoader` copy `serverMessageId` and `createdAt` from `mappedMessages` into the captured display, matching by id.
  4. Add a unit test that a native capture returns `serverMessageId === id` and `createdAt > 0`.
  5. Add a Playground integration test that a linear server chat opened from the sidebar offers Save to Notes and feedback.

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XP-01 | "Save to Notes" is hidden, not disabled, on answers in unsynced chats. That covers every new WebUI chat in the default "Saved" mode and every new side-panel chat, because new chats never get a `serverChatId` (`Message.tsx:1014-1021`; `usePlaygroundPersistence.tsx:250-256`). The workaround (Copy, then the Notes Dock) loses provenance. Server chats opened directly in the side panel do show the option. | 2 | M | Always offer the item on assistant messages. When there are no server ids, fall back to a direct Notes API create: reuse NoteQuickSaveModal, take the title from the user prompt, add a "from-chat" tag and a provenance footer. If neither path works, show the item disabled with the reason. Fixing CS-03 closes most of the gap. |
| XP-03 | Provenance runs one way and is hard to follow. The toast has no "Open note" action. Every save is titled "Snippet: &lt;chat title&gt;" with no tags (`tldw_Server_API/app/api/v1/endpoints/chat.py:6927-6929`). The backlink is a blue span that does nothing when clicked and shows a raw message UUID. "Open conversation" does not scroll to or highlight the message (`NotesManagerPage.tsx:1940-2011`). Lowered from 3: the save works and the note can be found. | 2 | M | Return `note_id` and toast "Saved to Notes · Open note", plus a "Saved" chip on the message. Title the note from the user message and default to a "from-chat" keyword. Make the backlink a real button. Scroll to and highlight the message via `backlinkMessageId`. Optional: "Notes from this chat (n)" in the chat header. |
| XP-04 | There is no "Chat about this note" action. The only route is the composer's Search & Context panel → Notes chip. Multi-word notes search fails: "reflog dropped commits" returns 0 hits, "reflog" returns 1. There is no `/note` command, and the Notes Dock's "Open Notes page" drops the active note id. Lowered from 3: a workaround exists. | 2 | M | Add "Chat about this note" to the note's More menu and the bulk bar; it opens chat with the note attached as a removable context chip. Add a `/note` slash command and a Notes picker in the Attach menu. Add "Insert into chat" to the Dock. Treat multi-word notes search as AND terms rather than a phrase. |

#### 7.2.2 Side panel ↔ full page continuity

#### XP-08 · Sidepanel tabs never refresh from the server; continuing a chat after working on it in the full page silently forks it and hides the other branch on both surfaces
**Priority** P0 · **Severity** 3/4 · **Effort** L · **Surfaces** Extension side panel, Extension options page (the fork is also invisible in the WebUI) · **Persona** Power user · **Heuristic** Nielsen #1 Visibility of system status; #5 Error prevention · **Verification** confirmed (live+code); upheld by independent skeptic
- **What happens:**
  - **Repro.** Open a server chat in the side panel and choose "Expand in full page". Send a turn in the full page, then return to the still-open panel. The panel still shows the old transcript, ending before the new turn.
  - **The fork.** Sending from the panel creates a second child of the same parent message. Server data for the chat (28 messages) shows two sibling user branches under parent `a286f2a3`, and no surface warns about it.
  - **What renders.** Re-expanding the chat, or opening it in a fresh WebUI profile, shows only the side-panel branch (26 messages). The full-page turn and its answer are missing, and there is no branch pager.
  - **Cause.**
    - `applySnapshot` (`routes/sidepanel-chat.tsx:874-880`) restores the `chrome.storage` snapshot and the pinned `historySelectionReference` without checking the server version or latest message.
    - The only `visibilitychange` handler (1553-1557) saves state when the panel is hidden and never refreshes it when the panel is shown again.
    - `formatSelectedHistory` (`db/dexie/helpers.ts:403-414`) builds variants only for alternative assistant replies, so sibling user branches can never be navigated.
  - **Evidence:** `pu-ext/21c-sidepanel-after-fullpage.png`, `pu-ext/23a-full-fork.png`, `skeptic-XP/s07-fork-fresh.png`.
- **Why it matters:** The hand-off buttons actively invite users to move between panel and full page, and doing so creates turns that exist on the server but appear on no surface. To the user this is data loss without warning or recovery. The workflow is narrow (hand off, then continue in the panel), but it is exactly the one power users adopt. A real side panel stays open beside the full-page tab, so the panel is out of date by default.
- **Recommendation:**
  1. Treat a side-panel tab with a `serverChatId` as a view of the server chat.
  2. On `visibilitychange`/focus, on tab switch and before every send, fetch the chat's version or last message id and compare it with the snapshot. `GET /api/v1/chats/{id}` already returns `version`.
  3. If the server copy is newer, show an inline banner, "Updated in another window — Reload (n new) / Continue here as a branch", and make the server's latest message the default parent for the next send.
  4. In `formatSelectedHistory`, group sibling user messages so the full page shows a branch pager instead of silently picking one path.

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XP-09 | Each hand-off keeps only half the context. "Expand in full page" carries the conversation but drops the draft. "Continue in WebUI" carries the draft but no conversation, so the draft lands in whichever chat the full page last restored (`Playground.tsx:1868-1877`). It also always opens `options.html`, not the WebUI. The originally reported "Focus mode with no navigation" was an artifact of the 420px test viewport. | 2 | M | Merge both into one "Open in full page" action. Add `historyId`, `serverChatId` and the selection reference to the hand-off package, and include the draft when expanding. Check for a hand-off before restoring the last session. Offer the configured WebUI as a second target. |
| XP-12 | The side panel and the full page feel like two products. Placeholders, control sets and assistant menus all differ; for example, the full page shows "Continue as user", "Impersonate user" and "Force narrate" on plain, non-character chats. The difference in model auto-select may be partly environmental. | 2 | L | Use one placeholder string key for both surfaces. Show the steering items only when a character or persona is active. Share the default-model resolver. For first-run users, collapse MCP, OpenUI and the rails behind Advanced. |
| XP-18 | Pins are stored separately on each surface: note pins in `tldw:notesPinnedIds`, full-page chat pins in `tldw:server-chat-pins`, and side-panel tab pins in the tab store. The WebUI, options page and panel therefore disagree. Casual/Pro is shared between the panel and the options page only. Message pins, by contrast, are stored on the server. | 2 | M | Store chat and note pins on the server (as a flag or a reserved keyword) and keep a local cache. Map a side-panel tab pin to the server pin when the tab has a `serverChatId`. Label the Casual/Pro toggle with its scope. |

#### 7.2.3 Navigation, addressability and global search (severity ≤ 2)

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XP-05 | The selected note never reaches the URL. `routes/option-notes.tsx` only reads `source_ref_id`. Back leaves /notes, Forward returns to an empty editor, and returning to Notes does not reopen the last note. Chats have no shareable URL: `useSelectServerChat` navigates to a bare `/chat`. The skeptic found that per-tab chat restore works, so two chats can be open side by side. | 2 | M | Use `/notes?note=<id>` and `/chat?chat=<id>`. Push a history entry on explicit selection and replace it on restore or autosave. On mount, read the URL first, then fall back to the last opened note. Set a per-note page title. |
| XP-06 | The header's "Search ⌘K" is only a command palette. Searches for "asyncio" and "LoRA" return "No results found" even though matching chats and notes exist. `useOmniSearchDeps`, which can open notes and chats, is dead code. Capturing a note from the keyboard takes about 15 keystrokes, partly because Enter in the title field does nothing. Lowered from 3: search works in the chat sidebar and in Notes. | 2 | M | Add Notes and Chats result sections (debounced, top 5 each, recent items when the query is empty) and a "Create note '&lt;query&gt;'" action, reusing `useOmniSearchDeps`. Make Enter in the note title move focus to the body. |
| XP-11 | Global navigation is chat-centric. A "Chats" sidebar with a New Chat "+" appears on every route, including /notes, beside Notes' own "+". Notes and the Notes Dock use the same icon, and the Dock's list is called "Archive". /chat shows two theme toggles. On mobile the drawers are nested (prior evidence only). | 2 | M | Split the sidebar into an app rail with a context-aware "New…" button and a chat list shown only on /chat. Give the Dock a distinct "Quick note" icon and rename "Archive" to "Notes". Keep one theme toggle and one mobile drawer header. |
| XP-19 | On the extension options page the title is always "tldw Assistant — Options", on #/notes and #/chat alike, and it keeps the last chat's title after returning to Notes (`entries/shared/options-app.tsx:30-34`). | 1 | S | On every route change, set `document.title` from a route→title map shared with the WebUI ("Notes \| tldw"), and let pages refine it. |

#### 7.2.4 Keyboard layer (severity ≤ 2; see also XS-08)

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XP-07 | The shortcut sheets contradict actual behaviour. "?" opens the page navigator, because `Layout.tsx:276-301` handles it before Notes can. Ctrl+B toggles the global sidebar mid-typing instead of making text bold; Notes has no Mod+B/I handler. Ctrl+K is documented as "focus notes search" but opens the palette. The WebUI sheet lists Ctrl+E and Alt+W, which exist only in the side panel. The navigator shows ⌘1-9 while the header sheet shows Alt+1-7. Lowered from 3: toolbar buttons are a workaround. | 2 | M | Short term: set `ignoreIfEditable` on `toggleSidebar`, add Mod+B/I to the Notes textarea, let a route register its own "?" help, and remove or wire the undocumented keys. Medium term: one scoped shortcut registry that generates the sheets, plus a test that every documented shortcut has a handler. |
| XP-10 | The Alt+1…0 area shortcuts never fire from /chat. The composer is autofocused and `ignoreIfEditable` skips the shortcut (`hooks/keyboard/useKeyboardShortcuts.ts:63-73, 150`). Matching on `event.key` probably also breaks Option+digit on macOS (plausible; a headless browser cannot reproduce Mac keyboard layouts). | 2 | S | Match on `event.code` (`Digit5`, `KeyN`) and fall back to `event.key`. Allow Alt/Meta+digit chords inside editable fields. Show ⌥ on macOS. Add a unit test with `{key:'∞', code:'Digit5', altKey:true}`. |

#### 7.2.5 Visual system, copy, consistency and performance (severity ≤ 2)

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| XP-13 | Vocabulary drifts across the app: chat/conversation/session/thread; Temporary/Temp/Private; tags vs keywords. "Saved" is a persistence mode on /chat and is shown on chats with no server record, but on /notes it means a completed save. Date formats are mixed, and some copy is written for developers ("Version pending", "0 + ~0 = 0 tokens"). | 2 | M | Write a short glossary and apply it through the i18n keys. Rename the mode to "Save chats"/"Temporary", and use "Saved" only for a confirmed save ("Saved to server" vs "Saved on this device"). Use one relative-date helper. Add an i18n lint for banned synonyms. |
| XP-14 | Raw IDs leak into the UI. The Notes header shows "· msg &lt;uuid&gt;" and wraps to 4 lines. List rows show the raw conversation UUID when the title is not yet resolved. Auto-titles look like "Helpful AI Assistant (20261003_033823)" (`tldw_Server_API/app/core/Chat/chat_helpers.py:305-307`). The Buddy picker shows licence captions. | 2 | S | Move ids into a tooltip or a copy action. Show "Linked chat" while the title resolves. Title new conversations from the first user message. Hide licence captions behind an info icon. |
| XP-15 | All `type="secondary"` helper text renders as 14px text in the full body colour. A rule at `assets/tailwind-shared.css:419` (specificity 0,4,0, `!important`) overrides `.ant-typography-secondary` at :477-478, and the antd `fontSize` token overrides `text-[11px]`. As a result, hints are louder than section headings; there are 712 such usages app-wide. | 2 | S | Add `:not(.ant-typography-secondary)` to the rule at :419. Add a small `MetaText` component that sets the size. Add visual snapshot tests in both themes. |
| XP-17 | Destructive actions follow different safety patterns. Single note delete has a confirm plus a 10s Undo. Bulk delete has a confirm only, runs sequentially, and only counts failures (`NotesManagerPage.tsx:1335-1383`). Message delete uses `Modal.confirm` with no undo (`Message.tsx:1380`). Toasts appear top-centre on Notes and top-right on Chat. | 2 | M | Make bulk delete work like single delete: one "Moved N notes to Trash · Undo", progress when N > 5, and a list of failed titles with Retry. Make message delete undoable if the API allows. Route all toasts through one helper at one position. |
| XP-20 | There are two primary blues: antd derives #517bdc from the seed colour, while Tailwind uses `--color-primary` #5c8dff. A teal accent competes with them, and green marks both a mode ("Saved") and a status ("Healthy"). | 1 | S | Pin the antd primary tokens to `--color-primary` after the algorithm runs. Reserve the accent for focus rings and green for statuses. |
| XP-16 | `IndependentBuddyHost` polls `/buddies` and `/buddies/attachment` every 5s on every page, with no visibility check (`Common/PersonaBuddy/IndependentBuddyHost.tsx:224-225`). `persona/profiles` is fetched 3x when /chat loads; the other 2x duplicates are React StrictMode artifacts of the dev build. The polling was reported in an earlier review. | 2 | S | Skip polling while the page is hidden, refetch on `visibilitychange`, and back off to 30-60s (or use SSE). Use shared react-query keys with `staleTime` of 60s or more. Measure duplicates on a production build. |

---

### 7.3 Accessibility (AX)

#### 7.3.1 Keyboard operation and focus management

#### AX-03 · Keyboard-only use of Notes is impractical: 35 Tabs to the list, 3 tab stops per row, and no way from the list into the editor or back
**Priority** P1 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, Extension options page · **Persona** Power user · **Heuristic** WCAG 2.2 SC 2.1.1 Keyboard, SC 2.4.3 Focus Order; Nielsen #7 · **Verification** confirmed (live+code); upheld by independent skeptic
- **What happens:**
  - **Reaching the list.** At 1440x900 it takes 54 Tabs from a fresh load to the first note row (35 from the header in the original run). "Skip to notes list" exists, but its target is still 25 Tabs from the first row.
  - **Tab stops per row.** Each row has three tab stops (Select checkbox, Pin, Open). That gives about 330 focusables for the 100 rows that render. The rows render 100 at a time because the backend ignores the "20 / page" control: the frontend sends `page`/`results_per_page`, but `tldw_Server_API/app/api/v1/endpoints/notes.py:2311` reads `limit` with a default of 100. That is a separate bug.
  - **Moving through the list.** ArrowUp/Down move between rows; Home/End do not.
  - **Opening a note.** Enter opens the note but leaves focus on the row. 150-200 further Tabs never reach the editor body.
  - **Workaround.** About 30 Shift+Tabs back to "Skip to editor", then Enter, then 26 Tabs through the toolbar to the "Note content" textarea: roughly 56-59 keystrokes per note switch.
  - **Search field.** Esc neither clears the search field nor moves focus out of it.
  - **Cause.** `components/Notes/NotesListPanel.tsx:491-557` renders three tabbable controls per row and handles only ArrowUp/Down. `NotesManagerPage.tsx:2212-2279` has no binding to focus the editor, move to the next or previous note, or return to the list.
  - **Evidence:** `pu-notes/03a-opened-by-keyboard.png`, `skeptic-AX/k01-after-enter.png`.
- **Why it matters:** Sighted keyboard-only users and switch users cannot browse and edit notes without a mouse, because every note switch costs about 60 keystrokes. Screen-reader users can jump by landmark, but still face hundreds of row stops. Power users expect a mail-client pattern (arrow to a row, press Enter, start typing) and find the editor out of reach.
- **Recommendation:**
  1. Make the results list a single Tab stop: `role=listbox` on `[data-notes-list]` with `role=option` rows and `aria-activedescendant`, or a roving tabindex on the Open buttons.
  2. Set `tabIndex=-1` on the per-row checkbox and pin. Expose them through keys on the active row (Space/x to select, p/s to pin) and a Shift+F10 row menu.
  3. Add Home/End/PageUp/PageDown to the existing handler at `NotesListPanel.tsx:544`.
  4. Make Enter open the note and move focus to the editor body once `loadDetail` resolves; expose an `editorRef` from NotesEditorPane.
  5. In NotesManagerPage (2212+), add Esc from the editor to return focus to the active row, and Alt+ArrowUp/Down for previous/next note.
  6. In search, the first Esc clears the query and a second Esc moves focus to the list.
  7. Move date, tags and preview text into `aria-describedby`, so they are not part of each row's accessible name.
  8. Add a Playwright keyboard test: "/" → type → Esc → ArrowDown → Enter → assert that the textarea is focused.

#### AX-05 · Notes off-canvas list (mobile or desktop at 200% zoom) keeps ~285 invisible controls in the Tab order when closed; when open it takes no focus and Esc doesn't close it
**Priority** P1 · **Severity** 3/4 · **Effort** S · **Surfaces** WebUI, Extension options page · **Persona** First-time and power users · **Heuristic** WCAG 2.2 SC 2.4.3 Focus Order, SC 2.4.7 Focus Visible, SC 2.4.11 Focus Not Obscured, SC 2.1.1 Keyboard; Nielsen #3 · **Verification** confirmed (live+code); upheld by independent skeptic
- **What happens:**
  - **Closed drawer.** At 640x400 the closed drawer `#notes-list-region` is moved -420px off-screen but still holds about 330 focusables, about 290 of them off-screen. It is not `inert`, not `aria-hidden` and not visibility-hidden. 58 of the first 60 Tabs landed on invisible controls (the first was "Sort notes" at x=-170). At 390x844 the figure was 55 of 60.
  - **Opening it.** "Browse notes" has no `aria-expanded`/`aria-controls`. Enter opens the drawer, but focus stays on the trigger behind the overlay. The next Tabs go to editor controls behind the backdrop.
  - **Closing it.** Esc does not close the drawer; the backdrop button is the only close control (`NotesManagerPage.tsx:2370-2372`). Choosing a note with Enter closes the drawer but leaves focus on the row, which is now off-screen.
  - **Desktop at 100%.** After "Collapse sidebar" the aside becomes `w-0 overflow-hidden` (`components/Notes/NotesSidebar.tsx:383-385`) but keeps about 325 focusables. 34 of 40 Shift+Tabs from "Note title" landed inside it.
  - **Correction to the original report.** "Skip to notes list" does open the drawer (`NotesManagerPage.tsx:2343-2360`).
  - **Evidence:** `lens-a11y/dr-open.png`, `verify-AX/09-640-open.png`.
- **Why it matters:** Keyboard users at zoom, mobile screen-reader users swiping through the page, and desktop users who collapse the list all spend most of their keystrokes on controls they cannot see. None of them can open, enter or dismiss the list predictably. The only workaround is Shift+Tab backwards into the drawer, which nobody would discover.
- **Recommendation:**
  1. In `NotesSidebar.tsx:370-395`, set `inert` (and `aria-hidden`) on the `<aside>` whenever it is hidden: `isMobileViewport ? !mobileSidebarOpen : sidebarCollapsed`.
  2. When it is open on mobile, render it as a modal dialog: `role=dialog`, `aria-modal`, a visible "Notes" heading as its label, a visible Close button and a focus trap. antd `Drawer placement="left"` provides all of these.
  3. On open, move focus to the search input. On Esc or Close, return focus to "Browse notes", or to the editor after a note is chosen.
  4. Add `aria-expanded` and `aria-controls="notes-list-region"` to "Browse notes" (`NotesEditorPane.tsx:545-557`) and to the desktop sidebar toggle.
  5. Add a test: at 640x400, press Tab 30 times and assert that no focused element has a bounding rect with `right <= 0`.

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| AX-01 | The composer model picker cannot be operated from the keyboard. Enter opens it, but focus stays on the trigger. ArrowDown and typing do nothing, and Tab closes it (`Option/Playground/ChatModelSelectorDropdown.tsx:69-203`, with `stopPropagation` at 96/127; `aria-haspopup=listbox` is set on what is actually a menu). The sort select has no label. Lowered from 3 because a workaround exists: ⌘K → "Switch Model" opens a settings drawer whose combobox works. The documented Mod+E has no handler. | 2 | M | Use a combobox popover, or reuse the side panel's `Common/ModelSelect`: on open, focus search; render a listbox of options with `aria-activedescendant`; Esc returns focus to the chip. Label the sort select. Wire "Switch Model" and Mod+E to the picker. Add a Playwright test. |
| AX-02 | Message "More actions" (New Branch, Continue, Summarize, Delete, Save to Notes) is a click Popover containing plain buttons. Focus stays on the trigger and arrow keys do nothing. The items are reachable only about 27 Tabs later, at the end of `<body>`, and Esc drops focus to `<body>` (`Common/Playground/MessageActionsBar.tsx:600-617`). The Info popover opens on hover only (581-595). Two different buttons are both named "More actions". | 2 | S | Use an antd `Dropdown` menu, as NotesEditorHeader does, which provides menu semantics and returns focus to the trigger. Open Info on hover, focus and click. Rename the "•••" chip to "Show message actions". |
| AX-09 | Focus is lost to `<body>` in three places: after closing the Notes "?" shortcuts modal (the overlay host unmounts, `NotesManagerPage.tsx:2666`; Tab also escapes the modal once per cycle), after Esc in chat thread search (`Playground.tsx:2622-2625`), and after deleting a note. | 2 | S | Keep the overlay host mounted once it has loaded, or restore the previously focused element in `afterClose`. Refocus the previous element after thread search, as the `shortcutsTriggerRef` pattern does. For delete, see AX-06. |
| AX-18 | The list scroller in the global "Keyboard shortcuts" modal (`Common/KeyboardShortcutsModal.tsx:235`) cannot receive focus, and the custom focus trap cycles Close → Close. About 160px of shortcuts cannot be reached by keyboard even at 1440x900. Raised from 1 on verification. | 2 | S | Make the scroller a focusable, labelled region with a focus ring and include it in the trap. Alternatively, make the dialog taller with a two-column layout so nothing scrolls. |

#### 7.3.2 Reflow, zoom and target size

#### AX-04 · At 200-400% zoom the chat composer fills the viewport: transcript is 0px at 320x256 CSS px and the 'Exit focus' pill covers the input
**Priority** P1 · **Severity** 3/4 · **Effort** M · **Surfaces** WebUI, Extension options page · **Persona** First-time and power users · **Heuristic** WCAG 2.2 SC 1.4.10 Reflow. The original SC 2.4.11 claim was downgraded: the textarea is only partly covered, which fails only SC 2.4.12 (AAA). · **Verification** confirmed (live+code); upheld by independent skeptic
- **What happens:**
  - **Measurements.** Measured with CSS-pixel viewports equivalent to browser zoom:

    | Viewport | Equivalent | Transcript (`role=log`) | Composer |
    |---|---|---|---|
    | 640x400 | 200% zoom on 1280x800 | 115px | 259px |
    | 640x360 | 200% zoom on a laptop, after browser chrome | 71px | not measured |
    | 320x256 and 360x225 | 400% zoom | 0px | 401-439px, in a 225-256px viewport |

    The page has no document scroll, so at 400% the conversation cannot be read at all.
  - **Cause.** Below 768px, `Option/Playground/Playground.tsx:473-475` defaults to focus layout. The composer dock has no max-height and is `shrink-0` (4403-4407), while the transcript is `flex-1 min-h-0` (4368) and is the element that collapses.
  - **Workarounds fail.** "Hide composer options" leaves 15px at 360x225, and "Exit focus" leaves 0px.
  - **Zooming mid-session.** A user who zooms in during a session stays in the cockpit layout, because the layout mode is not re-derived on resize. In that state the transcript is already 0px at 640x400, because the rail tiles ("Restore … sidechannel") fill the space.
  - **Exit focus pill.** At 320x256 the "Exit focus" pill covers the right part of the textarea.
  - **Evidence:** `lens-a11y/z2-chat-640.png`, `verify-AX/05-reflow-320x256.png`, `skeptic-AX/k05-def-640x400.png`.
- **Why it matters:** For low-vision users who zoom, the task is blocked, not merely cramped: they cannot read the answer they asked for. Zoom users are a minority, but this is a hard SC 1.4.10 failure on the product's main page, and at 200% it is triggered simply by zooming in after the page has loaded.
- **Recommendation:**
  1. Cap the composer dock (PlaygroundForm / PlaygroundCockpitShell) at about `40dvh` with internal scrolling.
  2. When the viewport height is under 500px, give the `role=log` container `min-h-[45dvh]`. Under the same media query, collapse the Modes, MCP, Search & Context, Prompt and usage chips into one "Composer options" button that opens a sheet, and hide the rail tiles.
  3. Render "Exit focus" as a header button instead of a pill floating over the textarea.
  4. Re-derive the layout mode when the window is resized.
  5. Add Playwright reflow tests at 320x256 and 640x400 that assert `[role=log]` is at least 100px tall and that `elementFromPoint` at the textarea centre returns the textarea.

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| AX-16 | Some targets are too small. The "Tags", "Organize" and "Connections" help icons are 11x11 (axe 2.5.8, serious). Mobile Notes rows pack a 16px checkbox, a 16x24 pin and the open button within about 8px of each other. The message chip is 37x22, and the side-panel switch is 28x16. The row controls pass AA through the spacing exception; the 44px touch figures are AAA or best practice. | 2 | S | Give help icons a hit area of at least 24x24, or replace them with one "About this section" button. Make the row checkbox and pin 24px (44px on coarse pointers). Change `sm:min-h-0` to `sm:min-h-6` on the chip. Use the default antd Switch size. Add axe `target-size` to the responsive smoke test. |

#### 7.3.3 Screen-reader semantics and announcements (severity ≤ 2)

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| AX-06 | antd `message` toasts have no live region, so delete/Undo, copy, "Saved to Notes" and most error toasts are silent. After a keyboard delete, focus lands on `<body>`, and Undo (portaled to the end of the DOM) expires after 10s. Lowered from 3: Notes save status has its own live region, "Chat saving is incomplete" is announced as `role=alert`, and deleted notes can be restored from Trash. | 2 | M | Add an app-level announcer with a polite and an assertive region, and call it from `hooks/useAntdMessage.ts`. After a delete, move focus to the next row. Show Undo as a persistent inline banner and announce "Note deleted. Undo available." |
| AX-10 | Markdown tables render without a `<table>` element. `Common/TableBlock.tsx:253-257` puts `thead`/`tbody` directly inside a div, so screen readers see orphan rows and cells, and the `[&_table]` styles never apply. This affects chat answers and the Notes preview. | 2 | S | Wrap the children in a real `<table>` inside a focusable, labelled scroll region. Add an RTL test that `getByRole('table')` finds the table and its column headers. |
| AX-11 | The transcript live region covers the whole scroll container (`Playground.tsx:4361-4367`). One send added about 100 nodes to it, including history controls, timestamps, toolbars and a markdown re-render after streaming. `aria-busy` is set only on the message article. The side panel nests two `role=log` regions (`Sidepanel/Chat/body.tsx:260-263`). The DOM mechanism is confirmed; actual NVDA/VoiceOver speech was not tested. | 2 | M | Put `role=log` on an inner wrapper that contains only messages. Set `aria-busy` on the log while a reply streams. Hide per-message chrome from the live region, and announce completion once. Remove the inner side-panel log. Add a screen-reader smoke step to the release checklist. |
| AX-12 | The slash-command menu and the [[wikilink]] suggestions are not exposed as comboboxes. The slash menu uses `role=option` buttons with no listbox parent. The textarea has no `aria-controls` or `aria-activedescendant`. The wikilink listbox contains buttons with `aria-selected` (axe critical). Sighted keyboard use works. | 2 | M | Add a shared `useComposerCombobox` hook that gives the textarea `role=combobox` with `aria-expanded`, `aria-controls` and `aria-activedescendant`. Render a listbox of `role=option` divs. Announce the number of matches when the list opens. |
| AX-13 | `#textarea-message` always carries `aria-expanded="true"`, tied to the composer's collapse state (`PlaygroundForm.tsx:5859` → `ComposerTextarea.tsx:139`; axe critical). Verification refuted the "unnamed Stop button" and "failures announced as status" claims. | 1 | S | Remove `aria-expanded` from the textarea and put it on the "Hide/Show composer options" toggle. |
| AX-14 | Landmarks and headings are malformed. /chat has a `<main>` nested inside another `<main>` (`Layouts/Layout.tsx:398` and `PlaygroundCockpitShell.tsx:425`). The chat history sidebar is not inside any landmark. /notes has no h1, and its only heading is the h5 note title. The side panel has no headings at all. | 2 | S | Change the cockpit `<main>` to a `<section>`. Wrap ChatSidebar in `<nav aria-label="Chat history">`. Add an sr-only h1 on Notes and in the side panel. Make Notes section labels h2 elements that contain their toggle buttons, and make the note title an h2. |
| AX-15 | The selected view or mode (Notes/Trash, List/Timeline/…, Markdown/WYSIWYG) is shown only by button fill colour. There is no `aria-pressed` or `aria-checked`, and by mechanism the state is lost in forced-colours mode. Two buttons are both named "Expand sidebar", and the sidebar toggles lack `aria-expanded`/`aria-controls`. | 2 | S | Use antd `Segmented` or add `aria-pressed`. Add a forced-colours outline for pressed and checked states. Give the toggles `aria-expanded`, `aria-controls` and distinct names. |
| AX-17 | Some accessible names do not contain the visible label. The "Context rail" and "Runtime rail" tabs are named "Restore … sidechannel" (`PlaygroundCockpitShell.tsx:181-188`). "Save & new" is named "Save and start another note" (`Notes/NotesEditorHeader.tsx:530-546`). | 1 | S | Use the visible label as the accessible name and put the action wording in the tooltip or description. Remove "sidechannel". |

#### 7.3.4 Contrast, focus visibility and motion (severity ≤ 2)

| ID | Issue | Sev | Effort | Recommendation |
|---|---|---|---|---|
| AX-07 | Text contrast fails at the token level in both themes. Measured ratios: <br>• Dark antd primary buttons, white on #517bdc: 4.02:1 <br>• Tailwind `bg-primary` buttons ("Start chatting", side-panel SEND), white on #5c8dff: 3.13:1 <br>• Light success badges at 9-10px: 2.94:1 <br>• `--color-text-subtle`: 4.43:1 <br>• Placeholders: 1.64:1 light, 2.09:1 dark <br>• Prism number tokens: 2.43:1 | 2 | M | Darken the subtle and success tokens, and set badge text to at least 11px. Set an explicit dark `colorPrimary` (about #3d6be0) and use the same value for Tailwind `bg-primary`. Set antd placeholder, description and tertiary text colours to at least 4.5:1. Switch to an accessible Prism theme. Add an axe colour-contrast check to CI for /notes, /chat and the side panel in both themes. |
| AX-08 | Focus outlines on antd buttons contrast with their background at only about 1.35-1.57:1. The side-panel "Open Persona Garden" and "Open settings" links, and the TableBlock Copy/Download buttons, show no focus indicator at all. A strong `--color-focus` token exists but is not mapped into antd. | 2 | S | Map `controlOutline` and `lineWidthFocus` to `--color-focus`. Add a global `:focus-visible` 2px outline. Darken light-theme `--color-focus` to at least 3:1. Add focus rings to the header links and to TableBlock. |
| AX-19 | Chat ignores `prefers-reduced-motion`: there are shake, bounce and pulse animations and smooth scrolling, and Playground has no `motion-reduce` variants. The app's animation setting does not default from the OS preference (`themes/apply-theme.ts:135`). This falls under SC 2.3.3 (AAA). | 1 | S | Add a global reduced-motion block in `tailwind-shared.css`. Choose the scroll behaviour based on the media query. Default animation speed to none when the query matches. |

#### 7.3.5 What already works

These patterns are already correct and are worth copying elsewhere:
- **Notes:** skip links ("Skip to notes list", "Skip to editor"); a polite/assertive live region for save status (`NotesEditorPane.tsx:810-826`); `aria-pressed` on Edit/Split/Preview; keyboard support in the header More menu (antd Dropdown); "/" to focus search; `motion-reduce` on the Notes drawer.
- **Chat:** an sr-only h1 "Chat"; `role=alert` on the send-error banner.
- **Side panel:** the side-panel `ModelSelect` focuses its search on open and returns focus on close.
- **Focus return:** the global shortcuts modal, the keyword picker, the role-play drawer and the chat Shortcuts panel all return focus to their trigger.
- **Reflow:** no horizontal page scroll was observed at 320px.

#### 7.3.6 WCAG 2.2 success-criteria summary

**Scope.** This table covers the success criteria exercised in this review: all relevant A/AA criteria, plus AAA where an issue cites one. It is not a full conformance audit.

**Status values:**
- **Fails:** a confirmed failure on at least one surface.
- **At risk:** the mechanism is confirmed, but the failure is partial, borderline, or not verified with assistive technology.
- **Not failed:** the criterion was examined and the related issue is not a violation of it.

| SC | Status | Issue IDs |
|---|---|---|
| 1.3.1 Info and Relationships (A) | Fails | AX-10, AX-12, AX-14 |
| 1.4.1 Use of Color (A) | At risk (selection shown by fill only; lost in forced colours by mechanism) | AX-15 |
| 1.4.3 Contrast (Minimum) (AA) | Fails | AX-07 |
| 1.4.10 Reflow (AA) | Fails | AX-04; side panel: XS-03, XS-04, XS-14, XS-18 |
| 1.4.11 Non-text Contrast (AA) | Fails | AX-08 |
| 1.4.13 Content on Hover or Focus (AA) | At risk (Info popover is hover-only; confirmed in code only) | AX-02 |
| 2.1.1 Keyboard (A) | Fails | AX-01, AX-02, AX-03, AX-05, AX-18 |
| 2.1.2 No Keyboard Trap (A) | Not failed (AX-09 is focus escaping a modal, not a trap) | AX-09 |
| 2.1.4 Character Key Shortcuts (A) | At risk (the hijacks are modifier chords, which are outside this criterion; tracked as usability issues) | XP-07, XS-08 |
| 2.2.1 Timing Adjustable (A) | At risk (Undo expires after 10s; Trash is an alternative) | AX-06 |
| 2.2.2 Pause, Stop, Hide (A) | At risk (not tested live) | AX-19 |
| 2.3.3 Animation from Interactions (AAA) | Fails | AX-19 |
| 2.4.1 Bypass Blocks (A) | At risk (Notes has skip links; landmarks on /chat are malformed) | AX-14 |
| 2.4.2 Page Titled (A) | Fails on the extension options page | XP-19 |
| 2.4.3 Focus Order (A) | Fails | AX-03, AX-05, AX-06, AX-09 |
| 2.4.6 Headings and Labels (AA) | At risk | AX-14 |
| 2.4.7 Focus Visible (AA) | Fails | AX-05, AX-08 |
| 2.4.11 Focus Not Obscured (Minimum) (AA) | At risk (partial cover only; focus moves behind the drawer overlay) | AX-04, AX-05 |
| 2.5.3 Label in Name (A) | Fails | AX-17 |
| 2.5.8 Target Size (Minimum) (AA) | Fails (11x11 help icons; other small targets pass through the spacing exception) | AX-16 |
| 4.1.2 Name, Role, Value (A) | Fails | AX-01, AX-02, AX-12, AX-13, AX-15 |
| 4.1.3 Status Messages (AA) | Fails (the AX-11 mechanism is confirmed in the DOM; actual speech is unverified) | AX-06, AX-11 |

---

## 8. Improvement opportunities (beyond defects)

The defect sections say what is broken. This section says what /notes and /chat could become once those defects are fixed. The 49 ideas from the ten reviewers (IDEA-01 to IDEA-49) are grouped into three horizons:

- **8.1 Quick wins (days).** Small, self-contained, and mostly reuse components that already exist.
- **8.2 Next (weeks).** Feature-sized work, grouped into four themes.
- **8.3 Strategic bets (larger redesigns).** Architectural or cross-surface changes that need a design spike first.

Every item has the same fields: persona, page, related issue IDs, *Why* (the rationale) and *Sketch* (the concrete interaction). "First-time" means the newcomer persona and "power" means the persona with about 160 notes and 49 chats. Where an idea only works once a P0 or P1 defect is fixed, a **Depends on** line says so.

### 8.0 North star: one knowledge workflow (capture → organize → converse → distill)

/notes is where knowledge lives and /chat is where the user works with it. They should feel like two views of one loop, not two products that happen to share a sidebar.

- **Capture.** Anything worth keeping becomes a note in one gesture, without leaving the current surface, and lands in Inbox with provenance and sensible defaults. That covers a web selection from the extension, a passing thought in Notes Dock, and a chat answer or part of one. Provenance means a source URL or "From chat: &lt;title&gt; · message"; defaults means a title taken from the question and a from-chat tag.
- **Organize.** Notes can be found and arranged at any scale:
  - the real total is shown, not a 100-row window;
  - tags mean the same thing on every page;
  - the folders the server already stores are visible;
  - smart views remember a query;
  - [[links]] resolve in both directions.
- **Converse.** Any note, tag, folder or selection can be brought into a chat as visible context, either through "Chat about this note" or the context strip. The user always sees exactly what the model will read. Each chat is a durable, addressable server object with the same history in the webui, the extension options page and the side panel.
- **Distill.** A good answer flows back into notes, as a new note or appended to an existing one, with a backlink to the exact message. Notes keep version history, so AI-assisted edits and bulk changes can be undone. The distilled note then becomes context for the next conversation or source material for a study pack.

Five shared rules hold the loop together:

- every object has a URL;
- one quick switcher finds any object;
- one status chip says where it is saved;
- one vocabulary names it;
- one conversation model backs every surface.

| Stage | What the user is trying to do | What blocks it today | Ideas that enable it |
|---|---|---|---|
| Capture | Save a page selection, a thought, or an answer | XP-01, XP-02, XS-13, NS-01 | IDEA-02, IDEA-32, IDEA-35, IDEA-20 |
| Organize | Find, tag, file and link at scale | NL-01, NL-03, NE-02, NL-10, CS-02 | IDEA-24, IDEA-26, IDEA-27, IDEA-22, IDEA-01, IDEA-07 |
| Converse | Ask with the right context, steer the model, compare | CS-01, CM-01, CC-04, CC-05, XP-04, CS-04 | IDEA-33, IDEA-13, IDEA-16, IDEA-10, IDEA-11, IDEA-06, IDEA-08 |
| Distill | Keep the good part, link it back, revise it safely, study it | XP-03, NS-03, CM-04 | IDEA-02, IDEA-03, IDEA-19, IDEA-14, IDEA-04 |
| Shared rules (all stages) | Return to anything, trust its state, read one language | XP-05, XP-06, XP-13, NS-06, XP-12 | IDEA-05, IDEA-01, IDEA-18, IDEA-43, IDEA-37, IDEA-45 |

---

### 8.1 Quick wins (days)

**Q1 · Example prompts in the chat empty state** (IDEA-31)
*First-time · /chat · Related: CO-02, CC-11*
- **Why:** An empty chat offers only "Start chatting" and a "Take a quick tour" button that does nothing. Starter prompts teach the product's capabilities by having the user use them.
- **Sketch:** Replace the "Start chatting" block with 3-4 chips tied to what tldw does well: "Summarize a video URL", "Ask about my notes", "Compare two models on a question", "Draft a note from this chat". Clicking a chip fills the composer without sending, and briefly highlights the control the prompt relies on (the Knowledge toggle or the model picker). The chips disappear after the first send. The existing "Explore chat modes" cards stay as a second row.

**Q2 · "Pick a model that works" on first run** (IDEA-10)
*First-time · /chat and side panel · Related: CC-01, XS-02, CC-02, CC-N2*
- **Why:** The first send is when a new user is most likely to give up. Today the webui picks an unreachable Ollama model and labels it "Healthy", and the side panel picks no model at all.
- **Sketch:**
  - *Day 1:* on first load, select the server's `default_provider`/`default_model` on both surfaces. Label it "Not checked yet" until a probe or a reply succeeds.
  - *Follow-up:* probe local providers in the background, and group the picker into "Ready", "Not reachable" and "Needs API key", with a status dot on each model.
  - *If the first send fails:* show an inline card in the reply slot: "Ollama isn't running at 127.0.0.1:11434. Switch to mock-gpt-4o and resend?" with [Switch & resend] and [Choose another model].

**Q3 · Respect OS preferences by default** (IDEA-42)
*First-time · cross-page · Related: AX-19*
- **Why:** `THEME_SETTING` defaults to dark, and animations ignore `prefers-reduced-motion`.
- **Sketch:** Default the theme to "System" and animations to the reduced-motion media query. Add "Reduce motion" and "Increase contrast" switches under Settings → Appearance. Each switch starts in the OS state and says so: "Following your system setting".

**Q4 · Local draft journal for notes** (IDEA-20)
*Both · /notes · Related: NS-01, NS-05*
- **Why:** A safety net against the 5s debounce, unmount and offline loss paths, which still matters after NS-01 adds a flush on unmount. This is the cheapest way to guarantee that typing is never lost.
- **Sketch:** On every change, write `{noteId, title, content, baseVersion, ts}` to IndexedDB, throttled to every 500ms. Clear the entry once the server confirms a save. When a note opens and its journal entry is newer than the server copy, show a banner: "Unsaved changes from 7:52 PM were recovered on this device." with Keep, Discard and Compare. Compare reuses the diff view from N15 once that ships.

**Q5 · Selection mini-toolbar inside answers** (IDEA-32)
*Both · side panel and /chat · Related: XP-01, XS-13*
- **Why:** Selecting text in an answer does nothing today, yet selecting is the most natural way to capture while reading.
- **Sketch:**
  - On selection, show a floating pill: **Save to Notes · Quote in reply · Copy · Explain**.
  - Save to Notes opens the existing `NoteQuickSaveModal`, prefilled with the selection, the chat title, the model and the source.
  - Quote in reply inserts `> selection` into the composer.
  - Shift+F10 or the menu key opens the same menu for keyboard users.

**Q6 · Code block toolbar** (IDEA-34)
*Power · /chat (and the Notes preview) · Related: CM-07, NE-N1*
- **Why:** `CodeBlock.tsx` already has a header with a Copy button, but the chat's `compact` and `github` variants bypass it. This is mostly a routing fix.
- **Sketch:** Send both variants through `<CodeBlock>`. Give it a slim header: language label · Copy (with check-mark feedback) · Wrap · Save to note. The header shows on hover or focus on desktop and is always visible on touch screens. "View code" should open in place and never trigger a download.

**Q7 · Denser chat history with date groups** (IDEA-07)
*Power · /chat · Related: CS-08, CS-N1*
- **Why:** With 40-150 chats, the ~120px four-line rows show about 4 chats per screen.
- **Sketch:**
  - Add a Compact/Comfortable toggle. Use two-line rows: the title, then a meta line such as "Resolved · 8m · retrieval".
  - Group rows under Today, Yesterday, This week and Older.
  - Show the source as an icon only when sources are mixed.
  - Add filter chips built from existing metadata (`state`, `topic_label`, `keywords`).
  - Never hide a filter because its view is empty: show "No chats in Trash" with a "Show all chats" link.

**Q8 · Context menu on notes rows** (IDEA-25)
*Power · /notes · Related: NL-04, AX-03, NL-03*
- **Why:** A row can only be selected, starred or opened. Rename, tag, move, copy link and delete all require opening the note first.
- **Sketch:** Right-click, a hover kebab, or Shift+F10 on the focused row opens: Open in Dock · Rename · Duplicate · Add tags… · Move to folder… · Copy [[link]] · Export · Move to Trash. Rename edits the title inline in the row. "Add tags…" reuses the chip input in a popover and **adds** tags, never replacing existing ones (contrast with NL-03).

**Q9 · One Tab stop for the notes list** (IDEA-40)
*Power, keyboard · /notes · Related: AX-03, AX-05*
- **Why:** Each row has three Tab stops, so a 20-row page costs 60 Tab presses (300 while the paging bug shows 100 rows).
- **Sketch:**
  - Make the list a single Tab stop with roving focus. ↑/↓ moves between rows.
  - Enter opens the note and moves focus to its title. Esc from the editor returns focus to the same row.
  - Space toggles selection, P pins, and Shift+↑/↓ extends the selection.
  - Each row carries its preview, date and tags through `aria-describedby`.
  - The existing "Skip to notes list" link lands on the active row.

**Q10 · Progressive disclosure of Notes views and AI actions** (IDEA-28)
*First-time · /notes · Related: NL-12, NL-13, NE-08, NL-14*
- **Why:** An empty workspace shows seven view buttons, ORGANIZE, FILTERS, Import/Sync folder/Export, ASSIST and a primary "Create study pack" button.
- **Sketch:**
  - With fewer than 3 notes, show only search, the list and "+ New note".
  - Reveal Timeline once notes span more than one day. Reveal Graph once a [[link]] exists, with a one-time hint ("New: see how your notes connect → Graph"). Reveal Boards once tags exist.
  - Group Suggest tags, Summarize, Make flashcards and Study pack under one "Assist ▾" button.
  - Split navigation into a Notes/Trash toggle and a separate "View: List ▾" dropdown that describes each view in one line.

**Q11 · Task-based first-run checklist for Notes** (IDEA-29)
*First-time · /notes · Related: NO-02, NO-N1, NO-04*
- **Why:** The coach-mark tour describes screen regions rather than the user's goals, scrolls targets off-screen and ends silently. Doing teaches better than reading.
- **Sketch:**
  - In the empty editor area, show a dismissible card: [ ] Write your first note · [ ] Add a tag · [ ] Link two notes with [[ · [ ] Find it with search.
  - Items tick themselves when the real event happens.
  - "Create a sample note" inserts a "Welcome to Notes" note that shows headings, a checklist and a link.
  - The tour stays available under Help but never scrolls the page.

**Q12 · Three-step first run in the side panel** (IDEA-30)
*First-time · side panel · Related: XS-17, XS-02*
- **Why:** The side panel has no onboarding, and a fresh panel opens on Companion "Setup required" cards with Chat about 3,000px further down.
- **Sketch:** Open the panel on Chat. Above the composer, show three inline cards one at a time:
  1. "Your model: mock-gpt-4o ▾. Change it here."
  2. "Ask about this page or anything else."
  3. "Save any answer to Notes from ⋯."

  The cards are dismissed for good after the first successful send or "Got it".

**Q13 · Adaptive knowledge search** (IDEA-36)
*Power · side panel (and Knowledge on /chat) · Related: none (newly observed)*
- **Why:** On this server, the default Balanced preset took about 91s while the client timed out at about 45s. Fast returned in about 5s.
- **Sketch:** On connect, probe whether the server has embeddings and a reranker loaded. If not, default to Fast, and make the preset chip say why: "Fast (keyword): semantic search isn't set up on this server". On timeout, make "Retry with Fast search" the primary action.

**Q14 · Glossary rollout and copy lint** (IDEA-43)
*First-time · cross-page · Related: XP-13, XP-14, CO-03, XS-12, CC-08*
- **Why:** Core objects have two to four names each, and jargon fills primary labels.
- **Sketch:** Make §10 the source of truth and run one rename pass over `defaultValue` strings. Add an i18n lint rule that rejects the banned terms in §10.8, UUID patterns and `snake_case` tokens in user-facing copy. Expert terms move into tooltips or "Learn more" links.

**Q15 · Request budget in CI** (IDEA-49)
*Both · cross-page · Related: XP-16, NS-04*
- **Why:** Polling every 5s, 2-4 duplicate fetches on load, and a ~700 KB list refetch on every autosave all grew unnoticed.
- **Sketch:** Add a Playwright check on seeded /notes and /chat. It fails on more than N API calls in the first 10s, duplicate GETs within 2s, or more than one request per 10s while idle. It also logs the request count and payload size of each autosave. Target: one PUT per autosave and no list refetch.

---

### 8.2 Next (weeks)

#### A. Close the capture ↔ converse loop

**N1 · Answer → Notes with smart defaults, append mode and provenance** (IDEA-02)
*Both · cross-page · Related: XP-01, XP-02, XP-03, XS-13*
**Depends on:** fixing XP-02 (loaded messages keep `serverMessageId`), or a Notes API path that doesn't require a server-backed chat.
- **Why:** Moving an answer into notes is the core research loop. Today the action is hidden. When it does work, it produces untagged notes with identical titles and no link back.
- **Sketch:**
  - **Where it appears:** "Save to Notes…" on every assistant message (always visible on the latest one), on text selections (Q5), and via Alt+S.
  - **The popover:** Title is prefilled from the user's question or the answer's first heading. Tags are prefilled with `from-chat` plus the chat's keywords. Destination is **New note | Append to…**, where Append searches recent notes and defaults to the last target used from this chat. Two checkboxes: "Include my question" and "Selection only".
  - **Fast path:** Shift-click saves with the defaults.
  - **Provenance:** the note gets a footer, "From chat: &lt;title&gt; · mock-gpt-4o · Oct 2", that links to `/chat/<id>#msg-<id>`.
  - **Confirmation:** an announced toast, "Saved to Notes", with an "Open note" action.

**N2 · Two-way links between chats and notes, plus "Chat about this note"** (IDEA-03)
*Power · cross-page · Related: XP-03, XP-04*
**Depends on:** N4 (addressable routes).
- **Why:** Provenance only flows from note to chat, through an overflow item, and never lands on the actual message. The only way to start a chat from a note is buried in Search & Context.
- **Sketch:**
  - The chat header shows "Notes (3)", a popover listing the notes derived from or attached to this chat, each with Open and Unlink.
  - The note header shows "From chat: &lt;title&gt; · '&lt;message excerpt&gt;'", which opens the chat at that message and highlights it for 2s.
  - The Connections panel gains a "Source chats" group.
  - "Chat about this note" in the note header opens /chat with the note attached as a context chip (N7).

**N3 · Notes inside the side panel** (IDEA-35)
*Both · side panel · Related: XS-13*
- **Why:** The panel can only create notes, through the browser context menu. It can't search them or open them.
- **Sketch:** Add a Notes tab to the panel with search, Recent captures, an Inbox count and "Open in full page". The message overflow's "Save to Notes" uses the N1 defaults, and its toast's "Open note" opens the note in the panel.

**N4 · Addressable routes and per-item page titles** (IDEA-05)
*Power · cross-page · Related: XP-05, XP-19, CS-02*
- **Why:** Bookmarks, sharing, working in several tabs and Back/Forward all need the URL to identify the selected note or chat. Most of the cross-page ideas build on this.
- **Sketch:**
  - Routes: `/notes/<id>`, `/chat/<id>`, `/chat/<id>?branch=<leaf>` and `/chat/<id>#msg-<id>`. Use pushState when the selection changes and replaceState for in-place updates, so Back/Forward walk through selections.
  - Tab titles: "&lt;note title&gt; · Notes" and "&lt;chat title&gt; · Chat".
  - "Copy link" in the note, chat and message overflow menus.
  - Extension hash routes (`#/notes/<id>`) and "Open conversation" use the same paths.
  - Returning to /notes reopens the last note.

**N5 · Universal quick switcher** (IDEA-01)
*Power · cross-page · Related: XP-06, CS-02, XP-07*
- **Why:** Power users navigate by name. Today ⌘K holds only commands, even though Notes' shortcut sheet documents it as notes search.
- **Sketch:**
  - ⌘K / Ctrl+K with an empty query shows Recent: the last 8 notes and chats mixed, with type icons and relative dates.
  - Typing searches, in parallel, note titles and bodies, chat titles and messages, prompts (server and local), characters and models. Results appear in grouped sections and are fully keyboard-operable.
  - Enter opens the item's URL (N4) and ⌘Enter opens it in a new tab. Tab cycles a type filter.
  - A `>` prefix shows today's commands and `@` shows entities. The last row offers 'Create note "&lt;query&gt;"' and 'New chat about "&lt;query&gt;"'.
  - ⌘P is an alias that opens pre-filtered to notes and chats.
  - The shortcut sheet describes ⌘K as "Search everything".

**N6 · One "Open in full page" hand-off, with a way back** (IDEA-09)
*Power · side panel · Related: XP-09*
- **Why:** "Expand in full page" and "Continue in WebUI" sit side by side, each keeps half the context, and neither has a return path.
- **Sketch:** Replace both with one "Open in full page ↗" (⌘⇧O). It carries the chat ID, the draft, attachments and page context, and opens in the normal layout, not Focus mode. The full page offers "Continue in side panel", which focuses the panel's tab through runtime messaging. If a WebUI URL is configured, the overflow offers "Open in WebUI" as a separate, accurately named item.

**N7 · A visible context strip above the composer** (IDEA-33)
*Both · /chat and side panel · Related: XS-08, CC-10, CC-04, CC-05*
- **Why:** What the model will read is scattered or invisible: the system prompt, pinned knowledge, attachments, the character, page context, and which messages are in the history.
- **Sketch:**
  - One collapsible strip above the composer: [Page: Spaced repetition – Wikipedia ×] [System: Socratic tutor ×] [Note: Glossary of RAG terms ×] [History: 2 of 6 messages · Reset] [+ Add context ▾ (Note, Tag, Media, This page, Prompt)]. Hovering a chip shows its token cost.
  - It replaces the always-on "Review conversation history" bar and the hidden Ctrl+E mode.
  - "Pin" in knowledge results adds a chip instead of taking over the viewport.
  - The Prompt picker lists the server prompt library alongside local prompts.

#### B. Trustworthy chat turns

**N8 · Stopped, incomplete and alternative replies as real message states** (IDEA-13)
*Power · /chat · Related: CM-01, CM-02, CM-03, CM-N2*
**Depends on:** fixing CM-01.
- **Why:** Stop, Regenerate and Edit & resend are how users steer a model. Today they fail, or they look like errors.
- **Sketch:**
  - **Stop** keeps the partial text, tagged "Stopped", with [Continue] and [Regenerate].
  - A reply that hit the length limit (`finish_reason=length`) gets "Incomplete · Continue".
  - **Regenerate** creates an alternative with a "‹ 2/3 ›" pager (Alt+Shift+←/→). Only the selected alternative is sent as context.
  - **Edit & resend** branches from the edited message. Later answers are dimmed and marked "outdated" but stay viewable.

**N9 · "Regenerate with…" and a model chip on every message** (IDEA-14)
*Power · /chat · Related: CM-04, CM-01*
- **Why:** Users who work with several models want to re-run one answer on another model without entering compare mode, and to see which model wrote each answer after a reload.
- **Sketch:** Make Regenerate a split button: the main click uses the same model, and ▾ lists recent and favourite models. Each assistant message header shows its model chip, and the pager reads "mock-gpt-4o · 2/3". This needs the server to store model metadata (CM-04).

**N10 · Branch map** (IDEA-15)
*Power · /chat · Related: CM-12, CM-N2*
- **Why:** A fork currently becomes an orphan chat named "Forked conversation".
- **Sketch:** A "Branches (2)" chip on the message where the fork happened opens a tree popover built from `/conversations/{id}/tree`. Each child is titled "&lt;title&gt; — from message 5" and shows when it was last updated. One click switches branch, which updates `?branch=`. The forked chat's header shows a breadcrumb back to its parent ("← &lt;parent title&gt;").

**N11 · A send queue that actually queues** (IDEA-17)
*Both · /chat and side panel · Related: CC-03, XS-10, XS-11*
- **Why:** While offline, both surfaces show a QUEUE button that doesn't queue.
- **Sketch:**
  - While offline or connecting, Send reads "Send when online". The message appears as a dimmed bubble, "Waiting for connection", with Cancel and Edit. It sends automatically on reconnect, and the queue persists per chat.
  - While a reply is generating, the primary button is always **Stop**. Queueing a follow-up becomes a secondary ▾ option: "Send after this reply".

**N12 · Composer restructure: one row of intent, Casual/Pro sets the density** (IDEA-11)
*Both · /chat and side panel · Related: CC-07, CO-03, CC-08, CO-06, CC-09, XS-15*
- **Why:** The default composer shows about 20-22 controls, Send is the weakest of them, and the existing Casual/Pro switch doesn't change how many controls appear.
- **Sketch:**
  - Row 1: [textarea] [**Send** (filled primary)].
  - Row 2: [+ Attach ▾ (Image, Document, From knowledge)] [Knowledge] [Web] [Model: mock-gpt-4o ▾] … [⋯ More: Compare models, Character, Tools, Response style, System prompt, Voice, Interactive answer]. Hovering the model chip shows the token estimate.
  - Casual is this layout. Pro adds the full toolbar, the right panel and a token chip. The choice is remembered per user.
  - The side panel uses the same composer and the same placeholder.

**N13 · Mobile composer as a bottom sheet** (IDEA-12)
*Both · /chat · Related: CO-01, AX-04, CO-N1, CO-05*
- **Why:** At 390px the composer takes about 46% of the height, and Focus mode hides navigation.
- **Sketch:**
  - A one-row composer: +, input, Send. "+" opens a bottom sheet with Model, Modes, Sources, Tools and Settings.
  - The header keeps ☰ (history), the title and "+ New chat". The toolbar hides while the thread scrolls.
  - Phones never open in Focus mode by default.
  - At 200-400% zoom the composer is capped at 40% of the viewport height and scrolls internally.

#### C. Notes at scale, safely

**N14 · One sync and status chip per surface** (IDEA-18)
*Both · cross-page · Related: NS-06, NS-03, CS-03, XS-05, NS-05*
- **Why:** Save and connection state appears three to five times per screen and contradicts itself. Users can't tell where their data lives.
- **Sketch:**
  - One chip next to the title: Saved · Saving… · Saved on this device · Conflict · Offline.
  - Clicking it opens a popover with details, for example: "Saved to your tldw server (127.0.0.1:8000) · Version 3 · Edited 2m ago · View history". The popover offers the one action that fits the state (Retry, Resolve or Diagnostics).
  - Other banners and toasts for the same event are suppressed.
  - /chat uses the same component, so "Saved" there also means a confirmed write.

**N15 · Note version history with diff and restore** (IDEA-19)
*Power · /notes · Related: NS-03, NS-N1, NL-03*
- **Why:** The server already versions every note. Showing that history makes conflicts, AI-assist replacements and bulk tag changes recoverable.
- **Sketch:** More actions → History opens a side panel of versions, each with a timestamp, an origin (Typed, Chat or Assist) and a word-count change. Each version offers a diff against the current text, "Restore this version" (which adds a new version and never overwrites one) and Copy. The 409 conflict flow reuses the diff, with "Keep mine", "Keep theirs" and "Merge manually".

**N16 · Notes browsing beyond 100 notes** (IDEA-24)
*Power · /notes · Related: NL-01, NL-05, NL-09, NS-04*
**Depends on:** fixing NL-01.
- **Why:** With 150+ notes, paging and Timeline must reflect the real total.
- **Sketch:**
  - Offset/limit paging feeds a virtualized list that shows the true total ("163 notes").
  - Timeline gets sticky month headers in the user's local time zone and "Jump to month".
  - Selection survives paging, with "Select all 163 matching".
  - An autosave updates its row in place instead of refetching the list.

**N17 · Search syntax and saved smart views** (IDEA-26)
*Power · /notes · Related: NL-07, NL-N1, NL-08, NL-10*
- **Why:** The tag filter, text search and the "Captured" filter are separate controls, and saved filters store only keywords.
- **Sketch:**
  - Support `tag:rag -tag:archive "exact phrase" in:title updated:<30d has:links`, with chips mirroring the parsed query.
  - Tag filters match whole tags. Several tags combine with AND by default, with OR available.
  - Results keep the server's relevance order and show a "Matched in: body" snippet.
  - "Save view" stores the query, sort and view as a named smart view with a live count.

**N18 · Folder tree** (IDEA-27)
*Power · /notes · Related: NL-10*
- **Why:** The backend stores folder paths and the extension's Clipper writes to them, but /notes has no folder UI.
- **Sketch:** A Folders section with counts. Notes can be dragged onto folders, or moved in bulk with "Move to…". The editor shows a breadcrumb ("Research / Papers"). Folders the Clipper creates appear automatically.

**N19 · Wikilinks that work: create from unresolved links, backlinks with context, unlinked mentions** (IDEA-22)
*Power · /notes · Related: NE-02, NE-10, NL-11*
**Depends on:** fixing NE-02.
- **Why:** This supports Zettelkasten-style capture and shows immediately whether a link resolves.
- **Sketch:**
  - Resolved links are solid. Unresolved links are dashed, with the tooltip 'No note titled "Idea". Click to create.' Creating the note opens it in the Dock or a peek view, so the user keeps their place.
  - Backlinks list each referencing note with a one-line snippet around the link.
  - "Unlinked mentions" finds the note's title in plain text and offers a one-click Link.
  - Autocomplete searches all notes on the server, not just the current sidebar page.

**N20 · Outline rail, focus mode and responsive layout for long notes** (IDEA-23)
*Power · /notes · Related: NE-03, NO-03, NO-01*
- **Why:** Long notes need persistent orientation without giving up editor space. Today the inline table of contents pushes the editor below the fold, and lines run unbounded on wide screens.
- **Sketch:**
  - The editor is centred, at most about 70 characters wide.
  - At 1280px and wider, a sticky right outline highlights the current section. Clicking an outline entry scrolls the editor pane, not the page. Below 1280px the outline becomes a collapsed "Contents" disclosure.
  - Mod+Shift+F hides the sidebar and chrome.
  - Pane widths and open/closed state are remembered.
  - On phones the formatting and Assist toolbars collapse into one "Aa" button above the keyboard.

#### D. Keyboard, accessibility and design-system foundations

**N21 · One keyboard-shortcut registry and sheet, aware of platform and context** (IDEA-37)
*Power · cross-page · Related: XP-07, XP-10, XS-08*
- **Why:** The documentation drifts from actual behaviour. "?" is bound three times. Ctrl+E and Ctrl+B collide with macOS text editing. The side panel and full page behave differently.
- **Sketch:**
  - One registry records each shortcut's scope, keys, description and whether it works inside text inputs (`allowInInput`). The side panel and full page share it.
  - Matching uses `event.code`, macOS defaults use ⌘, and conflicts are detected when a shortcut registers.
  - "?" opens one generated sheet, listing the current page's shortcuts first and then the global ones. Settings gets a shortcut editor.
  - Side-panel bindings: Alt+M opens the model picker, Alt+H recent chats, Alt+N a new chat, and Alt+S saves the last answer to Notes.

**N22 · Keyboard navigation between chat messages** (IDEA-38)
*Power · /chat · Related: AX-02, CM-09*
- **Sketch:** Alt+↑/↓ moves focus between messages, as do j/k when the composer is empty. On a focused message: c copies, e edits, r regenerates, b branches and n saves to Notes. ↑ in an empty composer edits the last user message. The message toolbar becomes visible on focus, not only on hover. Every binding appears in the shortcut sheet (N21).

**N23 · A chat transcript built for screen readers** (IDEA-39)
*Both · /chat · Related: AX-11, AX-14*
- **Sketch:** Each turn gets a visually hidden h2 ("You, 8:48 PM" / "Reply from mock-gpt-4o"). Markdown headings inside messages are demoted to h3 and below. The live region announces only the reply text, once streaming settles. A focused message announces its role, its position ("message 4 of 16") and the model.

**N24 · Shared accessible building blocks: MenuButton, Combobox, Announcer** (IDEA-41)
*Both · cross-page · Related: AX-02, AX-06, AX-09, AX-12*
- **Why:** Notes and Chat solve the same problems (overflow menus, inline suggestions, status toasts) with ad-hoc markup that behaves differently.
- **Sketch:** Add `packages/ui/src/components/a11y/` containing:
  - `<MenuButton>`, wrapping the antd Dropdown with `aria-haspopup` and focus return;
  - `useComboboxTextarea()`, used by the slash menu, @mentions and the [[wikilink]] menu;
  - `useAnnounce()` plus a `<LiveAnnouncer/>` mounted in AppShell, with an announcing wrapper around antd `message`.

  Model the webui picker on the side panel's `ModelSelect.tsx` (§9).

**N25 · One type and control scale, enforced by lint** (IDEA-44)
*Both · cross-page · Related: CC-07, XP-15, XP-20, CM-13*
- **Why:** The composer toolbar alone uses five font sizes, five control heights and corner radii from 0 to fully rounded. antd overrides change sizes silently.
- **Sketch:**
  - Type scale 12/13/14/16/20, with 11 reserved for uppercase section labels.
  - Control heights 28, 32 and 40, and one 8px radius. Fully rounded pills are reserved for status chips.
  - An ESLint/Tailwind rule forbids arbitrary `text-[Npx]`.
  - antd `ConfigProvider` tokens (`fontSize`, `colorPrimary`, `colorTextSecondary`) are read from the same CSS variables.
  - One interactive blue. Green is used only for success status.

**N26 · One Markdown renderer and one plain-text preview utility** (IDEA-46)
*Both · cross-page · Related: CM-08, NE-05, NE-07, NL-04, AX-10*
- **Why:** The renderers have drifted apart. Chat inline code shows its backticks, Notes code blocks drift into a "staircase", list previews show raw syntax, and tables lose their semantics.
- **Sketch:** One `<Markdown>` component with variants (`message`, `document`, `compact`), shared code, inline-code and table styles, and real `<table>` output. A `toPlainPreview(md)` utility feeds notes rows, chat search results and Notes Dock. Snapshot tests use one fixture containing a table, code, inline code, a checklist, an ordered list and LaTeX.

**N27 · Visual-regression and accessibility gates in CI** (IDEA-47)
*Both · cross-page · Related: AX-07, AX-08, NE-07, CO-06*
- **Sketch:**
  - Playwright screenshots at 1440×900, 1280×800 and 1024×768, in both themes, using seeded fixtures (a long note, a code note, a 16-message chat) and compared by perceptual diff.
  - `@axe-core/playwright` runs over /notes (empty state, editor, More menu, shortcuts modal) and /chat (empty state, an answer, the slash menu, the model picker). Violation counts are ratcheted down from today's numbers.
  - Every popover gets a keyboard end-to-end test.

**N28 · Typed API client and contract tests** (IDEA-48)
*Both · cross-page · Related: NL-01, NL-02*
- **Why:** The pagination and export bugs exist because frontend mocks accepted parameters the real API ignores.
- **Sketch:** Generate a TypeScript client for notes and chats from FastAPI's OpenAPI schema. Add a CI smoke test against a seeded backend with more than 100 notes, asserting that page 2 differs from page 1 and that the export count equals the total.

---

### 8.3 Strategic bets (larger redesigns)

Each bet gets a *First slice*, a small piece that delivers value alone and lowers the risk of the rest, and a *Risk / guardrail* that ties it to the strengths in §9.

**S1 · One chat history, searchable, across devices and surfaces** (IDEA-06)
*Both · cross-page · Related: CS-02, CS-03, XS-06, XS-05, XP-18*
- **Why:** History is split between local IndexedDB, the server, side-panel tabs, and a list buried under navigation.
- **Sketch:**
  - When connected, conversations default to the server, with offline queueing (N11).
  - The webui, options page and side panel share one Chats panel: Pinned, Today, Yesterday, Last 7 days.
  - Each chat shows a cloud or device badge. Search covers titles and message text.
  - Right-click or ⋯ offers Rename, Pin, Move to folder, Export and Move to Trash.
  - The side panel shows its open tabs plus a "Recent (server)" section navigable with ↑/↓ and Enter.
  - App navigation moves to the icon rail, so history is never below the fold.
- **First slice:** persist new webui chats to the server and make the "Saved" chip truthful (CS-03), then add message-text search.
- **Risk / guardrail:** keep the safe soft delete in chat Trash, the read-only share links, and Temporary chat staying genuinely local.

**S2 · One conversation model across the side panel, extension full page and webui** (IDEA-08)
*Power · side panel and full page · Related: XS-01, XS-07, XP-08, XP-18, XP-12*
**Builds on:** S1.
- **Why:** Most of the extension's power-user defects come from the side panel keeping its own tab snapshots, pins, labels and delete semantics.
- **Sketch:**
  - Each side-panel tab is a lightweight view onto either a `serverChatId` or a local draft.
  - When a tab gains focus, it reconciles with the server's version and latest message. If the chat changed elsewhere, a banner says "3 new messages from another window. Show".
  - Rename, pin and Trash happen on the server and apply everywhere.
  - Local-only chats show "Not synced" with a one-click "Save to server".
- **First slice:** make opening a past chat open a new tab instead of overwriting the current one (XS-01), and make Rename and Delete act on the server (XS-07).
- **Risk / guardrail:** the per-tab drafts that survive reloads and the clean per-chat Export are both strengths, and must survive the change.

**S3 · Reply generation that runs on the server and can be resumed** (IDEA-16)
*Both · /chat · Related: CS-04, CS-N3, CM-N1, CM-06*
- **Why:** Navigating away, reloading or a network blip discards the turn. With long local-model generations, this happens often.
- **Sketch:**
  - The client posts the turn. The server saves the user message, runs the generation under an `operation_id`, and writes deltas to history.
  - When the user returns, the UI re-attaches through an SSE resume (`Last-Event-ID`), or shows the finished reply.
  - The waiting state shows elapsed time ("Still working… 42s"), and the startup timeout matches the server's.
  - The existing "retained send outcome" plumbing could serve as the backing store.
- **First slice:** save the user message immediately and add a leave/reload guard while a reply is in flight. That alone ends the silent loss of the question.
- **Risk / guardrail:** the composer draft autosave and the `role=log` transcript with `aria-busy` while streaming must keep working when a reply resumes.

**S4 · Side-by-side research mode (chat + note)** (IDEA-04)
*Power · cross-page · Related: XP-04, XP-11*
**Builds on:** N1, N2, N4.
- **Why:** Research alternates between asking and writing. Notes Dock floats over the transcript and isn't linked to the chat.
- **Sketch:**
  - "Pin note beside chat" docks the note as a resizable right split on /chat.
  - Messages or selections can be dragged into the note.
  - "Ask about selection" in the note sends the selection, with a link, to the composer as a context chip (N7).
  - Each chat remembers which note it was paired with, and `/chat/<id>?note=<id>` restores the pair.
- **First slice:** "Open in split" from N1's toast and from N2's "Notes (3)" popover.
- **Risk / guardrail:** Notes Dock's fast, low-ceremony capture from any page must not become heavier.

**S5 · One block editor with Markdown shortcuts and a slash menu** (IDEA-21)
*Both · /notes · Related: NE-01, NE-06, NE-05*
- **Why:** A broken WYSIWYG mode plus a Markdown-only toolbar forces newcomers to pick an editing mode they can't evaluate.
- **Sketch:**
  - Replace the Markdown/WYSIWYG toggle with a TipTap or Milkdown editor that stores Markdown.
  - Typing `# `, `- `, `[] ` or `1. ` formats automatically, and lists continue on Enter.
  - `/` opens Heading 1-3, Checklist, Quote, Code, Link to note and Table.
  - "View source" remains for power users.
- **First slice:** ship the [[link]] and slash menus as comboboxes (N24) in the current Markdown editor, and add checklist and numbered-list buttons.
- **Risk / guardrail:** keep Markdown as the storage format, keep autosave with optimistic locking, and keep the measured performance: a 25k-character note opens in about 80-135ms at about 20ms per keystroke. Ctrl/Cmd+B must mean bold inside the editor (XP-07).

**S6 · A shared "object page" layout for Notes and Chat** (IDEA-45)
*Both · cross-page · Related: XP-11, NO-01, CO-06, AX-14*
- **Why:** Notes and Chat use different header, status and empty-state patterns, and on /notes the app navigation is the Chats sidebar.
- **Sketch:** Five columns, left to right:
  1. **App rail:** labelled navigation, titled "tldw", with a context-aware "New…" menu.
  2. **Object list:** search at the top, filter chips, compact rows and a density toggle.
  3. **Workspace header:** editable title · status chip (N14) · primary action · ⋯.
  4. **Content.**
  5. **Optional right panel:** Outline and Connections for notes, Context and Model settings for chat.

  One h1 per page and one `<main>`.
- **First slice:** separate global navigation from the Chats list, so /notes no longer has a "New chat" "+" next to its own "+".
- **Risk / guardrail:** keep the working skip links, the coherent light and dark themes, and the no-horizontal-overflow guarantee from 320px to 1920px.

### 8.4 Sequencing and dependencies

1. **Fix first.** The P0 defects CS-01, CS-03, CM-01, NL-01, NS-01 and XP-02 are prerequisites for N1, N8, N9, N16 and S1. Ship the quick wins alongside these fixes. Q2, Q4, Q14 and Q15 directly reduce the chance of the same defects recurring.
2. **Enablers.** N4 (routes), N14 (status chip), N21 (shortcut registry), N24 (accessible building blocks) and N26 (renderer) unblock several other items each. Schedule them early in the "Next" horizon.
3. **Loop features.** N1, N2, N3, N5 and N7 together make capture → organize → converse → distill feel like one product. They are the most visible return on the foundations.
4. **Bets.** S1, then S2, is the data-model path. S3 is independent and can run in parallel. S4 needs N1, N2 and N4. Prototype S5 and S6 behind a flag with usability sessions before committing.

---

## 9. Strengths to preserve

These work well today. Each group ends with a **Guardrail**: what any change from §8 must keep, and which patterns should become the template for the rest of the app.

### 9.1 Notes data safety and recovery
- **Delete and recover is solid.** A single delete offers a working 10s "Note deleted · Undo". Trash lists "Deleted · &lt;timestamp&gt;" with Restore, and restoring reopens the note with a "Note restored" toast.
- **Autosave with versions and optimistic locking.**
  - About 5s after typing stops, the status shows "Saved just now" and "Version n". Ctrl/Cmd+S saves.
  - Every save sends the version it expects, so conflicts are detected rather than silently overwritten.
  - A clear "This note was updated elsewhere… Reload note" banner appears within 30s.
  - `beforeunload` guards full page reloads, and double submits are prevented (double-click plus Ctrl+S sends one POST).
- **Switching notes saves reliably.** Five rapid switches, each within 250ms, each saved to the right note with no cross-contamination. Edits typed offline synced once the connection returned.

**Guardrail:** The undo-plus-Trash pattern and the "expected version" save contract are the reference model for chat deletion (XP-17), bulk tagging (NL-03) and the block editor (S5). N15 (history) and Q4 (draft journal) should build on the version numbers, not replace them.

### 9.2 Performance and robustness
- A 25k-character note opens in about 80-135ms, with about 19-22ms per keystroke in Edit and Split.
- A 30-message chat renders in about 0.6s. The timeline virtualizes above 100 blocks, and composer input latency is about 23ms.
- There is no horizontal overflow at 320, 390, 640, 768, 1024 or 1920px on either page, and chat has dedicated viewport handling for the mobile composer.
- About 80 scripted sessions on the webui, options page and side panel produced no console errors, page errors or failed API calls, apart from the specific broken actions reported.

**Guardrail:** Lock these numbers in as budgets in Q15 and N27, so S5 (editor), N12 (composer) and N16 (virtualized list) can't regress them unnoticed.

### 9.3 Finding and organizing notes
- **Tagging is smooth.** A chip input with Enter, usage counts ("thesis (1)", "research (32)") and a Browse tags modal sorted by frequency. Search and tag filtering page correctly through the server.
- **Search is genuinely full-text.** It highlights matched terms in titles and previews. The no-results state is excellent ('No notes match "…"' plus "Clear search & filters"), and a Search tips popover helps.
- **Import is strong.** It accepts multiple files, previews the note count per file, offers a duplicate strategy (copy, skip or overwrite), and honours YAML front-matter tags and H1 titles.
- **The list's error card is good.** Its title is specific, it keeps the search and filters, and it offers "Retry connection" and "Health & diagnostics".
- **"More actions" is well grouped.** Create, Organize, AI, Export and Delete each have icons and submenus, and the menu is fully keyboard-operable. The bulk bar shows a clear count, "Clear selection" and a danger-styled Delete, and Shift-click range selection works.
- **Smart collections state their rule in plain words** ("Auto: anything tagged rag").

**Guardrail:** The no-results state and the error card are the templates for chat's empty, error and loading states (CS-06, CS-07, NL-16). The "More actions" grouping is the model for the chat message overflow (CM-09). N17's query syntax must stay optional: chips and plain text search keep working alone.

### 9.4 Accessibility groundwork
- Working skip links: "Skip to main content", "Skip to notes list", "Skip to editor".
- The notes list is a real list, with specific row names and `aria-current`. Editor fields are properly named. Edit/Split/Preview use `aria-pressed`, and save status uses `role=status`.
- The chat transcript is a labelled `role=log` of named articles, with `aria-busy` while streaming and a visually hidden "Response complete".
- The role-play drawer, shortcuts panel and keyword picker manage focus correctly. Textareas use a high-contrast teal focus ring. axe found no `button-name`, `link-name` or `image-alt` violations.
- The side panel's model picker (`Common/ModelSelect.tsx`) already handles focus correctly: it focuses its search field on open and returns focus to the trigger on close. The side panel's composer and message buttons meet the 44px target size and have `aria-label`s.

**Guardrail:** Use `ModelSelect.tsx` as the template for the webui picker fix (AX-01) and for N24's MenuButton. The `role=log` transcript stays the base for N23. Narrow its live region, but don't remove it.

### 9.5 Chat answer quality and composer craft
- **Answers render well on both surfaces:** headings, lists, blockquotes, syntax highlighting, LaTeX, and tables with View (fullscreen), Copy as CSV and Download CSV. Message Copy shows clear check-mark feedback and copies the Markdown source.
- **The June 2026 empty-model-picker race no longer reproduces:** the picker loaded on about 20 cold loads. It offers "Usable configured models" and "Help me choose a model". The cockpit rails are now collapsed by default, with a screen-reader note.
- **"Explore chat modes" uses plain language a first-timer can follow:** General chat, Compare AI models, Character chat, Search your documents, Deep research.
- **Composer menus show each toggle's current state** ("Off", "Closed") under clear group headers. "Clear conversation" is disabled when there is nothing to clear.
- **Composer drafts autosave** with a quiet "Draft saved" indicator in the webui and the side panel. Per-tab drafts persist across reloads, so text typed during transient failures survives.
- **Temporary chat gives strong feedback through several channels:** the label "Temporary chat (not saved)", a tinted header, an explanatory toast, and a "Temp" button with a tooltip.
- **Keyboard and search basics work:**
  - The slash-command menu is discoverable and gives one-line descriptions.
  - Shift+Enter inserts a newline, and Shift+Esc or Ctrl+Shift+U focuses the composer.
  - Ctrl/Cmd+F opens a labelled thread search with a match count and Prev/Next, and handles 80-message threads without lag.
- **Character chat entry is well done:** a Character list in the header, a greeting picker with Reroll and Select, "Include greeting in context", and character and context chips.

**Guardrail:** The "Explore chat modes" copy sets the register for §10, so new labels should sound like it. The table tools and Copy feedback should carry over to the code-block header (Q6). "Draft saved" and the multi-channel Temporary-chat feedback must survive N12 (the composer restructure). The text "Temporary chat" itself is the target term (§10).

### 9.6 Cross-surface plumbing that already works
- **Capture surfaces:**
  - Notes Dock is a solid in-context capture surface on every page: a title, tag chips, unsaved state, a "Note saved" toast, and the note appears in /notes immediately.
  - The side panel's quick-save modal is clean and fast, with the title prefilled and the source URL shown.
- **Chat-to-note provenance:**
  - Origin metadata is recorded ("Origin: Saved from Chat", `conversation_id`/`message_id`).
  - "Open conversation" reliably loads the server chat, and refuses safely when the current chat has unsaved work.
  - `/notes?source_ref_id=<id>` deep-links to a note in both the webui and the extension.
- **Server chats on /chat:**
  - Revisiting restores the last open server chat and uses its title as the tab title.
  - Sending into a server-loaded chat saves both messages. Renaming in the header saves to the server.
  - Chat Trash is a safe soft delete with Restore and a confirmation before permanent delete.
  - Share links are read-only, explain the viewer's role, and expire.
- **Knowledge (Search & Context)** already treats Notes and Chats as retrieval sources, with source-type chips, Preview, Pin and Insert on each result, and a saved-results tray. The side panel adds Fast/Balanced/Thorough presets and a clear Retry on timeout. This is a strong base for "chat with my notes".
- **Hand-off between surfaces:**
  - "Expand in full page" carries the conversation, the selected model and the title into the extension full page.
  - "Continue in WebUI" preserves the draft exactly.
  - The infrastructure works; the two actions only need to be merged (N6).
- **Side panel details:**
  - Per-chat Export produces clean, timestamped Markdown or JSON.
  - The command palette works in the narrow panel.
  - The connection pill and the "Can't reach your tldw server" message are plain-language and specific.
- **Shell consistency:**
  - The extension options page's Notes matches the webui feature for feature.
  - The webui and options page share one app shell with coherent light and dark themes.
  - Webui tab titles follow the route.

**Guardrail:** N1, N2, N4, N6 and S4 should extend this plumbing, not replace it. Reuse the existing provenance fields, the `source_ref_id` deep link and the hand-off payloads. "Can't reach your tldw server" is the voice to copy for every error in §10.4.

---

## 10. Terminology and UX-writing guide

This guide consolidates the lens-visual reviewer's glossary (19 entries, finding "Inconsistent vocabulary for core objects", issue XP-13) with the jargon that the first-time reviewers (ft-notes, ft-chat, ft-ext) hit on their journeys. It is meant as the single source of truth for Q14 (the rename pass and the copy lint).

### 10.1 Principles

1. **One object, one name.** The core nouns are *note, tag, folder, board, chat, message, character, model, knowledge, prompt*. Each is used the same way on /notes, /chat, the options page and the side panel.
2. **Plain words in primary labels; the expert term goes in a tooltip.** For example, the label is "Tools" and the tooltip says "Uses MCP tools". Never the other way round.
3. **Name what the user gets, not the mechanism.** Don't use *turn, path, interpretation, overlay, rail, sidechannel, checkpoint, scope* or *route* in user-facing copy.
4. **A status is not a mode.** "Saved" appears only after a confirmed write, and says where: "to server" or "on this device". The persistence choice is a separate control, "Temporary chat".
5. **Errors say what happened, the likely cause and the next step, where the failure happened.** Raw codes and URLs belong in "Details".
6. **Destructive actions name the outcome and the way back.** Use "Move to Trash" for recoverable deletes, "Delete forever" for permanent ones, and say how to restore.
7. **Disabled controls say why, and how to enable them.** For example: "Set up speech in Settings to use Voice".
8. **No IDs, enums or `snake_case` in user-facing copy.** IDs go in an Info panel with a copy button. States use sentence case: "In progress", not "in-progress".
9. **Dates:**
   - Relative in lists ("8m ago", "Sep 17"), absolute in tooltips ("Oct 2, 2026, 8:33 PM").
   - No seconds.
   - Group by the user's local time zone.
10. **The visible label starts the accessible name** (WCAG 2.5.3). Tooltips and `aria-label`s reuse the visible words.
11. **One product name.** Decide between "tldw" and "tldw Assistant" and use it in every title and header.

### 10.2 Core objects and states (the glossary)

| Current term(s) | Where | Problem | Recommended term / plain-language rewrite |
|---|---|---|---|
| chat / conversation / session / thread | Sidebar "Chats" vs "Recent conversations" and "Load conversations"; "Share conversation"; "Search this thread"; "Session insights"; "Character sessions" | Four names for one object. The sidebar heading and its own section disagree. (XP-13) | **chat** for the object, **message** for each item in it: "Recent chats", "Load more chats", "Share chat", "Search in this chat", "Chat insights" |
| Temporary chat (not saved) / Temp / Private chat is locked | /chat header button, composer, lock message | Three names for one mode. "Private" suggests encryption or access control. (XP-13) | **Temporary chat** everywhere. A compact button can read "Temporary" with the tooltip "Temporary chat: not saved to history" |
| Saved (mode pill/chip) / Save chat to history / Saved · Locally + Server | /chat header pill and composer chip (shown before anything is sent); side-panel switch | "Saved" is used as a mode and as a status, and claims server saves that never happen. (CS-03, XS-05, NS-06) | Mode: the **Temporary chat** toggle (or "Save chats: On / Off"). Status: **Saved · 2m ago** only after a confirmed write, **Saved on this device** when stored locally, **Not saved** otherwise |
| keywords / tags | Notes "Tags" and "Assign tags" vs "Keyword filter", "Comma-separated keywords"; test ID `notes-bulk-assign-keywords` | Two names for one concept. Newcomers wonder whether they are different. (XP-13) | **tags** everywhere: "Filter by tag", "Add tags (press Enter after each)" |
| Character / Persona / Buddy / Assistant ("Buddy & Persona", "Conversation Persona", "Use this Buddy", "Open Persona Garden") | Composer toolbar, Buddy modal, side-panel header, message headers | Four overlapping names. "Artwork is independent of the conversation's Persona" is unreadable to a newcomer. (XP-13, CO-03) | **Character** for who replies and **Avatar** for its artwork. Use "Assistant" only as the default character's name. If Persona remains a separate product feature, define it once and keep it off chat surfaces |
| Search & Context / Knowledge panel / Manage in Knowledge Panel / Knowledge QA / RAG / `/search` | Composer toolbar, Modes menu ("Knowledge panel: Closed"), More tools, Attach menu, nav, slash command | Five names, and panel state listed as if it were a mode. (CC-08, CC-10) | **Knowledge**: the toggle "Knowledge" and the panel "Knowledge: search your notes and media". `/search` stays as an alias |
| Compare responses / Compare models | Modes menu vs More tools (styled as a link) | Two names for one feature. (CC-08, CM-05) | **Compare models** |
| Collection (moodboard) / Saved filter (`collections` in the backend) / notebook | Notes Views, ORGANIZE section | One word covers both hand-picked groups and rule-based filters. | **Board** for hand-picked groups, **Smart filter** for rule-based ones ("Auto: anything tagged rag") |
| Inbox (view) / Captured (filter chip) | Notes Views grid; FILTERS chip "Show notes saved from browser capture" | Two controls do the same thing. (NL-12) | **Inbox** only, described as "Notes saved from the browser extension" |
| Shortcuts (nav section) / Shortcuts (keyboard panel) / Show shortcuts (header strip) | Sidebar section, header "?" and "Shortcuts", signpost strip | One word, three meanings. (CC-08, XP-07) | **Go to** for navigation, **Keyboard shortcuts** for keys |
| in-progress / resolved / backlog | Chat history row chips (lowercase, the same grey for every state) | Raw enums that don't tell the states apart. (CS-08) | **In progress** (blue dot), **Resolved** (green check), **Backlog** (grey dot) |
| tldw Assistant / tldw | Header and extension tab title vs webui tab "Notes \| tldw" | Two product names. (XP-19) | Pick one (suggested: **tldw**). Tab titles follow "&lt;item&gt; · &lt;page&gt; · tldw" |

### 10.3 Chat controls and features

| Current term(s) | Where | Problem | Recommended term / plain-language rewrite |
|---|---|---|---|
| Cockpit rails / Context rail / Runtime rail / "Restore context sidechannel" / "Cockpit rails hidden" | /chat vertical tabs and their tooltips, screen-reader note | Engineering metaphors. The accessible name doesn't match the visible label. (CO-04, AX-17) | **Side panels: Context, Run info** (or "Details"). Tooltip: "Show context panel" |
| Composition / Scope: tldw:gemma3:1b / Provider route / Turn idle / Context stack | Rail contents | Internal state written as labels. (CO-04, CC-06) | Plain summaries: "Model: gemma3:1b (Ollama)", "Idle", "What the model will read: 3 items". Move the rest to Details |
| MCP None / Tool choice Auto · Required · None / Tool run: Idle | Composer toolbar chip, tools menu | "MCP" means nothing to most users. "None" reads like an error. (CO-03) | **Tools: Auto / Always / Off**, with the tooltip "Uses MCP tools configured on your server" |
| OpenUI / "Answer with an OpenUI interface for the next message" | Composer toolbar | Unexplained brand term. (CO-03) | **Interactive answer**: "Reply with buttons and forms where useful" |
| Review conversation history / Use no prior messages / "Complete source / included path: 2 / 2" / Path position 1 / Omitted alternative / "Confirm creates a saved interpretation of this complete source" | Bar above every chat; review panel (webui and side panel) | Exposes an internal history-path model. One click silently blanks the context. (CC-04) | Move into ⋯ as **"Choose which messages the AI sees…"**. Rename the empty choice **"Start fresh from here"**. When active, show the chip **"Using 2 of 6 messages · Reset"**. In the panel, show messages in chronological order: "Included", "Left out" |
| Buddy & Persona / Role-play setup / Apply overlay / Start tracked character chat / Start tracked persona chat / TRACKED SESSIONS | Composer toolbar; side-panel Character popover | Jargon ("overlay", "tracked") and duplicated concepts. (XS-12, CO-03) | **Chat as a character…**: a simple picker first. "Remember this character's memory across chats" goes under Advanced |
| Continue as user / Impersonate user / Force narrate (single-letter hints C, I, N) | Message ⋯ menu in plain chats | Role-play steering shown in normal chats, with unexplained letter hints. (CM-09) | Show only when a character is active, under a **Role-play** group: "Write my next message", "Narrate a scene". Explain the letters in the shortcut sheet |
| Redo (Pro mode) / Regenerate | Message action pills | Two names, and "Redo" suggests undoing an undo. (CM-13) | **Regenerate** (icon plus tooltip in tight layouts) |
| Advanced controls → four unlabelled icons (sparkles, scales, target, sliders) | Composer footer | Meaning is available only on hover. (CC-09) | **Response style: Creative · Balanced · Precise · Custom**, as a labelled segmented control |
| Connection dot styled like a switch | Advanced controls row | Looks like a toggle but is a status indicator. (CC-09) | Remove it (status lives in the model chip), or show the text "Connected" |
| "Type a message... (/ commands, @ mentions)" vs "Ask anything... (/ for commands)" | Webui vs side-panel placeholder | Different placeholders, and the webui promises @mentions that don't work. (CC-11, XP-12) | One placeholder on both surfaces: **"Ask anything. Type / for commands"**. Add "@ to mention" only once it works |
| "0 + ~0 = 0 tokens" | Composer footer | Developer arithmetic. | **~375 tokens**, with the breakdown in a tooltip |
| Standard chat / Conversation timeline / Composer (pill-styled labels) | /chat chrome | Look like buttons but do nothing. (CO-03) | Plain text, or remove |
| Usable configured models / Configured \| All known models | Model picker | Opaque scope names, and the toggle doesn't visibly switch. (CC-06) | **Ready to use / All models**, as a labelled segmented control |
| Expand in full page / Continue in WebUI; tooltip "Opens this selected history in the extension full page. Active streaming stays in this panel." | Side-panel header and chips | Two hand-offs with similar names. "Continue in WebUI" actually opens the extension. (XP-09) | **Open in full page ↗** (tooltip "Open this chat in a full tab"). Use "Open in WebUI" only when it truly opens the web app |
| Config (opens Voice chat settings) / Open dashboard (opens Flashcards) / TTS clips | Side-panel header and composer | Labels don't match where they go. (XS-12) | **Voice settings**, **Flashcards**, **Audio clips** |
| Start a conversation in Console | Buddy modal | "Console" is an undefined place. (CO-03) | Name the destination ("Start a chat"), or remove |

### 10.4 Errors, status and recovery copy

| Current term(s) | Where | Problem | Recommended term / plain-language rewrite |
|---|---|---|---|
| "Turn needs review / The original send outcome is retained separately from history. It will not be sent again automatically. / User input accepted; response not saved" with "Copy recovered text" and "Dismiss recovery" | Send failures, Stop, and reload mid-reply (webui, options, side panel) | Jargon with no cause and no next step. Also used for a user-initiated Stop. (CC-02, CM-02, CS-04) | In the reply slot: **"Couldn't get a reply from gemma3:1b (Ollama). Your server couldn't reach 127.0.0.1:11434."** with [Retry] [Switch model ▾] [Edit & resend] [Details]. After Stop: the tag **"Stopped"** with [Continue] [Regenerate]. After a reload: **"Your last question didn't get a reply. Send it again?"** |
| "Generated response needs review" | Returning to /chat mid-reply | Same jargon. (CS-04) | **"The reply was interrupted when you left this page."** [Regenerate] |
| `request_config_scope_changed` and other raw error codes (toasts) | Stop pressed early, New chat, Regenerate, Edit | Machine code shown to the user; the message is lost. (CS-01, CM-01, CM-02) | Explain, and keep the user's text: **"Settings changed while sending, so nothing was sent. Your message is back in the box."** The code goes in Details |
| "Something went wrong while talking to your tldw server" (provider failure) | Send failure toast | Blames the server for a model provider's failure. (CC-N2) | **"mock-gpt-4o (Custom OpenAI) didn't respond: &lt;cause&gt;."** [Retry] [Switch model] |
| "Still generating response (checkpoint N)" every 5s | Screen-reader announcements during long waits | Noisy and meaningless. (CM-06) | Visible text **"Still working… 25s"**, announced at most at 10s and 30s, then **"This model is slow to start. Keep waiting or switch model?"** |
| Healthy (on a model nobody has checked, or that just failed) | Model chip and picker | Claims a health check that never ran. (CC-01, CC-N2) | **Ready** only after a successful probe or reply. Otherwise **Not checked yet** or **Not reachable** |
| QUEUE (primary button while pending or offline) | /chat and side-panel composer | Promises queueing that doesn't happen, and hides Stop. (XS-10, CC-03, XS-11) | While generating: **Stop**. While offline: **Send when online** (only once N11 ships) |
| "You're offline" on first load | /chat composer | A false state while the app connects. (CC-03) | **Connecting…** for the first few seconds. **Offline** only after a failed check |
| Loading selected history | First send in a new chat | Internal step exposed, and the user's message appears late. (CM-14) | No copy needed: show the user's message immediately |
| "Version pending · Not saved yet" / "No server save status yet" / "Unsaved changes" ×3 | Notes editor header, under Tags, footer | Three status displays with developer wording. (NS-06) | One chip: **Not saved yet → Saving… → Saved · 2m ago**. The version number goes in the chip's popover |
| Remote changes detected / "Save anyway" / five conflict messages | Notes 409 conflict | Contradictory messages, and "Save anyway" fails. (NS-03, NS-N2) | **"This note changed on another device."** [Compare] [Keep mine] [Use theirs] |
| A note without a title fails with a network error | Notes save | Blames the network for a validation problem. (NS-02) | Default the title to **"Untitled note · Oct 2"**, or show **"Add a title to save"** next to the field |
| Raw request URL in error cards | Chat load failure, notes list error | A machine string that overflows its box. (CS-06, NL-16) | **"Couldn't load this chat."** [Retry] [Details] |
| "Please confirm / Delete this note?" | Notes delete confirmation | Generic, and doesn't mention that Trash can restore it. (NL-15, XP-17) | **"Move 'Thesis outline' to Trash? You can restore it from Trash."** [Move to Trash] |
| "Conversation cleared" (when nothing was cleared) / "Chat restored." (when the messages weren't restored) | /chat | Success copy for actions that didn't happen. (CS-05, CS-N2) | Report only what actually happened. After restore: **"Chat restored with 16 messages."** |

### 10.5 Notes-specific copy

| Current term(s) | Where | Problem | Recommended term / plain-language rewrite |
|---|---|---|---|
| "Portable markdown with best-effort task continuity" | Under Tags, once a note has checkboxes | Engineering jargon. (NE-09) | Remove it, or put **"Checkboxes stay in sync with the note text"** in a help tooltip |
| Origin: Typed manually / Saved from Chat | Note footer (also shown on blank notes and side-panel quick saves) | System voice, and wrong for side-panel saves. (XS-13) | **Created by you** / **From chat: &lt;title&gt;** (as a link). Hide on blank notes |
| "Linked to conversation: &lt;uuid&gt; · msg &lt;uuid&gt;" (link-blue, not clickable) | Notes list rows and editor header | Raw IDs styled as a link that doesn't work. (XP-03, XP-14) | **From chat: RAG chunking strategy · Open message** (a real link) |
| "Snippet: &lt;chat title&gt;" (automatic title) | Notes created from chat | Identical, unsearchable titles. (XP-03) | Title from the user's question or the answer's first heading, for example **"How LoRA fine-tuning works"** |
| Related notes / Manual links / Backlinks (listed twice) | Connections panel | Three unexplained concepts, plus duplicates. (NE-10) | **Linked notes**, split into **Links from this note** and **Links to this note**. Each row tagged "typed link", "added by you" or "suggested" |
| Canvas / Relationships / Radius / Max nodes 120 / Layout: Dagre, Circle, Grid, Concentric / Focused, All notes | Graph view | Graph-library vocabulary. (NL-14) | Under a **Graph settings** popover: **Layout: Tree / Circle / Grid**, **Show up to 120 notes**, **Show: This note's links / All notes**. Add a legend: circle = note, square = tag |
| Create study pack (primary button) / "Advanced/manual source reference… paste the item ID" | Notes editor header; Flashcards drawer | Too prominent for a niche action, leaves Notes, and asks for raw IDs. (NE-08) | **Make flashcards…** under **Assist ▾**. Hide the manual-ID option under "Advanced" |
| ORGANIZE / FILTERS / ASSIST / RESULTS (uppercase section labels) | Notes sidebar | Acceptable as labels, but ASSIST hides AI actions behind a vague verb. | Keep **Organize** and **Filters**. Rename ASSIST to **Assist ▾ (AI)**, with items named by outcome: "Suggest tags", "Summarize", "Make flashcards" |
| Sync folder | Notes header | Doesn't say which direction or where, and hints at local storage while notes live on the server. (NS-06) | **Sync with a folder…**, with one line describing the direction and location. The status chip says notes are stored on the server |
| "Showing notes in pages for faster loading." | Notes list footer | Developer voice; adds nothing. | Remove |
| "Select or create a note" shown above a live editor | Notes empty state | Contradicts the editor beneath it. (NO-04) | One state at a time: **"No note open. Pick one from the list or start a new note."** [New note] |
| Notes and Notes Dock (identical icons) | App rail | Two features look like one. (XP-11) | Keep the names, but give Notes Dock a distinct "docked panel" icon and the tooltip "Quick note (Ctrl/Cmd+Shift+N)" |

### 10.6 Side panel and onboarding copy

| Current term(s) | Where | Problem | Recommended term / plain-language rewrite |
|---|---|---|---|
| Companion Home with "Setup required" on every card | Fresh side panel | The first screen is a dashboard of blockers. Chat is about 3,000px down. (XS-17) | Open on **Chat**. Companion becomes a tab whose empty state reads **"Turn on personalization in Settings to see insights here."** |
| Unlabelled brain icon (the model control) | Side-panel composer | The only model control has no name, and disappears in Casual mode. (XS-02) | A visible **Model: mock-gpt-4o ▾** chip, in Casual mode too |
| Casual / Pro | Sidebar and side-panel switch | The names don't say what changes, and today little does. (CO-03, XS-15) | Keep the names, but add one line under each: **Casual: just the essentials** / **Pro: all controls and panels**. The switch must actually change density (N12) |
| Focus mode (default on mobile) / "Exit focus" pill | /chat on phones, full page after hand-off | Hides navigation by default, and the pill covers text. (CO-01, CO-N1, XP-09) | Use Focus only when the user chooses it. Put **Exit focus** in the header, not floating over content |
| Take a quick tour (does nothing) / an 8-step coach-mark tour | /chat, /notes | A dead link, and a tour of screen regions. (CO-02, NO-02) | **Get started** checklist (Q11, Q12). Show "Take a tour" only when a tour exists |
| Voice (disabled, no reason) / Read aloud (disabled, no reason) | Side-panel composer, message ⋯ | Unexplained disabled state. | Tooltip **"Set up speech in Settings to use this"** with a link |

### 10.7 Identifiers, numbers and dates

| Current term(s) | Where | Problem | Recommended term / plain-language rewrite |
|---|---|---|---|
| `custom_openai_api:mock-gpt-4o`, `tldw:gemma3:1b`, "Custom OpenAI API / …" (model name truncated) | Streaming bubble header, rails, composer chip | Three formats, and the part the user needs is the part that gets truncated. (CC-06, XP-14) | **mock-gpt-4o** (bold) · Custom OpenAI (secondary). Truncate the provider, never the model |
| "Captain Morrow Chat (20261003_033822)", "Helpful AI Assistant (20261003_033823)" | Chat list, Buddy modal picker | Timestamps inside titles, and duplicate titles. (XP-14) | Title from the first user message ("Ask Captain Morrow about the voyage"), falling back to **"Chat · Oct 2, 8:38 PM"** |
| `1fbba07d`, `LicenseRef-User-Supplied`, `bundled` | Buddy modal | Internal metadata in user-facing copy. (XP-14, CO-03) | Hide behind an info icon. Licence text goes in Details |
| "Server" on every chat row, plus a "Server (49)" tab | Chat history | Redundant. (CS-08) | Show the source only when sources are mixed, as an icon (cloud or device) |
| "Forked conversation" | New branch title | Says nothing about where it came from. (CM-12) | **"&lt;parent title&gt;: branch from message 5"** |
| "10/2/2026, 8:33:50 PM" / "Updated 6 minutes ago" / "Saved 8m ago" / "8:38 PM" | Notes rows, chat rows, status, messages | Four date styles; seconds precision. (XP-13) | One helper: **8m ago**, **Yesterday**, **Sep 17** in lists. Absolute date and time without seconds in tooltips. Month grouping in local time (NL-09) |
| "Save & new" (visible) / "Save and start another note" (accessible name); "Context rail" / "Restore context sidechannel" | Notes header, /chat rails | The accessible name doesn't start with the visible label. (AX-17) | The accessible name starts with the visible words: "Save & new note", "Context panel" |

### 10.8 Copy patterns and the lint list

**Templates.**
- **Error:** "&lt;What failed&gt;. &lt;Likely cause in user terms&gt;." + [Primary fix] [Secondary] [Details]. Example: "Couldn't save this note. You're offline. We'll retry when you're back." [Retry now]
- **Status chip:** a state word, plus where, plus when: "Saved to server · 2m ago", "Saved on this device", "Saving…", "Conflict", "Offline".
- **Destructive confirmation:** a verb that names the outcome, the object's name, and the way back. "Move 'Thesis outline' to Trash? You can restore it from Trash." [Move to Trash] [Cancel]. Prefer Undo over a confirmation when the action can be reversed.
- **Empty state:** what this place is for, plus one primary action, plus at most one learning link. "No chats yet. Ask anything to start." [New chat]
- **Buttons:** a verb plus an object, in sentence case ("Save to Notes", "Move to folder…"). The ellipsis means "asks for more input".
- **Icon-only buttons:** always a tooltip and an accessible name that match. Add a visible label when the panel is 400px wide or more.
- **Disabled controls:** a tooltip with the reason and the fix.

**Banned in user-facing copy** (enforced by the Q14 lint). Each term may appear in tooltips, "Learn more" or Details where noted:
- **Object synonyms:** *conversation, session, thread* (when naming the object), *keyword(s)* (meaning tags), *Temp, Private* (meaning temporary chat), *Buddy, Persona* (meaning character).
- **Engineering terms:** *turn, path, interpretation, overlay, tracked, cockpit, rail, sidechannel, checkpoint, scope, route, composition, context stack*.
- **Brand and protocol terms:** *OpenUI, MCP* (tooltip only), *RAG* (tooltip only).
- **Unverified health:** *Healthy* (unless backed by a probe).
- **Raw strings:** any UUID pattern (`[0-9a-f]{8}-[0-9a-f]{4}-`), `snake_case` tokens, `provider:model` IDs, and raw HTTP status codes or URLs.

---

## 11. Suggested remediation roadmap

The roadmap has five waves. Each wave groups issues that share a root cause, so one fix clears several of them. Waves can overlap. The P1 accessibility issues in Stage 5 (AX-03, AX-04, AX-05) should start alongside Stages 2-3 rather than wait. The improvement items in §8 (Q*, N*, S*) slot in where their **Depends on** lines allow (§8.4).

**Cross-cutting enablers.** Start these in Stage 1, because every later stage relies on them:
- Contract tests against the real FastAPI app (N28). They cover notes list paging, wikilink syntax, `serverMessageId`, prompts and default provider.
- One Playwright smoke test per Notes editing mode.
- A Playground integration harness that mounts a `HistorySelectionProvider`.
- A request budget in CI (Q15).

### Stage 1: Stop data loss and false "saved" claims

**Goal.** No user action silently loses, corrupts or overwrites data. No status message claims a save, delete or restore that didn't happen.

**Issue IDs.**
- Notes save pipeline: NE-01, NS-01, NS-N1, NS-03, NS-N2, NS-05, NS-02
- Notes list contract and bulk safety: NL-01, NL-02, NL-03
- Chat persistence: CS-03, XS-05, CS-04, CS-N2
- Extension conversation integrity: XS-01, XS-07, XP-08

**Dependencies.**
- **NL-01 before NL-02.** Both come from the same list-API mismatch (`page`/`results_per_page` → `limit`/`offset`; `total_items` → `total`). Fix the shared list client first. Export then pages correctly, and NS-04, NL-09, NL-11 and wikilink autocomplete (NE-02) benefit immediately.
- **Notes save fixes ship together.** Unify NS-01, NS-N1, NS-03, NS-N2, NS-05 and NS-02 behind one save state machine (debounce, flush on unmount and route change, retry policy by status code, conflict and offline). Patching each one separately recreates the contradictions.
- **The chat-saving false claim has two parts.** The CS-03 and XS-05 copy fix ("Saved on this device") can ship on day one. The real server promotion belongs to the Stage 2 controller workstream.
- **One root cause for the extension issues.** XS-01, XS-07 and XP-08 all come from the missing reconciliation between three copies of a conversation. A minimal fix opens a new tab rather than overwriting, labels the action "Close tab" or deletes for real, and refreshes on focus. That fix comes before the longer S2 work in §8.

**Success criteria.**
- WYSIWYG typing produces text in the right order in a real-browser test.
- Navigating away 1 s after typing persists the edit, or prompts.
- A two-tab conflict test never overwrites the other tab's change.
- Exporting a 500-note library yields exactly 500 unique notes in no more than ⌈500/page size⌉ list requests, with a Cancel.
- Bulk "Add tags" preserves existing tags and offers Undo.
- Reloading mid-reply keeps the question.
- Restore from Trash returns every message.
- Every "Saved", "Deleted" and "Restored" string is rendered only after a server acknowledgement.

### Stage 2: Repair broken core flows

**Goal.** The primary actions on each page work every time, and when they fail they say why and offer a way forward.

**Issue IDs.**
- History-selection workstream: CS-01, CS-05, CM-01, CM-N2, CC-04, CS-N3, CM-14
- Failure and stop recovery: CC-02, CM-N1, CM-02, CM-06, CM-03, CC-N2
- Finding chats: CS-02, CS-N1, CS-06, CS-07
- Libraries and links: CC-05, XP-02, NE-02, NE-04, NL-N1

**Dependencies.**
- Build the Playground integration harness first.
- **CS-01 before CS-05.** Clear conversation's fallback and the user's natural "New chat" workaround both depend on a clean reset. The side panel's reset (`routes/sidepanel-chat.tsx:1220`) is the reference.
- **CC-02 first among the failure issues.** CC-02 defines the one recovery panel (cause, Retry, Switch model, keep partial output) that CM-N1, CM-02, CM-06 and CS-04 reuse.
- **XP-02 before N1** (Answer → Notes with provenance).
- **NE-02 before N19 and the NL-11 graph edges.** Align on one wikilink syntax server-side.

**Success criteria.**
- After New chat or Clear, the request payload contains zero prior messages.
- Regenerate, Continue and Edit → "Save & Send" pass end to end on a server chat.
- A 60 s-to-first-token model completes.
- Every failed or stopped turn shows its cause and a Retry.
- The Prompt picker lists the 11 server prompts.
- Save to Notes is available on all 5 sampled server chats.
- `[[Title]]` links click through and produce backlinks.
- Print produces a PDF.
- Selecting two tags narrows the results.

### Stage 3: First-run clarity

**Goal.** Sam can send a first message, save a first note and understand the page without reading docs or changing settings.

**Issue IDs.**
- First send works: CC-01, XS-02, XS-17, CC-03
- Commands that don't misfire: CC-N1, XS-08
- Onboarding: CO-02, NO-02, NO-N1, NO-04
- Chat chrome and density: CO-03, CO-04, CO-05, CO-06, CO-01, CC-06, CC-07, CC-09, CC-11, CC-12, CM-05, CM-09, CM-10
- Notes views: NL-12, NL-13, NE-06, NE-08, NE-09
- Copy and status: XP-11, XP-13, XP-14, XP-15, XS-12, NS-06, NL-16, XS-11

**Dependencies.**
- **NO-N1 before NO-02.** Fix the runner first, because the tour can't complete until it does.
- **NO-01 before NO-02 positioning.** Constrain the page frame (NO-01, in Stage 4) before fixing tour positioning, or schedule it here.
- **Copy rollout.** Roll out the §10 glossary and copy lint (Q14) here. XP-13, XP-14 and XS-12 are mostly copy changes once the glossary exists.
- **Composer.** The composer restructure (N12) supersedes piecemeal fixes to CO-03, CO-06 and CC-07.

**Success criteria.**
- On a fresh install, the first send succeeds with no settings changes, and the model chip shows "Ready" only after a successful probe.
- The Notes tour reaches step N of N and is recorded as completed (e2e test).
- The composer shows at most one row of primary controls in Casual mode.
- The copy lint finds no engineering terms or raw IDs in user-facing strings.
- The side panel opens to Chat with a working model.

### Stage 4: Power-user efficiency and cross-surface continuity

**Goal.** Riley can browse, organize and move between notes and chats at library scale, on any surface, by keyboard.

**Issue IDs.**
- Notes at scale: NL-04, NL-05, NL-06, NL-07, NL-08, NL-09, NL-10, NL-11, NL-17, NS-04, NE-03, NE-10, NO-01, NO-03
- Chat at scale: CS-08, CS-09, CS-10, CS-11, CM-04, CM-11, CM-12, CC-08, CC-10
- Chat ↔ notes loop: XP-01, XP-03, XP-04, XS-13
- Addressability and navigation: XP-05, XP-06, XP-07, XP-10
- Surface parity: XP-09, XP-12, XP-16, XP-17, XP-18, XS-06, XS-10, XS-16

**Dependencies.**
- NL-01 (Stage 1) gates NL-10, NL-11, NS-04 and N16.
- XP-05 (addressable routes, N4) gates XP-06 (quick switcher, N5), XP-04 and N2.
- XP-02 (Stage 2) gates XP-03.
- XP-07 and XP-10 are best fixed through one shortcut registry (N21).
- XS-06, XP-12 and XP-18 converge on the one-conversation-model bets (S1 and S2 in §8.3). Ship the minimal parity fixes first.

**Success criteria.**
- A 1,000-note library shows the true total and pages through every note.
- Autosave sends no list refetch.
- Every note and chat has a URL that survives reload and Back/Forward.
- ⌘K finds notes and chats by title.
- "Chat about this note" exists.
- A saved answer links back to its exact message.
- Side-panel chats appear in the full page's history.
- Background polling stays within the CI request budget.

### Stage 5: Accessibility and polish

**Goal.** WCAG 2.2 AA conformance for the criteria in §7.3.6, and a consistent visual system on all three surfaces.

**Issue IDs.**
- Accessibility: AX-01, AX-02, AX-03, AX-04, AX-05, AX-06, AX-07, AX-08, AX-09, AX-10, AX-11, AX-12, AX-13, AX-14, AX-15, AX-16, AX-17, AX-18, AX-19
- Side-panel layout: XS-03, XS-04, XS-09, XS-14, XS-15, XS-18
- Rendering and visual polish: CM-07, CM-08, CM-13, CO-07, CO-N1, NE-05, NE-07, NE-N1, NL-14, NL-15, XP-19, XP-20

**Dependencies.**
- **Start the P1 items early.** AX-03, AX-04 and AX-05 are P1 and should run in parallel with Stages 2-3.
- **Colour and focus issues are token fixes.** AX-07, AX-08 and XP-20 come from the shared antd theme tokens, so fix them once.
- **Shared components.** AX-02, AX-06 and AX-12 need the shared MenuButton, Announcer and Combobox (N24). AX-10, NE-05, NE-07 and CM-08 need the single Markdown renderer (N26).
- **Regression gates.** Add axe and visual-regression gates in CI (N27) so the fixes stay fixed.

**Success criteria.**
- axe shows zero serious or critical violations on /notes, /chat and the side panel, in empty, populated and error states.
- Keyboard-only users can reach any note from the list in at most 3 Tab presses after "Skip to notes list".
- The transcript remains visible at 320×256 CSS px.
- Text contrast is at least 4.5:1 and focus indicators at least 3:1 in both themes.
- A VoiceOver and NVDA pass confirms the announcements.

### Quick wins (ship anytime)

These are small (S effort), self-contained fixes that don't depend on the stage work. Each one can go out in its own PR:
- **Notes:** NE-04 (Print), NE-05 (list markers), NE-07 (code-block staircase), NE-N1 (View code), NL-09 (Timeline timezone), NS-02 (untitled note save).
- **Chat:** CS-N1 (Trash filter lock-out), CS-10 (delete toast + Undo), CO-02 (tour button), CO-07 (scroll chevron), CM-07 (code copy), CM-08 (inline code), CC-12 (wand menu background), CC-N1 (slash commands run on Enter only).
- **Side panel:** XS-08 (Ctrl+E), XS-11 (offline Retry).
- **Accessibility:** AX-08 (focus ring token), AX-09 (focus return), AX-10 (table semantics), AX-17 (label in name), AX-18 (scrollable modal).
- **Extension options:** XP-19 (page title).

**Small P0 fixes.** Several P0s are also S effort. They belong to Stages 1-2, but each is small enough to ship on its own ahead of the stage work: NL-02 (export paging, best with NL-01), NS-N1 (conflict toast), CM-N1 (time-to-first-token timeout), XP-02 (keep `serverMessageId` on loaded messages), XS-07 (honest side-panel Delete and Rename) and CS-05 (Clear conversation, once CS-01's reset exists).

Also from §8.1: Q2 (pick a working model), Q3 (respect OS preferences), Q4 (local draft journal) and Q6 (code-block toolbar).

---

## Appendix A. Findings rejected during verification

No finding was rejected outright. Claims that verification narrowed are noted in each issue's entry in §§5-7.

| ID | Claim | Verdict | Why |
|---|---|---|---|
| — | none | — | — |

## Appendix B. Evidence index

**Screenshots.** About 1,300 screenshots (~154 MB) were captured during the live review run and are **not committed**. This report cites them as `shots/<reviewer-tag>/<file>.png`; a bare filename in a section entry refers to the folder of the reviewer who reported it, and paths such as `verify-NS/...` or `skeptic-CS/...` refer to the verification folders. The 63 screenshots cited in this report are viewable on the published review page linked from the PR; the full set is kept in the maintainer's local evidence bundle (`.worktrees/ux-review-notes-chat-dev/.ux-review-evidence/`).

| Folder(s) | Role | Screenshots |
|---|---|---|
| `ft-notes/`, `ft-chat/`, `ft-ext/` | First-time (Sam) walkthroughs: Notes, Chat, Extension | 87, 131, 78 |
| `pu-notes/`, `pu-chat/`, `pu-cross/`, `pu-ext/` | Power-user (Riley) walkthroughs: Notes, Chat, Cross-page, Extension | 68, 85, 75, 87 |
| `lens-a11y/` | WCAG 2.2 AA audit (also holds axe JSON) | 82 |
| `lens-visual/` | Visual design and consistency lens | 48 |
| `lens-states/` | Loading, empty, error, offline, conflict and performance lens | 73 |
| `verify-<AREA>/` (AX, CC, CM, CO, CS, NE, NL, NO, NS, XP, XS) | Adversarial verification for each area code | 407 |
| `skeptic-<AREA>/` (AX, CC, CM, CS, NE, NL, NS, XP, XS) | Independent skeptic re-check of severity ≥ 3 issues | 99 |
| folder root (`smoke-*.png`, `ext-*.png`) | Environment smoke checks | 4 |

**Scripts.** The review used a Playwright helper (`uxlib.mjs`: `open`, `openExtension`, `shot`, `dump`, `controls`), per-reviewer scripts, and a power-user seed script under `apps/tldw-frontend/scripts/ux-review/` in the review worktree. They are not committed: they were written as one-off probes. Turning the verified reproductions into committed regression tests is the first stage of the remediation plan (`Docs/superpowers/plans/2026-10-03-notes-chat-ux-remediation-plan.md`).

**Code maps and working data.** The first-pass code maps (`map_notes.md`, `map_chat.md`, `map_extension.md`), the seed report, and the raw reviewer outputs (`reviews.json`, `findings_flat.json`, including each reviewer's `coverage_gaps`) are in the evidence bundle under `wf1/`. The verified issue list is also committed in machine-readable form as `Docs/Design/2026-10-02-notes-chat-ux-review.issues.json`.
- Prior chat review (June 2026): `Docs/Design/2026-06-13-chat-page-uat-review.md`

**Test data left on the shared backend.** Reviewer-created items have titles prefixed with the reviewer tag (for example `[ft-notes]`, `[pu-chat]`, `[verify-NS]`). The exceptions are server-titled items: a "Snippet: Explain LoRA fine-tuning" note from pu-cross and a "Forked conversation" chat from pu-chat. pu-ext appended 4 `[pu-ext]` messages to the seeded chat "Prepare for a system design interview", which forked it. Seeded items were otherwise not modified.
