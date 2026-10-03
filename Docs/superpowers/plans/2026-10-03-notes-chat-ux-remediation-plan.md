# Notes & Chat UX Remediation Plan

**Created:** 2026-10-03
**Source review:** `Docs/Design/2026-10-02-notes-chat-ux-review.md` (PR #3100). Machine-readable list: `Docs/Design/2026-10-02-notes-chat-ux-review.issues.json`
**Tracking:** #3101 (epic). Defect groups #3102–#3124 (G01–G23); enhancement groups #3125–#3132 (E1–E8). Label: `ux-review-2026-10`
**Scope:** `/notes` and `/chat` on the WebUI, the extension options page and the extension side panel. Shared UI lives in `apps/packages/ui/src`; backend changes are called out per item.
**Goal:** Resolve all 151 verified issues (P0 22 · P1 9 · P2 97 · P3 23) and schedule all 54 improvement ideas (49 from the review plus 5 added by the 2026-10-03 enhancement audit). Order of work: stop data loss and false status first, then repair broken core flows, then first-run clarity, then power-user scale and cross-surface continuity, then accessibility and polish.

Each stage can ship on its own and has testable exit criteria. Within a stage, the groups (GitHub issues) bundle issues that share a root cause, and each group is cut into PR-sized slices listed in dependency order.

---

## How the work is organized

**Groups, not single issues.** Every verified issue (`NL-01`, `CS-01`, …) belongs to exactly one defect group, and every idea (`IDEA-xx`, labelled Q/N/S in report §8) belongs to exactly one enhancement group. Each group issue has a checklist, acceptance criteria, dependencies and per-item fix notes with root causes on `dev @ 86e287fee7`.

**Lanes run in parallel.** Stages set priority, not strict serialization. Three or four people can work concurrently:

| Lane | Groups | Notes |
|---|---|---|
| Notes | G01, G02, G09, G13, G15, G16 | G02 (list contract) gates G09 and G15 |
| Chat | G03, G05, G06, G07, G10, G11, G12, G17 | G05 (history-selection reset) gates most chat iteration fixes |
| Extension | G04, G19 | Shares the history-selection controller with Chat; test both surfaces |
| Cross-cutting | G08, G14, G18, G20 | G08 needs XP-02; G18's routes unlock ⌘K and "chat about this note" |
| Accessibility | G21, G22, G23 | P1 items (AX-03, AX-04, AX-05) start alongside Stages 2–3 |
| Platform | E1, E8 | E1 first; E8's glossary and shared primitives unblock G14 and G21–G23 |

**Sizing.** Effort codes from the review: S ≈ a day in one component; M ≈ a few days to two weeks; L ≈ multi-week, cross-cutting or backend + frontend.

| Stage | Groups | Items | P0 | P1 | P2 | P3 | S | M | L |
|---|---|---|---|---|---|---|---|---|---|
| 1 | G01–G04 | 21 | 12 | 0 | 9 | 0 | 6 | 13 | 2 |
| 2 | G05–G09 | 29 | 8 | 3 | 15 | 3 | 13 | 14 | 2 |
| 3 | G10–G14 | 41 | 1 | 3 | 28 | 9 | 24 | 17 | 0 |
| 4 | G15–G20 | 41 | 1 | 0 | 32 | 8 | 25 | 14 | 2 |
| 5 | G21–G23 | 19 | 0 | 3 | 13 | 3 | 12 | 7 | 0 |
| **All** | 23 groups | **151** | **22** | **9** | **97** | **23** | **80** | **65** | **6** |

Two P0s sit in later-stage groups: NE-04 (Print, G16) and CC-05 (server prompt library, G11). NE-04 ships early through the quick-win track, and CC-05 is pulled into Stage 2.

---

## Working agreements (definition of done for every PR)

1. **Red first.** Start each PR with a failing test that reproduces the verified issue: a Playwright test in the UX harness for UI behaviour, a contract test for client/server drift, or a Vitest test for pure logic. Name tests after the issue ID (for example `ne-01-wysiwyg-typing-order.spec.ts`).
2. **Fix the root cause the group names.** Shared root causes (the notes save pipeline, the list client, the history-selection controller, three copies of a conversation) get one fix, not per-symptom patches.
3. **Re-locate before fixing.** Line references are from `dev @ 86e287fee7`. Confirm the code path on current `dev` first.
4. **Verify live.** Before merge, re-run the relevant persona scenario in the UX harness against the seeded library and attach a screenshot to the PR.
5. **Copy follows the glossary.** User-facing strings follow report §10; no raw IDs, URLs or engineering terms.
6. **No new accessibility debt.** axe shows no new serious or critical violations on touched surfaces.
7. **Traceability.** PR titles name the group and issue IDs (for example `fix(notes): G01 NE-01 uncontrolled WYSIWYG editor`). Tick the item in the group issue. Because `dev` is not the default branch, close group issues manually after merge.

---

## Decisions (resolved 2026-10-03)

| # | Question | Affects | Decision |
|---|---|---|---|
| D1 | Should WebUI and side-panel chats save to the server by default? | G03, G04, G05, E2 | **Yes.** Chats save to the server by default when connected; "Temporary chat" stays local. Honest labels ("Saved on this device" until the server acknowledges) ship first in Stage 1; server-by-default lands in Stage 2. |
| D2 | Canonical wikilink syntax: `[[Title]]` or `[[id:UUID]]`? | G09, E7 | **Both.** Users write `[[Title]]`; the server resolves titles to ids, and the client, renderer and graph parser accept both forms. |
| D3 | Permanent delete and Empty Trash for notes (NL-06)? | G15 (backend) | **Yes.** Add hard delete and Empty Trash, with optional retention-based auto-purge. |
| D4 | Compare mode (CM-05): enable it or remove its entry points? | G11, E5 | **Hide until verified.** Remove the four entry points now; re-enable behind the flag once Compare passes its own verification. |
| D5 | Interim WYSIWYG mode while NE-01 is open? | G01 | **Hide until fixed.** Hide the WYSIWYG toggle in the first Stage 1 PR and restore it with the NE-01 fix. |
| D6 | A full Notes surface in the side panel (N3)? | G08, E3 | **Decide after the D7 spike.** Ship the quick-save improvements (XS-13) now. |
| D7 | When to run the design spike for one conversation model (S1/S2) and server-side, resumable generation (S3)? | E2 | **During Stage 2**, so Stage 4 builds on a decided model. The spike's output also settles D6. |

### Composer and switcher decisions (resolved 2026-10-03)

The enhancement audit found design conflicts between reviewers that consolidation had settled without discussion. The owner decided them:

| # | Question | Affects | Decision |
|---|---|---|---|
| D8 | Which shortcut opens the quick switcher? | E4 (IDEA-01, N5) | **Cmd/Ctrl+K is one switcher for content and commands**, with a ">" prefix for commands. Cmd/Ctrl+P and Cmd/Ctrl+O open the same switcher pre-filtered to notes and chats. |
| D9 | What does Pro mode add to the composer? | E5 (IDEA-11, N12) | **Pro mode adds a persistent right panel** instead of more toolbar buttons; the panel absorbs today's cockpit rails. |
| D10 | How many rows does the Casual composer use? | E5 (IDEA-11, N12; IDEA-12, N13) | **Two rows on desktop** (input + Send; then attach / Knowledge / Web / Model / More). **One row on phones and the narrow side panel** (+, input, model chip, Send), with the bottom-sheet composer (N13). |
| D11 | What does Cmd/Ctrl+Enter do on a switcher result? | E4 (IDEA-01, N5) | **Cmd/Ctrl+Enter opens a result beside the current page**: notes in the Notes Dock now, chats in split view once S4 lands. **Shift+Enter opens a new browser tab.** |

---|---|---|---|
| D1 | Should WebUI and side-panel chats save to the server by default, or stay local-first with explicit promotion? | G03, G04, E2 | Save to the server by default when connected; keep "Temporary chat" local. Ship honest labels ("Saved on this device") immediately either way. |
| D2 | Canonical wikilink syntax: `[[Title]]` or `[[id:UUID]]`? | G09, E7 | Users write `[[Title]]`; the server resolves titles to ids and the graph parser accepts both forms. |
| D3 | Permanent delete and Empty Trash for notes (NL-06)? | G15 (backend) | Add hard delete and Empty Trash, plus optional retention-based auto-purge. |
| D4 | Compare mode (CM-05): enable it or remove its entry points? | G11, E5 | Remove the four entry points until Compare passes its own verification, then enable behind the flag. |
| D5 | Interim WYSIWYG mode while NE-01 is open? | G01 | Hide the toggle (or label it Beta) in the first PR of Stage 1. |
| D6 | Notes surface in the side panel (N3, XS-13)? | G08, E3 | Start with quick-save improvements (tags, open note); decide on a full mini-surface after E2's design spike. |
| D7 | Server-side, resumable generation (S3) and one conversation model (S1/S2) | E2 | Run a design spike during Stage 2 so Stage 4 work can start on a decided model. |

---

## Stage 0: Enablers — #3125 (E1)

**Goal:** The test and CI scaffolding that every later stage uses to prove its fixes.

**Work items (in order):**
1. **UX regression harness.** Commit a Playwright harness under `apps/tldw-frontend/e2e/` that launches the WebUI and the built extension (options page and side panel), seeds auth, and seeds a deterministic power-user library through the API (150+ notes with folders, tags, wikilinks, long notes and checklists; 40+ chats including forks and character chats; prompts; characters). The review's helper (`uxlib.mjs`) and seed logic are the starting point; they live only in the review worktree and need cleanup before committing.
2. **Deterministic mock LLM fixture.** An OpenAI-compatible mock that streams markdown and can be told to delay the first token, truncate (`finish_reason: length`), fail with provider errors, or hang. Needed for G06 and G05 tests.
3. **Contract tests.** Exercise the client calls the review found drifting against the real FastAPI app: notes list paging, sort and total; wikilink syntax; `serverMessageId` on loaded messages; prompt listing; default provider. Prefer generated types (`apps/tldw-frontend/scripts/generate-api-types.mjs`) for notes and chat calls so parameter drift fails at compile time.
4. **Playground integration harness.** A Vitest harness that mounts `Playground` with a real `HistorySelectionProvider`, for the G05 and G06 flows.
5. **CI gates.** An axe job on `/notes`, `/chat` and the side panel (empty, populated, error states), and a request budget (calls in the first 10 s; duplicate GETs within 2 s) on seeded `/notes` and `/chat`. Start both as warnings and switch to failing once their baselines are clean.

**Success criteria:** Each P0 has a committed failing reproduction before its fix lands. Contract tests run in CI against the real app. axe and request-budget jobs report on every PR.
**Status:** Not Started

---

## Stage 1: Stop data loss and false "saved" claims — #3102, #3103, #3104, #3105

**Goal:** No user action silently loses, corrupts or overwrites data, and no status message claims a save, delete or restore that didn't happen.

**PR slices (in order):**
1. **G01 (#3102) — hide WYSIWYG until fixed (D5).** Hide the WYSIWYG toggle; slice 4 restores it with the NE-01 fix. S.
2. **G02 (#3103) — fix the notes list contract (NL-01).** Send `limit`/`offset`, read `pagination.total`, and sort server-side (or sort across the full result). Add a contract test. M. *Unblocks NL-02, NS-04, G09 and G15.*
3. **G02 — export pages correctly (NL-02).** Terminate on the true total, show progress with Cancel, or switch to the server export endpoints. S.
4. **G01 — fix WYSIWYG typing (NE-01).** Make the `contentEditable` uncontrolled and write `innerHTML` only on explicit external revisions. Add an e2e that types at human speed. Re-enable the toggle. M.
5. **G01 — notes save state machine (NS-01, NS-N1, NS-03, NS-N2, NS-05, NS-02, NS-06).** One state machine for debounce, flush on unmount and route change (plus a `beforeunload` guard), retry by status code (no retry on 400/409), a conflict flow with keep mine / take theirs / copy my text, the offline queue for network errors, server auto-title for untitled notes, and one acknowledged status. NS-N1 (S) can ship first as a stopgap. M overall.
6. **G02 — bulk "Add tags" (NL-03)** that merges keywords instead of replacing them, with Undo. **NS-04:** update the edited row in place instead of refetching the whole list. M.
7. **G03 (#3104) — honest persistence labels (CS-03, XS-05).** Derive labels from acknowledged state: "Saved on this device" until the server acknowledges a save. S, day one. Server-by-default saving (D1) follows in Stage 2.
8. **G03 — interrupted replies (CS-04, CS-N3).** Persist the user turn before streaming, keep partial output marked "Interrupted" with Retry, add a leave guard, and never create empty server chats. L.
9. **G03 — restore from Trash (CS-N2).** Restore every message; a backend check may be needed. M.
10. **G04 (#3105) — side-panel integrity (XS-01, XS-07, XP-08, XS-06).** Opening a past chat opens a new tab; Delete and Rename act on the real chat (with Undo) or are relabelled; tabs refresh on focus and before send, with stale-leaf detection; recents appear without searching. M–L.

**Success criteria:**
- WYSIWYG typing produces text in the right order (e2e).
- Navigating away 1 s after typing persists the edit or prompts.
- A two-tab conflict test never overwrites the other tab.
- A 500-note export yields 500 unique notes in ⌈500/page size⌉ requests, with Cancel.
- Bulk tagging preserves existing tags and offers Undo.
- Reloading mid-reply keeps the question.
- Restore returns every message.
- Every "Saved", "Deleted" and "Restored" string renders only from an acknowledged state change.

**Tests:** Harness e2e for NE-01, NS-01, NS-N1/NS-03 (two pages), NL-02 (seeded 500 notes), CS-04 (reload mid-stream), XS-01 (two side-panel tabs); contract test for NL-01; Vitest for the save state machine transitions.
**Status:** Not Started

---

## Stage 2: Repair broken core flows — #3106, #3107, #3108, #3109, #3110

**Goal:** The primary actions on each page work every time, and when they fail they say why and offer a way forward.

**PR slices (in order):**
1. **G05 (#3106) — reset history selection (CS-01 then CS-05).** New chat and Clear reset the `HistorySelection` controller and the persisted session; Send stays disabled until the controller is idle. Add a defensive check in `useChatActions`. Cover the header, sidebar "+" and Ctrl+Shift+U paths. M + S.
2. **G06 (#3107) — one recovery panel (CC-02).** Plain-language cause, Retry, Switch model, and partial output kept. M. **CM-N1 (S, P0):** apply the 120 s startup timeout to time-to-first-token. Can ship before CC-02.
3. **G05 — iterate on answers (CM-01).** Route regenerate, continue and edit-and-send through the controller for local and server chats. Then CM-N2 (variants rendered as alternatives of one turn), CC-04 (show history-path controls only while reviewing), CM-12 (linked branches with a breadcrumb) and CM-14 (no flash on first send). L overall.
4. **G06 — stop, waits and truncation (CM-02, CM-06, CM-03, CC-N2).** Stop keeps the message and partial answer, marked "Stopped"; long waits show elapsed time; `finish_reason: length` marks the answer incomplete with Continue; provider errors name the provider and update the model status.
5. **G07 (#3108) — finding past chats (CS-02).** History visible by default and remembered, local and server chats listed, content search. Then the S items: CS-N1, CS-06, CS-07, CS-08, CS-09, CS-10, CS-11.
6. **G08 (#3109) — Save to Notes everywhere (XP-02, S, P0).** Keep `serverMessageId` when formatting loaded history. Then XP-01 (unsynced chats), XP-03 (provenance: title, tags, "Open note", readable back-link), XP-04 ("Chat about this note"), XS-13 (side-panel quick-save).
7. **G11 (#3112) — server prompt library (CC-05, P0).** Pulled forward from Stage 3: the Prompt picker lists server prompts. M.
8. **G09 (#3110) — wikilinks (NE-02).** Per D2, accept both `[[Title]]` and `[[id:UUID]]`: render and follow links, create backlinks and graph edges, and autocomplete across the whole library. Then NE-10 and, once G02 has landed, NL-11.
9. **G03 (#3104) + G05 (#3106) — save chats to the server by default (D1).** When connected, create the server conversation on first send and write each turn; "Temporary chat" stays local; local-only chats from before the change can be promoted with one action. Builds on the G05 controller reset. M.
10. **E2 (#3126) — design spike (D7).** One conversation model across the side panel, extension full page and WebUI (S1/S2), and server-side resumable generation (S3). Output: a design doc with the chosen model, migration path, and the D6 decision on a side-panel Notes surface. Timebox: one to two weeks.

**Success criteria:**
- After New chat or Clear, the request contains zero prior messages.
- Regenerate, Continue and Edit → Save & Send work on server chats.
- A 60 s-to-first-token model completes.
- Every failed or stopped turn shows its cause and a Retry.
- The Prompt picker lists all server prompts.
- Save to Notes is available on all sampled server chats.
- `[[Title]]` links click through and create backlinks.
- Selecting two tags narrows the results (G15 NL-N1 may ride along here).
- A new chat sent while connected appears in server history (D1).

**Tests:** Playground integration tests (E1.4) for every reset and iteration path; server-history e2e for D1; mock-LLM fixtures for slow first token, truncation and provider errors; e2e for XP-02 on seeded server chats; contract test for prompts.
**Status:** Not Started

---

## Stage 3: First-run clarity — #3111, #3112, #3113, #3114, #3115 (+ E5 #3129, E6 #3130)

**Goal:** A first-time user can send a first message, save a first note and understand both pages without docs or settings changes.

**PR slices (in order):**
1. **G10 (#3111) — first send works (CC-01, CC-03, XS-02, XS-17, CC-06).** Use the server's default provider, mark the model "Ready" only after a probe or reply, remove the false offline state, give the side panel a default model and a labelled selector and open it on Chat, and use one model label format. Pairs with E6 Q2.
2. **G11 (#3112) — commands that don't misfire (CC-N1, XS-08; S, P1).** Slash commands run on Enter or selection; Ctrl+E shows its mode or is removed. Then CM-05 (hide Compare's entry points until verified, per D4), CC-11, CC-12, CC-10, CC-08. Fold CC-07 and CC-09 into the composer restructure.
3. **E5 (#3129) N12 + G12 (#3113) — composer and chrome.** Restructure the composer per D9/D10: Casual uses two rows on desktop and one row on phones and the narrow side panel; Pro adds a persistent right panel that absorbs the cockpit rails instead of more toolbar buttons. This resolves CO-03, CO-06, CC-07 and CC-09 together. Then CO-04 and CO-05 (rails auto-collapse; no 2x2 grid on tablets), CO-01 and CO-N1 with N13 (mobile bottom sheet), CO-02 (mount the help-modal host; S, can ship anytime) and CO-07.
4. **G13 (#3114) — Notes layout and onboarding.** NO-01 first (add `/notes` to the viewport-constrained routes), then NO-N1 (the tour runner skips missing targets), NO-02 (tour positioning), NO-04 (empty state and editor mutually exclusive), NO-03 (tablet and phone), NL-12, NL-13, NL-14, NE-08, NE-09. Pairs with E6 Q10 and Q11.
5. **E8 (#3132) Q14 + G14 (#3115) — words and tokens.** Adopt the §10 glossary and add a copy lint. Then XP-13, XP-14, XS-12, XP-11, XP-17 (one destructive-action pattern), XP-15 and XP-20 (with AX-07 and AX-08 in G23, as one theme-token change), and XP-19.

**Success criteria:**
- On a fresh install the first send succeeds with no settings changes.
- The model chip says "Ready" only after a successful probe.
- The Notes tour reaches its last step and records completion (e2e).
- Casual mode shows one row of primary composer controls.
- The copy lint finds no engineering terms or raw IDs in user-facing strings.
- The side panel opens on Chat with a working model.

**Tests:** First-run e2e on fresh databases (WebUI and side panel); tour completion e2e; copy-lint CI job; viewport e2e at 390, 768, 1024 and 1440 px.
**Status:** Not Started

---

## Stage 4: Power-user scale and cross-surface continuity — #3116–#3121 (+ E2 #3126, E3 #3127, E4 #3128, E7 #3131)

**Goal:** A power user can browse, organize and move between notes and chats at library scale, on any surface, by keyboard.

**PR slices (in order):**
1. **G18 (#3119) — addresses and navigation (XP-05 first).** Notes and chats get URLs that survive reload and Back/Forward, and page titles name the item. Then XP-06 (⌘K finds and creates notes and chats; with E4 N5 per D8/D11: one ⌘K switcher with a ">" command prefix, ⌘P/⌘O aliases filtered to notes and chats, ⌘Enter opens beside, ⇧Enter opens a new tab), and XP-07 and XP-10 through one shortcut registry (E4 N21).
2. **G15 (#3116) — notes at scale.** NL-04 (row density and snippets), NL-07 (keep server ranking, show the match), NL-N1 (AND, exact tag matching), NL-08, NL-09 (local-date Timeline; S, quick win), NL-10 (folders and correct counts; with E7 N18), NL-05 (select-all, progress, Undo), NL-06 (permanent delete and Empty Trash with optional auto-purge; D3, backend), NL-15, NL-16, NL-17.
3. **G16 (#3117) — editor and preview.** NE-04 (Print) ships early through the quick-win track. Then NE-05, NE-07, NE-N1, NE-06 (list continuation; checklist and numbered-list buttons), and NE-03 (collapsible TOC; with E7 N20).
4. **G17 (#3118) — messages.** CM-04 (persist model and generation metadata server-side; backend), CM-07 and CM-08 (with E8 N26, one Markdown renderer), CM-09 (actions without hover), CM-10, CM-11 (scroll to your message on send), CM-13.
5. **G19 (#3120) — side-panel layout and parity.** Layout first: XS-03, XS-04, XS-09, XS-15, XS-14, XS-18. Then states: XS-10 (Stop while pending), XS-11 (Retry when offline), XS-16 (one parameter editor). Then hand-offs and parity: XP-09 (with E2 N6), XP-12, XP-18.
6. **G20 (#3121) — polling (XP-16).** Pause polling when hidden or disabled and coalesce duplicate GETs, enforced by the E1 request budget.
7. **Enhancements.** E3 (capture ↔ converse: N1 answer → note with append mode, N2 two-way links, N7 context strip, Q5 selection toolbar, Q6 code-block toolbar, Q13 adaptive knowledge search). E4 (N22 message navigation, Q9 single Tab stop). E7 (N16 browsing beyond 100, N17 smart views, N15 version history, Q4 local draft journal, Q8 row menus, N19 wikilinks with click-to-create). E2 and the S-bets after their design spikes (D7).

**Success criteria:**
- A 1,000-note library shows the true total and pages through every note.
- Autosave sends no list refetch.
- Every note and chat has a URL that survives reload and Back/Forward.
- ⌘K finds notes and chats by title.
- A saved answer links back to its exact message.
- Side-panel chats appear in the full page's history.
- Background polling stays within the CI request budget.

**Tests:** Seeded 1,000-note e2e; URL and deep-link e2e; ⌘K e2e; side-panel e2e at 360 and 420 px; contract test for model metadata.
**Status:** Not Started

---

## Stage 5: Accessibility and polish — #3122, #3123, #3124 (+ E8 #3132)

**Goal:** WCAG 2.2 AA for the criteria in report §7.3, and one consistent visual system across the three surfaces.

**Start early:** AX-03, AX-05 (G21) and AX-04 (G23) are P1. Run them alongside Stages 2–3.

**PR slices (in order):**
1. **G21 (#3122) — keyboard and focus.** AX-03 (reach any note in ≤3 Tabs; list → editor binding), AX-05 (`inert` off-canvas list; Esc and focus trap when open), AX-01 and AX-02 (with E8 N24 MenuButton and Combobox), AX-09 (focus return), AX-18.
2. **G23 (#3124) — visual.** AX-04 (transcript visible at 320x256 CSS px; capped composer), then AX-07, AX-08 and XP-20 as one token change, then AX-15, AX-16, AX-19 (with E6 Q3).
3. **G22 (#3123) — screen readers.** AX-06 (one Announcer for toast outcomes), AX-10 (keep `<table>` semantics; with N26), AX-11 (narrow live region; with E8 N23), AX-12 (comboboxes), AX-13, AX-14, AX-17.
4. **Gates.** Switch the E1 axe gate to failing, and add visual-regression baselines (E1 N27).

**Success criteria:**
- axe shows zero serious or critical violations on `/notes`, `/chat` and the side panel in empty, populated and error states.
- Any note is reachable in ≤3 Tabs after "Skip to notes list".
- Text contrast is ≥4.5:1 and focus indicators ≥3:1 in both themes.
- A VoiceOver and NVDA pass confirms announcements.

**Status:** Not Started

---

## Stage 6: Strategic bets — E2 #3126, E3 #3127, E7 #3131, E8 #3132

**Goal:** Deliver the larger redesigns the review proposed, each starting from a design spike so the build follows a decided model.

**Work items (in order):**
1. **S1, S2, S3 (E2) — build on the D7 spike from Stage 2.** One searchable chat history across devices and surfaces, one conversation model across the side panel, extension full page and WebUI, and server-side resumable generation. N10 (branch map) and N3 (Notes inside the side panel, per D6) follow the spike's decisions.
2. **S4 (E3) — side-by-side research mode (chat + note).** Design spike after N2 two-way links and XP-05 routes land, then build.
3. **S5 (E7) — one block editor with Markdown shortcuts and a slash menu.** Design spike after NE-01 and N20 land, then build; keeps Markdown as the storage format.
4. **S6 (E8) — shared "object page" layout for Notes and Chat.** Design spike after N4 routes and N12 composer restructure, then migrate both pages.
5. **N31 (E5) — Compare models in the side panel**, once Compare passes verification (D4).

**Success criteria:** Each bet has an approved design doc before build; each build ships behind a flag with the UX harness scenarios passing on all three surfaces.
**Status:** Not Started

---

## Quick-win track (ship anytime)

Small (S), self-contained fixes that don't wait for their stage. Good first issues; each is one PR. Tick the item in its group issue.

| Area | Items (group) |
|---|---|
| Notes | NE-04 Print (G16, **P0**), NE-05 list markers (G16), NE-07 code staircase (G16), NE-N1 View code (G16), NL-09 Timeline timezone (G15), NS-02 untitled save (G01) |
| Chat | CS-N1 Trash filter lock-out (G07, P1), CS-10 delete toast + Undo (G07), CO-02 tour button (G12), CO-07 scroll chevron (G12), CM-07 code copy (G17), CM-08 inline code (G17), CC-12 wand menu background (G11), CC-N1 slash commands on Enter (G11, P1) |
| Side panel | XS-08 Ctrl+E (G11, P1), XS-11 offline Retry (G19) |
| Accessibility | AX-08 focus ring token (G23), AX-09 focus return (G21), AX-10 table semantics (G22), AX-17 label in name (G22), AX-18 scrollable modal (G21) |
| Extension options | XP-19 page title (G14) |
| Small P0s | NL-02 export (G02, with NL-01), NS-N1 conflict toast (G01), CM-N1 TTFT timeout (G06), XP-02 `serverMessageId` (G08), XS-07 honest Delete/Rename (G04), CS-05 Clear (G05, after CS-01) |

Enhancement quick wins from §8.1: Q2 working model on first run (E6), Q3 OS preferences (E6), Q4 local draft journal (E7), Q6 code-block toolbar (E3).

---

## Enhancement sequencing

| Group | Issue | Starts | Depends on |
|---|---|---|---|
| E1 Quality gates | #3125 | Stage 0 | — |
| E5 Composer, message states, branching | #3129 | Stage 3 (N12 with G12) | G05, G06, G11 |
| E6 First-run experience | #3130 | Stage 3 | G10, G13 |
| E8 Design-system and accessibility foundations | #3132 | Stage 3 (Q14), continues Stage 5 | — |
| E3 Capture ↔ converse loop | #3127 | Stage 4 | G08 (XP-02), G18 (XP-05) |
| E4 Navigation and keyboard | #3128 | Stage 4 | G18, G21 |
| E7 Notes at scale | #3131 | Stage 4 | G01, G02, G09, G15 |
| E2 One conversation model | #3126 | Design spike during Stage 2 (D7), build in Stage 4 | G03, G04, D1 |

### Enhancement schedule (every idea)

Every idea has an explicit slot. Codes match report §8 (Q16, Q17 and N29–N31 come from the §8.5 audit addendum); details restored by the audit are listed on each group issue.

| Code | Idea | Group | Stage | Slot | Depends on |
|---|---|---|---|---|---|
| Q1 | Example prompts in the chat empty state | E6 #3130 | 3 | with G10 / G12 (chat empty state) | CO-02 |
| Q2 | First-run 'pick a model that works' handshake | E6 #3130 | 3 | with G10 slice (first send works) | CC-01 |
| Q3 | Respect OS preferences by default (theme, reduced motion, contrast) | E6 #3130 | Quick win | anytime; pairs with AX-19 (G23) | — |
| Q4 | Local draft journal with restore banner | E7 #3131 | 1 → 2 | follow-on to the G01 save state machine (local draft journal) | G01 slice 5 |
| Q5 | Selection mini-toolbar inside answers | E3 #3127 | 4 | selection mini-toolbar inside answers | N1 |
| Q6 | Code block toolbar with copy, wrap, save to note and open in workspace | E3 #3127 | Quick win | anytime; with CM-07 (G17) | — |
| Q7 | Density, grouping and filters for the chat history list | E5 #3129 | 2 | with G07 (CS-08 density, date groups, filters) | CS-02 |
| Q8 | Row context menu and inline row actions in the notes list | E7 #3131 | 4 | with G15 (row context menu) | — |
| Q9 | Reduce Tab burden in the notes list | E4 #3128 | 2–3 | with AX-03 (G21, P1 — started early) | — |
| Q10 | Progressive disclosure of advanced Notes views and AI actions | E6 #3130 | 3 | with G13 (Views and AI actions) | NL-12 |
| Q11 | Task-based first-run checklist instead of an area tour (Notes) | E6 #3130 | 3 | with G13 (replaces the area tour once NO-N1 lands) | NO-N1 |
| Q12 | Three-step sidepanel first-run coachmarks | E6 #3130 | 3 | with G10 (XS-17 side panel opens on Chat) | XS-02, XS-17 |
| Q13 | Adaptive knowledge search for servers without embeddings | E3 #3127 | 4 | adaptive knowledge search defaults | — |
| Q14 | Terminology glossary and copy lint | E8 #3132 | 3 | with G14 (glossary + copy lint) | — |
| Q15 | Performance and request budget in CI | E1 #3125 | 0 | Stage 0 item 5 (request budget) | — |
| Q16 | Configurable side-panel start page | E6 #3130 | 3 | with XS-17 (G10) | XS-17 |
| Q17 | Ctrl/Cmd+Enter sends when "Send on Enter" is off | E4 #3128 | 3 | with G11 (verify, then fix) | — |
| N1 | Answer → Notes with smart defaults, append mode and provenance | E3 #3127 | 4 | with G08 follow-up (answer → note with append mode and provenance) | XP-02, XP-03 |
| N2 | Bidirectional chat↔note backlinks | E3 #3127 | 2 → 4 | "Chat about this note" in Stage 2 (XP-04); two-way backlinks in Stage 4 | XP-04, XP-05 |
| N3 | Notes mini-surface in the side panel | E3 #3127 | after D7 | decided by the D7 spike (D6) | D6, D7 |
| N4 | Addressable routes and per-entity titles everywhere | E4 #3128 | 4 | G18 slice 1 (XP-05 addressable routes and per-item titles) | — |
| N5 | Universal Cmd/Ctrl+K quick switcher for notes, chats, prompts, characters and models | E4 #3128 | 4 | with G18 (XP-06 universal quick switcher) | N4 |
| N6 | Single 'Open in full page' hand-off with round-trip | E2 #3126 | 4 | with G19 (XP-09 single hand-off with a way back) | XP-09 |
| N7 | Visible context strip above the composer (this page, notes, system prompt, model) | E3 #3127 | 4 | visible context strip above the composer | N12 |
| N8 | Stopped, interrupted and variant states as first-class message states | E5 #3129 | 2 | with G06 (Stopped / Interrupted / Variant as real message states) | CC-02, CM-N2 |
| N9 | 'Regenerate with…' and a per-message model chip | E5 #3129 | 4 | with G17 (per-message model chip after CM-04 persists model metadata) | CM-04 |
| N10 | Branch map using the server /conversations/{id}/tree endpoint | E5 #3129 | 4 | after D7 spike (branch map on the server tree endpoint) | CM-12, D7 |
| N11 | Honest pending-send queue (offline/connecting) | E2 #3126 | 3 | with G10 / G19 (honest pending-send queue: CC-03, XS-10) | CC-03 |
| N12 | Composer restructure: one row of intent, Casual/Pro governs density | E5 #3129 | 3 | with G12 (composer restructure; absorbs CO-03, CO-06, CC-07, CC-09) | G11 slice 2 |
| N13 | Mobile composer as a bottom sheet | E5 #3129 | 3 | with G12 (CO-01, CO-N1 mobile) | N12 |
| N14 | Single sync and status centre (one chip per surface) | E2 #3126 | 3 | with G14 (one sync/status chip per surface, after D1 lands) | D1, CS-03 |
| N15 | Note version history with diff and restore | E7 #3131 | 4 | note version history with diff and restore | G01 |
| N16 | Notes browsing that scales past 100 (virtualized list, true totals, jump to month) | E7 #3131 | 4 | with G15 (browsing beyond 100, virtualized list) | NL-01 |
| N17 | Search query syntax and saved smart views | E7 #3131 | 4 | with G15 (search syntax and saved smart views) | NL-07 |
| N18 | Folder tree with drag-and-drop matching the server model | E7 #3131 | 4 | with G15 (NL-10 folder tree) | NL-01 |
| N19 | Wikilinks done right: click-to-create unresolved links, backlinks with context and unlinked mentions | E7 #3131 | 4 | after G09 (click-to-create, backlinks with context, unlinked mentions) | NE-02, D2 |
| N20 | Outline rail, focus mode and responsive ergonomics for long notes | E7 #3131 | 4 | with G16 (NE-03 outline rail, focus mode) | — |
| N21 | One platform-aware, context-aware keyboard shortcut registry and sheet | E4 #3128 | 4 | with G18 (XP-07, XP-10 one shortcut registry) | — |
| N22 | Keyboard message navigation in chat | E4 #3128 | 4 | keyboard navigation between chat messages | N21 |
| N23 | Screen-reader-friendly chat reading model | E8 #3132 | 5 | with G22 (AX-11 screen-reader transcript) | N24 |
| N24 | Shared accessible primitives: MenuButton, Combobox, Announcer | E8 #3132 | 3 → 5 | start in Stage 3 (needed by AX-01, AX-02, AX-12) | — |
| N25 | One typographic and control scale, enforced by lint | E8 #3132 | 3 | with G14 (type and control scale lint; XP-15, XP-20 tokens) | — |
| N26 | Single Markdown renderer and plain-text preview utility | E8 #3132 | 4 | with G17 (CM-07, CM-08) and AX-10 (single Markdown renderer) | — |
| N27 | Visual-regression and accessibility gates in CI | E1 #3125 | 0 → 5 | Stage 0 item 5 (axe as warning) → Stage 5 gate switch + visual-regression baselines | — |
| N28 | Typed API client and contract tests for Notes and Chat | E1 #3125 | 0 | Stage 0 item 3 (contract tests, generated API types) | — |
| N29 | Graph view at library scale | E7 #3131 | 4 | after NL-11 (G09) and NL-01 (G02) | NL-11, NL-01 |
| N30 | Knowledge panel as a retrieval workbench | E3 #3127 | 4 | with CC-10 follow-up (G11) and N7 | CC-10, N7 |
| N31 | Compare models in the side panel | E5 #3129 | 4+ | after Compare is verified (D4) | D4 |
| S1 | One unified, searchable chat history across devices and surfaces | E2 #3126 | 4+ | build after D7 spike (one searchable history across surfaces) | D7 |
| S2 | One conversation model across side panel, extension full page and webui | E2 #3126 | 4+ | build after D7 spike (one conversation model) | D7 |
| S3 | Server-side, resumable reply generation | E2 #3126 | 4+ | build after D7 spike (server-side resumable generation) | D7 |
| S4 | Side-by-side research mode (chat + note) | E3 #3127 | 6 | Stage 6 design spike, then build (side-by-side chat + note) | N2, XP-05 |
| S5 | Single block editor with Markdown shortcuts and a slash menu | E7 #3131 | 6 | Stage 6 design spike, then build (single block editor) | NE-01, N20 |
| S6 | Shared 'object page' layout for Notes and Chat | E8 #3132 | 6 | Stage 6 design spike, then build (shared object-page layout) | N4, N12 |

---

## Risks and mitigations

- **Very large components.** `PlaygroundForm.tsx` (~6.7k lines), `Playground.tsx` (~4.6k), `NotesManagerPage.tsx` (~2.7k) and `Sidepanel/Chat/form.tsx` (~4.5k) make regressions likely. Keep PRs small, land the harness first, and extract only where a fix needs a seam (for example the notes save state machine).
- **Shared controller.** The history-selection controller serves the WebUI, the options page and the side panel. Every G05/G06 change runs the side-panel e2e too.
- **Backend changes.** Notes list contract, model metadata (CM-04), restore with messages (CS-N2), permanent delete (NL-06) and server-side generation (S3) need backward-compatible APIs and migrations. Keep clients tolerant of both shapes during rollout.
- **Extension release cadence.** Side-panel fixes ship with extension releases. Group extension changes so a release carries a coherent set.
- **Copy churn.** Glossary changes touch locale files across packages. Land the copy lint with an allowlist and burn it down.

---

## Closing the loop

- After each stage, re-run the review's persona scenarios (first-time and power user, WebUI and extension) in the UX harness against seeded data. Attach the results to the epic (#3101).
- A group issue closes when all its checklist items and acceptance criteria are done.
- After Stages 3 and 5, run a short follow-up review against the then-current `dev` to catch regressions and new issues.

## Status

| Stage | Groups | Status |
|---|---|---|
| 0 Enablers | E1 #3125 | Not Started |
| 1 Data safety | G01–G04 #3102–#3105 | Not Started |
| 2 Core flows | G05–G09 #3106–#3110 | Not Started |
| 3 First run | G10–G14 #3111–#3115 (+E5, E6) | Not Started |
| 4 Scale and continuity | G15–G20 #3116–#3121 (+E2, E3, E4, E7) | Not Started |
| 5 Accessibility and polish | G21–G23 #3122–#3124 (+E8) | Not Started |
| 6 Strategic bets | S1–S6, N31 (E2, E3, E5, E7, E8) | Not Started |
