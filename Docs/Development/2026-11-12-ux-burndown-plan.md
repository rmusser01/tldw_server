# UX remediation burn-down (2026-11-12)

Owner request (2026-11-12): "burn down the rest" of the /notes and /chat UX
review (#3101) after the chat-recovery lane (#3356, #3368) landed.

Scope: the 17 open review groups (G05–G09, G11–G15, G17–G23) and the
standalone issues #3149, #3157, #3182 and #3343. The enhancement epics E1–E8
(#3125–#3132) are product work and are planned separately.

## Re-triage against dev @ 28b6806698

Five read-only passes checked every remaining item in code. The tables, with
file:line evidence, are in `2026-11-12-ux-burndown/`:

| Area | File | Items | Fixed | Partly | Open |
|---|---|---|---|---|---|
| Chat core (G05, G06, G17) and D7 frontend | `retriage-chat-core.md` | 11 | 0 | 1 | 10 |
| Composer and chrome (G11, G12) | `retriage-composer-chrome.md` | 11 | 1 | 6 | 4 |
| Notes (G09, G13, G15, #3149, #3157) | `retriage-notes.md` | 14 | 0 | 3 | 11 |
| Cross-surface (G07, G08, G14, G18, #3182, #3343) | `retriage-cross-surface.md` | 24 | 0 | 4 | 20 |
| Side panel, performance, a11y (G19–G23) | `retriage-ext-perf-a11y.md` | 18 | 0 | 7 | 11 |

One more open follow-up: an empty notes list shows "Loading…" on every refetch
(`NotesListPanel.tsx:194`).

### D7 frontend status

Wave 4 was deferred "until D7 frontend Phase 1 and the Sync v2 outbox". On dev:
P4 and P6 are mostly done, P5, P9 and P10 are partly done, and P7 and the
outbox (P15–P17) have not started. Only three item parts still need D7:

- CM-03 (a), the lost output-limit marker after reload, and CM-04's model and
  generation info for client-settled turns need P5 finished.
- XP-18's side-panel tab pins need P7.

Everything else can be done now.

## Waves

Each wave lands as one integration PR (the merge queue is off and `dev`
requires 8 checks on an up-to-date head). Within a wave, lanes are built in
separate worktrees, red test first, then merged into the wave branch.

### Wave 5: small fixes, no decisions

**Lane 5A — Notes data and list**
- #3157: a tags-only PATCH bumps the version (data loss with two clients).
- SQLite: non-ASCII capitalised tags match (copy the Media fix `3300ab7b87`;
  lower the keyword ratchet baseline).
- #3149: Review notes paging uses the real list contract.
- Connections: cancel superseded neighbour requests; don't repeat a 403.
- An empty list shows its empty state, not "Loading…", on a refetch.

**Lane 5B — Chat actions and side-panel Stop**
- CM-01 follow-ups: a refused Continue withdraws what the refusal covers
  (server `reason` on the refusal); Continue's recovery action, disabled
  reason and error codes use Continue wording; the Continue prefix drops
  trailing whitespace; no role-play steering in saved character chats.
- "Copied" is announced; clipboard failures say so.
- CM-04 (part): saved RAG sources show again after a reload.
- G19 follow-ups: Stop on the pending reply bubble before streaming starts;
  Stop no longer replaces attach and dictation for the whole request; the
  scroll-to-latest pill sits on the composer only when it is pinned.

**Lane 5C — Navigation, readiness and titles**
- G18 follow-ups: New chat keeps the previous chat's history entry; no stale
  pending-navigation marker; a failed history handoff doesn't stop address
  writes; a visible tutorial prompt survives a same-page address change;
  correct the review's XP-10 recommendation for macOS.
- G14: title-restoring cleanups (AdminRouteShell, voice chat) no longer undo
  XP-19's title reset.
- #3343: side-panel model dot without evidence; Pro + v5 blocked send opens
  the picker; readiness counts only provider-shaped failures and clears on an
  account change; the "All models" catalog shows its failure.
- XP-14: raw IDs and machine strings.

**Lane 5D — Accessibility and composer chrome**
- AX-01, AX-02, AX-14 (remaining parts).
- Contrast: the a11y ratchet also scans the light theme and the side panel;
  the status and primary `/10` tints meet 4.5:1 through the tokens.
- Smooth scroll: the 24 conflict-free files (plus
  `option-quick-chat-popout.tsx`) respect reduced motion; the 4 files that
  #3435/#3071 change wait.
- CC-11 (quick part): the web app stops promising "@" mentions, and the
  "@tab" setting shows only in the extension.
- CO-03: remove the decorative "Composer" pill. CO-06: a 1024×768 test.

### Wave 6: medium items, no decisions

Started 2026-11-13 on `ux/burndown-wave6` from dev @ 4b6b9ec446 (#3461).

**Lane 6A — Notes list, bulk and Trash**
- NL-04: a plain-text one-line preview, a "From chat" link instead of the
  conversation id and message id, batch-resolved chat titles, relative dates
  and denser rows. Moving search above Views waits for decision 16 (NL-12).
- NL-05: select all on the page and all matching, a running bulk state with
  progress, failures with Retry, and in Trash checkboxes, search and
  "Restore selected". Restore stays in Trash. Undo for bulk delete waits for
  decision 8.

**Lane 6B — Notes search, Graph and ⌘K**
- NL-07: a backend snippet and server-side sort, "Best match" while a query is
  active, and the snippet in the row.
- NL-14: tag labels, a node legend, a "Graph settings" popover with plain
  labels, and a hint when the focus note is a fallback. The cap and filters
  wait for decision 15 (NL-11).
- XP-06 slice 1: Chats and Notes groups in the global palette, and "New note".

**Lane 6C — Chat**
- CM-03 (b): Continue and saved-status actions in the side panel.
- CM-12: a fork keeps its parent, with a title, lineage and a way back.
- CM-N2: legacy branched chats default to the newest-leaf path, with a hint
  that alternative replies exist.
- The full-page pending Stop (a Wave 5 follow-up).
- CS-09: per-chat Export (read from the server) and Move to folder on the
  full page, plus a visible ⋯ on side-panel tabs.

**Lane 6D — Composer and accessibility**
- CO-01 with CO-N1: a slim top bar in phone focus mode (chat list, title, New
  chat, Exit focus).
- CC-11 slice (a): `/new`, `/prompt` and the server commands in the WebUI.
- AX-11: a narrower transcript live region.
- AX-12: the slash, mention and wikilink suggestions become comboboxes.

**Moved out of Wave 6:**
- XP-03, XS-13 slice 1 and XP-04 go to Wave 8. They edit the files #3226
  rewrites (`NotesEditorHeader.tsx`, `NotesDockPanel.tsx`,
  `NoteQuickSaveModal.tsx`, `notes-capture.ts`), and the re-triage puts them
  after #3226.
- Measuring AX-04 and XS-15 goes to Wave 7, with decisions 12 and 14.

### Wave 7: after owner decisions

CC-08, CC-10, toolbar size tokens, CO-04, CC-11 "@ Notes" (with XP-04),
XP-01 (temporary chats), XP-11, XP-13, XP-17 with the Trash Undo follow-up
and AX-06, `text-subtle`/`text-muted`, XS-12 "Open dashboard", NL-11, NL-12,
the tag-match follow-up, NL-06, NL-10, XP-09, XP-12, XP-18, AX-04.

### Wave 8: blocked by other work

- XP-16 duplicate reads: after #3435 (webui HTTP architecture) merges.
- NO-03, NL-06, NL-10 and the G08 Notes side: after #3226 (Notes structured
  editing) lands or is rebased.
- Finishing P5 (CM-03 (a), CM-04 model info) and CM-14: they change the send
  files that #3071/#3159 rewrite; needs the owner's call on those PRs.
- XP-18 tab pins: D7 P7.
- #3182: the Sync outbox (backend, L), after the owner approves the
  migration.

## Owner decisions

Chat and composer:
1. CC-08 names: the review's glossary ("Knowledge", "Compare models",
   "Temporary chat") or the October redesign's ("Search & Context", "Add
   sources", "Runtime")? Does the Modes menu stay?
2. CC-10: does search stay in the composer (capped) or move to the Context
   rail or a drawer? Should the generated answer be opt-in?
3. Sizes and layout: 32px desktop controls (tokens) or 44px? Do D9 (one Pro
   right panel) and D10 (one-row phone composer) still stand?
4. Web "@": attach a note's content or insert a reference? Does `/note` attach
   a note?
5. XP-01: may a temporary chat's answer be saved to Notes?

Navigation and copy:
6. XP-11: should the chat list leave non-chat routes?
7. XP-13: adopt the review's glossary (Character/Persona/Buddy, product name)?
8. XP-17 and AX-06: one delete policy and one undo pattern (persistent inline
   Undo or a longer toast); a message-restore endpoint or keep the confirm?
9. Keep `text-subtle` and `text-muted` as two light-theme tiers, or merge?
10. XS-12: "Open dashboard" becomes "Flashcards"?

Side panel:
11. XP-09: one "Open in full page" action or two? May the extension open a
    configured WebUI?
12. XP-12: align strings and menus only, or rebuild the side-panel composer on
    the #3309 composer?
13. XP-18: a server-side pin field for conversations and notes? Does UI mode
    stay shared across surfaces?
14. AX-04: which composer rows collapse in short viewports?

Notes:
15. NL-11: raise the 100-note Graph cap, add filters, or both? Does picking a
    note re-centre the graph or open the editor?
16. NL-12: keep the Inbox view or the Captured chip?
17. NL-06 (under D3): the auto-purge default, and how a permanent delete
    reaches Sync v2 devices.
18. NL-10: server-side pins or a synced setting? Folders now or in E7 (#3131)?
19. Tag match: do saved filters always match all their tags?

Other work:
20. #3182: approve the outbox migration; refuse the unversioned save meanwhile?
21. #3071/#3159 (chat workspace, untouched since 11-02, conflicting with dev):
    keep, rebase or close? Finishing P5 and CM-14 touch the same files.

## Process

- Each lane: a failing test first, then the fix; one commit per item.
- Gates per lane and on the wave branch: shared UI lint
  (`apps/tldw-frontend/scripts/check-shared-hooks.mjs`, frozen-file
  ceilings), `tsc --noEmit` (12 GB heap), affected Vitest suites with no
  unhandled errors, backend tests for touched modules, the CI contracts step.
- Code review of the wave branch before the PR (superpowers code-reviewer).
- Land with the strict-dev loop: merge dev when it moves, re-run a flaky
  E2E Critical at most twice.

## Wave 5 log

Four lanes were built in worktrees and merged into `ux/burndown-2026-11` with
no conflicts. Each lane was reviewed separately, and the review fixes were
merged on top.

- **Integration fixes:** the CI contracts step caught two ratchets.
  - `build_context_and_messages` grew past its 821-line ceiling. The
    `history_selection` imports moved to module level and the Continue strip
    moved into `_history_for_request`, so the function is back at 821.
  - `bump_note_version` grew the CharactersRAGDB surface. Callers now use
    `db.note_store.bump_note_version`. The private-member ceiling went from 551
    to 553, with the reason recorded in the baseline.
- **Lane choices accepted during integration:**
  - Continue whitespace: a continuation that starts with whitespace replaces
    the saved trailing whitespace; otherwise the saved spacing is kept.
  - Role-play steering is offered only where the server accepts it, which
    means before a character chat is saved and in temporary chats.
  - The fallback chat title uses server local time with its zone.
  - `continue` recovery records are forward-only: an older bundle refuses that
    conversation's recovery list until the record is opened or dismissed.
  - Tag and folder edits advance the note version. With semantic indexing on,
    that also re-embeds the note.
- **Pre-existing failures seen locally, unchanged by this wave:**
  - `chat-document-processing.test.ts` "reconstructs upload files" (a jsdom
    Blob).
  - `test_chat_service_streaming_tool_autoexec.py::test_a_stream_dropped_before_any_output_is_refunded_and_writes_no_row`
    (times out on the base too).
  - `test_merge_queue_gate_workflows.py` needs a `python` on PATH with
    `defusedxml`.
  - `TutorialRunner.real-joyride.test.tsx` can log an unhandled "window is not
    defined" under heavy load. The cause is react-joyride's own scroll, which
    updates the popper 10 ms after its animation ends.

### Follow-ups from Wave 5

- **Chat:**
  - The full-page Playground has no Stop on the pending reply bubble
    (Wave 6, with CM-03 (b)).
  - Note backlinks have no "resolving" state; this waits on #3226.
  - Chatbooks import detects conflicts by title, so server-titled chats and
    voice fallback titles can collide.
- **Notes:**
  - Advance a note's semantic record without re-embedding when only its
    version moved.
  - `update_note` metadata-only bumps don't mark the note semantic-dirty.
  - The keyword link/unlink endpoints don't bump the version.
- **Contrast:**
  - Light status text on `/15` and `/20` tints is still below 4.5:1, and the
    derivation margin is thin (about 4.55). Derive to about 4.6, or move those
    classes to `/10`.
  - The light-mode `text-primary` shade is now darker than the primary fills;
    it needs a visual sign-off.
- **Shortcuts:** the side panel's Ctrl+E ("chat with current page") and the
  full page's Mod+E (Switch model) share a help label.

## Wave 6 log

The four lanes were built in the Wave 5 worktrees on new branches, merged into
`ux/burndown-wave6`, reviewed per lane, and their review fixes merged on top.

**Integration:**
- Lanes 6A and 6B both changed `NotesListPanel.tsx` and
  `useNotesListManagement.tsx`. The merge keeps 6B's search snippet in front
  of 6A's plain-text preview, and 6A's `oldestFirst` is folded into 6B's
  `NotesPageFetchOptions`. The OpenAPI fingerprint was regenerated.
- A Wave 5 regression was fixed: CO-03 removed the Composer pill, but
  `Playground.responsive-parity.guard` still required it. That guard reads
  source files and isn't in a CI shard, so it went unnoticed.
- The log test for unknown search sorts now only requires that nothing is
  logged at WARNING or above, because request logging echoes the query at
  INFO.

**Choices made during integration:**
- **Notes search:**
  - Unknown `sort_by` and `sort_order` values fall back to relevance and
    desc, with no 422.
  - Relevance pages end with `id ASC`, so paging is stable.
  - `include_snippets=false` skips the excerpts.
  - Note text can't forge a highlight.
- **Graph wording:** the English values of the existing keys were reworded:
  "Distance from focus", "Most items shown", "Rings" and the focus hint.
- **⌘K "New note":** opens `/notes?new=<nonce>`.
- **Notes list and Trash:**
  - Restore stays in Trash, and row Restore is disabled while a bulk run is
    active.
  - "Select all matching" pages oldest first and is capped at 1000.
- **Chat export:** JSON is `format: "tldw-chat-export"`, `version: 1`, with
  every message, and `images_omitted` when the chat has images. Markdown
  follows the newest path and says when alternatives are omitted.
- **Folders:** "Add to folder…" (one chat) shows and edits the chat's current
  folders.
- **Composer:**
  - The composer and Notes textareas keep the textbox role, with
    `aria-autocomplete`, `aria-controls` and `aria-activedescendant` on the
    textbox. `role=combobox` isn't allowed on a textarea.
  - ☰ in the phone Focus bar opens the chat list in a drawer, without
    changing the saved layout.
  - `/new` and `/prompt` are standalone. Any text after `/new` moves to the
    new chat, and `/prompt <text>` searches the picker.
  - Server commands are fetched when the slash menu first opens and are
    listed after the local commands.
- **AX-11:**
  - There is one polite log on the message rows, busy while a reply is
    pending or streaming, with one "Response complete" status outside it.
  - Hiding per-message chrome from the announced text was tried and
    reverted: it duplicated reply text in the DOM. It is a follow-up.
  - The empty log has no height, so the e2e checks for it use
    `toBeAttached()`.

### Follow-ups from Wave 6

- **AX-11 and AX-12:**
  - Keep the per-message chrome (name, time, toolbar) out of the announced
    text.
  - Verify the log and busy handling and the textbox autocomplete pattern
    with NVDA+Chrome and VoiceOver+Safari.
- **XP-06 slice 2:** recent items for an empty query, "Create note <q>", and
  Enter in the note title moving to the body. Also: the extension's options
  layout passes no palette content providers.
- **Chat export:** offer images (`include_images`) as an option.
- **Flaky locally under load:** `MessageActionsBar.real-popover.test.tsx`,
  which passes alone.
- **Waiting on owner decisions:**
  - NL-11 Graph cap and filters (decision 15);
  - moving notes search above Views (decision 16);
  - Undo for bulk delete (decision 8).

## Status

| Wave | Status |
|---|---|
| 5 | Landed (#3461, 2026-11-13) |
| 6 | In progress |
| 7 | Waiting on owner decisions |
| 8 | Blocked |
