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

Started 2026-11-19, after the owner's answers (see "Owner decisions"). It is
built on `ux/burndown-wave7`, from the Wave 6 head 45188b5e13, while #3526
landed. Dev, with #3526, was merged in on 2026-11-22. The numbers in brackets
are the decision numbers.

**Lane 7A — Composer names, search and size**
- CC-08 [1]:
  - Keep the redesign's names ("Search & Context", "Runtime") and update the
    glossary.
  - Move the Modes menu's toggles to where they belong.
  - Give web search one home and one spelling.
- CC-10 [2]:
  - Search stays in the composer, with its own height cap.
  - Sources are shown first.
  - The generated answer is opt-in.
- Size tokens [3]: 32px desktop controls and 44px touch, applied to the
  remaining toolbar controls. D9 and D10 are dropped.
- CO-04: the rails at 1024px (width-aware, one rail at a time), and the
  leftover jargon, using the names from [1].
- AX-04 [14]:
  - In short viewports the toolbar row collapses first; the input and Send
    stay.
  - Reflow tests at 320×256 and 640×400.
  - Measure the composer share.

**Lane 7B — Notes views, Graph and filters**
- NL-11 [15]:
  - Graph filters, and an All-notes cap of 500.
  - Picking a note re-centres the graph and opens a preview without leaving
    Graph.
- NL-12 [16]: keep the Inbox view, and fix the tag-match toggle counting
  Inbox's hidden token.
- Move notes search above Views.
- Saved filters always match all of their tags [19].

**Lane 7C — Delete policy, Undo and shared copy**
- XP-17 and AX-06 [8]:
  - One delete policy: soft deletes get a persistent inline Undo, and only
    permanent deletes ask for confirmation. Message delete keeps its confirm.
  - A shared undo helper, which also covers the G07 Trash Undo keyboard
    follow-up and Undo for the Notes bulk delete.
- XP-13 [7]: the glossary terms (Character, Persona, Buddy and the product
  name).
- XP-11 [6]: the chat list leaves routes that aren't chat.
- `text-subtle` merges into `text-muted` in the light theme [9].
- XS-12 [10]: "Open dashboard" becomes "Flashcards".

**Lane 7D — Side panel, temporary chats and notes in chat**
- XP-09 [11]: one "Open in full page" action. The extension may open a
  configured WebUI.
- XP-12 [12]: align the side-panel composer's strings and menus with the full
  page; no rebuild.
- XS-15: the Pro composer's share of the side panel, as the review specifies
  it, plus the measurement.
- XP-01 [5]: a temporary chat's answer can be saved to Notes.
- Web "@" Notes and `/note` [4]: attach a note's content as a removable chip
  in the composer. XP-04's note-side action waits for #3226.

**Still blocked (Wave 8):**
- NL-06 [17] and NL-10 [18] wait for #3226.
- XP-18 [13] needs D7 P7 and the server pin field, which collides with #3226's
  migrations.
- #3182 [20]: the outbox is approved, but it is sequenced after #3226 and
  #3071/#3159.

### Wave 8: blocked by other work

- XP-16 duplicate reads: after #3435 (webui HTTP architecture) merges.
- NO-03, NL-06, NL-10 and the G08 Notes side: after #3226 (Notes structured
  editing) lands or is rebased.
- Finishing P5 (CM-03 (a), CM-04 model info) and CM-14: they change the send
  files that #3071/#3159 rewrite. **Unblocked 2026-11-23:** the owner parked
  those PRs [21], so these are Wave 8's first lanes.
- XP-18 tab pins: D7 P7.
- #3182: the Sync outbox (backend, L). The migration is approved [20]; it is
  sequenced after #3226.

## Owner decisions

**Answered 2026-11-19:** the owner accepted the recommended answer for every
question below ("go with your recommendations"):

1. CC-08: keep the redesign's names ("Search & Context", "Runtime") and
   update the glossary to match. The Modes menu's toggles move to where they
   belong.
2. CC-10: search stays in the composer, with its own height cap. Sources are
   shown first; the generated answer is opt-in.
3. Sizes: 32px controls on desktop, through the size tokens, and 44px on
   touch. D9 and D10 are dropped (the redesign replaced them).
4. Web "@" attaches a note's content as a removable chip. `/note` does the
   same.
5. XP-01: a temporary chat's answer may be saved to Notes.
6. XP-11: the chat list leaves routes that aren't chat.
7. XP-13: adopt the review's glossary (Character, Persona, Buddy and the
   product name).
8. XP-17 and AX-06: soft deletes get a persistent inline Undo, and only
   permanent deletes ask for confirmation. Message delete keeps its confirm.
9. Merge `text-subtle` into `text-muted` in the light theme.
10. XS-12: "Open dashboard" becomes "Flashcards".
11. XP-09: one "Open in full page" action. The extension may open a configured
    WebUI.
12. XP-12: align the side-panel composer's strings and menus only; no
    rebuild.
13. XP-18: a server-side pin field for conversations and notes. UI mode stays
    per surface.
14. AX-04: in short viewports the composer's toolbar row collapses first; the
    input and Send stay.
15. NL-11: add Graph filters and raise the All-notes cap to 500. Picking a
    note re-centres the graph and opens a preview, without leaving Graph.
16. NL-12: keep the Inbox view.
17. NL-06: Trash is auto-purged after 30 days. A permanent delete reaches
    Sync v2 devices as an id-only tombstone, as in the #3198 rule.
18. NL-10: server-side pins. Folders go with the notes epic E7 (#3131).
19. Saved filters always match all of their tags.
20. #3182: the outbox migration is approved. The current save behaviour stays
    until it lands.
21. #3071/#3159: the recommendation was to leave this to the owner, so it
    stayed open until 2026-11-23, when the owner parked both PRs. They stay
    open but no longer hold anything up, so finishing P5 and CM-14 goes ahead
    on dev. If they come back, they rebase onto that work.

The original questions:


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

## Wave 7 log

The four lanes were built in the Wave 5 worktrees on new branches, from the
Wave 6 head. They were reviewed per lane (two reviewers each), and their
review fixes went on top. The lanes were then merged into
`ux/burndown-wave7`.

**Integration:**
- **Lane 7C commit order:** its `UndoOutcome` export was moved ahead of the
  commits that import it, so every commit compiles. The tree is unchanged.
- **Lane 7D conflicts:**
  - 7D conflicted with 7A in the English locale files, and with 7C in the
    side-panel header and its test.
  - The header keeps 7D's shared `openOptionsRoute` and 7C's "Flashcards"
    name.
- **Note chip:** the remove button uses 7A's composer size tokens.
- **Side-panel strings:** two strings 7D no longer reads ("Web Search",
  "Knowledge Search") now say "Web search" and "Search & Context". 7A's
  one-name guard allowlists only the /knowledge page title.
- **`getNote`:** it lives only on the Notes domain client. A delegating copy
  on `TldwApiClientBase` failed the client ownership guard.
- **Dev merges (2026-11-22):** dev @ cee4433eb2 is merged in. It brings
  #3526, the ChaCha owner-scope work (TASK-13950.23) and the first
  CharactersRAGDB shrink (TASK-13950.22). The only conflict was the OpenAPI
  fingerprint, which was regenerated from the merged tree.
  - Wave 7 was built on an earlier Wave 6 head, so this also brings Wave 6's
    CI fixes: the NL-05 selection reset now happens during render, and the
    XP-06 test waits longer.
  - The frontend tests were rerun under Node 20, the version CI uses. Before
    the merge, the NL-05 Notes → Trash → Notes test failed there.
  - Dev @ 37cc558814 (#3557, TASK-13950.22 c2) followed. It moves the note
    and keyword stores onto ChaChaContext. The only conflict was the two
    keyword-token filter calls in `note_store.py`: dev's
    `self._ctx.map_table(...)` with Wave 7's `required_tokens` argument.
- **CI fixes on #3559 (2026-11-22):**
  - The real-server cockpit spec opens the Tools tray before it checks the
    MCP control. Its CC-08 update closed the tray after the web-search step,
    and the MCP control lives in the tray. It was rerun locally against a
    live server: 6 passed, and the check fails the same way as CI without it.
  - pre-commit's whitespace nits in `note-mentions.ts` and `ocr.ts` are fixed.

**Choices made during review and integration:**
- **Delete and Undo (7C):**
  - Soft deletes of notes, chats and side-panel tabs no longer confirm.
  - They get one inline Undo that stays until it is used, dismissed or
    replaced, and that reports what was actually restored.
  - The side-panel bulk delete still confirms when any selected tab has no
    server chat (local-only or Temporary).
  - Moving the open note to Trash saves its unsaved edits first.
- **XP-11:** off `/chat` the sidebar keeps navigation and settings, under one
  name ("Menu"), and reads no chat history. The mobile drawer is named to
  match.
- **CC-10 (7A):**
  - QA search never generates an answer.
  - "Generate answer" re-runs the searched query with the answer settings
    chosen at click time.
  - The button sits in the results header and moves focus to the answer.
  - The "Enable generation" switch no longer hides those answer settings.
- **Short viewports (7A):**
  - The transcript floor is `min(42dvh, 45%)`, and the dock floor is 4.5rem.
  - A rail panel saved open isn't restored there.
  - The overlay rail stops above the dock. Closing it keeps the other rail
    closed.
- **NL-11 (7B):**
  - A tag or source filter shows exactly its notes, with no one-hop
    neighbours.
  - Graphs with no focus note get 4 links per node.
  - A lower `NOTES_GRAPH_MAX_NODES` or `_MAX_EDGES` shrinks those budgets.
    Values of 0 or less count as unset.
  - "Linked notes only" reads `has_note_links` from
    `include_link_status=true`, computed before pruning. Similarity
    suggestions don't count as links.
- **Tag match (7B):** "Any" means required AND (any picked tag). A picked tag
  the view already requires satisfies it.
- **"@" and `/note` (7D):**
  - A note attaches to one message: the chip clears after a send or a queue,
    and a failed send keeps it.
  - Notes are sent even with file retrieval on, as a labelled, fenced block.
  - An unpicked `/note …` isn't sent.
- **XP-09 (7D):**
  - In the extension, "Open in full page" always opens the extension's page,
    with the chat and the draft.
  - A configured WebUI gets its own "Open in WebUI" item in Tools, which
    carries the draft only.
  - The web-hosted panel keeps the WebUI hand-off.
- **XP-12 (7D):** the side panel's "Tools" ends with a "More tools" group
  (Images, Advanced, Quick ingest, Open in full page), matching the full page.
- **XS-15 (7D):** the textarea and Send always stay in view, which outranks
  the 60% conversation target.
  - The optional dock content scrolls above them.
  - Side-panel controls are 32px on a fine pointer and 44px on a coarse one.
  - Measured on `/__debug__/sidepanel-chat`. Send was hidden in 26 of 27
    states before, and is visible in all 27 now. At 600px tall, the Pro
    composer takes 64–71% of the panel.

**For the owner:**
- **Pro side-panel composer:** a conversation share near 60% needs a shorter
  composer: a one-row control row, or the Send-row extras moved into Tools.
- **Persona on chat surfaces:** Persona-vs-Character labels were left alone.
  Removing the word would merge two kinds of reply source.
- **Extension name:** the extension name and the toolbar title are still
  "tldw Assistant", because renaming them changes the Chrome Web Store
  listing.
- **"Avatar settings":** the dialog also holds the conversation's Persona
  settings.
- **Modes:** it has one entry while Compare is hidden (D4). Hide it until
  Compare returns?
- **Short viewports:** at 500px tall or less, Send sits beside the textarea.
  This is close to the dropped D10.
- **"More tools":** the full page's overflow menu keeps that name and icon.
- **Side-panel QA search:** the shared Search & Context panel also makes the
  answer opt-in there.
- **Other theme presets:** they keep two light text tiers. Decision 9 was
  applied to the default preset.

### Follow-ups from Wave 7

- **Graph recency filter:**
  - Not built: `time_range` on `GET /notes/graph` is documented but can't be
    reached, because a nested `Query()` model returns 422.
  - It needs a flat parameter, and a seed query for libraries over the cap.
  - Filed as #3558.
- **Graph build:** `generate_graph` still runs on the event loop. The
  suggestion routes show the `run_in_threadpool` plus
  `release_context_connection` pattern.
- **NL-12:** only the Wave 7 slice is done (keep Inbox, drop the Captured
  chip, tag match). The segmented Notes/Trash control and the View select
  remain.
- **Saved filters:** they allow 25 tags, but the server caps each token list
  at 20, so a filter with 21 or more tags gets a 400. This predates Wave 7.
- **Older servers:** they ignore `required_tokens`, so Inbox or a saved filter
  plus "Any" of two or more tags widens to OR there. That is no worse than
  before.
- **AX-06:** an app-wide announcer for antd toasts.
- **Undo toasts:** Flashcards, WorldBooks, Watchlists and the Notes bulk-tag
  Undo still use the 8 s Undo toast.
- **XP-11:** the Notes Dock icon, the "Quick note" label and "Archive" →
  "Notes".
- **7D leftovers:**
  - carrying a draft into a new Character or Persona chat;
  - a client-side Notes permission check (the picker uses the `hasNotes`
    capability);
  - trimming the note list and search payloads, which needs a backend content
    flag.
- **Locale mirror:** `public/_locales/en/sidepanel.json` drifts about 435 lines
  from `assets/locale/en`.
- **`MessageActionsBar.real-popover.test.tsx`:** "opens a native button
  Popover as an independent trigger fixture control" now fails when run
  alone. It fails on the Wave 6 branch merged with dev as well, so it is not
  from Wave 7.
- **Real-server gate:** the advisory /chat real-server cockpit gate
  (`e2e:chat-cockpit:real:focused`) wasn't run locally. It needs a live
  server.

## Status

| Wave | Status |
|---|---|
| 5 | Landed (#3461, 2026-11-13) |
| 6 | Landed (#3526, 2026-11-22) |
| 7 | Landed (#3559, 2026-11-22) |
| 8 | In progress: P5 and CM-14; the rest blocked |
