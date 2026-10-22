# /notes + /chat UX remediation, Wave 3: integration log (2026-10-22)

Wave 3 of the /notes and /chat UX review remediation (tracking #3101). The
10 lanes were re-triaged against `dev` on 2026-10-19 (#3101 comment
6275372638). Each lane was built in its own worktree from `dev` `530e483b05`,
reviewed adversarially, then merged here with `origin/dev` (`d47c9ccf58`,
2026-10-22).

- Branch: `integration/ux-remediation-wave3`.
- Merged: 9 of the 10 lanes, plus NL-N1 (owner decision 2026-10-19: offer a
  toggle).
- Not merged: `ux3/chat-recovery` (section 3).

## 0. What each lane delivered

Ids are from the 2026-10-02 review. "Partly" items list what is left in
section 5.

| Lane | Fixed | Partly |
|---|---|---|
| chat-continue | CM-01 (Continue on saved chats: server-saved continuation as a sibling reply, plain refusals, shared availability) | CM-03 |
| composer | CC-N1, CC-03, CC-12, CM-05 (D4), CC-07, CC-09 | CC-11 |
| model-readiness | CC-01, CC-06, XS-02, XS-17 | — |
| messages | CM-07, CM-08, CM-09, CM-10 (with the CM-04 header), CM-11, CO-07 | — |
| notes-layout | NO-01, NO-02, NO-04, NO-N1, NL-13, NE-08, NE-09, NE-10 | — |
| notes-list | AX-03, AX-05, NS-04, NL-08, NL-09, NL-15, NL-16, NL-17; NL-N1 added here | — |
| a11y-visual | AX-07, AX-08, AX-09, AX-10, AX-15, AX-16, AX-17, AX-18, AX-19, XP-15, XP-20 (AX-13 was already fixed) | — |
| sidepanel | XS-04, XS-09, XS-10, XS-11, XS-14, XS-16, XS-18 | XP-16 |
| nav-history | XP-05 (`/chat?chat=`, `/notes?note=`), XP-19, XP-10, XP-07, XS-08, CS-06, CS-08, CS-10, CS-11 | — |

## 1. Review and fix-up

Each lane went through implement → adversarial review → fix.

**Usage limits.** Subagents hit their usage limits twice, so the following
steps were finished during integration instead:

- **messages.** Committed the unfinished CM-09 fix: role-play steering armed
  in a character chat is cleared once the chat is no longer one. Its test
  fails without the fix.
- **model-readiness.** Fixed all three review findings:
  - Bare Ollama ids such as `mistral:latest` keep their full name
    (`splitChatModelId`).
  - Provider health marks a model Unreachable only at the server's failure
    streak (3, as in `provider_manager.py`). Below that it makes no claim, and
    the copy names the provider when the evidence is the provider report.
  - The one-click switch moves focus to the chip and announces the switch.
- **sidepanel.** The reset button target is now 24px.
- **composer** and **a11y-visual.** A re-run had already fixed every blocking
  finding: the narrow `useFeatureFlags` mocks, the `Connecting…` empty card,
  the focus-token utilities, the steps/badge/menu contrast, and the
  stage21 focus-return determinism.
- **chat-continue** and **nav-history.** No agent review completed. Both
  were reviewed during integration: chat-continue's backend Continue path,
  and nav-history's address sync hooks (no write-back loop; `replace` when
  filling an empty address).

## 2. Conflict resolutions

### Between lanes

- **R1. messages × sidepanel, `sidepanel-chat.tsx`.** Keep the shared
  `ScrollToLatestButton` (CM-11/CO-07) and add `pinOnContentResize` (XS-09).
  The XS-09 test now finds the content observer among the hook's observers
  (`703d336c1d`).
- **R2. notes-layout × notes-list.**
  - Keep the NL-16 server-unavailable banner and drop the editor empty state
    that NO-04 removed.
  - The sidebar header keeps the mobile close button (AX-05) and the
    `notes-new-note-button` test id.
- **R3. sidepanel (XS-14) × a11y-visual (AX-10), `TableBlock`.**
  - A real `<table>` sits inside an always-focusable, labelled scroll region
    named by its caption or its columns, with the overflow hint and fade.
  - `useTableName` and the `regionLabel*` keys are removed.
  - The XS-14 test passes `thead`/`tbody` children, as Markdown does.
- **R4. notes-layout/notes-list × a11y-visual, Notes components.**
  - `TaskChecklistToggle` (NE-09) with the MetaText/help-button imports.
  - The NO-04 hint inside MetaText (AX-07).
  - The row checkbox keeps `tabIndex=-1` (AX-03) inside the 24px label (AX-16).
  - The sidebar toggle keeps `h-full` (NO-01) with the "notes sidebar" labels
    (AX-15).
  - stage23 keeps both tests. The XP-15 check reads the hint by id
    (`2f68dd8bed`).
- **R5. a11y-visual (AX-09) × nav-history (CS-11), thread search.**
  - Keep `useFocusReturn` and a11y's `closeThreadSearch`.
  - nav-history's `openThreadSearch` remembers the opener through the same
    hook. The duplicate ref and close function are removed.
- **R6. Locales.** Merged three-way by key. No value conflicts.

### With `origin/dev` (merge `eff0f1a2ce`)

- **D1. `chat_service.py` / `chat.py`.**
  - Continue shares the reused-reply branch with Regenerate.
  - dev's new server-existing-input retry runs first, and is refused with
    Continue as it is with Regenerate.
  - The runtime state carries the admission or the existing input, plus the
    continue prefix.
- **D2. Saved-reply availability (CM-01) vs dev's local Regenerate path.**
  - dev added Regenerate for local (device) histories after the lanes
    branched. The shared helper now offers Regenerate for a local owner whose
    capture ends with an answer to the question before it.
  - Native histories still need a parent-graph target, because
    `nativeRegenerateView` rejects legacy linear views.
  - The rail uses the helper instead of dev's inline rule, which would have
    enabled a Regenerate that then fails.
- **D3. Notes delete.**
  - dev's mutation-fenced flow, with the NL-15 confirm (title, Trash wording,
    linked-note warning).
  - Restore keeps the fence check and dismisses the stale "deleted" toast.
- **D4. Keyword suggestions test.** NL-17's assertions with dev's
  `userEvent` clicks.
- **D5. Model chip.**
  - The tooltip reads "Model status" when readiness shows, otherwise dev's
    "Status"; the connected label is dev's "Server connected".
  - dev's disabled-character-blocker condition joins `primaryDisabled`.
- **D6. Harness and imports.** Both sides.

### Integration fixes (`e50e5c34d3`)

- **Model chip without evidence.** The chip names only the server
  connection, following dev's UAT decision (TASK-13260.278.18.83.45).
  "Working" and "Unreachable" still show once a send or the provider report
  gives a signal.
- **Switch announcement.** It is a plain polite live region, not a second
  `role=status`.
- **Tests aligned with the merged behaviour.**
  - The cockpit rails name the model "gpt-4.1-mini (OpenAI)", with the raw
    route under Details (CC-06).
  - The Delete confirm uses the NL-15 Trash wording.
  - New note's title field becomes writable after the draft settles.
  - The CC-01 unknown-readiness test accepts either connection label (the
    harness has no knowledge service).

## 3. Not merged: chat-recovery (CC-04, CC-02, CM-02, CM-06, CC-N2)

`ux3/chat-recovery` conflicts in 13 files with dev's interrupted/failed-turn
work (`history-turn-retry.ts`, `085d266b95`, `eb0c8c9e52`).

- **Backend already overlaps.** dev now answers a provider failure after
  admission with `_admitted_native_provider_error_response` and retries
  saved questions with `tldw_history_server_existing_input_v1`. That covers
  part of chat-recovery's backend change.
- **Needs rebuilding.** The lane must be re-triaged against dev and rebuilt
  on that design; this is not a mechanical merge.
- **Where its work is.** The lane (8 commits) is on `ux3/chat-recovery`.
  Its merge with the other lanes, including the de-duplication against
  model-readiness, is kept on `ux3/wave3-with-chat-recovery`.

## 4. Generated files, gates and test runs

- **OpenAPI fingerprint** regenerated twice: NS-04 `legacy_aliases`, and
  NL-N1 `tokens_match`. The drift gate passes locally. The path union and
  the privilege route snapshot are unchanged, and the shard-coverage guard
  passes.
- **UI (Vitest).** Every test file the branch changes, plus all Playground
  integration and Notes stage suites: 244 files, 2,868 tests, all passing.
  All Notes suites after NL-N1: 152 files, 1,354 tests.
- **Backend.**
  - Chat history, continuation and existing input: 246 passed; the 5 skips
    are the Postgres variants.
  - NL-N1: the endpoint suite (5 of 8 red on dev), a real PostgreSQL test
    (pgserver) and the SQL-shape owner test.
- **Ratchet baselines, measured live.** Three runs of the accessibility and
  request-budget ratchets (`--grep`, which skips the deferred43
  instrumentation: its pinned node_modules hashes do not match this machine).
  - The first run found an empty-library scroller with no focusable content,
    a faded Send label that failed contrast, and lower request counts. All
    three were fixed or recorded (`af6019d98a`).
  - The next two runs passed all 7 checks.
  - `a11y.json` is empty: no serious or critical violations in any scanned
    state.
  - Request budget: /notes 21 and /chat 22 API requests; the buddy polling
    entries are removed.

## 5. Follow-ups

- **Chat.**
  - chat-recovery rebuild on dev's retry design (section 3).
  - #3283: frequency/presence penalty, Top K and Min P are never sent.
  - CM-03: a client-settled reply in a saved chat loses its live output-limit
    marker when the view re-reads the server copy. This needs the settlement
    to carry finish metadata (D7 P5). The side panel offers no Continue on
    these markers.
  - CM-05: before Compare is re-enabled, "Add models" must open the compare
    popover.
  - CC-11 (large): `@` mentions for Notes, Media, Chats and Prompts; `/prompt`,
    `/note` and `/new`; server commands in the composer.
  - CC-07: move the other toolbar controls onto the size tokens (with D9/D10).
- **Side panel.**
  - XP-16: remaining duplicate reads (auth/me on /notes, users/me/profile on
    /chat). Measure a production build first.
  - XS-10: inline Stop on the pending reply bubble.
  - The side-panel model dot still says "Not checked" without evidence, while
    the full page now names the connection.
- **Accessibility.** AX-19: 28 files outside chat still pass a literal smooth
  scroll behaviour.
- **Owner attention (behaviour changes).**
  - NE-10: after Connections is opened once, it stays open across note and
    Graph switches and fetches neighbours on each switch.
  - NO-02: Joyride's scroll-parent fix is off on every viewport-framed route
    (/chat, /media, research workspace), not only /notes.
  - NL-N1: tag tokens now match exactly. Several tokens default to "all", so
    Inbox plus a tag shows captured notes that have the tag, and a notebook
    with several tags shows notes that have all of them.
