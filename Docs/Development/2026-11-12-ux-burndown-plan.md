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

- Notes: NL-04 (row density, no raw UUIDs), NL-05 (bulk actions and Trash at
  scale), NL-07 (search relevance), NL-14 (Graph for newcomers).
- Chat: CM-03 (b) side-panel Continue and actions on saved markers, CM-12
  fork lineage, CM-N2 for older branched chats.
- Navigation: CS-09 per-chat Export (from the server) and Move to folder on
  the full page; XP-06 ⌘K finds and creates notes and chats (first slice).
- Composer and layout: CO-01 phone navigation with CO-N1; CC-11 slice (a):
  `/new`, `/prompt` and server commands.
- Accessibility: AX-11 transcript live region, AX-12 suggestion comboboxes;
  measure AX-04 and XS-15 live.
- XP-03 provenance, XS-13 side-panel quick-save slice 1, XP-04 "Chat about
  this note" (after #3226, see below).

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

## Status

| Wave | Status |
|---|---|
| 5 | In progress |
| 6 | Not started |
| 7 | Waiting on owner decisions |
| 8 | Blocked |
