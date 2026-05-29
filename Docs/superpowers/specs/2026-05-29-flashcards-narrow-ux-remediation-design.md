# Flashcards Narrow UX Remediation Design

Date: 2026-05-29
Owner: Codex collaboration session
Status: Draft for user review
Backlog: TASK-483

## Summary

This design narrows the broader flashcards UX audit into two reviewable PRs.
The work stays scoped to `/flashcards` and directly connected extension
flashcard handoffs. It does not reopen the full flashcards audit backlog.

The selected approach is:

1. PR 1 fixes the direct entry and review-recovery issues: extension sidepanel
   route registration, clearer create/import labeling, visible re-rate after a
   rating, deck prefill when creating from a selected Study deck, and accurate
   completion actions.
2. PR 2 fixes the remaining power-user behavior: all-deck Study starts at the
   deck dashboard, and recent session history preserves user-facing deck names.

## Goals

1. Make the actual extension sidepanel able to reach the direct `/flashcards`
   handoff when the route is still missing.
2. Replace the user-facing `Transfer` label with clearer task language.
3. Keep the undo/re-rate affordance visible after rating a card.
4. Preserve deck context when a user creates a card from a selected Study deck.
5. Avoid showing `Practice again` when no cram or repeatable cards exist.
6. Make all-deck Study dashboard-first for power users.
7. Preserve last-known deck names in recent session history.
8. Add focused regression coverage for the two most fragile flows.

## Non-Goals

- Do not redesign the full flashcards product surface.
- Do not build a new extension capture workflow.
- Do not change Quiz, Study Pack, or scheduler behavior except where a direct
  `/flashcards` handoff requires it.
- Do not rename internal component files, import/export helper types, or
  backend fields just because the visible label changes.
- Do not add advanced analytics, spaced-repetition policy changes, or new review
  modes.
- Do not solve every item in `Flashcards-UX-Fix-List.md`.

## Current Evidence

Observed and inspected anchors:

- `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx` does not
  currently register `/flashcards`.
- `apps/packages/ui/src/routes/sidepanel-route-registry.tsx` does register
  `/flashcards` and imports `sidepanel-flashcards`.
- `apps/packages/ui/src/routes/sidepanel-flashcards.tsx` opens
  `/options.html#/flashcards`.
- `apps/packages/ui/src/components/Flashcards/FlashcardsManager.tsx` owns the
  top-level tab model and Study to Manage/Create entry point.
- `apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx` owns review
  queue state, rating, undo/re-rate state, completion actions, and the current
  all-deck active/empty review rendering. The current checkout does not expose a
  reusable `DeckStudyDashboard` component, so PR 2 must either adapt the
  existing all-deck empty/review setup area or introduce the smallest focused
  dashboard component needed for the selected behavior.
- `apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx` opens
  `FlashcardCreateDrawer`.
- `apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx`
  owns manual card creation fields and currently initializes its own deck state
  when opened.
- `apps/packages/ui/src/components/Flashcards/components/RecentStudySessions.tsx`
  renders recent session labels and currently falls back to technical deck
  identifiers when no user-facing name is available.

The main checkout is dirty with unrelated changes, so implementation should use
a clean worktree before editing code.

## PR 1: Flashcards Entry And Review Recovery

### Scope

PR 1 should include only low to moderate risk route and interaction fixes:

- Register the actual extension sidepanel `/flashcards` route if it is still
  missing.
- Rename the create/import tab to `Create & Import`. In the current checkout the
  visible tab already appears as `Import / Export` while stale test/docs wording
  still references `Transfer`; the target acceptance criterion is the final
  visible label, not the starting label in a particular branch.
- Keep undo/re-rate visible for the existing undo window after any rating.
- Preselect the active Study deck when opening Create from a selected deck.
- Suppress or keep absent `Practice again` when there are no cram or repeatable
  cards. If the target branch already omits this button, treat this as
  regression coverage rather than inventing new UI work.
- Add focused tests for the route registration, re-rate flow, and create-drawer
  deck prefill.

### Interaction Design

The extension sidepanel route fix should mirror the shared sidepanel behavior:
register `/flashcards` in the app extension registry and open the existing
options `/flashcards` page. This is a handoff route, not a new sidepanel-native
flashcard editor.

The visible tab label should be task-first. `Create & Import` is the preferred
label because the flashcards setup workflow spans creation handoffs, import,
export, generation, and image occlusion tooling. Internal names such as
`ImportExportTab` and transfer-summary event types can remain as implementation
details. This label change must not move manual card creation out of the
existing Manage/Create flow; it only makes the top-level flashcards IA clearer.

After a rating, the review UI should display a visible recovery affordance such
as `Re-rate last card` or `Undo rating` for the same undo window currently used
by shortcut logic. It must remain visible even when the review advances to the
next card or reaches completion. `Ctrl/Cmd+Z` remains an accelerator, but the
visible button is the primary recovery path.

When a user studies a specific deck and clicks Create, the create drawer should
open with that deck selected. The user can still change decks. The selected deck
context should not leak into unrelated manual creation after the drawer closes or
after a later tab change.

Completion should only offer `Practice again` when the current review context
has cram or repeatable cards. If no cram cards exist, the UI should remove that
button, keep it absent if already absent, or replace it with an accurate next
action such as `Create card`, `Manage deck`, or `Open scheduler`.

### Implementation Shape

- Add the missing lazy import and route entry in
  `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`.
- Update locale keys and tab-label tests for the new user-facing label.
- Render the undo/re-rate control independently from the current card display
  branch in `ReviewTab.tsx`, reusing existing undo state:
  `showUndoButton`, `lastReviewedCard`, `undoCountdown`, and
  `handleUndoReview`.
- Add a deck handoff state in `FlashcardsManager.tsx` for Study to
  Manage/Create. Pass the selected deck ID through `ManageTab` into
  `FlashcardCreateDrawer` as an optional initial deck ID.
- Reset consumed create-deck handoff state after the drawer opens or closes.
- Gate completion `Practice again` using the same review/queue data that
  indicates whether cram can produce cards. If the current implementation has no
  `Practice again` control, add focused regression coverage or an explicit manual
  verification note instead of adding and then hiding new UI.

### Tests

Focused automated coverage:

- App extension sidepanel registry contains `/flashcards` and references
  `sidepanel-flashcards`.
- Rate card, observe visible re-rate/undo affordance, trigger re-rate, and
  confirm the previous card can be rated again.
- From Study with a selected deck, open Create and confirm the create drawer
  preselects that deck.
- Completion does not show `Practice again` when no cram cards are available.
- Existing tab-label consistency tests expect `Create & Import`.

Expected likely suites:

- Flashcards component Vitest tests under
  `apps/packages/ui/src/components/Flashcards`.
- Route registry tests under `apps/packages/ui/src/routes/__tests__` and/or the
  app extension route tests if present.
- Focused Playwright e2e for the rate to visible re-rate path if the flashcards
  e2e harness can seed cards deterministically.

## PR 2: Power-User Study And Session History

### Scope

PR 2 should contain the behavior changes that affect Study defaults and session
history semantics:

- All-deck Study shows the deck dashboard first instead of auto-starting a card.
- Explicit selected-deck review still starts efficiently.
- Recent session history displays the last-known user-facing deck name.
- Recent session history falls back gracefully when no name is available.

### Interaction Design

When no specific deck is selected, Study should land on a deck dashboard. The
dashboard can be an extracted component or a focused restructuring of the
existing no-active-card Study setup area; implementation planning should choose
the smallest option after inspecting current `ReviewTab.tsx`. The dashboard
should help power users choose the right next action: review all due cards, pick
a specific due deck, inspect counts, or navigate to management. The important
change is that users are not dropped directly into an all-deck card without
context.

Direct deck flows should remain fast. If the user starts from a deck card, deep
link, or selected deck context, they should not be forced through the all-deck
dashboard unless they explicitly clear the selection.

Recent sessions should show the best available user-facing deck name. If a deck
is renamed or deleted after the session, the session row should preserve the
name known at review time. If no name can be resolved, the fallback should be
plain language, such as `Deleted deck` or `Deck unavailable`, not `Deck 1` or a
raw mode key.

### Data Design

The preferred data contract is a preserved deck-name snapshot at the session
summary boundary. The API or normalized frontend model should expose an optional
field such as `deck_name_snapshot`, `deckNameSnapshot`, or an equivalent
existing name if one already exists. Any new field must be nullable and
backward-compatible for existing session records.

Resolution order in the UI should be:

1. Preserved session deck-name snapshot.
2. Current deck lookup by ID when the deck still exists.
3. Plain fallback copy for unavailable or deleted deck names.

Frontend-only current-deck lookup is not sufficient by itself because it cannot
handle deleted or renamed decks reliably. If implementation proves the backend
already returns enough data, use that contract instead of adding a new field.

### Implementation Shape

- Adjust `ReviewTab.tsx` so all-deck Study keeps `activeCard` empty and renders
  a deck dashboard until the user chooses a deck or an explicit all-due action.
  Because no reusable `DeckStudyDashboard` exists in the current checkout, PR 2
  should either extract a small dashboard component from the existing setup area
  or adapt the existing no-card branch directly.
- Preserve selected-deck and resume behavior so explicit contexts still reach a
  review card quickly.
- Add or normalize the optional deck-name snapshot in the recent-session model.
- Update `RecentStudySessions.tsx` to use the resolution order above and remove
  raw technical deck labels from normal user-facing rows.
- Add tests for dashboard-first all-deck Study, selected-deck fast path, and
  preserved session deck names.

### Tests

Focused automated coverage:

- All-deck Study renders a deck dashboard first.
- Selecting a specific deck from the dashboard starts the intended deck review.
- Starting from an already selected deck bypasses the all-deck dashboard when
  that is the existing fast path.
- Recent sessions render a preserved deck name when the current deck is missing
  or renamed.
- Recent sessions render graceful fallback copy when no name exists.

Backend or API-level coverage may be needed if a session-summary field is added.
If no backend field is required, add model-normalization and component tests
instead.

## Sequencing

1. Create a clean worktree for PR 1 from latest `dev`.
2. Create a Backlog.md implementation task for PR 1 before edits.
3. Implement PR 1 with tests and browser verification.
4. Merge PR 1 or rebase PR 2 on its completed branch, depending on review flow.
5. Create a clean worktree for PR 2 from latest `dev` after PR 1 is merged or
   otherwise stabilized.
6. Create a Backlog.md implementation task for PR 2 before edits.
7. Implement PR 2 with tests and browser verification.

PR 2 should not start from stale PR 1 assumptions. If PR 1 changes the review
state model or route handling in review, inspect that final code before changing
dashboard or session history behavior.

## Verification Gates

For both PRs:

- Run focused Vitest suites for changed flashcards components.
- Run focused route-registry tests for extension sidepanel route changes.
- Run a browser verification pass for the changed `/flashcards` workflows.
- Run TypeScript or package checks required by the touched package if feasible.
- Run Bandit only for touched Python/backend code. Document a skip for
  frontend-only PRs.
- Record known skipped checks in the Backlog.md task and final PR summary.

For PR 1 specifically:

- Browser or e2e proof that rating a card leaves visible re-rate available and
  re-rate works.
- Browser or component proof that Study selected deck to Create drawer preserves
  deck selection.

For PR 2 specifically:

- Browser or component proof that all-deck Study shows the deck dashboard first.
- Test proof that recent sessions preserve or resolve deck names without raw
  IDs in normal rows.

## Risks And Mitigations

- Risk: visible undo UI duplicates existing shortcut state.
  Mitigation: reuse the existing undo state and handler rather than adding a
  second undo model.

- Risk: deck prefill leaks into unrelated creates.
  Mitigation: treat deck prefill as a one-shot handoff and clear it after the
  drawer is opened or closed.

- Risk: all-deck dashboard-first breaks resume or selected deck fast paths.
  Mitigation: scope the default change to no selected deck and no explicit
  resume context. Add tests for the selected-deck path.

- Risk: preserving deck names requires backend schema work.
  Mitigation: first inspect the existing session summary payload. Add the
  smallest optional field only if current data cannot represent renamed/deleted
  deck names.

- Risk: extension route registries have diverged.
  Mitigation: add a direct app-extension registry test, not only a shared
  package route test.

## Acceptance Criteria

PR 1 is done when:

- The actual extension sidepanel registry exposes `/flashcards`.
- The visible create/import tab label is `Create & Import`.
- A user can rate a card, see visible re-rate/undo, and re-rate the previous
  card within the supported window.
- Create from a selected Study deck preselects that deck.
- `Practice again` is not offered when it cannot produce cards.
- Focused tests cover the above behavior.

PR 2 is done when:

- All-deck Study opens the deck dashboard first.
- Selected-deck Study still remains efficient.
- Recent session rows show preserved or resolved deck names.
- Recent session fallback copy is non-technical.
- Focused tests cover dashboard-first behavior and deck-name history.
