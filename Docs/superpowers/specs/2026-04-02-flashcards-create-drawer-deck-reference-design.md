# Flashcards Create Drawer Deck Reference Design

Date: 2026-04-02
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Add a deck-specific reference section to the flashcard create drawer so users can see cards they already created while authoring a new one. The new surface stays read-only, appears inside the existing create flow, and is intentionally lightweight: it shows recently created cards from the selected deck plus a searchable list of cards from that same deck.

The goal is not duplicate prevention, editing, or active review. The goal is to let users reference existing deck content without leaving the authoring flow.

## Problem

The current flashcards experience separates authoring and reference too aggressively for users who build decks incrementally.

Today:

- [`FlashcardCreateDrawer.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx) lets users create a card, but it does not show any of the cards already in the selected deck.
- [`ReviewTab.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx) supports studying cards, but it is a separate surface from authoring.
- [`ManageTab.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx) supports browsing and editing existing cards, but opening it means leaving the create flow.

That makes it harder to:

- maintain consistent wording across related cards
- remember what has already been covered in the current deck
- quickly reference prior questions and answers while drafting the next one

The reported need is not “find duplicates automatically.” It is “let me see what I already created while I am adding new cards to this deck.”

## Goals

- Let users reference existing cards from the currently selected deck while creating a new card.
- Keep the reference surface inside the existing create drawer.
- Show recently created cards from the selected deck.
- Let users search within the selected deck without leaving the drawer.
- Show both front and back content for referenced cards.
- Refresh the recent list after `Create & Add Another` so the just-created card becomes visible immediately.
- Keep the feature read-only and low-friction.
- Preserve shared behavior across the WebUI and extension because both reuse the shared flashcards UI package.

## Non-Goals

- Turn the create drawer into a second `Manage` tab.
- Add edit, delete, move, or review-rating actions to the reference section.
- Add duplicate detection or similarity ranking in this feature.
- Add contextual search based on what the user is typing into `front` or `back`.
- Redesign the create drawer entry point or replace it with a new standalone authoring page.
- Add new bulk APIs or whole-deck browsing workflows for this change.

## Requirements Confirmed With User

- The reference area is for active reference while authoring, not for active study actions.
- The surface should be read-only.
- The default content should be:
  - recently created cards from the selected deck
  - plus a searchable list of all cards in the selected deck
- The reference area can live in an expandable section inside the drawer rather than always being visible.
- Referenced cards should show both front and back immediately.
- After `Create & Add Another`, the reference area should refresh so the new card appears in the recent list.

## Current State

### Authoring already has the right entry point

[`ManageTab.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx) already opens [`FlashcardCreateDrawer.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx) as the quick-create surface. Keeping that entry point is the correct scope for this feature.

### The create drawer already knows the selected deck

`FlashcardCreateDrawer` already watches `deck_id`, loads deck options, supports inline deck creation, and preserves deck selection across `Create & Add Another`. That makes it the natural place to add deck-scoped reference content.

### The existing query layer can already fetch deck-scoped cards

[`useFlashcardQueries.ts`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts) already exposes `listFlashcards`-backed query patterns keyed by deck and search parameters. The list API supports:

- `deck_id`
- `q`
- `limit`
- `offset`
- `order_by=due_at|created_at`

That is enough for a first pass at:

- deck-scoped recent cards
- deck-scoped search

### The recent-cards hook must not inherit Manage's created-sort behavior

The backend flashcards list query already returns `order_by=created_at` as newest-first.

However, the existing `Manage` query path re-sorts `"created"` client-side for its own UI behavior. That means this feature should not blindly reuse Manage's created-sort helper for the drawer reference section.

Design implication:

- the deck reference recent-cards hook should use the raw API response ordering for `order_by=created_at`
- the hook should not reverse or re-sort that response unless a proven API inconsistency is discovered during implementation

This keeps `Recently created` aligned with the actual requirement: newest cards first.

## Approaches Considered

### Approach 1: Add a collapsible reference section inside the existing create drawer

Show an `Existing cards in this deck` section once a deck is selected. Keep it collapsed by default. When expanded, it shows recent cards first, then a deck search field and matching cards.

Pros:

- Solves the exact request with minimal workflow change
- Reuses the existing create surface
- Keeps authoring and reference in one place
- Lower implementation and responsive-layout risk
- Fits the current shared UI architecture

Cons:

- Drawer width constrains how much reference content can be shown at once
- Requires careful control of vertical space so the form does not feel cramped

### Approach 2: Replace the create drawer with a wider two-pane create-and-reference surface

Keep the form on one side and show an always-visible deck browser on the other.

Pros:

- Best visibility for heavy authoring sessions
- Clear separation between form and reference content

Cons:

- Much larger UX and layout change
- More mobile and extension complexity
- Not needed for the user’s stated requirement

### Approach 3: Launch a separate modal or secondary browser from the create drawer

Keep the current drawer unchanged and add a button that opens another surface to browse current-deck cards.

Pros:

- Smallest layout disruption
- Easy to isolate technically

Cons:

- Reintroduces context switching
- Undercuts the goal of “see cards while authoring”
- Feels like a workaround rather than an integrated solution

## Recommendation

Use Approach 1.

Add a collapsible, read-only `Existing cards in this deck` section inside the current create drawer. This directly addresses the reported need without turning the create flow into a new multi-pane editor.

## Proposed Design

### 1. Add a deck reference section to the create drawer

Inside [`FlashcardCreateDrawer.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx), add an expandable section below deck selection and above the main content fields.

Proposed title:

- `Existing cards in this deck`

Visibility rules:

- hide the section when no deck is selected
- show it when a deck is selected
- keep it collapsed by default

Why collapsed by default:

- matches the user’s preference
- keeps the create drawer primarily focused on authoring
- prevents the form from feeling heavier for users who do not need reference content on every card

### 2. Expanded section layout

When expanded, the section shows two read-only subareas:

- `Recently created`
- `Search this deck`

#### Recently created

This block shows a short list of the newest cards for the selected deck.

Display rules:

- both front and back are shown for each card
- cards are compact but readable
- tags or metadata are optional and should stay secondary
- the list is intentionally bounded to a small count so it remains a reference aid, not a full browser

#### Search this deck

Below the recent list, show:

- a small search input
- a result list scoped to the selected deck

Behavior:

- empty search shows no separate result list beyond `Recently created`
- entering text triggers a debounced deck-scoped search
- v1 search results should request `order_by=created_at` so the newest matching cards appear first
- matching cards render with both front and back visible
- search results remain read-only

This gives users both fast passive recall from recent cards and explicit lookup when they want to find older deck content.

### 3. Keep the section read-only

The section must not include:

- answer reveal toggles
- review ratings
- edit buttons
- delete or move actions
- selection or bulk controls

Reasoning:

- the user explicitly wants reference, not in-drawer study actions
- adding operational controls would blur the create surface into a second manage surface
- keeping it read-only reduces state complexity and keeps the feature aligned with the request

### 4. Refresh behavior after creation

After `Create & Add Another` succeeds:

- preserve current behavior for form reset and deck/template retention
- refresh the deck reference section for the selected deck
- ensure the newly created card appears at the top of `Recently created`
- preserve the reference section's expanded or collapsed state across repeated authoring in the same open drawer session
- preserve the active deck-scoped search term across `Create & Add Another`; only deck changes or drawer close reset it

After plain `Create` succeeds:

- no special extra behavior is required beyond normal invalidation, because the drawer closes

### 5. Deck-change behavior

When the selected deck changes:

- clear the deck search input
- clear any deck-specific search results from the prior deck
- swap the recent cards list to the newly selected deck

When inline deck creation succeeds:

- select the newly created deck as today
- show the reference section in its empty state for that deck

## Data And Query Design

### Drawer query responsibilities

Add a small drawer-scoped query layer that depends on `selectedDeckId`.

The feature needs two data paths:

- recent cards by deck
- search results by deck and query text

### Recent cards by deck

For v1, use the existing flashcards list API with:

- `deck_id=<selected deck>`
- `order_by=created_at`
- a bounded `limit`

Use the response in its native backend order for the recent slice.

Why this is acceptable for v1:

- the backend already returns `created_at` newest-first for flashcards
- the drawer only needs a small recent reference slice
- the feature does not require additional sort-direction API work up front

Future-friendly note:

- if flashcards list ordering gains an explicit direction parameter later, the drawer should adopt it
- this design does not require that backend change up front

### Search results by deck

Use the existing flashcards list API with:

- `deck_id=<selected deck>`
- `q=<debounced query>`
- `order_by=created_at`
- a modest `limit`
- `due_status=all` if needed by the current client contract

The search scope is intentionally limited to the selected deck. This matches the user request and avoids turning the create drawer into a global search surface.

### Query keys and cache behavior

Use dedicated React Query keys so the drawer can refresh reference data without broad invalidation.

At minimum, include:

- `deckId`
- `searchTerm`

Post-create behavior:

- after `Create & Add Another`, refresh the active deck’s recent query
- also refresh the active search query if a search term is present

This can be implemented either through targeted invalidation or by patching the newly created card into the recent cache for the active deck. The design does not require optimistic insertion; it only requires that the user sees the new card quickly and reliably.

## Component Boundaries

### Keep `FlashcardCreateDrawer` focused on authoring orchestration

[`FlashcardCreateDrawer.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx) should remain responsible for:

- form state
- create mutations
- selected deck state
- inline deck creation behavior
- post-create reset behavior

### Add one focused deck-reference component

Introduce a focused read-only component, for example:

- `FlashcardDeckReferenceSection`

Responsibilities:

- render the collapsible section
- render recent cards
- own the deck search input
- render loading, empty, and error states
- render read-only card previews

### Add small deck-reference hooks

Use one small pair of hooks or one hook with two query modes in the existing flashcards hooks area, for example:

- `useFlashcardDeckRecentCardsQuery`
- `useFlashcardDeckSearchQuery`

This keeps data concerns out of the create-drawer form logic.

### Prefer a small read-only preview row over reusing heavy manage/review rows

Do not directly reuse the `Manage` row or `Review` active-card components for this feature.

Reason:

- those components carry extra behaviors and assumptions that this surface does not need
- the create drawer needs a smaller, denser, read-only presentation
- a simple preview row is easier to control visually and behaviorally

## Error Handling And Empty States

### Empty states

No deck selected:

- hide the whole section

Selected deck has no cards:

- show a small message such as `No cards in this deck yet`

Search has no matches:

- keep `Recently created` visible
- show a `No matching cards` message in the search results area

### Error states

If recent or search fetches fail:

- show a compact inline error within the section
- provide a local retry affordance
- do not block card creation
- avoid promoting routine section fetch failures to global flow-stopping errors

### Loading states

When loading recent cards:

- show a small inline loading state within the expanded section

When a search is in flight:

- keep the search input usable
- show a local loading indicator for search results

## Testing Strategy

### Unit and component tests

Add focused tests for the new reference section covering:

- hidden when no deck is selected
- collapsed by default when a deck is selected
- empty-deck state
- recent cards render both front and back
- deck search renders matching cards
- no-results state
- local error state

### Hook and service tests

Add tests for:

- deck-scoped recent query parameters
- deck-scoped search query parameters
- bounded recent-card reordering to newest-first
- post-create refresh behavior for the active deck reference queries

### Drawer integration tests

Add integration coverage for:

- selecting a deck and expanding the reference section
- performing `Create & Add Another`
- seeing the new card appear in the recent list after success

### E2E coverage

If this workflow already has flashcards create-flow E2E coverage, extend it to verify:

- the drawer can show current-deck reference content
- the recent list updates after repeated authoring inside one drawer session

## Acceptance Criteria

- A user creating a flashcard can expand an `Existing cards in this deck` section without leaving the create drawer.
- The section appears only when a deck is selected.
- The section is collapsed by default.
- The expanded section shows recently created cards from the selected deck.
- Each referenced card shows both front and back content.
- The section includes a search input that searches only within the selected deck.
- Search results are read-only.
- After `Create & Add Another`, the recent list refreshes and the newly created card becomes visible.
- Changing decks resets the section to the new deck’s content and clears the old deck’s search term.
- Failures in the reference section do not block flashcard creation.

## Implementation Notes For Planning

- Keep the scope in the shared UI package so the WebUI and extension stay aligned.
- Do not reuse `applyManageClientSort(..., "created")` for the drawer reference queries; that would invert the intended recent-cards ordering.
- Prefer a bounded, drawer-specific solution over a general-purpose deck browser.
- Do not add duplicate detection, inline edit affordances, or study actions unless separately requested.
- Pin the exact default recent-slice size and search debounce delay in the implementation plan so the shared UI behaves consistently across surfaces.
