# Flashcards Global Tag Suggestions Design

Date: 2026-04-02
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Add global flashcard tag suggestions to the flashcard create and edit drawers used by both the WebUI and extension. The current drawers only allow free typing, which forces users to manually retype tags that already exist elsewhere in their flashcards. The design introduces a dedicated backend endpoint for listing flashcard tags across all flashcards the current user can access, then reuses that endpoint through one shared tag-picker component in the shared UI package. The picker remains hybrid: users can choose existing tags from suggestions or type a brand-new tag when needed.

## Problem

Flashcard tag entry currently works as raw free typing in the two flows where users most expect selection from an existing list.

Today:

- [FlashcardCreateDrawer.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx) renders the tag field as `Select mode="tags"` with `open={false}` and no suggestion source.
- [FlashcardEditDrawer.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx) does the same.
- The only existing tag-suggestion logic is [useFlashcardQueries.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts), where `useTagSuggestionsQuery` scans paginated flashcard results and is only used by [ManageTab.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/tabs/ManageTab.tsx) for filter chips.
- The backend in [flashcards.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/endpoints/flashcards.py) exposes per-card tag update and fetch operations, but it does not expose a collection-level endpoint for listing flashcard tags.

The result is a product inconsistency:

- filtering has suggestions
- create and edit do not
- existing tags are hard to reuse
- users must manually remember and re-enter tags instead of selecting from the available list

This is especially visible because both the WebUI page and extension route already reuse the same shared flashcards workspace. The issue is not surface-specific; it is a shared component gap.

## Goals

- Let users select from existing flashcard tags when creating a card.
- Let users select from existing flashcard tags when editing a card.
- Make suggestions global across all flashcards the current user can access, not limited to one deck.
- Preserve the ability to type a brand-new tag that does not yet exist.
- Keep WebUI and extension behavior identical by changing the shared flashcards UI components.
- Use a dedicated backend contract for create/edit typeahead suggestions instead of reusing the current client-side page scan.

## Non-Goals

- Redesign flashcard management filters outside the create and edit drawers.
- Replace or redesign the current Manage-tab filter suggestion implementation in this change.
- Change the underlying persistence format for flashcard tags.
- Restrict users to selection-only tags.
- Introduce deck-scoped tag ownership rules.
- Redesign broader keyword management across notes, prompts, or media.

## Requirements Confirmed With User

- Suggestions should cover all accessible flashcard tags, not just the current deck.
- The fix should apply to both create and edit, not only create.
- The field must remain hybrid: existing tags should be selectable, and new tags should still be manually enterable.
- The issue matters for both the WebUI and extension.

## Current State

### Shared UI owns flashcard behavior

The route wrappers in [apps/tldw-frontend/pages/flashcards.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/pages/flashcards.tsx) and [apps/tldw-frontend/extension/routes/option-flashcards.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/extension/routes/option-flashcards.tsx) both land in the shared flashcards workspace implemented under [apps/packages/ui/src/components/Flashcards](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards).

That means one shared component change is the correct place to restore parity.

### Existing suggestion logic is in the wrong place for this use case

[useTagSuggestionsQuery](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/hooks/useFlashcardQueries.ts#L399) already exists, but it currently:

- scans flashcards page by page through the list API
- is bounded by client-side paging limits
- only powers Manage-tab filtering
- does not feed create or edit

This is an acceptable stopgap for filter chips, but it is not the right long-term source for global create/edit suggestions.

### The backend already models flashcard tags as real linked keywords

The flashcards database layer in [ChaChaNotes_DB.py](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py) already has:

- a `keywords` table
- a `flashcard_keywords` link table
- `flashcards.tags_json` as a mirrored serialized field
- helper methods such as `get_keywords_for_flashcard` and `set_flashcard_tags`

This is enough structure to support a first-class flashcard tag listing endpoint without scanning every card.

## Approaches Considered

### Approach 1: Reuse the current client-side page scan in create and edit

Wire the existing `useTagSuggestionsQuery` into the create and edit drawers and keep the field hybrid.

Pros:

- Fastest frontend-only change
- No backend contract change
- Minimal initial implementation scope

Cons:

- Suggestions are only as complete as the scan cap allows
- Typeahead still requires page-scanning card lists
- Duplicates flashcards-list concerns into a picker concern
- Poor fit for a global suggestion feature expected to scale

### Approach 2: Add a dedicated backend flashcard tag suggestions endpoint and shared picker

Expose a collection-level tag list endpoint backed by flashcard keyword links, then use it in one shared picker for create and edit.

Pros:

- Correct source of truth for global suggestions
- Efficient query path
- Works cleanly for both WebUI and extension
- Supports ranking and query filtering without card scans
- Easier to test and evolve later

Cons:

- Requires coordinated backend and frontend work
- Slightly larger initial implementation

### Approach 3: Suggest only recent or local tags

Keep suggestions entirely client-side and show tags from recent edits, current deck, or cached local state.

Pros:

- Lowest backend effort
- Small UI change

Cons:

- Does not satisfy the complaint about selecting from the available list
- Weak consistency across sessions and surfaces
- Easy for users to miss valid existing tags

## Recommendation

Use Approach 2.

Add a dedicated backend endpoint for flashcard tag suggestions and consume it through a shared hybrid picker used by both create and edit drawers.

## Proposed Design

### 1. Add a flashcard tag suggestions endpoint

Add a collection-level route under the flashcards API:

- `GET /api/v1/flashcards/tags`

Proposed query parameters:

- `q`: optional search string for typeahead narrowing
- `limit`: optional result limit with a small default such as `20` or `50`

Visibility contract:

- the initial endpoint is intentionally global for the current user
- it returns tags across all non-deleted flashcards the current user can access
- it must include both general-scope flashcards and workspace-scoped flashcards, not just decks where `workspace_id IS NULL`
- it does not inherit current deck filters, workspace filters, or Manage-tab visibility toggles
- create and edit drawers call it without deck or workspace scoping because the agreed behavior is “all accessible flashcard tags”

Implementation note:

- the endpoint should not blindly reuse the default `list_flashcards` visibility behavior because that path defaults to general-scope items when no workspace flags are provided

Proposed response shape:

```json
{
  "items": [
    { "tag": "biology", "count": 42 },
    { "tag": "chapter-1", "count": 17 }
  ],
  "count": 2
}
```

Per-item `count` means the number of distinct non-deleted flashcards currently using that tag. Top-level `count` means the number of suggestion items returned in the response.

Rationale:

- `tag` gives the selectable value
- `count` supports stable ranking by popularity
- query filtering allows typeahead without fetching the full corpus
- the narrow contract keeps this feature focused on create/edit instead of broad filter refactors
- the explicit global visibility rule prevents planners from accidentally coupling the picker to current workspace-filter state

### 2. Back the endpoint with flashcard keyword links, not flashcard list scanning

The endpoint should query active flashcard tags using the existing flashcard-keyword link model:

- include only non-deleted flashcards
- include only non-deleted keywords
- exclude flashcards whose joined deck row is deleted, while still allowing deckless flashcards
- aggregate usage counts by keyword text
- apply `q` as a trimmed, case-insensitive substring filter
- sort by `count DESC`, then tag text case-insensitively

This avoids the current client-side `MAX_SCAN` approach and aligns the API with the actual domain model already present in `ChaChaNotes_DB`.

Scope note:

- this new endpoint is for create/edit tag suggestions only in the initial change
- the existing Manage-tab filter suggestion flow remains unchanged for now
- do not silently repurpose the current Manage-tab page-scan hook unless its existing behavior is preserved for that tab

Routing note:

- because the same router also exposes a later `GET /{card_uuid}` alias route, the new static `GET /tags` route must be registered before that dynamic route so the literal path `tags` is not captured as a `card_uuid`

### 3. Add one shared `FlashcardTagPicker` component

Create one shared picker component in the flashcards component layer and reuse it in:

- [FlashcardCreateDrawer.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/FlashcardCreateDrawer.tsx)
- [FlashcardEditDrawer.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/FlashcardEditDrawer.tsx)

Expected behavior:

- wraps Ant `Select` in `mode="tags"`
- loads suggestions from the new endpoint
- shows a normal dropdown instead of `open={false}`
- updates suggestions as the user types with a short debounce
- forwards the active request abort signal so stale typeahead requests can be cancelled cleanly
- allows selecting existing tags from the dropdown
- allows typing a new tag and confirming it with `Enter`
- trims whitespace and dedupes case-insensitively within the current value

This keeps the create/edit UX aligned instead of letting the two drawers drift.

### 4. Keep the field hybrid

The picker must not become selection-only.

Required behavior:

- if a user types an existing tag, they can select it
- if a user types a brand-new tag, they can still create it
- if there are no matches, the absence of suggestions must not block submission

This preserves current flexibility while solving the reuse problem.

### 5. Query lifecycle and invalidation

Create and edit flows should invalidate the flashcard tag-suggestions query after a successful mutation.

This ensures:

- a newly added tag appears immediately in later create or edit sessions
- the suggestion list stays in sync without full page reloads
- both WebUI and extension see the same refreshed suggestion state because they share the same query client logic in shared UI

Implementation note:

- if the new suggestion query key stays under the existing `flashcards:` namespace, the current shared invalidation helper can keep handling this without special-case mutation logic
- the suggestion query should only be enabled while the drawer and picker are active, and its request helper should accept React Query abort signals to avoid stale results during rapid typing

### 6. Preserve current create/edit behavior if suggestions fail

Suggestion loading is an enhancement layer, not a hard dependency.

If the tag suggestion request fails:

- create and edit drawers must still open normally
- users must still be able to type tags freely
- card save/create must not be blocked
- the picker should fall back to an empty non-blocking suggestion state while preserving normal free typing

This keeps the blast radius small. The complaint is about missing convenience, not data loss or failed persistence.

## UX And Interaction

### Create drawer

When the user opens the advanced options section:

- the tag field can be focused and searched
- existing global tags appear as selectable options
- selecting an option adds a tag chip
- typing an unmatched value and pressing `Enter` adds a new chip

The rest of the create flow does not change.

### Edit drawer

When the user opens a card for editing:

- existing tags load as chips
- the same global suggestion dropdown is available for adding more tags
- unmatched values can still be added manually

The edit drawer should use the same picker instead of its own separate configuration.

### Parity across WebUI and extension

Because the web page and extension route both render the shared flashcards workspace from `apps/packages/ui`, the design intentionally avoids web-only and extension-only variants. One picker implementation should cover both product surfaces.

## Error Handling

- Suggestion fetch failure must not block create or edit.
- Empty results must still allow manual tag entry.
- The picker should ignore empty strings and whitespace-only values.
- Duplicate values should be collapsed case-insensitively within the current field value.
- Backend create/update validation remains authoritative. Suggestions are convenience data only.

## Testing

### Backend tests

Add unit or integration coverage for the new flashcard tags endpoint verifying:

- unique tags are returned only once
- deleted flashcards do not contribute suggestions
- deleted keywords do not contribute suggestions
- flashcards attached to deleted decks do not contribute suggestions
- `q` filters results correctly
- results are ordered by usage count, then alphabetically
- `limit` is respected

### Frontend component tests

Add focused tests for the shared picker verifying:

- existing tags render as selectable suggestions
- users can still create a brand-new tag not in the list
- duplicate tag entry is normalized case-insensitively
- whitespace-only tags are ignored
- fetch failure falls back to non-blocking free typing
- the picker exposes stable test ids or labels that create, edit, and E2E coverage can target consistently

### Drawer tests

Add or update tests for:

- create drawer suggestion selection and submission
- edit drawer suggestion selection and submission
- edit drawer preservation of pre-existing tags
- suggestion query invalidation after successful create or edit

### E2E coverage

Add one focused end-to-end flow that proves the actual complaint is fixed:

- create a flashcard with an existing tag by selecting from suggestions instead of retyping it
- edit a flashcard and add another existing tag from suggestions

Because the route surfaces are shared, one web flow plus one extension-path smoke/parity check is sufficient if it proves the shared component is reachable in both shells.

The extension-path parity check is required, not optional.

## Risks

### Overloading the suggestions endpoint

If the picker fetches on every keystroke without debounce or caching, the new endpoint could become noisy. The frontend should debounce searches and the query should stay small and cacheable.

### Query semantics drift

If the backend uses one normalization rule and the frontend uses another, duplicate tags could appear in suggestions or chips. The implementation should align on trimming and case-insensitive dedupe behavior.

### Hidden future scope creep

Adding deck/workspace filters to the endpoint contract is acceptable for reuse, but the initial implementation should not expand the current feature into deck-scoped UX changes or broader keyword-management work.

## Decision

Proceed with a shared create/edit parity fix:

- add `GET /api/v1/flashcards/tags`
- back it with flashcard keyword aggregation rather than card-list scans
- add one shared hybrid `FlashcardTagPicker`
- replace the raw tag fields in create and edit drawers with that picker
- invalidate suggestions after successful mutations
- cover backend, component, drawer, and parity paths with tests
