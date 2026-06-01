# Companion Home Header Shortcut Design

Backlog task: TASK-492

## Goal

Add a fast, explicit path to Companion Home from the shared WebUI chat header and make Companion Home discoverable in the page directory launcher, including the legacy sheet view.

## Current Context

The shared options shell renders `apps/packages/ui/src/components/Layouts/Header.tsx`, which delegates the chat-route top bar to `ChatHeader.tsx`. In that header, the signpost icon toggles `HeaderShortcuts`, the page directory modal. `HeaderShortcuts` renders both the current two-panel launcher and the legacy sheet view from the same shortcut data in `header-shortcut-items.ts`.

The `/` route is already defined as the home resolver for options and sidepanel surfaces. For this change, `/` is the desired Companion Home destination.

## Recommended Approach

Implement Companion Home as a first-class shared shortcut and reuse that shortcut in both entry points:

1. Add a `Companion Home` shortcut item targeting `/` to the header shortcut metadata.
2. Place it in a small top-level `Start` group so it appears before Chat and the other workspace groups.
3. Add a `House` icon button in `ChatHeader` immediately to the left of the existing signpost button.
4. Navigate to `/` when the home button is clicked.
5. Use accessible labels and tooltips that clearly say `Companion Home`.

This keeps the launcher and legacy sheet behavior data-driven. Once the shortcut metadata includes the new item, both launcher views should list it without duplicate modal-specific rendering.

## Components And Boundaries

`ChatHeader.tsx` owns the visible header controls and should only receive a navigation callback or use the existing router hook in a narrow way. The new button should follow the existing signpost styling, focus ring classes, tooltip pattern, and icon sizing.

`header-shortcut-items.ts` owns launcher content. It should add the new item and import the Lucide `House` icon. The `HeaderShortcutId` union in `ui-settings.ts` should include the new stable id, likely `companion-home`.

`HeaderShortcuts.tsx` should not need structural changes because both current and legacy sheet views already render `getHeaderShortcutGroups()`. Any change there should be limited to tests if needed.

## Data Flow

The header home button navigates directly to `/`. The page directory modal reads selected shortcut ids from `HEADER_SHORTCUT_SELECTION_SETTING`, resolves groups through `getHeaderShortcutGroups()`, and renders the resulting list in either current or legacy mode. Adding `companion-home` to `HEADER_SHORTCUT_IDS` and `BASE_HEADER_SHORTCUT_GROUPS` makes it available in default selections and in reset-to-all behavior.

Hosted deployment filtering should include `/` only if Companion Home is valid and useful there. If hosted mode does not expose the same home surface, the hosted shortcut filter can omit it while the direct header button remains scoped to the shared options shell only after confirming route availability.

## Accessibility And UX

The button should be icon-only with:

- `aria-label="Companion Home"`
- matching `title`/Tooltip copy
- the same focus-visible ring classes as the signpost and settings controls
- placement immediately before the signpost icon, as requested

The launcher listing should show `Companion Home` with a short description such as `Return to your Companion Home dashboard`. It should be searchable by `home` and `companion`, and should render in both the current modal and legacy sheet view.

## Testing

Add focused tests in the existing frontend test suite:

- `ChatHeader.test.tsx`: verifies the Companion Home button is next to the shortcut toggle, has the correct accessible name, and calls navigation to `/`.
- `HeaderShortcuts.test.tsx`: verifies `Companion Home` appears in the current launcher and in the legacy sheet view.
- `header-shortcut-items.hosted.test.ts` or a nearby shortcut metadata test: verifies the shortcut has id `companion-home`, target `/`, and is included or intentionally excluded in hosted mode.

Manual verification after implementation should open the chat route, confirm the home icon appears immediately left of the signpost, confirm it navigates to `/`, and confirm the launcher legacy sheet lists Companion Home.

## Non-Goals

This change should not redesign the header, alter the signpost behavior, change the `/` route resolver, or add a new home route. It should not refactor the launcher modal beyond what is required to add the listing.

## Acceptance Criteria

- Chat header shows a Companion Home icon immediately left of the signpost icon.
- Clicking the icon navigates to `/`.
- The icon has accessible label and tooltip copy naming Companion Home.
- The page directory current view lists Companion Home.
- The legacy sheet view lists Companion Home.
- Existing shortcut filtering and reset behavior continue to work.
- Focused unit tests cover the header button and launcher listing.
