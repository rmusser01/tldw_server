# Chat Empty State Unified Onboarding Design

**Date:** 2026-04-21
**Surface:** Web `/chat` page and extension chat surface
**Status:** Approved in-session

---

## Goal

Turn the current first-run chat empty state from three visually separate blocks into one cohesive onboarding panel without changing the underlying actions or routes.

## Problem

The current empty state is fragmented:

- A hero card introduces chat and exposes the primary actions.
- A second section presents the five guided modes.
- A third footer section contains tips and the quick tour link.

This separation makes the first-run experience feel assembled from multiple pieces instead of feeling like one deliberate entry point.

## Non-Goals

- No changes to the starter mode behavior.
- No new onboarding flow, modal, or wizard.
- No feature-flag or settings gate.
- No migration of the chat empty state onto the newer generic `EmptyState` primitive in this change.

## UX Direction

Use a single onboarding card that contains four internal layers:

1. Hero block
   - Keep the icon, title, and one short descriptive sentence.
   - Preserve the disconnected and demo copy branches.

2. Primary action row
   - Keep `Start chatting`.
   - Keep `Quick Ingest`.
   - Preserve existing click behavior.

3. Guided mode deck
   - Present all five modes with equal visual weight.
   - Keep them inside the same card as the hero instead of in a separate external section.
   - Use subtle internal structure such as a divider and section label, not a second bordered box.

4. Support footer
   - Keep the helper tips and `Take a quick tour` inside the same card.
   - Reduce contrast so they support the primary choices instead of competing with them.

## Interaction Requirements

- `Start chatting` must still dispatch the general starter event and focus the composer.
- `Quick Ingest` must still open the ingest flow.
- `General chat`, `Compare AI models side-by-side`, `Chat as a character`, and `Search your documents` must keep dispatching their existing starter events.
- `Deep research` must still navigate to the research launch route.
- `Take a quick tour` must still open the help modal.
- `Open Settings` in the disconnected state must still navigate to `/settings/tldw`.

## Visual Requirements

- One outer onboarding shell only.
- No standalone `Start with a guided mode:` block below the hero.
- No standalone tips footer outside the main card.
- Guided mode tiles remain equally visible.
- Internal hierarchy should read in this order:
  - start a chat
  - choose how to begin
  - skim lightweight help

## Implementation Scope

Keep the change scoped to:

- `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

`FeatureEmptyState` remains untouched unless a small helper prop becomes unavoidable, which is not expected for this change.

## Testing Strategy

- Add a regression test that expects a unified onboarding shell.
- Assert the old `Start with a guided mode:` heading is gone.
- Keep or update existing tests for starter telemetry, deep research navigation, and help modal behavior.
- Preserve disconnected-state coverage for `Open Settings`.
- Run focused Vitest coverage for the touched empty-state tests.

## Risks

- Layout-only changes can break tests that depended on the old mocked `FeatureEmptyState` boundary.
- The new unified card must stay readable on narrow widths and in the extension surface.

## Acceptance Criteria

- The empty chat state renders as one cohesive card.
- All five guided modes remain equally visible.
- Existing actions behave the same as before.
- The old stacked three-section feel is removed.
