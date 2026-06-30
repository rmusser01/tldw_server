# /chat Siderail Collapse Design

## Task

Backlog: `TASK-485`

## Context

The `/chat` route is rendered by `apps/tldw-frontend/pages/chat/index.tsx`, which loads the shared `Playground` route. The page sits inside `OptionLayout` from `apps/packages/ui/src/components/Layouts/Layout.tsx`.

Desktop `/chat` currently has two side-adjacent surfaces:

- Left chat navigation rail: `ChatSidebar` in `apps/packages/ui/src/components/Common/ChatSidebar.tsx`, controlled by `chatSidebarCollapsed` in `apps/packages/ui/src/store/layout-ui.ts`.
- Right artifact rail: rendered from `Playground` when `useArtifactsStore().isOpen` is true.

Live layout inspection showed the current collapsed states are not discoverable:

- With the left rail collapsed and the right artifact rail open, there is no clear same-side affordance showing that the left rail can be expanded.
- With both rails collapsed, there is no visible indication that either rail exists.
- The chat body should gain horizontal space when a rail collapses; it should not move downward or lose its vertical anchoring.

## Goal

Make collapsed desktop `/chat` rails disappear from layout while leaving a same-side edge-mounted expand button. Expanding a rail should restore that rail on the side where the affordance appeared.

For this task, desktop side-rail behavior means `lg` and wider viewports, matching the breakpoint where the artifact panel is currently eligible to render as a right-hand rail.

## Non-Goals

- Do not redesign the chat cockpit, composer, header, or empty state.
- Do not change mobile drawer behavior except where shared code needs a guard to avoid desktop-only controls on mobile.
- Do not change artifact creation, artifact content rendering, or chat history loading.
- Do not add a new persisted settings model unless an existing store cannot express the required open/collapsed state.

## UX Contract

### Left Rail

When the left chat rail is collapsed on desktop:

- The left rail does not reserve its previous full width.
- A compact expand button is mounted on the left content edge.
- The expand button uses a clear accessible label, such as "Expand chat rail".
- Activating the button restores the left chat rail on the left side.
- The chat transcript and composer widen into the released horizontal space.

When the left rail is expanded on desktop:

- Existing `ChatSidebar` content and collapse behavior remain available.
- The collapse control inside the rail collapses the rail back to the left-edge expand button state.

### Right Rail

When the right artifact rail is collapsed on desktop:

- The artifact rail does not reserve its previous width.
- A compact expand button is mounted on the right content edge when an artifact is available.
- The expand button uses a clear accessible label, such as "Expand artifacts rail".
- Activating the button restores the artifact rail on the right side.
- The chat transcript and composer widen into the released horizontal space.

When the right artifact rail is expanded on desktop:

- The existing artifact panel content remains unchanged.
- The close control collapses the artifact rail back to the right-edge expand button state when an artifact remains available.

### Both Rails Collapsed

When both rails are collapsed on desktop:

- The left-edge expand button is visible, and the right-edge expand button is visible when an active artifact exists.
- The chat body occupies the combined available width.
- The composer remains docked at the bottom and the transcript remains vertically anchored.
- The page does not introduce a blank horizontal gutter where either rail used to be.

## Layout Model

Use the current desktop flex layout at `lg` and wider viewports, but make open rails conditional layout participants:

- Left rail open: render `ChatSidebar` as a left flex child.
- Left rail collapsed: omit the full-width `ChatSidebar` flex child and render a left-edge expand control over the chat shell.
- Right rail open: render the artifact panel as the current right flex child.
- Right rail collapsed: omit the artifact panel flex child and render a right-edge expand control over the chat shell when an active artifact exists.

Edge controls should be positioned within the `/chat` page shell rather than in the document body. This keeps them scoped to the route, avoids covering global browser UI, and makes route teardown straightforward.

## Component Boundaries

Expected touch points for implementation planning:

- `apps/packages/ui/src/components/Layouts/Layout.tsx`
  - Decides whether the desktop left rail is rendered as a layout participant.
  - Owns the left-edge expand control because the left rail belongs to the outer option layout.
  - Any layout changes must be scoped to `/chat`/`Playground` so other option routes keep their current rail behavior.
- `apps/packages/ui/src/components/Common/ChatSidebar.tsx`
  - Keeps expanded rail content and the internal collapse button.
  - Should not own the collapsed edge button if the collapsed rail is no longer mounted.
- `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
  - Decides whether the right artifact rail is rendered as a layout participant.
  - Owns the right-edge expand control because the artifact rail belongs to the chat playground shell.
- `apps/packages/ui/src/store/layout-ui.ts` and `apps/packages/ui/src/store/artifacts.tsx`
  - Existing state should remain the source of truth unless implementation planning finds a real gap.

## Accessibility

- Edge expand controls are real buttons.
- Buttons have explicit `aria-label` text.
- Buttons remain keyboard reachable in normal tab order.
- When a rail is expanded from an edge button, focus should move to the restored rail's existing collapse button or first meaningful control.
- When a rail is collapsed from inside the rail, focus should move to the corresponding edge expand button if possible.

## Responsive Behavior

- Desktop edge-button behavior applies at `lg` and wider viewports.
- At `md`-only widths, 768px through 1023px, preserve the existing medium/tablet behavior. Do not introduce the new desktop edge expand buttons there in this task.
- Mobile should continue to use existing drawer or sheet behavior.
- Edge buttons should not appear below `lg` or when the corresponding rail is not a desktop layout participant.

## Verification Plan

Implementation should include focused regression coverage plus browser verification:

- Unit or component coverage for `ChatSidebar`/layout state showing the collapsed left rail is not rendered as a full rail and that a left-edge expand button exists.
- Unit or component coverage for `Playground` artifact rail state showing the collapsed right rail is not rendered as a full rail and that a right-edge expand button exists when an artifact is available.
- Browser smoke at desktop viewport:
  - Left collapsed, right open: left edge expand button visible; chat uses freed left width.
  - Right collapsed, left open or collapsed: right edge expand button visible; chat uses freed right width.
  - Both collapsed: both edge buttons visible; chat body/composer remain vertically anchored.
  - Record layout measurements, not only screenshots: chat shell width increases after each rail collapse, chat shell top remains stable, and composer bottom remains docked.
- Browser smoke at mobile viewport:
  - No desktop edge buttons appear unexpectedly.
  - Existing drawer/sheet controls still work.
- Browser smoke at medium/tablet viewport, 768px through 1023px:
  - No new desktop edge buttons appear.
  - Existing medium layout behavior is preserved.

## Open Questions

None. The requested collapsed state is an edge-mounted expand button, not a persistent narrow icon rail.
