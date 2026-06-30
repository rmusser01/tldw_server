# Chat Workspace Terminal-Literal Design

Date: 2026-05-03
Owner: Codex collaboration session
Status: Reviewed with user, pending implementation planning

## Summary

Create a new visible web UI page, `/chat-workspace`, that recreates the generated Chatbook-inspired operator-console layout inside the existing tldw server web shell.

The page keeps the current web UI titlebar/header and left application sidebar. The new content area uses a dense Terminal-Literal layout: workspace/library/source panels on the left, chat as the center of gravity, and an inspector rail on the right for scope, staged sources, model/persona, approvals, task progress, and runtime state.

The first implementation should be a usable prototype, not a static mock. The core chat/composer/source-staging path should use existing app state and behavior where practical. Secondary panels may start as honest read-only v1 scaffolds, but they must not imply inactive capabilities are already working.

## Problem

The existing web UI has separate surfaces for chat and workspace-style research:

- `/chat` is the primary chat playground.
- `/workspace-playground` has sources, workspace state, chat, and generated outputs.
- The web shell already provides the global header, sidebar, command palette, model controls, quick ingest, notes dock, and backend recovery behavior.

The generated target design is different from both current surfaces. It treats chat as the main agentic control surface while keeping source scope, context staging, study outputs, and runtime status visible at all times. Recreating that experience directly in the web UI is feasible, but it should not require changing the Chatbook TUI or destabilizing the existing `/chat` and `/workspace-playground` routes.

## Goals

1. Add a first-class `/chat-workspace` route visible in navigation.
2. Preserve the existing web UI titlebar/header and left app sidebar.
3. Render a Terminal-Literal operator console inside the page content area.
4. Make the center chat panel the primary workflow surface.
5. Support visible staged context that is not silently sent.
6. Reuse existing chat, composer, workspace, source, model, persona, and backend-status behavior where feasible.
7. Keep inactive v1 capabilities clearly labeled instead of pretending they work.
8. Design component boundaries so the prototype can grow into the fully functional version.
9. Keep `/chat`, `/workspace-playground`, and the Chatbook TUI unchanged except for shared improvements that are explicitly needed and safe.

## Non-Goals

1. Replacing `/chat` in the first implementation.
2. Replacing `/workspace-playground` in the first implementation.
3. Changing the Textual Chatbook TUI.
4. Rebuilding the full chat streaming/request stack.
5. Building a complete approvals/tool-execution console in v1.
6. Building a complete study-materials management surface in v1.
7. Adding server-side APIs unless implementation planning finds an existing UI requirement cannot be met from current contracts.
8. Turning this into a marketing-style dashboard or landing page.

## Requirements Confirmed With User

1. The target is the web UI in `tldw_server2/apps`, not `tldw_chatbook`.
2. The new page should reuse the current web UI header/titlebar and sidebar.
3. The new page should follow the generated Terminal-Literal layout direction rather than the more web-native card direction.
4. The route should be visibly exposed in navigation immediately.
5. The first version should combine a real core interaction path with prototype scope.
6. The long-term goal is a fully functional `Chat Workspace` page.
7. The first version may use read-only v1 scaffolds for secondary panels if the core chat/source flow works.

## Current Web UI Context

### Shell

`apps/tldw-frontend/pages/_app.tsx` wraps authenticated pages in the shared `OptionLayout` from `components/layout/WebLayout.tsx`.

The shell already provides:

- persistent app header
- global command palette
- header shortcuts
- current model settings drawer
- chat/sidebar toggle behavior
- quick ingest modal host
- notes dock host
- backend unavailable handling
- responsive mobile drawer behavior

The new page should rely on that shell instead of creating another global frame.

### Chat

`apps/tldw-frontend/pages/chat/index.tsx` dynamically loads `@/routes/option-chat`, which renders `components/Option/Playground/Playground`.

Relevant existing capabilities include:

- transcript rendering
- composer behavior
- streaming and abort handling
- chat persistence
- model and system prompt state
- character/persona hooks
- attachments and context files
- route context coordination
- research-context attachment handling

The new route should reuse these capabilities where a wrapper is enough. It should avoid forking the chat stack unless a smaller shared extraction is required.

### Workspace Playground

`apps/tldw-frontend/pages/workspace-playground.tsx` dynamically loads `@/routes/option-workspace-playground`, which renders `components/Option/WorkspacePlayground/WorkspacePlayground`.

Relevant existing capabilities include:

- workspace store and persistence
- sources pane
- selected source state
- workspace chat pane
- generated artifacts/studio pane
- workspace status bar
- source transfer and split workflows
- responsive pane behavior

The new route should reuse workspace/source state and selected-source semantics where feasible. It should not restyle or replace `/workspace-playground` as part of this feature.

## Approaches Considered

### Approach 1: New isolated `Chat Workspace` route

Add a visible `/chat-workspace` route with new page-level layout components. Reuse shared chat and workspace internals where practical.

Pros:

- matches the requested new page
- keeps existing routes stable
- lets the design intentionally follow the Terminal-Literal direction
- creates clean boundaries for phased functionality

Cons:

- requires orchestration across two existing feature families
- some chat/workspace internals may need small shared extractions

### Approach 2: Restyle `/workspace-playground`

Turn the existing research studio into the new operator-console interface.

Pros:

- source/workspace/studio behavior is already present
- fewer new routes

Cons:

- risks disrupting existing research workflows
- current mental model is research studio, not chat-first console
- visual rewrite would be large

### Approach 3: Extend `/chat` with workspace rails

Add source and inspector rails around the existing chat page.

Pros:

- lowest risk for core chat behavior
- minimal route proliferation

Cons:

- would rebuild source/workspace behavior around a page that does not own it today
- risks making `/chat` carry two mental models
- harder to preserve the existing simple chat experience

## Recommendation

Use Approach 1.

Create `/chat-workspace` as a visible, dedicated route. The page can borrow stable behavior from `/chat` and `/workspace-playground` without forcing either existing route to become the new experience. This keeps the first implementation small enough to plan while aligning with the long-term goal of a fully functional chat-first workspace.

## UX Design

### Visual Direction

The page intentionally follows the Terminal-Literal option selected by the user:

- compact borders
- dense panel grid
- monospace-heavy labels and status rows
- explicit section headers
- visible keyboard hints
- clear system state
- subdued dark-console styling derived from existing theme tokens

The page should still respect the existing app theme system. A local `ChatWorkspaceConsole` visual layer may define the denser console treatment, but it should use semantic tokens such as `bg`, `surface`, `surface2`, `border`, `text`, `text-muted`, `primary`, `success`, `warn`, and `danger`.

### Information Architecture

Desktop layout has four major regions below the existing web header:

1. Left workspace rail
2. Center chat panel
3. Right inspector rail
4. Bottom status strip

The center chat panel is the main work area. Left and right rails provide context and controls without displacing chat.

### Left Workspace Rail

The left rail contains:

- active workspace name
- source search or source filter
- staged source summary
- library shortcuts
- source list or recent source list
- study section with flashcards/quizzes/notes outputs
- explicit source-row staging actions

V1 behavior:

- real workspace name and real staged sources when available
- real source list if existing workspace/source store integration is straightforward
- read-only study summaries or empty states if study actions are not wired in the first slice

Selection and staging are different states. Selecting a source in the left rail only highlights or focuses it for browsing. It must not automatically attach that source to the next chat request. A source becomes staged only through an explicit `Stage for Chat` or `Use in Chat` action, or through an explicit external handoff that opens `/chat-workspace` with context already staged.

### Center Chat Panel

The center panel contains:

- chat header within the console surface
- transcript/messages
- staged context card when sources are staged
- composer
- send/abort/loading states

The `Context staged - not sent` card is a primary control, not a passive note.

Required actions:

- `Clear`: removes staged context
- `Insert`: inserts a textual source/context summary into the composer without sending, then unstages the structured source metadata to prevent duplicate context submission
- `Send`: submits with staged context

The card must make source scope and send state explicit.

### Right Inspector Rail

The right rail contains:

- scope
- sources
- model/persona
- approvals/tool policy
- task progress
- runtime/backend status

V1 behavior:

- scope and staged sources should be real
- model/persona should be real when available from existing hooks
- approvals/tool policy and task progress may be non-interactive v1 scaffolds
- inactive scaffolds must be labeled with states such as `Not configured`, `Unavailable`, or `No active task`

### Bottom Status Strip

The status strip shows:

- current route/system state
- staged context state
- streaming state
- backend availability
- concise keyboard hints

Examples:

- `Ready`
- `Context staged`
- `Streaming`
- `Server unavailable`
- `Ctrl+K command`
- `Ctrl+Enter send`
- `Esc clear focus`

### Responsive Behavior

Desktop:

- three-column console grid with fixed/minmax rails and a flexible chat center

Tablet:

- left rail may collapse
- inspector becomes a drawer or stacked panel
- chat remains primary

Mobile:

- chat-first layout
- source/study/inspector regions become tabs or drawers
- composer remains reachable
- panels collapse before text becomes unreadable

## Nielsen Norman Group UX Principles Applied

The design applies the following usability heuristics:

- Visibility of system status: status strip, staged context card, inspector runtime state.
- Match between system and real world: source scope, persona, approvals, and model labels use product terms already present in the app.
- User control and freedom: staged context is visible, clearable, insertable, and only sent intentionally.
- Consistency and standards: global shell, navigation, model settings, backend handling, and theme tokens remain shared with the current web UI.
- Error prevention: source context is staged before send; unavailable sources remain visible with warnings.
- Recognition rather than recall: source scope, selected model/persona, and runtime state stay visible.
- Aesthetic and minimalist design: the page is dense but task-focused; no hero sections, marketing cards, or decorative visuals.
- Help users recover from errors: failed sends preserve draft text and staged context.

## Component Design

### New Route

Add:

- `apps/tldw-frontend/pages/chat-workspace.tsx`
- `apps/packages/ui/src/routes/option-chat-workspace.tsx`

The Next page should follow existing page-wrapper conventions and disable SSR through `dynamic(..., { ssr: false })`.

### New Shared Components

Add a component folder:

- `apps/packages/ui/src/components/Option/ChatWorkspace/`

Initial components:

- `ChatWorkspacePage`
- `ChatWorkspaceConsole`
- `WorkspaceRail`
- `WorkspaceChatPanel`
- `ContextStagingCard`
- `InspectorRail`
- `WorkspaceStatusStrip`

### `ChatWorkspacePage`

Responsibilities:

- page-level orchestration
- route context setup
- responsive panel state
- bridge existing chat/workspace stores into the console
- pass explicit props to child panels

This component should avoid low-level chat request logic if existing hooks/components can own it.

### `ChatWorkspaceConsole`

Responsibilities:

- terminal-like page frame
- three-column desktop grid
- mobile/tablet responsive shell
- local console styling primitives
- status strip placement

### `WorkspaceRail`

Responsibilities:

- active workspace summary
- source search/filter shell
- staged source summary
- library/source/study section rendering
- empty states for no workspace or no staged sources

### `WorkspaceChatPanel`

Responsibilities:

- wrap or compose existing chat transcript/composer behavior
- render `ContextStagingCard` near the chat turn flow
- preserve existing send/stream/error behavior where feasible

Implementation planning should decide whether this wraps `Playground` internals, wraps `WorkspacePlayground/ChatPane`, or extracts a narrower shared chat surface.

### `ContextStagingCard`

Responsibilities:

- show source count, titles, scope, and send state
- expose `Clear`, `Insert`, and `Send`
- keep user control explicit
- show stale/unavailable source warnings

### `InspectorRail`

Responsibilities:

- show real scope and staged source state
- show model/persona state when available
- show inactive v1 sections honestly
- show backend/runtime state

### `WorkspaceStatusStrip`

Responsibilities:

- show concise route/system state
- show keyboard hints
- surface backend unavailable and streaming states

## Data Flow

### Route Load

On mount:

1. Set chat surface route context to `routeId: "chat-workspace"` and `surface: "webui"`.
2. Restore or initialize chat state using existing chat behavior where practical.
3. Restore or initialize workspace/source state using existing workspace store behavior where practical.
4. Render empty console state if there is no active workspace or no staged source.

### Source Staging

Source staging in v1 should operate on existing workspace/source state when possible:

1. User explicitly stages one or more sources through `Stage for Chat`, `Use in Chat`, or an external handoff into `/chat-workspace`.
2. `ContextStagingCard` renders staged source metadata.
3. `InspectorRail` mirrors the staged source set and scope.
4. Staged sources remain unsent until the user sends.

Selecting or focusing a source for browsing is not staging. The implementation should name these states separately so the UI, tests, and request builder cannot accidentally treat source selection as source attachment.

The staged context model should be small and explicit:

- source ids
- display titles
- source type
- scope/workspace id or label
- availability state
- optional token/word estimate when already available

### Insert

`Insert` should add a concise source summary into the composer without sending. It should not mutate chat history.

After insertion, the structured staged context should be cleared. The user can still edit and send the inserted text manually, but a later `Send` must not submit both the inserted text and the previously staged source metadata unless the user explicitly stages the source again.

### Send

`Send` should submit through the normal chat path where feasible.

If staged context exists, the request should include explicit source/scope metadata using existing contracts. If the exact request contract needs implementation investigation, planning should prefer adapting existing workspace chat behavior before inventing a new request shape.

### Clear

`Clear` removes staged context and leaves composer text intact.

### Errors

Failed sends should preserve:

- draft composer text
- staged context
- visible source warnings

## Navigation

The route is visible immediately.

Add `Chat Workspace` near `/chat` in the existing shortcut/navigation system. It should use an icon that reads as a console/workspace if available in lucide; otherwise use the closest existing message/workspace icon.

The route should also be added to viewport-constrained route handling so the workspace fills the available area below the header instead of becoming a long scrolling dashboard.

## Implementation Phasing

### Phase 1: Visible usable prototype

Ship:

- route and navigation
- terminal-literal console layout
- core chat/composer path
- staged context card
- basic workspace/source integration
- inspector rail with real scope/sources and honest inactive sections
- responsive desktop/mobile behavior
- focused tests and route smoke coverage

### Phase 2: Deepen real workspace behavior

Add:

- richer source browsing/search in the left rail
- better external source handoff into `/chat-workspace`
- study output integration
- richer runtime/task progress
- tighter model/persona controls in inspector

### Phase 3: Full agentic workspace

Add:

- real approvals/tool policy controls
- task progress and tool execution timeline
- deeper automation and workflow integration
- possible migration path if `/chat-workspace` becomes the preferred chat surface

## Error Handling

Backend unavailable:

- keep global shell modal behavior
- show local warning in status strip or inspector

No model configured:

- keep composer visible
- disable or route send through existing model settings flow
- provide a local status warning

Staged source unavailable:

- keep source visible with warning state
- allow `Clear`
- do not silently remove it

Send failure:

- preserve draft text
- preserve staged context
- show retry/clear choices where existing chat behavior supports it

Inactive v1 panel:

- show honest state labels such as `Not configured`, `No active task`, or `Unavailable`
- do not show `Ready` for inactive capabilities

Mobile overflow:

- collapse panels into tabs/drawers before text overlaps or becomes unreadable

## Accessibility

Requirements:

- all icon buttons have labels/tooltips
- keyboard focus order follows header, workspace rail, chat, inspector, status strip
- staged context actions are keyboard reachable
- inspector and collapsed panels are reachable on mobile
- contrast meets existing dark/light theme expectations
- panel labels and status text do not rely only on color
- no text overlap at desktop, tablet, or mobile widths

## Testing

Unit/component tests:

1. `ContextStagingCard` renders staged sources and exposes clear/insert/send actions.
2. `ContextStagingCard` shows unavailable source warnings.
3. `InspectorRail` renders real scope/source state and inactive v1 states honestly.
4. `WorkspaceStatusStrip` renders ready, streaming, staged-context, and backend-unavailable states.
5. `ChatWorkspaceConsole` applies desktop and mobile layout classes/states.

Route/integration tests:

1. `/chat-workspace` renders inside the existing shell.
2. `/chat-workspace` appears in visible navigation/shortcuts.
3. Selecting a source for browsing does not create a staged context card.
4. Stage source -> context card appears -> clear removes it.
5. Insert staged source summary writes to composer without sending and clears structured staged metadata.
6. Send with staged context calls the shared chat path.
7. Failed send preserves draft and staged context.
8. `/chat` still renders.
9. `/workspace-playground` still renders.

Accessibility/visual checks:

1. Desktop screenshot validates Terminal-Literal layout.
2. Mobile screenshot validates chat-first layout.
3. Keyboard tab order reaches staged context controls and panel toggles.
4. Buttons and tabs have accessible names.
5. No overlapping text in dense panel states.

## Risks

1. Reusing existing chat internals may expose coupling in `Playground`.
   Mitigation: extract small shared hooks/components only where needed.
2. Reusing workspace state may pull in assumptions specific to `/workspace-playground`.
   Mitigation: keep `ChatWorkspacePage` as an adapter and do not mutate workspace contracts in the first slice.
3. Terminal-Literal styling can drift from the web app theme.
   Mitigation: use semantic theme tokens and local visual primitives.
4. Visible navigation can make users expect full functionality immediately.
   Mitigation: real core interaction path, honest inactive states, and focused scope.
5. The page could become a second implementation of chat.
   Mitigation: use shared chat path where feasible and treat forks as temporary only if implementation planning justifies them.

## Open Implementation Questions For Planning

1. Whether `WorkspaceChatPanel` should wrap existing `Playground` pieces, existing `WorkspacePlayground/ChatPane`, or a smaller extracted shared chat surface.
2. Which existing source selection/staging contract is the cleanest v1 fit for `/chat-workspace`.
3. Whether the visible navigation entry belongs only in header shortcuts or also in the persistent collapsed sidebar shortcut defaults.
4. Whether route naming should remain `/chat-workspace` or use a shorter alias later.

These are planning questions, not design blockers. The approved product direction is a visible `/chat-workspace` route with Terminal-Literal layout and real core chat/source behavior.
