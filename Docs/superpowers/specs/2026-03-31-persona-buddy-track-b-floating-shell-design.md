# Persona Buddy Track B Floating Shell Design

**Date:** 2026-03-31

## Goal

Define phase 2 of the persona buddy system as a shared desktop floating shell that gives the active persona a persistent visual presence across persona-aware surfaces without replacing existing inline avatars or profile pictures.

This track should turn the buddy from a backend-only identity facet into a consistent UI layer while staying short of Track C's richer Clippy-style interaction model.

## Context

Relevant existing repo context:

- `Docs/superpowers/specs/2026-03-31-persona-buddy-facet-design.md`
- `apps/packages/ui/src/components/Common/NotesDock/NotesDockHost.tsx`
- `apps/packages/ui/src/components/Common/NotesDock/NotesDockPanel.tsx`
- `apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx`
- `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx`
- `apps/packages/ui/src/hooks/useMessage.tsx`

Existing patterns already establish:

- persona is a first-class assistant artifact with its own route and live workflow
- floating desktop UI already exists through the notes dock host/panel pattern
- lightweight global UI preferences already go through the shared settings registry
- current persona imagery still relies on existing `avatar_url` and profile-picture paths

Track B should build on those patterns rather than invent a separate shell framework or replace inline persona imagery outright.

## Problem

Track A gives each persona a canonical buddy identity, but the product still does not present that identity in a consistent visible way. The missing layer is not "another avatar component." It is a shared persona-owned shell that can follow the active persona through the product.

The user has now clarified an important constraint: the buddy should not replace the existing inline profile picture in chat and related UI. Instead, the buddy should exist separately, more like a draggable desktop companion shell that is present across persona-aware pages.

That changes the Track B shape:

- inline persona avatars continue to exist
- the buddy becomes a distinct floating visual layer
- the shell must still remain persona-owned and state-driven, not an independent assistant

If Track B is implemented as per-surface custom widgets, it will drift in behavior, targeting, and placement. If it is implemented as a full interaction agent, it will collapse into Track C too early.

## User-Confirmed Product Rules

1. Track B should land as a thin slice across the main persona-aware surface groups rather than one isolated page.
2. The buddy should use a lightweight static popover in phase 2.
3. Buddy data for list-heavy surfaces should be hydrated through persona payload projection, not N+1 `/buddy` fetches.
4. Existing inline avatars and profile pictures remain in place.
5. The buddy should render separately as a draggable modal or shell across persona-aware pages.
6. There should be one global user setting controlling buddy-shell presence.
7. That global setting should live in app settings, outside persona surfaces.
8. The shell should default to a compact docked state and expand on click.
9. The shell should represent the current active persona for the workspace.
10. If no persona is active, the shell should stay dormant.
11. Drag position should persist per surface or layout class, not as one universal coordinate.
12. Narrow/mobile layouts are out of scope for phase 2.
13. The day-one static popover should show only:
    - buddy visual preview
    - persona name
    - one-line role or summary

## Scope

Track B should stay focused on shared rendering, shell behavior, and data hydration for the compact buddy presence.

### In Scope

- a shared layout-level buddy shell host for desktop persona-aware surfaces
- compact docked buddy rendering
- a static read-only popover with minimal persona summary content
- active-persona targeting rules for chat, Persona Garden, and persona selection surfaces
- compact buddy summary projection on persona payloads used by phase 2 surfaces
- global settings support for enabling or disabling the buddy shell
- persisted per-surface shell position for supported desktop layouts
- desktop-only tests covering shell state, targeting, and hydration behavior

### Out Of Scope

- inline avatar replacement
- richer bubble systems, reactive copy, proactive hints, or persona micro-chat
- mobile or narrow-layout buddy rendering
- end-user buddy customization editing UI
- full resolved buddy-detail fetches as the main render path
- multi-buddy rendering for persona lists
- autonomous buddy actions or persona-setting mutation

## Non-Goals

- Do not replace existing persona avatars or profile pictures in chat or list UI.
- Do not introduce a separate pet or assistant identity.
- Do not make every persona card render its own floating buddy.
- Do not force Track B to carry Track C interaction complexity.
- Do not issue per-persona buddy fetches from the shell host.

## Recommended Approach

Implement Track B as a shared `BuddyShellHost` pattern mounted at each app root that can host persona-aware desktop surfaces.

That host owns:

- whether the floating shell is mounted at all
- which persona is currently active for the current workspace
- compact vs open shell state
- per-surface remembered shell position
- dormant behavior when no persona is active

Persona-aware surfaces do not each build independent buddy UIs. They only surface or derive the active persona context already present in their own state. The host then renders a single buddy shell from a compact buddy summary already present on the active persona payload.

This is the recommended shape because it:

- matches the existing notes-dock host/panel model
- preserves consistency across chat, Persona Garden, and selectors
- avoids Track B splitting into slightly different surface-specific buddy behaviors
- keeps the API contract cheap by using projected persona summary data
- leaves Track C free to deepen the shell later without replacing the host architecture

## Design

### 1. Component Model

Track B should introduce four focused frontend units plus one explicit surface-to-host adapter contract.

#### Buddy Shell Host

An app-root-level host component mounted alongside other global desktop overlays.

Responsibilities:

- read the global buddy-shell enabled setting
- suppress itself on unsupported layouts for Track B
- resolve the active persona for the current workspace
- choose the correct persisted position bucket for the current layout class
- render dormant, docked, or open buddy states

The host should mount once per app root rather than once per page panel.

Day-one roots should be explicit:

- the main authenticated web app layout root
- the extension sidepanel app root

Track B should not assume one process-wide singleton host spanning both roots.

#### Buddy Shell Store

A small persisted UI store, similar in scope to the notes dock store but narrower.

Day-one state should include:

- transient open or closed shell state
- per-surface position memory

Track B should not persist speculative future interaction state, chat history, or persona-authored content in this store.

The shell should default back to its compact docked state when it mounts for a new session or fresh page load. Phase 2 should persist position, not persistent expanded-session UI state.

#### Persona Buddy Render Context

Track B should define an explicit frontend adapter contract between surfaces and the host.

Recommended day-one shape:

- `surface_id`
- `surface_active`
- `active_persona_id`
- `position_bucket`
- `persona_source`

This contract exists to prevent the host from inferring buddy activation from unrelated global state alone.

Surface adapters should publish context into their own app root only. The host for a given root reads only that root's active render context.

#### Buddy Dock

The compact default rendering state.

Behavior:

- shown as a small desktop corner dock when the feature is enabled and an active persona exists
- draggable by its shell frame
- clamped to viewport bounds
- visually dormant or hidden when no active persona exists

The dock is the always-available entry point for Track B. It should feel like a persistent face, not a notification badge.

#### Buddy Popover

The expanded read-only state opened from the dock.

Day-one content is intentionally minimal:

- buddy visual preview
- persona display name
- one-line role or summary

The popover is not a second chat window. It is a static identity card for phase 2.

### 2. Surface Ownership And Targeting

Track B should render exactly one buddy shell per active layout context.

The host should target the active persona rather than whichever persona the user last hovered.

#### Activation Gate

The shell should render only when both of these are true:

- the current surface has explicitly marked buddy rendering as active
- the surface has resolved an active persona for that context

Persisted global assistant selection by itself must not activate the buddy shell on unrelated routes. If the app root is mounted on a page that is not participating in Track B, the host should remain dormant even if a persona was selected elsewhere earlier.

#### Active Persona Precedence

Track B should define a strict precedence rule so "active persona" is not ambiguous across surfaces:

1. route-local or surface-local active persona state for the current surface
2. route bootstrap state for the current surface
3. current-surface hydrated persona catalog entry matching the active surface selection
4. persisted global assistant selection only as a same-root fallback when the surface is already marked active

This means Persona Garden route-local persona state beats persisted chat assistant selection inside Persona Garden, and chat-selected persona state beats stale stored persona data inside chat.

Each host should resolve persona only from the render context for its own root. A Persona Garden root and a chat root may legitimately point at different personas at the same time without conflicting.

#### Chat

In chat surfaces, the shell should follow the currently selected active persona. Existing message avatars and selected-assistant inline imagery remain unchanged.

#### Persona Garden

In Persona Garden, the shell should follow the persona loaded into the route or current selected persona state.

#### Persona Selection Surfaces

On selector-style surfaces that list multiple personas, the shell should only wake up if the workspace already has an active persona. It should not rebind to every row interaction by default.

#### No Active Persona

When no active persona exists, the shell should stay dormant rather than guessing a target persona.

### 3. Rendering Rules

Track B rendering behavior should follow these rules:

- desktop only
- compact dock by default
- static popover on click
- drag from shell chrome, not content body
- remember position per supported layout class
- preserve current inline avatar rendering everywhere

The relevant layout classes for day one should be grouped broadly, not per route, so position memory stays predictable. A main web-app desktop bucket and an extension-sidepanel bucket are sufficient for phase 2 unless implementation discovers a clearer existing split.

### 4. Data Contract And Hydration

Track B should not drive rendering from `GET /persona/profiles/{id}/buddy` on every surface.

Instead, the persona payloads already consumed by phase 2 surfaces should project a compact buddy summary sufficient for shell rendering.

Track B should define one canonical frontend summary type rather than allowing each surface to invent its own partial shape.

Recommended day-one canonical type:

- `has_buddy: boolean`
- `persona_name: string`
- `role_summary: string | null`
- `visual`
  - `species_id: string`
  - `silhouette_id: string`
  - `palette_id: string`
  - `accessory_id?: string | null`
  - `eye_style?: string | null`
  - `expression_profile?: string | null`

Recommended compact buddy summary fields:

- `buddy_summary.visual`
- `buddy_summary.persona_name`
- `buddy_summary.role_summary`
- `buddy_summary.has_buddy`

The exact field names may adapt to repo conventions, but the contract should remain compact and render-oriented.

Day-one payloads that should carry this canonical summary are:

- `/api/v1/persona/catalog` list responses normalized into shared persona summary clients
- `/api/v1/persona/profiles/{persona_id}` detail responses used by Persona Garden when route detail is loaded
- any route-local persona catalog shape that currently shadows the shared client summary type

Track B should prefer converging those frontend persona summary types onto a shared contract instead of adding parallel ad hoc buddy fields in multiple local type aliases.

`AssistantSelection` may mirror `buddy_summary` when a persona is selected from hydrated catalog data, but that mirrored copy is a cache or fallback only. The shell host must prefer current-surface persona payloads over stale persisted assistant-selection data.

#### Why Summary Projection

Summary projection avoids:

- N+1 fetches for persona lists
- host-owned detail fetch loops
- inconsistent timing between selector, chat, and Persona Garden surfaces

Full buddy detail from the Track A sub-resource should remain available for later tracks, but Track B should treat it as a deeper contract rather than the default render dependency.

### 5. Settings Model

The global user preference should live in app settings outside persona surfaces, as requested.

Track B should add a new UI setting through the shared settings registry instead of writing directly to ad hoc localStorage keys.

Day-one setting behavior:

- default enabled when the feature is available
- user can disable the floating buddy shell globally
- disabling the setting unmounts the shell host
- disabling the shell does not modify persona data or inline avatar behavior

This is a presence toggle, not a full identity-mode switch. Since inline avatars remain unchanged, the setting controls whether the floating buddy system is active.

### 6. Desktop-Only Boundary

**Update (TASK-13423, 2026-10-02):** The browser-width suppression below is superseded. An active enabled Buddy now remains mounted when a web pane narrows below 1024 pixels, using the existing position clamping and controls. This preserves its live session and unsent draft through resizing. The surface and ownership gates still apply.

Track B should explicitly suppress the buddy shell on responsive web narrow/mobile layouts.

This should be treated as a deliberate rollout boundary, not a degraded desktop implementation. Mobile shell behavior can be designed later with its own interaction and layout assumptions.

To avoid ambiguity, day-one support should be:

- main web app desktop layouts only at desktop breakpoint (`>=1024px`)
- extension sidepanel root as its own supported desktop surface class, even though its visual width may be narrow

The extension sidepanel is in scope for Track B because it is a desktop shell, not a mobile layout. Responsive web tablet and mobile layouts remain out of scope.

The host should therefore:

- not mount on responsive web layouts below desktop breakpoint
- mount in the extension sidepanel root only when that root has an active Track B surface
- not attempt a fixed mobile dock fallback in phase 2
- avoid persisting mobile-specific shell positions

### 7. Failure Handling

Track B should fail soft.

If the active persona lacks usable buddy summary data, the host should:

- stay dormant, or
- render only a minimal compact fallback if a clean placeholder exists

It should not:

- block page rendering
- trigger emergency fetch loops
- break existing persona selection or avatar flows

This is possible because inline persona rendering remains unchanged.

### 8. Testing

Track B should be tested across three layers.

#### Store And Shell Tests

- default docked state
- open and close behavior
- active-persona-only targeting
- explicit `surface_active` gating
- dormancy with no active persona
- per-surface position persistence
- mobile suppression
- sidepanel-root support without responsive-web leakage

#### Surface Integration Tests

- chat surfaces publish or resolve the active persona correctly
- Persona Garden publishes or resolves the active persona correctly
- selector surfaces do not retarget the shell to arbitrary hovered rows
- route-local persona state beats stale persisted assistant selection inside Persona Garden
- existing inline avatar rendering remains intact

#### Data Contract Tests

- shared persona summary consumers normalize the same canonical `buddy_summary` shape
- persona payloads used by phase 2 surfaces expose compact buddy summary fields
- shell consumers do not regress into per-persona `/buddy` fetch loops
- missing buddy summary data degrades safely

## Rollout Notes

Track B is a rendering-and-shell track with a deliberately static interaction ceiling.

It should land before richer Track C behavior, but it also slightly prepares the ground for Track C by establishing:

- a shared host
- a stable persona-targeting rule
- a durable shell presence model

Track C can later add richer buddy reactions or direct interaction inside this same shell architecture, but Track B should not pre-commit to those behaviors beyond the minimal static popover.

## Planning Contract

The implementation plan for this spec should stay limited to:

- compact summary payload projection
- shared desktop buddy host and store
- explicit render-context adapters and active-persona precedence
- shared persona summary normalization for `buddy_summary`
- settings wiring
- tests and failure-safe rollout

The plan should not include:

- animation choreography
- proactive buddy speech
- command execution
- micro-chat
- mobile rendering
