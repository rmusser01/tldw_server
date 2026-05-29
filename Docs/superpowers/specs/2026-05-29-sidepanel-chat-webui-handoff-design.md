# Sidepanel Chat WebUI Handoff Design

**Date:** 2026-05-29
**Surface:** Browser extension sidepanel chat to WebUI `/chat`
**Status:** Approved design
**Backlog:** TASK-546

## Goal

Add an explicit sidepanel-to-WebUI handoff that lets a user continue the current sidepanel draft in the full `/chat` experience with the visible page context attached.

This must not replace the existing full-screen/open action. The current open action remains route-only and continues to communicate that sidepanel draft, current page context, and unsaved sidepanel state stay in the sidepanel.

## Approved Decisions

- Add a separate **Continue in WebUI** action near the existing full-app/open action.
- Transfer only the current sidepanel draft and already-visible page context.
- Prefill the WebUI `/chat` composer; do not send automatically.
- Show a visible context indicator in `/chat` before send.
- Use a one-time, short-lived handoff package keyed by an opaque ID.
- Preserve the original sidepanel draft after handoff.
- Keep role-play route intent compatible with the handoff URL when active.

## Current Evidence

- `SidepanelHeaderSimple` opens `/options.html#/chat` for the full-chat button and describes the action as route-only.
- `ControlRow` opens `/options.html#${fullAppChatPath}` and uses route-only accessible copy for normal and role-play handoff.
- `buildSidepanelFullAppChatPath` already preserves normal `/chat` and character route intent.
- Sidepanel and WebUI composers already use separate draft keys through `useComposerText`, so sidepanel draft preservation matches the existing storage model.
- `chat-surface-coordinator` currently tracks route/surface and optional panel engagement only; it is not a cross-surface state-transfer mechanism.
- Existing sidepanel handoff tests assert route-only behavior and role-play route preservation, so the new behavior must be additive instead of mutating the current route-only contract.

## User Experience Contract

### Existing Route-Only Action

The existing full-screen/open action keeps its current semantics:

- Opens `/chat` in the WebUI.
- Does not transfer sidepanel draft text.
- Does not transfer current page context.
- Does not transfer unsaved sidepanel chat state.
- Keeps accessible copy that states this limitation.

### New Explicit Transfer Action

Add a separate **Continue in WebUI** action.

Suggested copy:

- Label: `Continue in WebUI`
- Tooltip/description: `Opens /chat with this draft and visible page context. Nothing is sent automatically.`

The action should be enabled when there is either:

- non-empty sidepanel draft text, or
- visible page context that can be attached.

If neither exists, the action should be disabled or visually deemphasized with a short disabled-state label such as `No draft or page context to continue`.

### WebUI Arrival

When `/chat?handoff=<id>` opens and the handoff is valid:

- WebUI consumes the handoff once.
- The composer is prefilled with the transferred draft.
- The imported page context appears as a visible context attachment/indicator near the composer or active context rail.
- The user can edit the draft before sending.
- Nothing is submitted until the user explicitly sends.

If the WebUI composer already has an unsent local draft, the handoff must not silently overwrite it. Show a small import conflict choice instead:

- `Insert handoff draft`
- `Replace current draft`
- `Cancel import`

The context indicator must expose enough information for user trust:

- page title when available,
- origin/domain or URL,
- snippet/selection count or short snippet preview when available,
- a remove/dismiss control before send.

## Handoff Package

The URL carries only an opaque handoff ID:

`/options.html#/chat?handoff=<handoffId>`

If role-play route intent is active, merge the handoff parameter into the existing route query rather than replacing it:

`/options.html#/chat?mode=character&characterId=<id>&handoff=<handoffId>`

The package should be stored in extension/browser storage, not encoded into the URL.

Proposed package shape:

```ts
type SidepanelChatHandoffPackage = {
  id: string
  source: "sidepanel-chat"
  createdAt: string
  expiresAt: string
  consumedAt?: string
  draft: {
    text: string
  }
  pageContext?: {
    title?: string
    url?: string
    snippets: Array<{
      kind: "selection" | "visible-context" | "captured-snippet"
      text: string
      label?: string
    }>
  }
  routeIntent?: {
    path: "/chat" | string
    mode?: "character"
    characterId?: string
  }
}
```

For the first implementation slice, `pageContext` must be limited to context already visible or explicitly captured in the sidepanel. Do not trigger a fresh readable-page/body capture at handoff time.

Payloads should be bounded:

- cap the number of snippets,
- cap each snippet length,
- truncate with an explicit indicator when needed,
- reject malformed package shapes before prefill.

## Lifecycle Rules

- Handoff packages are one-time consume.
- Default expiry should be short, for example 10 minutes.
- Handoff IDs should be unguessable, for example `crypto.randomUUID()`.
- Consuming `/chat?handoff=<id>` marks the package consumed or removes it.
- Expired or consumed packages must not prefill composer state.
- The sidepanel draft remains unchanged after handoff creation and after WebUI consumption.
- Handoff cleanup should remove expired records opportunistically when creating or consuming a package.

## Error And Edge States

If the handoff is missing, expired, malformed, or already consumed:

- `/chat` still opens normally.
- Show non-blocking feedback such as `The sidepanel handoff expired. Start a new handoff from the sidepanel to continue that draft.`
- Do not block normal chat usage.

If the draft can be recovered but page context cannot:

- Prefill the composer draft.
- Show a warning on the context indicator or toast: `Draft imported. Page context could not be attached.`

If storage write fails in the sidepanel:

- Keep the user in the sidepanel.
- Show a failure notification.
- Do not open `/chat?handoff=<id>` with a missing package.

## Privacy And Security Constraints

- Do not put draft text or page snippets in the URL.
- Do not store handoff packages longer than the expiry window.
- Do not capture new page body text during handoff.
- Do not send the draft automatically.
- Treat handoff context as user-visible input, not hidden model metadata.
- Avoid logging package contents in errors or analytics.

## Accessibility Requirements

- Both actions need distinct labels and descriptions:
  - route-only open action,
  - state-transfer Continue in WebUI action.
- Disabled state must be announced when no transferable state exists.
- The imported context indicator in `/chat` must be keyboard reachable.
- The context remove/dismiss action must have an accessible name that identifies the imported page context.
- Non-blocking error feedback must be available to assistive technology, not only visual toast color.

## Test Contract

Focused regression tests should cover:

- Existing full-screen/open action still opens `/options.html#/chat` with no `handoff` parameter.
- Existing route-only accessible copy remains present for full-screen/open.
- **Continue in WebUI** creates a handoff package and opens `/options.html#/chat?handoff=<id>`.
- Draft text and page snippets are not serialized into the URL.
- Sidepanel draft is preserved after handoff creation.
- Role-play route intent preserves `mode=character&characterId=...` alongside `handoff=<id>` when applicable.
- `/chat` consumes a valid handoff once and pre-fills the composer.
- `/chat` displays a visible imported-context indicator before send.
- Existing WebUI composer draft is not silently overwritten by an arriving handoff.
- Expired, missing, malformed, and already-consumed handoffs open `/chat` normally with non-blocking feedback.
- A storage write failure does not open a broken handoff URL.

## Implementation Boundaries

This design does not require:

- automatic send,
- clearing or moving the sidepanel draft,
- fresh readable-page extraction during handoff,
- server-backed draft/session records,
- cross-device resume,
- changing the current full-screen/open route-only action,
- broader `/chat` layout redesign.

## Suggested Implementation Slices

1. Add a storage service and unit tests for creating, reading, consuming, expiring, and cleaning sidepanel chat handoff packages.
2. Add the sidepanel **Continue in WebUI** action and tests that assert route-only open remains unchanged.
3. Add `/chat` handoff consumption, composer prefill, visible context indicator, and error feedback tests.
4. Run packaged extension smoke to verify sidepanel action, WebUI arrival, role-play route compatibility, and stale handoff behavior.

## Open Questions

- Which existing context indicator component should render the imported sidepanel page context in `/chat`: composer attachment UI, context rail, or a small imported-context banner?
- Should handoff packages be stored in `browser.storage.local`, an existing extension storage helper, or a new thin service wrapping the current storage abstraction?
- Should imported context become part of the next chat request only, or should it also be visible in saved conversation metadata after send?
- Should there be a small visual acknowledgement in the sidepanel after successful package creation, or is opening the WebUI enough feedback?

## Verification For This Spec

- `git diff --check`
- Backlog task `TASK-546` references this design.
- Bandit is skipped for this design-only Markdown task; no Python code is touched.
