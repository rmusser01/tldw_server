# Sidepanel Chat WebUI Handoff Design

**Date:** 2026-05-29
**Surface:** Browser extension sidepanel chat to WebUI `/chat`
**Status:** Approved design
**Backlog:** TASK-546

## Goal

Add an explicit sidepanel-to-WebUI handoff that lets a user continue the current sidepanel draft in the full `/chat` experience with the visible page context attached.

This must not replace the existing full-screen/open action. The current open action remains route-only and continues to communicate that sidepanel draft, current page context, and unsaved sidepanel state stay in the sidepanel.

## Approved Decisions

- Add a separate **Continue in WebUI** action in the sidepanel composer `ControlRow` quick-actions area, near the existing full-app/open action.
- Transfer only the current sidepanel draft and already-visible page context.
- Prefill the WebUI `/chat` composer; do not send automatically.
- Show a visible imported-context banner in `/chat` before send.
- Include imported context in the next chat request unless the user removes it before sending.
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
- `SidepanelHeaderSimple` does not currently receive composer draft/context props, while `ControlRow` already receives conversation-context composition props. The transfer action belongs in `ControlRow`, not the header.
- Existing `createLocalRegistryBucket` storage writes swallow `set` errors. The handoff service must not reuse that helper for package creation because this flow must fail closed before opening `/chat?handoff=<id>`.

## User Experience Contract

### Existing Route-Only Action

The existing full-screen/open action keeps its current semantics:

- Opens `/chat` in the WebUI.
- Does not transfer sidepanel draft text.
- Does not transfer current page context.
- Does not transfer unsaved sidepanel chat state.
- Keeps accessible copy that states this limitation.

### New Explicit Transfer Action

Add a separate **Continue in WebUI** action in the sidepanel composer `ControlRow` quick-actions menu.

Suggested copy:

- Label: `Continue in WebUI`
- Tooltip/description: `Opens /chat with this draft and visible page context. Nothing is sent automatically.`

The action should be enabled when there is either:

- non-empty sidepanel draft text, or
- visible page context that can be attached.

If neither exists, the action should be disabled or visually deemphasized with a short disabled-state label such as `No draft or page context to continue`.

### WebUI Arrival

When `/chat?handoff=<id>` opens and the handoff is valid:

- WebUI reads and validates the handoff package.
- The composer is prefilled with the transferred draft.
- The imported page context appears as a visible imported-context banner above the composer.
- The imported context is included in the next chat request unless the user removes it.
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

If the user removes the imported context, the prefilled draft should remain. Removing context only removes the contextual attachment from the next request.

## Handoff Package

The URL carries only an opaque handoff ID:

`/options.html#/chat?handoff=<handoffId>`

If role-play route intent is active, merge the handoff parameter into the existing route query rather than replacing it:

`/options.html#/chat?mode=character&characterId=<id>&handoff=<handoffId>`

The package should be stored in extension/browser storage, not encoded into the URL.

Implementation storage contract:

- Use a dedicated sidepanel chat handoff service backed by extension local storage, preferably through `createSafeStorage({ area: "local" })`.
- Package creation must return success or throw/return a typed failure. Do not use storage helpers that silently swallow write failures for package creation.
- Package creation should verify the package can be read back before opening `/chat?handoff=<id>`.
- Package contents must not be logged.

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
      truncated?: boolean
    }>
    truncated?: boolean
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

- max 4 snippets,
- max 4,000 characters per snippet,
- max 16,000 total snippet characters,
- max 32,000 draft characters in the handoff package,
- drafts longer than the current `PASTED_TEXT_CHAR_LIMIT` of 1,500 characters should prefill through the existing large-draft/collapse behavior instead of expanding the composer unexpectedly,
- truncate with `truncated: true` when needed,
- reject malformed package shapes before prefill.

## Imported Context Request Semantics

Imported page context is not only a visual banner. It is first-class composer context for the next send:

- The banner is rendered above the composer so the user can review or remove it before send.
- The next chat request should include the imported context in the same request-composition path used for explicit composer context, or in a dedicated sidepanel-handoff context block if that is the smallest safe integration.
- Imported context is consumed by one user send or by user dismissal. It should not silently persist across unrelated drafts after send.
- The first implementation slice should not write imported context as durable saved-conversation metadata beyond whatever is already captured by the sent message/request path.
- Tests must assert that the outgoing chat request includes imported page title, URL, and snippets when the banner is present, and omits them after removal.

## Lifecycle Rules

- Handoff packages are one-time consume.
- Default expiry should be short, for example 10 minutes.
- Handoff IDs should be unguessable, for example `crypto.randomUUID()`.
- Opening `/chat?handoff=<id>` should not immediately destroy the package. Read and validate first, then mark consumed or remove it only after one of these terminal outcomes:
  - successful composer prefill/import,
  - user chooses `Cancel import` in an existing-draft conflict,
  - user acknowledges a malformed/unsupported-package error.
- Expired or consumed packages must not prefill composer state.
- The sidepanel draft remains unchanged after handoff creation and after WebUI consumption.
- Handoff cleanup should remove expired records opportunistically when creating or consuming a package.

## Error And Edge States

If the handoff is missing, expired, malformed, or already consumed:

- `/chat` still opens normally.
- Show non-blocking feedback such as `The sidepanel handoff expired. Start a new handoff from the sidepanel to continue that draft.`
- Do not block normal chat usage.
- Remove the `handoff` query parameter from the React Router location after the feedback is shown, so reloads do not repeat the same stale handoff attempt.

If the draft can be recovered but page context cannot:

- Prefill the composer draft.
- Show a warning on the context indicator or toast: `Draft imported. Page context could not be attached.`

If storage write fails in the sidepanel:

- Keep the user in the sidepanel.
- Show a failure notification.
- Do not open `/chat?handoff=<id>` with a missing package.

If `/chat` already has a local unsent draft:

- Show the import conflict choice before changing composer text.
- `Insert handoff draft` appends or inserts the handoff text according to the existing composer insertion behavior.
- `Replace current draft` replaces the local draft with the handoff draft.
- `Cancel import` leaves the local draft unchanged and consumes/removes the handoff package so reload does not prompt repeatedly.

## Privacy And Security Constraints

- Do not put draft text or page snippets in the URL.
- Do not store handoff packages longer than the expiry window.
- Do not capture new page body text during handoff.
- Do not send the draft automatically.
- Treat handoff context as user-visible input, not hidden model metadata.
- Avoid logging package contents in errors or analytics.
- Do not silently overwrite an existing WebUI draft.

## Routing Requirements

- Parse handoff parameters through React Router route state, for example `location.search` for `/chat?handoff=<id>`.
- Do not parse handoff parameters from `window.location.search`; this app uses hash routes such as `/options.html#/chat?handoff=<id>`, where the meaningful query string lives inside the hash route.
- Remove `handoff` from the route after a terminal outcome with router-aware navigation or `history.replaceState` that edits the hash-route query, not the outer document query.
- Preserve existing role-play query parameters when adding or removing `handoff`.

## Accessibility Requirements

- Both actions need distinct labels and descriptions:
  - route-only open action,
  - state-transfer Continue in WebUI action.
- Disabled state must be announced when no transferable state exists.
- The imported context indicator in `/chat` must be keyboard reachable.
- The context remove/dismiss action must have an accessible name that identifies the imported page context.
- The existing-draft conflict choice must be keyboard reachable and must not trap focus.
- Non-blocking error feedback must be available to assistive technology, not only visual toast color.

## Test Contract

Focused regression tests should cover:

- Existing full-screen/open action still opens `/options.html#/chat` with no `handoff` parameter.
- Existing route-only accessible copy remains present for full-screen/open.
- **Continue in WebUI** creates a handoff package and opens `/options.html#/chat?handoff=<id>`.
- Draft text and page snippets are not serialized into the URL.
- Sidepanel draft is preserved after handoff creation.
- Role-play route intent preserves `mode=character&characterId=...` alongside `handoff=<id>` when applicable.
- Handoff package creation fails closed when storage write or read-back verification fails.
- `/chat` reads and validates a valid handoff, pre-fills the composer, then consumes it after successful import.
- `/chat` displays a visible imported-context banner before send.
- The next outgoing chat request includes imported page title, URL, and snippets while the banner is present.
- Removing the imported-context banner removes the imported context from the outgoing chat request without clearing the draft.
- Existing WebUI composer draft is not silently overwritten by an arriving handoff.
- Existing-draft import conflict supports insert, replace, and cancel outcomes.
- Expired, missing, malformed, and already-consumed handoffs open `/chat` normally with non-blocking feedback.
- A storage write failure does not open a broken handoff URL.
- Hash-router cleanup removes only `handoff` and preserves role-play route parameters.

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
2. Add the sidepanel **Continue in WebUI** action in `ControlRow` and tests that assert route-only open remains unchanged.
3. Add `/chat` handoff read/validate/import/consume, composer prefill, imported-context banner, request inclusion, hash-route cleanup, and error feedback tests.
4. Run packaged extension smoke to verify sidepanel action, WebUI arrival, role-play route compatibility, request context inclusion, and stale handoff behavior.

## Open Questions

- Whether imported sidepanel context should later be promoted into durable saved-conversation metadata. First slice keeps it request-scoped.
- Whether a future slice should support fresh readable-page capture at handoff time. First slice uses only already-visible or explicitly captured context.
- Whether a future slice should add a success acknowledgement in the sidepanel after package creation. First slice treats opening the WebUI tab as success feedback and reserves notifications for failure.

## Verification For This Spec

- `git diff --check`
- Backlog task `TASK-546` references this design.
- Bandit is skipped for this design-only Markdown task; no Python code is touched.
