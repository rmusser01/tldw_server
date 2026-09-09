# ADR-005: Independent Buddy bindings and interaction ownership

Status: Accepted
Date: 2026-09-08
Task: TASK-13226

## Context

The approved Buddy/Persona UX review found that artwork, editing, and live interaction share Persona and page lifetimes. Users need one visible Buddy attached to a conversation or workspace while they use other pages. Choosing artwork must not require assigning a Persona or changing a conversation's identity.

## Decision

Buddy profiles own validated, immutable native artwork snapshots and attribution. A source Persona or catalog entry is provenance, not the artwork's lifetime owner. Persona remains optional and separate from artwork selection. A principal-owned, versioned attachment selects an existing conversation or workspace; the client slot is a preference key, never an authorization credential. Every interaction resolves the authenticated principal and rechecks the target's current access and membership.

The shared application shell owns the visible Buddy, its interaction state and speech queue. Page changes and closing a popover do not mean Stop. Conversation mode identifies one exact target and may offer explicitly started voice input. Workspace mode projects its conversations, never offers microphone input, and speaks opted-in results serially with the conversation title. Acknowledgements reference exact result identities. Pending decisions retain their existing approval authority; displaying or speaking them cannot approve them.

Accepted work must have an owner independent of a page or transport subscription. Existing Persona Live connection-owned work (Docs/ADR/046) cannot be presented as durable. Any new detached execution adapter must preserve the authenticated Chat admission boundary, quota/moderation/accounting, FIFO order and publication fencing. Credentials may remain in memory only. A process-owned adapter must mark interrupted work terminal after restart and must not automatically retry uncertain provider/tool effects. Persisting credentials or silently bypassing Chat admission is rejected.

Workspace default Persona selection is resolved when creating a new conversation. Omitted choice may inherit; explicit None and explicit identity override the default. Existing, restored, moved and copied conversation identities are preserved. Workspace settings and management surfaces use the existing workspace defaults contract.

Static/Dynamic is a user setting; reduced-motion forces static rendering. Positioning has keyboard and reset controls. Editing a Persona is independent of the Persona connected to a live session. Apply/Cancel surfaces stage all fields until Apply.

## Alternatives

- Keeping references to Persona-owned artwork would lose a Buddy when the Persona is deleted.
- Keeping a page mounted to simulate a background worker would leave ownership dependent on client lifetime and must not be called durable execution.
- Calling provider adapters directly would duplicate or bypass Chat admission policy.
- Copying workspace defaults into old conversations would unexpectedly change their identity.
- A separate workspace microphone would create ambiguous reply routing; workspace interaction instead uses explicit conversation selection.

## Verification

Ownership/isolation, source deletion, optimistic concurrency, explicit-None inheritance, navigation continuity, exact result acknowledgements and stale target handling require focused tests. Rendered desktop and narrow walkthroughs verify discoverability, target labels, keyboard access and motion controls. Claims about provider continuation require actual server execution evidence; component tests alone are insufficient.

Related: `Docs/superpowers/specs/2026-09-08-buddy-persona-ux-remediation.md`, `Docs/ADR/046-persona-live-conversation-and-voice-runtime.md`.
