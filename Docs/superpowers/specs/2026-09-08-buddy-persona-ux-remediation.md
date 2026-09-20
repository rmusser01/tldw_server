# Buddy and Persona UX remediation

Approved scope: TASK-13226. Server and shared WebUI/extension only. The user's Flashcards reference was a mistake; no Flashcards functionality is included.

## User outcomes

1. A new user can find Buddy & Persona from the conversation menu, preview ready-made artwork, select a Buddy, choose an attachment, and Apply without understanding manifests or creating a Persona.
2. Persona identity and Buddy artwork are separate choices. Cancel, Escape and closing setup discard staged changes. Existing tracked conversations retain explicit identity semantics.
3. One Buddy remains available while navigating the application. Its title always identifies the conversation or workspace it represents; incoming activity never changes the user's main page.
4. The Buddy shows conversation context and accepts a reply in place. Workspace mode lists relevant conversations, makes the reply destination explicit, and queues optional spoken results prefaced by conversation title. Viewing a pending decision never accepts it.
5. Workspace settings expose the existing optional default Persona for new conversations, its effective state, and explicit None. Server creation enforces the same precedence as the client.
6. Persona Garden keeps an editing selector visible independently of the live session. Connected identity remains frozen. Transcript, composer and pending decisions precede optional voice and diagnostic controls.
7. Ready-made Buddy choices show actual preview artwork before copying. Imports remain secondary; advanced authoring, validation, draft review and activation remain available.
8. All tabs, fields, movement controls and modal actions work with a keyboard and at narrow widths. Functional status text and contrast use readable theme tokens. Static/Dynamic and reduced-motion preferences are honored.

## Ownership and limits

ADR-005 defines independent artwork, attachment authority and accepted-work ownership. Source Persona deletion does not delete copied Buddy art. Private workspace bindings do not authorize access to an owner's private conversations through a shared workspace. Scope changes and stale access invalidate pending interactions and speech, retaining recoverable drafts without sending them elsewhere.

The existing Persona Live contract cancels connection-owned work on disconnect. Remediation must either establish a verified independent owner for accepted turns or clearly retain that limitation; a persistent render context alone is not proof of server work continuity. Provider/model and actual voice availability remain explicit; no synthetic transcript, automatic microphone or automatic tool approval is allowed.

## Review traceability

| Finding | Work |
| --- | --- |
| Cancel mutates identity; drawer overflows | TASK-13226.1 plus Apply integration |
| Obstructed/non-keyboard tabs; late previews; editing/live coupling; setup-first transcript | TASK-13226.2 |
| Independent art, explicit attachment, API defaults | TASK-13226.3 |
| Persistent management, target/reply UX, motion, movement, speech | Shared shell integration |
| Real continuation and complete first-time/power-user journey | Parent integration verification |

Latest dev already implements workspace default settings in Research Workspace and client creation inheritance. Reuse these rather than adding a second preference system. The audit's earlier draft-only characterization is superseded by source verification on 6cd2745f69.
