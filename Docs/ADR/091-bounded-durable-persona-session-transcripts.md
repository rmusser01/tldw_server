# ADR-091: Bounded durable Persona session transcripts

**Status:** Accepted
**Date:** 2026-11-02
**Backfilled from:** not backfilled
**Decision owner:** UAT repair session, under the requester's authorization to fix the selected workflows
**Related task:** TASK-13260.278.18.83.45
**Related spec/plan:** ../Design/Persona_Session_Transcript_Durability_20261102.md; ../Plans/IMPLEMENTATION_PLAN_uat_all_fixes_20261006.md#s06-specialized-tools-and-discovery-selected-closure--2026-11-02

## Decision

Retain future Persona session turns in the owning CharactersRAGDB Persona store, with the same bounded transcript identities and retention policy as the runtime, and read that history across process restart.

## Context

S06 produced a real typed answer and verified its exact two-turn transcript across a browser reload. After the API restarted, the same session returned zero turns. Session metadata and the explicit Companion check-in survived. Current detail/list/export read only the process-local SessionManager. Personalization records contain usage metadata and optional truncated summaries; they cannot reconstruct an exact transcript.

The Persona product design advertises persisted turns. ADR-046 keeps the Chat adapter's persistence disabled and preserves the existing Persona transcript/memory boundary. This decision adds durable Persona history within that boundary; it does not enable ordinary Chat persistence, semantic-memory learning, new tool authority, or approval replay. ADR-046's accepted rationale is unchanged. The decision implements the requester-authorized saved-history and restart-recovery repair; publication and human Change summary gates remain separate.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Reconstruct from usage events, summaries, screenshots or test journals | They lack the original exact text and turn identity. Reconstruction would manufacture history. |
| Enable ordinary Chat persistence for the adapter | It changes another domain's ownership and history and conflicts with ADR-046's adapter boundary. |
| Keep the runtime-only implementation and remove the persisted-history promise | It leaves the requested saved-conversation and recovery workflow incomplete. |
| Store transcript data in preferences or scope snapshots | It mixes content with mutable settings or security scope and risks lost updates or authority changes. |

## Consequences

- A dedicated versioned Persona transcript store uses owned session/persona/user checks and existing SQLite/PostgreSQL migration and transaction boundaries.
- Persist the actual bounded turn ID, timestamp, role, type, content and metadata before acknowledging a successful turn. Persistence failure must not publish a false successful reply.
- Preserve existing runtime bounds: 200 recent turns, 8192 content characters and 4096 metadata characters by default, plus the existing session TTL and per-user session cap. Define and test pruning and expiry consistently in the store and runtime.
- Detail, list counts and selected-session export share the owned durable read. Explicit resume hydrates an absent runtime session once, without duplicate turns.
- Pending plans, approval decisions, open sockets, voice authority and tool grants are never restored by transcript hydration.
- Old turns already lost from runtime cannot be recovered by this migration. Keep that limitation explicit; no synthetic backfill.
- Raw transcript content becomes durable in the local, user-owned database for the bounded retention window. It remains separate from optional personalization summaries and is subject to owned deletion and retention.
- Tests and review must cover both supported database backends, real producer-to-restart recovery, repeated resume, foreign owner/persona, terminal/deleted sessions, concurrent appends, retention, storage failure and cancellation.

## Follow-up

Complete live restart and mode qualification before accepting S06. Retain the separate publication and merge gates.

The metadata payload keeps its existing 4096-character limit plus at most 512 characters of retention diagnostics. This preserves the existing small internal-cap truncation markers; storage validates the complete serialized metadata and the full escaped snapshot size.

Expired reads mark the actual stored revision retired; they never allocate the next committed revision. Explicit resume renews only nonexpired history under the owner/session lock, advances the committed revision without changing any turn, and refreshes runtime activity after success. Ordinary GETs remain read-only.

Payload expiry is enforced on reads without mutating the database. Physical expired-payload cleanup runs on the next owned transcript append; retired headers remain to prevent stale runtime publication. No background cleanup service is introduced.
