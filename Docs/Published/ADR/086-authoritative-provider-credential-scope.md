# ADR-086: Authoritative Provider Credential Scope

**Status:** Proposed
**Date:** 2026-10-02
**Backfilled from:** not backfilled
**Decision owner:** MCP adapter workstream requester and reviewers
**Related task:** TASK-2294.3.2
**Related spec/plan:** `Docs/superpowers/specs/2026-07-23-mcp-skills-model-only-runner-design.md`; `Docs/superpowers/plans/2026-09-07-mcp-bounded-model-completion-adapter-implementation-plan.md`

## Decision

Bounded MCP model completion resolves provider credentials through an opt-in shared runtime mode that revalidates one exact authenticated user and optional active team/organization, permits broader credential fallback only after authoritative absence, and uses frozen server endpoint settings.

## Context

Legacy provider lookups can return no row for either a missing credential or lost authority. Treating both outcomes as absence would allow execution with a broader key after membership revocation. A user can belong to multiple teams and organizations, so inferred memberships cannot identify the active execution scope. The bounded completion adapter also needs a stable endpoint that stored credential metadata cannot replace.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Interpret every missing row as permission to fall back | Conflates absent credentials with revoked or unavailable authority. |
| Infer a team or organization from memberships | Selects a credential scope that was never explicitly authenticated. |
| Duplicate provider resolution inside MCP | Creates a second credential/security implementation and drifts from shared BYOK handling. |
| Change all legacy callers immediately | Widens the migration and regression surface beyond the bounded MCP adapter. |

## Consequences

The shared repositories expose distinct resolved, authorized-absent, unauthorized, and unavailable outcomes using one database snapshot per scope. Every supplied scope is validated before selecting user, team, organization, or server credentials. Revoked credentials and store failures fail closed. The strict runtime never uses a cached credential to skip a later authority check, and credential base-URL overrides are disabled. Existing callers retain their current default behavior until they explicitly adopt this mode.

Checks are authoritative at the database read; they do not promise a transaction spanning credential selection and an outbound network request. Later MCP admission and dispatch checks remain necessary. The additional reads are an accepted cost of preserving scope authority.

Amendment (TASK-15571, 2026-11-19): after every supplied scope is validated, a user, team or organization credential whose model the admin provider override's allowlist denies (`model_not_allowed`) is replaced by the server credential from the same frozen snapshot. This is the owner's rule that a denied model falls back to the server key. It is a policy fallback after authority is established, not a fallback on absence, and the reservation then counts and checks the call as a server-key call. A provider the override disables fails closed with its policy error, and an override store that cannot be read fails closed with `credential_store_unavailable`, as other store failures do. A server credential is not checked.

## Follow-up

Stage 3 adds conservative accounting; Stages 4 and 5 certify the bounded transport and compose the host adapter. ADR-025 continues to govern shared provider adapter routing.
