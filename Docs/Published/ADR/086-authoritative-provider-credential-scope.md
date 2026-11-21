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

Amendment (TASK-15581, 2026-11-19): the owner reversed the TASK-15571 rule, so the amendment above is withdrawn. The owner's rule: "if a user tries to use a model/service thats been explicitly disabled by the admin, they shouldn't be able to use it; if its just been disabled as in 'no credits/key enabled' then tehy should be able to use it via BYOK". An explicit disable is the admin provider override's `is_enabled = false` (`provider_disabled`), or a model left off its `allowed_models` when that list is set (`model_not_allowed`). After every supplied scope is validated, this runtime refuses such a call with its policy error on whichever credential the scope chose: the user's, a team's, an organization's or the server's. Nothing falls back to the server credential. An override store that cannot be read still fails closed with `credential_store_unavailable`. A provider the server has no key or credits for is not a disable, and a user's own credential still serves it. Every provider adapter call meets the same check in the adapter base class (ADR-067), for its model and each OpenRouter fallback model, before the quota check. The chat calls that do not use an adapter check it themselves: this runtime for the MCP completion tool, the Messages API's native path at resolution, Slides standalone-HTML generation before its request, and the llama.cpp inference endpoint before it runs the loaded model (a provider disable only, since that endpoint names no model). So no LLM chat path runs a disabled provider or model. Embeddings check the override at their endpoint. Image generation, speech-to-text and the OCR backends are outside this rule. Outside the app process, each standalone worker or CLI that can make an LLM call loads the override store when it starts.

Amendment (TASK-15582, 2026-11-21): the owner chose "Server key only" for the provider override's `allowed_models`, and added `denied_models` as the explicit model disallow. A model on `denied_models` is refused on every credential and every path with `model_denied`, and wins over `allowed_models`. A model left off `allowed_models` is only not offered on the server's credential: it is refused with `model_not_allowed` when the call runs on the server's credential, a team's or an organization's, or a user's credential sent to an endpoint the operator runs, and the user's own credential at its vendor's host (credential source `user`, ADR-083) runs it, as it runs a provider the server has no credential for. `is_enabled = false` stays the explicit provider disallow, refused on every credential. This runtime still checks after the scope check, against the credential the scope chose. Elsewhere, a check made before the call's credential is known refuses only what every credential refuses, and the credential runtime or the adapter base class checks `allowed_models` once the credential is known. Embeddings, text-to-speech, shared-workspace chat and the persona voice preflight check `allowed_models` before they know the credential, so there it still refuses every credential.

## Follow-up

Stage 3 adds conservative accounting; Stages 4 and 5 certify the bounded transport and compose the host adapter. ADR-025 continues to govern shared provider adapter routing.
