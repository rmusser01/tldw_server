# Phase 4.5 API Versioning Policy Design

**Date:** 2026-05-03

**Status:** Proposed design for maintainer review.

## Purpose

Define the policy that governs response-shape migrations after the completed
Phase 3 helper work and Phase 4 guardrail/decomposition stack. The goal is to
make future API contract changes predictable for both backend owners and
frontend/client consumers before any route family attempts to change its
default `v1` response shape.

This design is policy-first. It does not approve immediate runtime migrations.

## Current Constraints

Current versioning documentation in [Docs/API/api-versioning-strategy.md](../../API/api-versioning-strategy.md)
already defines path-based `/api/v1/` versioning and treats response-field
removal, error-format changes, auth changes, and path changes as breaking.

Current shared helper surfaces already exist:

- Backend canonical envelope schema uses `metadata`, not `meta`:
  [response_envelope.py](../../../tldw_Server_API/app/api/v1/schemas/response_envelope.py)
- Backend envelope builders and detection also use `metadata`:
  [response_envelope.py](../../../tldw_Server_API/app/api/v1/utils/response_envelope.py)
- Frontend shared unwrap helpers use the same `metadata` contract:
  [response-envelope.ts](../../../apps/packages/ui/src/services/response-envelope.ts)
- Canonical pagination docs currently describe additive nested `pagination`
  objects in `v1`:
  [Pagination.md](../../API/Pagination.md)

The older Phase 3 / Phase 4 planning notes are directionally useful but stale
in one important way: they refer to `meta.pagination`, which no longer matches
the helper contract that shipped. Phase 4.5 must resolve that drift and define
the canonical long-term rule.

## Design Goals

- Keep `/api/v1/` stable for current callers by default.
- Define when additive `v1` contract growth is allowed.
- Define when a change requires a sibling route or `/api/v2/`.
- Define the client-side migration contract so frontend/service code does not
  drift from backend policy.
- Make provider-compatible and special transport routes explicit exemptions
  instead of ad hoc exceptions.
- Keep the policy aligned with generated OpenAPI and shared helper code.

## Non-Goals

- No route implementation changes in this tranche.
- No immediate introduction of `/api/v2/` endpoints.
- No new public opt-in shipped by runtime code in this tranche.
- No mass rewrite of old planning documents beyond the docs chosen for this
  policy update.

## Recommended Policy

### 1. `v1` Remains Legacy-Default

`/api/v1/` keeps current default response shapes, default error bodies, default
pagination aliases, and current auth/status behavior unless a route family
documents an additive change that is non-breaking under the existing versioning
rules.

Implication:

- Default body-shape replacements in `v1` are not allowed.
- Default error-shape replacements in `v1` are not allowed.
- Default removal or rejection of accepted pagination aliases in `v1` is not
  allowed.

### 2. Header Opt-In Is Transitional, Not A New Major Version

If first-party JSON routes continue to support envelope-style or other additive
opt-in behavior within `v1`, that opt-in is transitional and subordinate to the
path-based versioning model.

Recommended rule:

- Header opt-in may be used for additive behavior inside `v1`.
- Header opt-in must not become the primary major-version mechanism.
- Future default-breaking behavior should move to `/api/v2/`, not to a new
  public header version.

This keeps the public major-version story consistent with the existing
path-based strategy.

### 3. Canonical Nested Metadata Uses `metadata.pagination`

The canonical nested location for additive pagination metadata inside canonical
envelopes is `metadata.pagination`, not `meta.pagination`.

Reason:

- It matches the backend schema/helper code already in the repo.
- It matches the frontend shared unwrap typing that already shipped.
- It avoids introducing a second nested metadata convention that would need its
  own migration later.

For non-envelope additive `v1` responses that expose nested canonical
pagination, the body-level field remains `pagination` as documented today in
`Docs/API/Pagination.md`.

### 4. Route Families Must Follow Explicit Migration Rules

Phase 4.5 should classify routes into policy families:

- **First-party JSON routes**
  - Can add non-breaking fields in place.
  - Can expose additive opt-in envelope behavior in `v1`.
  - Can move default-breaking body changes to `/api/v2/`.
- **Provider-compatible routes**
  - Must preserve upstream/provider body shapes by default.
  - Must not be silently normalized into tldw-specific default envelopes or
    pagination shapes in `v1`.
  - If tldw-specific normalization is desired, use a sibling route or `/api/v2/`.
- **Raw-list and custom legacy-envelope routes**
  - Must not be reshaped in place when the default body contract changes.
  - Require a sibling object route or `/api/v2/` for default-breaking shape
    upgrades.
- **Transport/special routes**
  - Streaming, file downloads, webhooks, WebSockets, and `204 No Content`
    remain envelope-exempt unless explicitly documented otherwise.

### 5. Frontend/Client Contract Must Stay Domain-Shaped By Default

Client behavior needs to be part of the policy, not an afterthought.

Recommended rule:

- Shared client/service layers may detect, send opt-in headers for, and unwrap
  canonical envelopes or transitional wrappers.
- UI/domain consumers should continue to receive stable domain-shaped data
  rather than raw transport envelopes unless they intentionally target a new
  versioned contract.
- Route-family migration is incomplete until backend docs, OpenAPI treatment,
  and frontend/client unwrap/typing behavior are all defined together.

This preserves the boundary established by the Phase 3 frontend unwrap helpers:
transport concerns live in shared client layers, not scattered UI components.

### 6. Deprecation Headers Are For Approved Deprecation Windows, Not Additive `v1` Pilots

The repo-wide deprecation header guidance stays valid, but it should not be
applied automatically to additive Phase 3/Phase 4 compatibility work.

Recommended rule:

- Do not emit deprecation headers merely because a legacy `v1` shape coexists
  with additive opt-in behavior.
- Emit `Deprecation` / `Sunset` headers only when maintainers approve a real
  deprecation window for a route, field, or versioned surface.

This avoids implying that every additive `v1` pilot is already on a removal
timeline.

## Decision Table

| Change | Allowed in default `v1`? | Allowed as additive `v1` opt-in? | Requires sibling route or `/api/v2/`? |
| --- | --- | --- | --- |
| Add optional response fields to first-party JSON route | Yes | Yes | No |
| Replace legacy default body with envelope | No | Yes, if additive only | Yes for default behavior |
| Replace legacy default error body/status contract | No | Yes, if additive only | Yes for default behavior |
| Add canonical nested `pagination` while preserving current fields | Yes | Yes | No |
| Remove legacy pagination aliases/fields | No | No | Yes |
| Normalize provider-compatible body into tldw envelope by default | No | No by default | Yes |
| Convert raw list response into object body by default | No | No | Yes |
| Change auth requirement or auth status behavior | No | No | Yes |

## OpenAPI And Documentation Policy

- `Docs/API/api-versioning-strategy.md` remains the canonical high-level
  versioning policy document.
- `Docs/API/Pagination.md` remains the canonical additive pagination contract
  document for `v1`.
- Phase 4.5 should decide how opt-in `v1` variants are documented:
  - narrative docs only;
  - alternate generated OpenAPI responses for selected routes; or
  - deferred until a route family is explicitly versioned.
- `/api/v2/` should remain path-based by default for major versioning.

Recommended default:

- Keep generated OpenAPI centered on default `v1` behavior.
- Document additive opt-in semantics narratively unless a route family
  explicitly needs generated alternate responses.

## Migration Trigger Checklist

Future route-family work should require `/api/v2/` or a sibling route if any of
these are true:

- default response fields are removed, renamed, or wrapped;
- default error shape changes;
- default auth/status behavior changes;
- a raw list must become an object response;
- a provider-compatible payload must become tldw-normalized by default;
- clients would need to change default deserialization logic for the same `v1`
  path.

## Concrete Phase 4.5 Docs Tranche

This policy tranche should produce:

- an update to `Docs/API/api-versioning-strategy.md`;
- a small update to `Docs/API/Pagination.md` clarifying how additive `v1`
  pagination relates to `metadata.pagination` and default-breaking migrations;
- a roadmap/tracker update in issue `#1116` summarizing that Phase 4.5 is now a
  policy/spec item and that prior Phase 4 implementation work is complete;
- no runtime behavior changes.

## Risks

- Leaving the `meta.pagination` wording uncorrected would bake a stale contract
  back into policy and create unnecessary future migration work.
- Over-specifying OpenAPI alternate responses too early could create a contract
  maintenance burden before route families are actually versioned.
- A backend-only policy would leave client migration ownership ambiguous and
  encourage UI-level transport leakage.

## Recommendation

Adopt the policy above and keep the first implementation tranche strictly
docs-only. After that lands, future route-family work can use this policy as
the decision gate for whether a change belongs in additive `v1`, a sibling
route, or `/api/v2/`.
