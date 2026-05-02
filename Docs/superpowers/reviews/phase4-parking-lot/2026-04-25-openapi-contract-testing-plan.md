# Phase 4.6 OpenAPI Contract Testing Plan

**Date:** 2026-04-25

**Status:** Planning artifact complete; implementation deferred until Phase 3 helper schemas are stable.

## Purpose

Define how OpenAPI contract testing should evolve after Phase 3 response-envelope and pagination helpers land. The goal is to catch backend/frontend contract drift without forcing unstable generic envelope schemas or file/streaming routes into unsuitable shapes.

## Current OpenAPI Guard

Current script:

- `apps/extension/scripts/verify-openapi-client-paths.mjs`

Current package entry points:

- `apps/extension/package.json`: `verify:openapi`
- `apps/packages/ui/package.json`: `verify:openapi`

Current behavior:

- Reads `apps/extension/openapi.json` when present.
- Otherwise generates a spec from `tldw_Server_API.app.main` with `app.openapi()`.
- Uses a synthetic single-user API key for generation.
- Verifies `ClientPath` entries from `apps/packages/ui/src/services/tldw/openapi-guard.ts`.
- Verifies `MEDIA_ADD_SCHEMA_FALLBACK` fields are a subset of `/api/v1/media/add`.
- Allows known missing client paths unless `TLDW_VERIFY_OPENAPI_STRICT=1`.

## Phase 3 Risks

Response envelope risks:

- Pydantic generic schemas can produce noisy or unreadable component names.
- Alternate response shapes may not be represented clearly in `v1` OpenAPI.
- Error envelopes might appear to replace legacy `{"detail": ...}` defaults if modeled carelessly.

Pagination risks:

- Canonical metadata may be additive for some route families and opt-in-only for others.
- Legacy aliases may be public but under-documented in generated schemas.
- Provider-compatible cursor shapes should not be normalized accidentally.

Auth dependency risks:

- OpenAPI security metadata could drift if auth aliases obscure existing dependencies.
- Setup-local, webhook-secret, and provider-compatible routes may have intentionally different auth behavior.

## Contract Testing Layers

### Layer 1: Existing Client Path Guard

Keep:

- `bun run verify:openapi` from `apps/packages/ui`
- `bun run verify:openapi` from `apps/extension`

Do not remove the current known-missing exception map until each exception is either restored or intentionally documented elsewhere.

### Layer 2: Schema Name Guard

Add after Phase 3.1 helpers exist:

- Generate OpenAPI from `app.openapi()`.
- Assert envelope schema names are readable and stable enough for clients.
- Assert `ErrorEnvelope` is non-generic.
- Assert response envelope generics do not create an excessive number of duplicate schema components for the same route family.

Candidate test:

- `tldw_Server_API/tests/Utils/test_openapi_phase3_contract.py`

### Layer 3: Exemption Guard

Add after the first pilot:

- Assert file, streaming, WebSocket, webhook, and `204` route families are not documented as standard JSON envelope responses.
- Assert OpenAI-compatible routes remain provider-shaped.
- Assert `skills` export remains `application/zip`.
- Assert `skills` delete remains `204` with no response body.

### Layer 4: Pilot Route Contract Guard

For the `skills` pilot:

- Assert legacy default response model remains the current route model.
- Assert opt-in envelope behavior is documented in either OpenAPI alternate responses or a dedicated docs note, depending on maintainer decision.
- Assert canonical pagination metadata schema exists once Phase 3.2 helpers are in use.

### Layer 5: Strict Client Drift Guard

Only after Phase 3 pilots are stable:

- Run `TLDW_VERIFY_OPENAPI_STRICT=1 bun run verify:openapi` in a non-required or nightly lane first.
- Promote to required only after known missing paths are intentionally resolved or explicitly excluded by policy.

## Proposed Implementation Sequence

1. Add backend OpenAPI schema-name tests after response envelope schemas land.
2. Add backend exemption tests after the `skills` pilot lands.
3. Extend the existing JS OpenAPI guard only if client path drift is not enough.
4. Add a strict-mode CI experiment for `TLDW_VERIFY_OPENAPI_STRICT=1`.
5. Promote strict mode after exception cleanup.

## Canonical Spec Source Decision

Recommended:

- Treat generated `app.openapi()` as the canonical backend truth for CI.
- Use `apps/extension/openapi.json` only as a checked-in snapshot when maintainers intentionally refresh it.
- When both exist, the guard currently prefers the snapshot. Any Phase 4 change should explicitly decide whether that priority remains correct.

Open question:

- Should PRs that change backend routes be required to refresh `apps/extension/openapi.json`, or should CI always generate from backend source?

## Test Commands

Existing:

```bash
cd apps/packages/ui
bun run verify:openapi
```

Strict experiment:

```bash
cd apps/packages/ui
TLDW_VERIFY_OPENAPI_STRICT=1 bun run verify:openapi
```

Backend contract tests after implementation:

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/Utils/test_openapi_phase3_contract.py -v
```

## Do Not Do

- Do not make strict OpenAPI verification required while reviewed exceptions still exist.
- Do not force standard envelopes onto provider-compatible routes.
- Do not rely on OpenAPI alone for runtime behavior; keep endpoint tests for legacy and opt-in behavior.
- Do not model `204` responses as JSON envelope responses.
- Do not hide breaking response-shape changes behind schema aliases.

## Handoff Checklist

- [ ] Phase 3.1 helper schema names are stable.
- [ ] Phase 3.2 pagination schema names are stable.
- [ ] Maintainers decide whether opt-in envelope variants appear in `v1` OpenAPI.
- [ ] A generated OpenAPI contract test is added for schema names and exemptions.
- [ ] Existing `verify:openapi` remains green.
- [ ] Strict OpenAPI mode is trialed before becoming required.
