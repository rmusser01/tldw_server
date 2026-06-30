# Phase 3.1 Standard Response Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking. Do not start endpoint rewrites until the frontend contract decision is explicitly accepted.

**Goal:** Introduce a standard success/error response envelope for API v1 without breaking existing frontend, extension, OpenAPI, streaming, file, or third-party-compatible routes.

**Architecture:** Add a shared envelope schema and builder first, then roll it out behind an explicit compatibility boundary. Existing route payloads remain stable until each endpoint family is migrated with frontend/client coverage. Streaming responses, file downloads, OpenAI-compatible endpoints, webhook callbacks, and `204 No Content` routes are exempt unless a later API-versioning phase opts them in.

**Tech Stack:** FastAPI, Pydantic generics, OpenAPI generation, shared UI API client, pytest, Vitest

---

## Current Inventory

Measured on 2026-04-25 from `tldw_Server_API/app/api/v1`:

- `1634` route decorators currently declare `response_model=...`.
- `134` route response models are bare `list[...]`/`dict` style.
- `175` route response models are named `*ListResponse` shapes.
- `81` route response models are generic status/message/envelope-like shapes.
- Prompt Studio already has a local `StandardResponse` and `ListResponse` in `prompt_studio_base.py`.
- `main.py` still returns global unhandled errors as `{"detail": "Internal server error"}`.
- Draft helper contract spec created: `Docs/superpowers/reviews/api-response-envelope/2026-04-25-helper-contract-spec.md`.

Observed response-shape families:

- `{"items": [...], "pagination": {...}}` for chat, media, kanban, and several newer endpoints.
- `{"data": ..., "metadata": ...}` in Prompt Studio.
- Top-level list metadata such as `total`, `limit`, `offset`, `page`, or `has_more`.
- Bare arrays for older list endpoints.
- Provider-compatible shapes for OpenAI, Anthropic, embeddings, audio, and paper-search providers.

## Contract Decision Required

Before code changes, confirm the standard success shape:

```json
{
  "success": true,
  "data": {},
  "meta": {
    "request_id": "optional",
    "pagination": null,
    "warnings": []
  }
}
```

Confirm the standard error shape:

```json
{
  "success": false,
  "error": {
    "code": "internal_error",
    "message": "Internal server error",
    "details": null
  },
  "meta": {
    "request_id": "optional"
  }
}
```

Compatibility rule:

- Existing clients continue receiving legacy route payloads by default during Phase 3.1.
- The standard envelope is exposed either by an explicit opt-in header/query flag or by new versioned routes. Because Phase 4.5 covers API versioning, the Phase 3.1 default should be opt-in unless maintainers approve a coordinated breaking change.

## File Structure

- Create: `tldw_Server_API/app/api/v1/schemas/response_envelope.py`
- Create: `tldw_Server_API/app/api/v1/utils/response_envelope.py`
- Create: `tldw_Server_API/tests/Utils/test_response_envelope.py`
- Modify: `tldw_Server_API/app/main.py`
- Modify: selected pilot endpoint family after approval
- Modify: `apps/packages/ui/src/services/tldw/request-core.ts` only after the backend opt-in contract is green
- Create/modify: focused UI API client tests for envelope opt-in behavior

## Task 1: Lock The Envelope Contract

- [x] Draft the helper, rollout switch, exemption, and error-handling contract spec. See `Docs/superpowers/reviews/api-response-envelope/2026-04-25-helper-contract-spec.md`.
- [ ] Confirm field names with frontend/client owners: `success`, `data`, `error`, `meta`.
- [ ] Confirm whether `message` belongs only inside `error.message` or also as top-level success copy.
- [ ] Confirm whether `request_id` is always in `meta` when available and mirrored to `X-Request-ID`.
- [ ] Confirm the rollout switch: header opt-in, query opt-in, or versioned route.
- [ ] Record exempt route categories: streaming, files, `204`, webhooks, and third-party-compatible API surfaces.

## Task 2: Add Shared Envelope Schemas And Builders

- [ ] Add Pydantic generic success/error envelope models in `response_envelope.py`.
- [ ] Add helper builders that accept existing payloads and optional metadata without mutating the payload object.
- [ ] Add a sanitizer for error details so Phase 3.3 raw-error work is not weakened.
- [ ] Add unit tests for success payloads, error payloads, metadata, warnings, and request IDs.
- [ ] Verify generated OpenAPI schemas are readable and do not explode into unusable generic names.

## Task 3: Normalize Framework Error Output Behind The Same Contract

- [ ] Add or update HTTP exception handling so opt-in requests receive the standard error envelope.
- [ ] Add validation-error handling so request validation failures map to `error.code="validation_error"`.
- [ ] Preserve legacy `{"detail": ...}` errors for non-opt-in requests during the compatibility window.
- [ ] Add regression tests for `HTTPException`, validation errors, unhandled exceptions, and client disconnect behavior.
- [ ] Ensure runtime CORS header behavior in `main.py` remains unchanged.

## Task 4: Pilot One Low-Risk Endpoint Family

Choose one family with:

- existing typed response models
- stable tests
- no streaming/file behavior
- limited frontend callers

Recommended pilot candidates:

- `slides` list/detail endpoints
- `skills` list/detail endpoints
- `data_tables` list/detail endpoints

Pilot steps:

- [ ] Add opt-in envelope responses without changing default legacy responses.
- [ ] Add focused backend tests for legacy default and standard opt-in modes.
- [ ] Update shared UI client request parsing only for opt-in calls used by the pilot.
- [ ] Add client tests showing legacy and envelope responses both parse correctly.
- [ ] Record any OpenAPI caveats before expanding the rollout.

## Task 5: Prepare The Expansion Backlog

- [x] Create a response-shape inventory table grouped by route family. See `Docs/superpowers/reviews/api-response-envelope/2026-04-25-response-shape-inventory.md`.
- [x] Rank route families by client coupling and migration risk. Initial recommendation: start with `skills`; avoid `media`, `chat`, `admin`, and provider-compatible surfaces first.
- [x] Create a pilot-readiness map for the selected `skills` candidate. See `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-readiness.md`.
- [x] Define a mechanical migration recipe for future slices. See `Docs/superpowers/reviews/api-response-envelope/2026-04-25-envelope-migration-recipe.md`.
- [x] Define the shared helper contract before runtime code changes. See `Docs/superpowers/reviews/api-response-envelope/2026-04-25-helper-contract-spec.md`.
- [x] Draft the `skills` pilot execution packet with maintainer decisions, frontend coordination, and verification gates. See `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-execution-packet.md`.
- [ ] Define a deprecation window for legacy payloads if maintainers want the standard envelope to become default.
- [ ] Update the closeout tracker with the pilot result and remaining route-family backlog.

## Verification

Minimum verification before any Phase 3.1 PR:

```bash
python3 -m pytest tldw_Server_API/tests/Utils/test_response_envelope.py -v
python3 -m pytest <pilot backend test files> -v
cd apps/packages/ui && bunx vitest run <pilot client tests>
cd apps/packages/ui && npm run verify:openapi
```

If Python source files outside schemas/utils are modified, run focused Bandit on touched Python paths.

## Out Of Scope

- Repo-wide endpoint wrapping in one PR.
- Changing third-party-compatible OpenAI/Anthropic payload shapes.
- Standardizing pagination fields; Phase 3.2 owns that.
- API versioning policy; Phase 4.5 owns that unless maintainers pull it forward.
