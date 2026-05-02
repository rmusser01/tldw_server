# Phase 3.1 Response Envelope Helper Contract Spec

**Date:** 2026-04-25

**Status:** Draft contract for implementation after PR #1125 and the Phase 2 bases are accepted stable.

## Purpose

Define the shared response-envelope helper contract before changing runtime behavior. This keeps Phase 3.1 implementation small, testable, and compatible with current clients.

## Current Constraints

- `main.py` currently returns unhandled server errors as `{"detail": "Internal server error"}`.
- Runtime CORS behavior exposes `X-Request-ID`, `traceparent`, and `X-Trace-Id`; Phase 3.1 must not change that header behavior.
- PR #1125 is still stabilizing sanitized error details, so envelope helpers must not reintroduce raw exception text.
- Streaming routes, file responses, `204 No Content`, webhooks, and third-party-compatible provider APIs need explicit exemption handling.
- Phase 4.5 owns broader API versioning, so Phase 3.1 should use opt-in behavior unless maintainers explicitly accept a breaking default change.

## Rollout Switch

Recommended Phase 3.1 rollout switch:

- Header opt-in: `X-TLDW-Response-Envelope: v1`
- Optional query flag for manual testing only: `response_envelope=v1`
- Legacy default: unchanged route payloads and unchanged FastAPI-style errors.

Rules:

- Header opt-in wins over query flag.
- Unknown envelope versions are rejected with legacy `400` by default unless the request already opted into a supported version.
- Exempt route categories ignore the opt-in and continue returning their established response shape.
- Do not make the envelope the default until Phase 4.5 API versioning or a maintainer-approved compatibility plan exists.

## Success Shape

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

Contract:

- `success` is always `true` for success envelopes.
- `data` contains the original route payload without mutating that payload object.
- `meta.request_id` is copied from `request.state.request_id` when present, otherwise from `X-Request-ID` when present.
- `meta.pagination` is either absent/null or populated by Phase 3.2 pagination helpers.
- `meta.warnings` defaults to an empty list only when warnings exist or the model default is serialized.

## Error Shape

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

Contract:

- `success` is always `false` for error envelopes.
- `error.code` is stable machine-readable text.
- `error.message` is safe user-facing text.
- `error.details` is structured only when the source is already safe, such as request validation field errors.
- Unhandled `5xx` errors always use `details: null` unless a route explicitly raises a sanitized public exception.

## Proposed Schemas

Create `tldw_Server_API/app/api/v1/schemas/response_envelope.py`.

Schema names:

- `EnvelopeMeta`
- `EnvelopeWarning`
- `EnvelopeError`
- `ResponseEnvelope[T]`
- `ErrorEnvelope`

Field guidance:

- Use Pydantic generics only for success payloads.
- Keep `ErrorEnvelope` non-generic so OpenAPI output remains readable.
- Keep `pagination` typed loosely at first, for example `dict[str, Any] | None`, until Phase 3.2 provides shared pagination schema types.
- Avoid top-level `message` on success envelopes. Success copy belongs in route payloads or warnings, not in the wrapper.

## Proposed Builders

Create `tldw_Server_API/app/api/v1/utils/response_envelope.py`.

Helper functions:

- `wants_response_envelope(request: Request) -> bool`
- `is_envelope_exempt_response(response: Response | Any, status_code: int | None = None) -> bool`
- `build_success_envelope(data: T, *, request: Request | None = None, meta: Mapping[str, Any] | None = None, warnings: Sequence[Any] | None = None, pagination: Any | None = None) -> ResponseEnvelope[T]`
- `build_error_envelope(code: str, message: str, *, request: Request | None = None, status_code: int | None = None, details: Any | None = None) -> ErrorEnvelope`
- `sanitize_error_details(details: Any, *, status_code: int | None = None) -> Any | None`

Builder rules:

- Never mutate `data`, `details`, or caller-provided `meta`.
- Do not serialize raw exceptions.
- For `status_code >= 500`, default `details` to `None`.
- For validation errors, allow structured field lists produced by FastAPI/Pydantic.
- For `HTTPException`, allow string detail and structured dict/list detail only when it is already intended as public API detail.

## Exception Handling Contract

Implementation should update exception handling only for opt-in requests:

- `HTTPException`: opt-in response uses `error.code` derived from status family or a safe detail code when available.
- `RequestValidationError`: opt-in response uses `error.code = "validation_error"` and safe field details.
- Unhandled `Exception`: opt-in response uses `error.code = "internal_error"`, `message = "Internal server error"`, and `details = null`.
- `ClientDisconnect`: preserve current `499` behavior and safe message.
- Non-opt-in requests keep legacy `{"detail": ...}` behavior.

Headers:

- Preserve `X-Request-ID`, `traceparent`, and `X-Trace-Id` behavior.
- Preserve auth headers such as `WWW-Authenticate`.
- Preserve existing CORS expose headers.

## Exemption Contract

Do not envelope these categories in Phase 3.1:

- `StreamingResponse`
- file downloads and generated binary payloads
- `204 No Content`
- WebSocket messages
- webhook callback payloads
- OpenAI-compatible chat, embeddings, audio, and eval list shapes unless a route-specific compatibility plan exists
- third-party provider passthrough payloads

Exempt routes can still use `request_id` headers and sanitized errors independently of envelope adoption.

## OpenAPI Contract

Before migrating a route family:

- Verify generic schema names are readable.
- Prefer explicit response models for pilot routes rather than dynamic wrapper responses that disappear from OpenAPI.
- Keep legacy response models documented during compatibility windows.
- Record whether the opt-in envelope appears in OpenAPI as an alternate response or is only documented until Phase 4.5.

## Test Matrix

Unit tests for helpers:

- success envelope with dict payload
- success envelope with list payload
- success envelope with Pydantic model payload
- request ID copied from `request.state.request_id`
- request ID fallback from `X-Request-ID`
- warnings preserved and copied
- pagination metadata preserved and copied
- error envelope for `400`
- error envelope for `422` validation details
- error envelope for `500` strips details
- raw `Exception` object details are stripped
- input payload/meta/detail objects are not mutated

Framework tests:

- legacy default `HTTPException` shape remains `{"detail": ...}`
- opt-in `HTTPException` returns `ErrorEnvelope`
- legacy default validation error shape remains compatible
- opt-in validation error returns `validation_error`
- opt-in unhandled error returns sanitized `internal_error`
- `ClientDisconnect` remains safe
- exempt streaming/file responses ignore opt-in

Pilot tests:

- `skills` legacy default response remains unchanged.
- `skills` opt-in response wraps the same payload in `data`.
- UI client parser accepts the pilot opt-in path without breaking legacy calls.

## Pending Decisions

- Whether the query flag should ship beyond local/manual testing.
- Whether OpenAPI should expose both legacy and envelope response models in Phase 3.1 or defer formal alternate response docs to Phase 4.5.
- Whether `meta.warnings` should serialize as an empty list or be omitted when empty.
