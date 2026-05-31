# Phase 3.1 Response Envelope Migration Recipe

Date: 2026-04-25

Scope: API v1 route-family migrations after the shared envelope helpers exist.

This recipe turns the response-shape inventory into a repeatable PR pattern. It does not decide the rollout switch; Phase 3.1 still needs an explicit contract decision for header opt-in, query opt-in, or versioned routes.

## Required Preconditions

- Phase 2 and PR #1125 are merged or explicitly accepted as stable bases.
- The standard envelope contract is approved:
  - success: `{"success": true, "data": ..., "meta": ...}`
  - error: `{"success": false, "error": {"code": ..., "message": ..., "details": ...}, "meta": ...}`
- The rollout switch is approved.
- Streaming, file, `204`, webhook, OpenAI-compatible, Anthropic-compatible, and provider-compatible route exemptions are recorded for the route family.
- Phase 3.2 canonical pagination metadata exists before wrapping list responses.

## Slice Shape

Use one small route-family PR at a time:

1. Add or reuse shared envelope helpers.
2. Add or reuse shared pagination helpers for list routes.
3. Add backend tests proving legacy default responses are unchanged.
4. Add backend tests proving opt-in envelope responses use the standard shape.
5. Update frontend parsing only for opt-in calls in the same route family.
6. Add frontend tests for legacy and opt-in payload parsing.
7. Run OpenAPI verification and focused backend/frontend tests.

## Endpoint Classification Checklist

For each route in the family, classify it as one of:

- `json-item`: named item/operation response model; eligible for `data=<current payload>`.
- `json-list`: named list response model; eligible after Phase 3.2 pagination metadata is available.
- `bare-list-or-dict`: eligible only after caller audit because top-level type changes are risky.
- `raw-response`: exempt until explicitly converted.
- `file-download`: exempt.
- `streaming`: exempt.
- `204`: exempt.
- `third-party-compatible`: exempt unless an API-versioning decision opts it in.
- `webhook`: exempt unless the external receiver contract is known.

## Compatibility Rules

- Legacy response body remains the default during Phase 3.
- Opt-in response wraps the existing payload under `data`; it does not mutate the payload model.
- List responses preserve legacy list metadata during the compatibility window.
- Canonical pagination metadata goes under `meta.pagination` for enveloped list responses.
- Status codes and headers remain unchanged.
- Error envelopes use sanitized details so Phase 3.3 raw-error protections are not weakened.
- `204 No Content` remains bodyless even when the request opts in.
- Binary and streaming responses ignore the envelope opt-in unless a future route-specific contract says otherwise.

## Backend Test Pattern

For each eligible route:

- Call without the rollout switch and assert the exact legacy top-level shape.
- Call with the rollout switch and assert:
  - `success is true`
  - `data` equals the legacy payload shape
  - `meta.request_id` is present when available
  - `meta.pagination` is present for list routes and absent or null for item routes
- For error paths, assert:
  - legacy request returns the current `detail` shape
  - opt-in request returns `success=false`
  - `error.code` is stable
  - raw exception text is not leaked

## Frontend Test Pattern

For each migrated client method:

- Keep a test for the legacy payload.
- Add a test for the opt-in envelope payload.
- Verify the method returns the same normalized UI type for both shapes.
- Verify file/download methods do not attempt envelope parsing.

## First Three Slices

1. `skills` helper-backed pilot:
   - Eligible: list, context, get, create, update, JSON import.
   - Exempt: delete `204`, export zip.
   - Defer: multipart import and execute until core list/detail behavior is proven.
2. `slides` list/detail pilot:
   - Eligible after `skills`: visual style list/detail and presentation get/create/update.
   - Exempt: presentation export, render artifacts/download-like paths.
   - Frontend risk: `listVisualStyles()` currently depends on `styles` plus `total_count`.
3. `data_tables` list/detail row-window pilot:
   - Eligible after `skills`: list and detail row-window routes.
   - Exempt: export.
   - Frontend risk: mapper accepts arrays, `tables`, `items`, or `results`, and reads `total`/`count`.

## Deprecation Window

Do not set a deprecation clock until:

- all known web UI and extension callers can parse envelope payloads
- OpenAPI output is stable
- provider-compatible and raw-response exemptions are documented
- route-family tests cover legacy and opt-in behavior

When maintainers choose a deprecation timeline, publish it as a separate API compatibility note rather than burying it in a route PR.
