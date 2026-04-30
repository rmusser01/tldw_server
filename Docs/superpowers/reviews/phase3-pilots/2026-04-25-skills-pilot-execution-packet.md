# Phase 3 Skills Pilot Execution Packet

**Date:** 2026-04-25

**Status:** Ready for implementation after Phase 2 PRs and PR #1125 are merged or explicitly accepted as stable bases.

## Purpose

Convert the Phase 3.1, Phase 3.2, and Phase 3.4 planning artifacts into an implementation-ready pilot packet for the `skills` route family. This packet is not a runtime change. It records the decisions, file touch points, sequencing, and verification gates needed for the future code PR.

## Preconditions

Do not start runtime implementation until these are true:

- Phase 2 PR bases are merged or explicitly accepted as stable bases.
- PR #1125 is merged or explicitly accepted as a stable base for sanitized errors.
- Maintainers accept the Phase 3.1 rollout switch and response envelope shape.
- Frontend owner accepts the temporary opt-in parsing approach for the `skills` client.

## Proposed Maintainer Decisions

### Phase 3.1 Response Envelope

Recommended decision:

- Use `X-TLDW-Response-Envelope: v1` as the implementation opt-in.
- Keep `response_envelope=v1` as a manual/debug-only query opt-in if included at all.
- Keep legacy route payloads as the default.
- Use `success`, `data`, `error`, and `meta` as the standard field names.
- Put request IDs in `meta.request_id` when available.
- Put pagination metadata in `meta.pagination` when Phase 3.1 and Phase 3.2 are both active for a route.
- Do not add a top-level success `message`.
- Preserve legacy `{"detail": ...}` errors for non-opt-in requests.

Accepted exemptions for this pilot:

- `DELETE /api/v1/skills/{skill_name}` remains `204 No Content` and is not enveloped.
- `GET /api/v1/skills/{skill_name}/export` remains a raw zip response and is not enveloped.

### Phase 3.2 Pagination

Recommended decision:

- Use canonical `limit` and `offset` for first-party offset pagination.
- Keep existing `GET /api/v1/skills/` query parameters unchanged.
- Add canonical metadata only for opt-in responses during the pilot.
- Keep top-level `skills`, `count`, `total`, `limit`, and `offset` in the legacy response body.
- Use `has_more = offset + count < total`.
- Use `next_offset = offset + limit` only when `has_more` is true.

### Phase 3.4 Auth Dependencies

Recommended decision:

- Add type aliases for principal-returning dependencies, but keep existing lower-case factory names for role/permission/scope guards.
- Use `CurrentPrincipal` only after contract tests prove request-state compatibility.
- Do not change `require_token_scope(...)` to return an `AuthPrincipal`; treat it as `TokenScopeGuard` where needed.
- Migrate `skills` auth in a separate PR after the response-envelope and pagination pilot is green.

## Backend Touch Points

Shared helper PR:

- Create `tldw_Server_API/app/api/v1/schemas/response_envelope.py`
- Create `tldw_Server_API/app/api/v1/utils/response_envelope.py`
- Create `tldw_Server_API/app/api/v1/schemas/pagination.py`
- Extend `tldw_Server_API/app/api/v1/endpoints/_pagination_utils.py`
- Add `tldw_Server_API/tests/Utils/test_response_envelope.py`
- Add `tldw_Server_API/tests/Utils/test_pagination_contract.py`
- Add `tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py`

Skills pilot PR:

- Modify `tldw_Server_API/app/api/v1/endpoints/skills.py`
- Modify `tldw_Server_API/app/api/v1/schemas/skills_schemas.py` only if `seed_builtin_skills` gets a typed response model in the same slice.
- Extend `tldw_Server_API/tests/Skills/integration/test_skills_api.py`

Auth cleanup PR:

- Modify `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/skills.py`
- Extend `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
- Extend or add AuthNZ request-state tests if `get_skills_service` changes from `User` to `AuthPrincipal`.

## Frontend Touch Points

Client files:

- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- `apps/packages/ui/src/types/skill.ts`
- `apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts`
- `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

Current client behavior:

- `workspace-api.ts` methods call `bgRequest(...)` directly.
- `bgRequest(...)` supports custom headers through its `headers` field.
- `listSkills(...)` currently returns the backend body as-is.
- `SkillsManager` reads `data.skills` and `data.total`.

Recommended frontend approach:

- Add the opt-in header only on the specific methods included in the pilot.
- Add a small `unwrapTldwEnvelope(...)` helper near the tldw client domain layer if no shared helper already exists at implementation time.
- Keep UI consumers receiving legacy-shaped `SkillsListResponse` from `listSkills(...)`.
- For the pilot, unwrap `response.data` in the client method and map `meta.pagination.total` only as a fallback. Do not make components envelope-aware.
- Do not opt in `exportSkill(...)` because it expects an `ArrayBuffer` and returns a `Blob`.
- Do not opt in `deleteSkill(...)` because it expects no response body.

## Implementation Sequence

### PR 1: Shared Helpers Only

Goal:

- Add reusable response envelope, pagination, and auth alias contracts without changing route behavior.

Backend work:

- Implement response envelope schemas and builders.
- Implement pagination schemas and normalization helpers.
- Keep `build_link_header(...)` backwards compatible.
- Add auth dependency aliases or documented helper exports.
- Add unit/contract tests.

Exit criteria:

- Existing endpoints behave the same without opt-in.
- Helper tests pass.
- OpenAPI generation remains readable.

### PR 2: Skills Envelope And Pagination Pilot

Goal:

- Prove opt-in envelope and canonical pagination metadata on a low-risk route family.

Backend work:

- Add opt-in envelope handling to:
  - `GET /api/v1/skills/`
  - `GET /api/v1/skills/{skill_name}`
  - `POST /api/v1/skills/`
  - `PUT /api/v1/skills/{skill_name}`
- Add canonical pagination metadata to opt-in `GET /api/v1/skills/`.
- Keep legacy default responses unchanged.
- Leave `DELETE /api/v1/skills/{skill_name}` and export zip responses exempt.
- Hold `POST /api/v1/skills/import/file` unless normal JSON routes are already green.
- Hold `POST /api/v1/skills/seed` unless it first gets a typed response model.

Frontend work:

- Add opt-in header only to the selected `skills` client methods.
- Unwrap envelopes inside `workspace-api.ts`.
- Keep component-facing return shapes unchanged.
- Add boundary tests proving header use and unwrapping.
- Add component test coverage showing `SkillsManager` still reads list rows and totals.

Exit criteria:

- Legacy backend tests still pass.
- Opt-in backend tests prove wrapped shapes and metadata.
- UI client tests prove callers still receive legacy-shaped data.
- OpenAPI verification passes or documented OpenAPI caveat is added to the plan.

### PR 3: Skills Auth Cleanup

Goal:

- Prove the Phase 3.4 alias surface on one low-risk route family without changing response shapes.

Backend work:

- Migrate only `skills` identity resolution after alias contract tests are green.
- Preserve `get_chacha_db_for_user` behavior.
- Preserve `execute_skill` `RequestContext.user_id` behavior.
- Preserve TEST_MODE overrides for `get_request_user`.
- Preserve single-user API-key and multi-user JWT behavior.

Exit criteria:

- `skills` auth/status-code tests pass.
- Request-state compatibility tests pass.
- Focused Bandit on touched Python paths is clean.

## Backend Test Additions

Add to `tldw_Server_API/tests/Skills/integration/test_skills_api.py`:

- list skills legacy default still returns top-level `skills`, `count`, `total`, `limit`, and `offset`.
- list skills opt-in returns `success=true`, `data.skills`, and `meta.pagination`.
- list skills opt-in pagination reports `mode=offset`, `has_more`, and `next_offset`.
- get skill opt-in wraps `SkillResponse` under `data`.
- create skill opt-in preserves `201` and wraps `SkillResponse`.
- update skill opt-in preserves `If-Match` behavior and wraps `SkillResponse`.
- delete skill ignores opt-in and returns empty `204`.
- export skill ignores opt-in and returns `application/zip`.
- service-layer `SkillsError` opt-in returns sanitized standard error after PR #1125 is stable.

Add to helper tests:

- response-envelope opt-in detection from header.
- response-envelope query flag behavior if included.
- offset pagination helper computes `has_more` and `next_offset`.
- alias helpers preserve `request.state.auth` and `_auth_user` compatibility.

## Frontend Test Additions

Add to `apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts`:

- `listSkills` sends `X-TLDW-Response-Envelope: v1` when the pilot is enabled.
- `listSkills` unwraps `{ success: true, data, meta }` to legacy-shaped data.
- `getSkill`, `createSkill`, and `updateSkill` unwrap pilot envelope responses.
- `exportSkill` does not send the envelope opt-in header.
- `deleteSkill` does not require envelope unwrapping.

Add to `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`:

- list rows still render when `listSkills` internally unwraps an envelope.
- pagination total still comes from the client-returned legacy-shaped `total`.

## Verification Commands

Backend helper PR:

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/Utils/test_response_envelope.py -v
python3 -m pytest tldw_Server_API/tests/Utils/test_pagination_contract.py -v
python3 -m pytest tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py -v
```

Skills pilot PR:

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -v
cd apps/packages/ui
bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx
npm run verify:openapi
```

Auth cleanup PR:

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/AuthNZ/test_auth_dependency_contract.py -v
python3 -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -v
python3 -m bandit -r tldw_Server_API/app/api/v1/API_Deps tldw_Server_API/app/api/v1/endpoints/skills.py
```

## Do Not Change In The Pilot

- Do not make envelopes the default.
- Do not wrap file, streaming, WebSocket, webhook, or `204` responses.
- Do not change OpenAI-compatible or provider-compatible response shapes.
- Do not remove legacy top-level `skills` pagination fields.
- Do not make components consume envelope wrappers directly.
- Do not remove `get_current_user`, `get_current_active_user`, or `get_request_user`.
- Do not change `require_token_scope(...)` return behavior.

## Remaining Questions

- Should `POST /api/v1/skills/seed` get a typed response model in PR 2 or wait for a later route cleanup slice?
- Should `POST /api/v1/skills/import` join PR 2 or wait until get/list/create/update are proven?
- Should `POST /api/v1/skills/import/file` remain out of the first pilot because multipart upload flows have a larger client surface?
- Should opt-in be enabled per method by hard-coded header during the pilot or through a temporary client option passed into each method?
