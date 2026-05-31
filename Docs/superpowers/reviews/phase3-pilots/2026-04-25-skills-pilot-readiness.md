# Phase 3 Pilot Readiness: Skills

Date: 2026-04-25

Scope:

- `tldw_Server_API/app/api/v1/endpoints/skills.py`
- `tldw_Server_API/app/api/v1/schemas/skills_schemas.py`
- `tldw_Server_API/tests/Skills/integration/test_skills_api.py`
- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- `apps/packages/ui/src/types/skill.ts`
- `apps/packages/ui/src/components/Option/Skills/*`

This note validates `skills` as the first Phase 3 pilot candidate across response envelopes, pagination, and auth dependency cleanup. It is a planning artifact only; no endpoint behavior has been changed.

Implementation packet:

- `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-execution-packet.md`

## Backend Route Map

| Method | Path | Current response | Pagination | Phase 3 treatment |
| --- | --- | --- | --- | --- |
| `GET` | `/api/v1/skills/` | `SkillsListResponse` | `limit`, `offset`, `count`, `total` | Good first offset pagination and response-envelope pilot. Keep legacy body by default; opt-in can wrap payload under `data` and put canonical pagination under `meta.pagination`. |
| `GET` | `/api/v1/skills/context` | `SkillContextPayload` | None | Can be opt-in envelope pilot after list/detail behavior is proven. |
| `GET` | `/api/v1/skills/{skill_name}` | `SkillResponse` | None | Good item-response envelope pilot. |
| `POST` | `/api/v1/skills/` | `SkillResponse`, `201` | None | Good create-response envelope pilot. Preserve `201`. |
| `PUT` | `/api/v1/skills/{skill_name}` | `SkillResponse` | None | Good update-response envelope pilot. Preserve `If-Match` behavior. |
| `DELETE` | `/api/v1/skills/{skill_name}` | `204 No Content` | None | Envelope-exempt. Preserve empty body. |
| `POST` | `/api/v1/skills/import` | `SkillResponse`, `201` | None | Good JSON import envelope pilot after create/get/list are covered. |
| `POST` | `/api/v1/skills/import/file` | `SkillResponse`, `201` | None | Multipart upload with JSON response. Eligible for opt-in envelope later, but keep upload parsing untouched. |
| `GET` | `/api/v1/skills/{skill_name}/export` | Raw zip `Response` | None | Envelope-exempt file response. Preserve `Content-Disposition` and `application/zip`. |
| `POST` | `/api/v1/skills/{skill_name}/execute` | `SkillExecutionResult` | None | Eligible for envelope after auth dependency behavior is mapped because it injects both service and current user. |
| `POST` | `/api/v1/skills/seed` | Bare `dict` with `seeded` and `count` | None | Add a typed response model before including in an envelope pilot. |

## Auth Dependency Map

All normal `skills` routes depend on `get_skills_service`, which currently depends on:

- `get_request_user`, returning a legacy `User`.
- `get_chacha_db_for_user`, returning the per-user ChaCha DB.

The execute route also injects `current_user: User = Depends(get_request_user)` directly so it can populate `RequestContext` for fork-mode execution.

Not present in this route family:

- `require_roles`
- `require_permissions`
- `require_token_scope`
- `require_api_key_scope`
- `rbac_rate_limit`
- org/team dependencies
- setup-local dependencies

Phase 3.4 implication: `skills` is a reasonable first auth cleanup candidate because it has one identity dependency and no role/permission/org/setup mix. The pilot still needs explicit tests because migrating `get_skills_service` from `User` to an `AuthPrincipal` alias must preserve:

- single-user API-key behavior
- multi-user JWT behavior
- TEST_MODE dependency overrides in `test_skills_api.py`
- `request.state.auth` and `_auth_user` reuse behavior from `get_request_user`
- the execute route's `RequestContext.user_id`

## Pagination Surface

Only `GET /api/v1/skills/` is paginated. Current response shape:

```json
{
  "skills": [],
  "count": 0,
  "total": 0,
  "limit": 10,
  "offset": 0
}
```

Recommended Phase 3.2 opt-in metadata:

```json
{
  "pagination": {
    "mode": "offset",
    "limit": 10,
    "offset": 0,
    "total": 0,
    "has_more": false,
    "next_offset": null
  }
}
```

During the compatibility window, keep `count`, `total`, `limit`, and `offset` in the legacy response body. If Phase 3.1 envelope opt-in is active, put canonical pagination in `meta.pagination` while wrapping the existing payload under `data`.

## Frontend Callers

Client methods live in `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`:

- `listSkills({ limit, offset })`
- `getSkill(name)`
- `createSkill(payload)`
- `updateSkill(name, payload, version)`
- `deleteSkill(name)`
- `importSkill(payload)`
- `importSkillFile(file)`
- `seedSkills({ overwrite })`
- `exportSkill(name)`
- `executeSkill(name, args)`
- `getSkillsContext()`

Typed consumer expectations live in `apps/packages/ui/src/types/skill.ts`. `SkillsListResponse` currently expects top-level `skills`, `count`, `total`, `limit`, and `offset`.

Primary UI consumer:

- `apps/packages/ui/src/components/Option/Skills/Manager.tsx`
  - Fetches with `tldwClient.listSkills({ limit: pageSize, offset })`.
  - Reads `data.skills` for table rows.
  - Reads `data.total` for pagination.
  - Calls import, file import, seed, delete, get, export, and execute through the same client domain.

Existing frontend tests that should be extended for opt-in parsing:

- `apps/packages/ui/src/services/__tests__/tldw-api-client.boundary-slices.test.ts`
- `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`

## Pilot Slice Recommendation

Use three small PRs instead of one broad cross-phase change:

1. Shared helpers only:
   - Add response-envelope schemas/builders and pagination metadata helpers.
   - Add unit tests for helper behavior.
   - No `skills` route behavior change yet.
2. `skills` list/detail opt-in:
   - Envelope opt-in for `GET /skills/`, `GET /skills/{skill_name}`, `POST /skills/`, and `PUT /skills/{skill_name}`.
   - Canonical offset metadata for `GET /skills/`.
   - Keep legacy response as default.
   - Leave delete/export exempt.
3. `skills` auth dependency cleanup:
   - Introduce or use the Phase 3.4 identity alias.
   - Migrate `get_skills_service` only after contract tests prove no behavior drift.
   - Keep `execute_skill` user context behavior intact.

## Verification Targets

Backend:

```bash
source .venv/bin/activate
python3 -m pytest tldw_Server_API/tests/Skills/integration/test_skills_api.py -v
```

Frontend:

```bash
cd apps/packages/ui
bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts src/components/Option/Skills/__tests__/Manager.test.tsx
```

Security if Python auth or endpoint code changes:

```bash
source .venv/bin/activate
python3 -m bandit -r tldw_Server_API/app/api/v1/endpoints/skills.py tldw_Server_API/app/api/v1/API_Deps
```

## Open Questions Before Implementation

- Which opt-in mechanism should Phase 3.1 use: header, query flag, or versioned route?
- Should `seed_builtin_skills` get a named Pydantic response before any envelope work?
- Should `import/file` be included in the first response-envelope pilot or held until normal JSON routes are proven?
- Should auth cleanup happen after envelope/pagination pilot, or should it be a separate Phase 3.4 PR with no response-shape changes?
