# API Response Shape Inventory

- Date: 2026-04-25
- Scope: `tldw_Server_API/app/api/v1/endpoints` and representative schemas under `tldw_Server_API/app/api/v1/schemas`
- Related plan: `Docs/superpowers/plans/2026-04-25-phase3-1-standard-response-envelope-implementation-plan.md`

## Summary

This inventory is the starting point for Phase 3.1. It counts route decorators and classifies their declared response shape so the standard response envelope work can be planned by route family instead of as a repo-wide rewrite.

Counts from a route-decorator scan:

- `2188` route decorators under `app/api/v1/endpoints`.
- `1200` use item or operation `response_model` classes.
- `175` use named `*ListResponse` models.
- `135` use bare `list[...]`, `List[...]`, or `dict` response models.
- `128` use status/message/envelope-like response models.
- `550` have no declared `response_model`.

The earlier Phase 3.1 plan count of `1634` counted only decorators that declared `response_model=...`; this review counts all route decorators, including file, stream, `204`, webhook, and raw-response routes.

## Shape Families

### Item Or Operation Response Models

Most API routes already use named Pydantic models for item or operation responses. These are the easiest to support behind an opt-in standard envelope because the current payload can be wrapped as `data` without changing the model itself.

High-count examples:

- `mcp_hub_management`: `52` item/operation response routes.
- `paper_search`: `57` item/operation response routes, but many are provider-compatible and should be deferred.
- `notes`: `40` item/operation response routes.
- `watchlists`: `41` item/operation response routes.
- `persona`: `31` item/operation response routes.
- `slides`: `21` item/operation response routes.

### Named List Responses

Named list response models are viable Phase 3.1 candidates when paired with Phase 3.2 pagination decisions. Most already carry either `items` or top-level list metadata.

Representative examples:

- `slides`: `5` named list response routes.
- `reading`: `5` named list response routes.
- `admin/admin_data_ops`: `3` named list response routes.
- `quizzes`: `4` named list response routes.
- `storage`: `4` named list response routes.
- `kanban/*`: several route families use nested pagination models.

### Bare List Or Dict Responses

Bare collection response models are higher risk because wrapping changes the top-level type that frontend callers may assume.

Representative examples:

- `acp_permissions`: `list[PermissionDecisionOut]`
- `admin/admin_api_keys`: `list[APIKeyMetadata]`
- `admin/admin_orgs`: `list[TeamResponse]`
- `characters_endpoint`: multiple `list[CharacterResponse]` routes
- `workspaces`: list-style workspace source/artifact/note routes

These should not be first pilots unless all callers are easy to update.

### No Declared Response Model

`550` route decorators do not declare `response_model`. This group includes intentional raw responses, downloads, streams, `204` routes, and routes that should eventually gain explicit response models.

Examples:

- `admin/admin_bundle_ops` download route.
- `admin/admin_events_stream` event stream route.
- `workflows` streaming and execution routes.
- `vector_stores_openai` compatibility surface.
- media processing routes that manually return `JSONResponse`.

These need route-by-route classification before any envelope migration.

## Exemption Candidates

The standard envelope should not be applied blindly to:

- `StreamingResponse` and SSE/websocket-like routes.
- `FileResponse` and download routes.
- `204 No Content` routes.
- OpenAI-compatible endpoints and schemas.
- Anthropic-compatible message endpoints.
- Provider-compatible paper/search routes where external shape compatibility matters.
- Webhook endpoints with externally expected response bodies.

## Candidate Pilot Families

Recommended first pilots:

1. `skills`
   - Small surface.
   - Uses conventional list/detail operation shapes.
   - Lower coupling than media/chat/admin.
2. `data_tables`
   - Medium surface, but already has named schemas and explicit jobs.
   - Useful because it exercises both operation and list/detail responses.
3. `slides`
   - Good list/detail coverage and typed schemas.
   - Larger than `skills`, but still narrower than media/chat/admin.

Avoid first:

- `media`, because it has multiple legacy page schemas and process routes.
- `chat`, because it mixes direct responses, streaming, token scopes, and compatibility logic.
- `admin`, because role/permission layering and audit side effects raise the blast radius.
- `paper_search`, because provider response shape compatibility is part of its contract.

## Migration Guidance

For each pilot route family:

- Keep legacy response bodies as default.
- Add the standard envelope only through the approved opt-in mechanism.
- Wrap the current response payload under `data`.
- Put request metadata under `meta`.
- Use the Phase 3.2 pagination object when wrapping list responses.
- Preserve current status codes and headers.
- Add backend tests for both legacy and envelope modes.
- Add client tests before enabling envelope parsing in frontend code.

## Next Action

Before implementation, choose one pilot family and confirm the opt-in mechanism from the Phase 3.1 plan. `skills` is the lowest-risk first candidate.
