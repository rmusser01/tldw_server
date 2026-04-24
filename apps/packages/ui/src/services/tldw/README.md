# Shared tldw Client Boundary

## Ownership Rules

- `TldwApiClient.ts` owns the shared facade, request/bootstrap primitives, cache state, auth/config helpers, and path-resolution utilities.
- `domains/*.ts` own feature-facing API methods. New endpoint methods should start in the appropriate domain module, not in the base class body.
- Focused helper modules are justified only for shared runtime normalizers or cloning helpers that would otherwise force a domain to import runtime code from `TldwApiClient.ts`.
- Transitional re-exports from `TldwApiClient.ts` exist for compatibility. Prefer importing shared helpers from their focused modules in new code.

## Deferred Follow-On Overlaps

- `models-audio` (23 overlaps)
- `characters` (23 overlaps)
- `collections` (45 overlaps)
- `media` (39 overlaps)
- `chat-rag` (101 overlaps)

## Maintenance Workflow

- When adding or changing API paths used by the UI package, update the client path guards and then run `bun run verify:openapi` from `apps/packages/ui`.
- `verify:openapi` prefers `apps/extension/openapi.json` when maintainers keep a local snapshot. If that file is absent, it derives the spec from the checked-out backend via `tldw_Server_API.app.main` using the project Python environment and a local-only synthetic API key.
- `verify:openapi` treats path-parameter names as equivalent during comparison, so `{key}` and `{archetype_key}` do not cause false drift.
- `verify:openapi` warns, but does not fail, for the current reviewed non-OSS/optional surfaces:
  - `/api/v1/billing/*` public billing routes, which the OSS backend explicitly does not expose
  - `/api/v1/media/bulk/keyword-update`, which already has a per-item fallback path
  - `/api/v1/media/statistics`, which remains a deferred cleanup item in the media slice
- When a domain starts sharing runtime-only transformation logic with another domain, extract a focused helper module rather than reintroducing class-body duplication.
