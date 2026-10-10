# @tldw/api-types

Single source of truth for the frontend API contract (ADR-065). Consumed by
both frontend shells (`apps/tldw-frontend`, `apps/extension` via
`apps/packages/ui`) — nothing else may introduce a second contract source.

## Artifacts

| File | Committed | What it is |
| --- | --- | --- |
| `paths.d.ts` | yes | Sorted path-literal union (`ApiPath`) over the current OpenAPI spec. Python-free: frontend-only checkouts and CI typecheck against this. |
| `openapi.fingerprint.json` | yes | Drift fingerprint checked by the `backend-required` CI gate and the `make openapi-drift-check` target. |
| `schema.d.ts` | no (gitignored) | Full `openapi-typescript` output (rich request/response types). Import as `@tldw/api-types/schema` after generating. |
| `openapi.json` | no (gitignored) | Canonical exported spec from the backend. |

## Regenerating

```bash
bun run generate          # in apps/packages/api-types (needs server-importable Python)
# or from the web shell:  bun run generate:api-types   (in apps/tldw-frontend)
```

Run whenever the backend API contract changes; the fingerprint gate in
`backend-required.yml` fails until the fingerprint (and `paths.d.ts`) are
refreshed in the same PR.

## Consumers

- `apps/packages/ui` → `import type { ApiPath } from "@tldw/api-types"`
  (`ClientPath` in `services/tldw/openapi-guard.ts` is derived from it).
- `apps/tldw-frontend/lib/api/generated/schema.d.ts` re-exports
  `@tldw/api-types/schema`.
