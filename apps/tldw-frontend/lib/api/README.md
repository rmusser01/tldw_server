# Backend API contract sync

This directory links the frontend to the backend's OpenAPI contract so a
backend model change can't silently drift from the frontend's view of the API
(the RF1 / #2590 defect class from `Docs/Audit/2026-07-04-test-suite-audit-round2.md`).

## Files

- **Contract artifacts live in `apps/packages/api-types` (`@tldw/api-types`,
  ADR-069)**: the committed `openapi.fingerprint.json` drift fingerprint and
  `paths.d.ts` path union, plus the gitignored regenerated `openapi.json` /
  `schema.d.ts`. The CI drift gate (`backend-required.yml` → "OpenAPI contract
  drift gate") checks the fingerprint and exact committed path union against
  the canonical schema, forcing both artifacts to be refreshed and reviewed.
- **`generated/`** — legacy output location; no longer produced. The generation
  script now writes into the package (`bun run generate:api-types` here still
  delegates to it).

## When the drift gate fails

The backend API contract changed. Regenerate and review:

```bash
# from apps/tldw-frontend (venv with server deps importable)
bun run generate:api-types
# review the fingerprint + paths.d.ts changes in apps/packages/api-types,
# update any affected frontend types/mocks, then commit them together
```

`make openapi-fingerprint` refreshes only the fingerprint; a changed path set
still requires generation. `make openapi-drift-check` checks both artifacts.

## Using the generated types

After `bun run generate:api-types`, import from the generated schema:

```ts
import type { paths, components } from "@tldw/api-types/schema";
type RoleResponse = components["schemas"]["RoleResponse"];
```

Decision (see the implementation plan): OpenAPI-generated TypeScript, not
mirrored zod schemas — the backend already emits OpenAPI 3 from FastAPI, so
codegen is zero-runtime-cost and single-source-of-truth.
