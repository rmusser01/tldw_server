# Backend API contract sync

This directory links the frontend to the backend's OpenAPI contract so a
backend model change can't silently drift from the frontend's view of the API
(the RF1 / #2590 defect class from `audits/2026-07-04-test-suite-audit-round2.md`).

## Files

- **`openapi.fingerprint.json`** — committed. A small sha256 + counts of the
  canonical backend schema. The CI drift gate (`backend-required.yml` →
  "OpenAPI contract drift gate") recomputes it and fails if it differs, forcing
  a backend contract change to be acknowledged.
- **`generated/`** — gitignored build artifacts: the full `openapi.json` (~5MB)
  and `schema.d.ts` (~6MB, from `openapi-typescript`). Not committed because
  they regenerate deterministically and would bloat the repo with unreviewable
  diffs.

## When the drift gate fails

The backend API contract changed. Regenerate and review:

```bash
# from apps/tldw-frontend (venv with server deps importable)
bun run generate:api-types
# review the fingerprint change, update any affected frontend types/mocks,
# then commit the updated openapi.fingerprint.json
```

Or just refresh the fingerprint from the repo root: `make openapi-fingerprint`.

## Using the generated types

After `bun run generate:api-types`, import from the generated schema:

```ts
import type { paths, components } from "@/lib/api/generated/schema";
type RoleResponse = components["schemas"]["RoleResponse"];
```

Decision (see the implementation plan): OpenAPI-generated TypeScript, not
mirrored zod schemas — the backend already emits OpenAPI 3 from FastAPI, so
codegen is zero-runtime-cost and single-source-of-truth.
