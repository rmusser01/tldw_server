# @tldw/api-types

Single source of truth for the frontend API contract (ADR-069). Consumed by
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

Generate with **Python 3.12**, matching the existing `backend-required` CI
contract. This is an artifact-generation requirement, not a server runtime
restriction. Different Python versions can change union ordering and default
HTTP response descriptions in the exported schema without an API change.

From the repository root, the Make targets default to `OPENAPI_PYTHON=python3.12`
and reject missing or non-3.12 interpreters before exporting. Override it with
a Python 3.12 environment containing the server dependencies; these targets do
not use `CI_LOCAL_PYTHON`:

```bash
make openapi-fingerprint OPENAPI_PYTHON=/path/to/py312-venv/bin/python
make openapi-drift-check OPENAPI_PYTHON=/path/to/py312-venv/bin/python
```

For full frontend type generation, pass that same Python 3.12 interpreter via
the generator's existing `PYTHON` setting:

```bash
# in apps/packages/api-types
PYTHON=/path/to/py312-venv/bin/python bun run generate
# or in apps/tldw-frontend
PYTHON=/path/to/py312-venv/bin/python bun run generate:api-types
```

Run whenever the backend API contract changes. The existing backend drift gate
checks both the fingerprint and the exact committed path union against the
canonical schema. A fingerprint-only refresh cannot certify stale `paths.d.ts`.
Package-only changes select both backend and frontend checks.

## Consumers

- `apps/packages/ui` → `import type { ApiPath } from "@tldw/api-types"`
  (`ClientPath` in `services/tldw/openapi-guard.ts` is derived from it).
- Rich-type consumers import from `@tldw/api-types/schema` after generation.
  The legacy web `lib/api/generated/` output is retired.
