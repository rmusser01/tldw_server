# ADR-065: One shared frontend API contract package (`@tldw/api-types`)

**Status:** Proposed
**Date:** 2026-10-09
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (direction given via TASK-13526 program start 2026-10-09)
**Related task:** TASK-13526
**Related spec/plan:** `Docs/Plans/2026-10-09-api-contract-package-13526-implementation-plan.md`

## Decision

The OpenAPI contract artifacts for the two frontend shells live in one neutral
workspace package, `apps/packages/api-types` (`@tldw/api-types`): the
generation script, the committed drift fingerprint (`openapi.fingerprint.json`),
a committed path-literal union (`paths.d.ts`), and the gitignored rich
artifacts (`schema.d.ts`, `openapi.json`). The shared client's
`ClientPath` union is derived from the generated paths instead of being
hand-maintained, and subsequent TASK-13526 stages unify request execution and
error models on top of this package rather than adding a second contract
source.

## Context

The 2026-10-06 architecture review (TASK-13511 companion findings, tracked in
TASK-13526) found two parallel API clients over one backend: the web shell's
`lib/api.ts` plus a generated `schema.d.ts` that lives inside the web shell
(gitignored, regenerated via `scripts/generate-api-types.mjs`), and the shared
8.7k-line `TldwApiClient` used by the extension and all shared components,
typed against ~8k lines of hand-copied Pydantic shapes plus a manually
maintained `ClientPath` union guarded only by the opt-in
`verify:openapi` script. Backend renames surface as runtime 404s in the
extension, not type errors, and every timeout/retry/cancellation fix lands
twice.

A neutral package is required because `packages/ui` cannot import from a shell
(the web shell) and the generated artifacts must be consumable by both shells
and any future `@tldw/api-client` extraction.

One constraint shapes the artifact split: regenerating the rich `schema.d.ts`
requires a Python environment with the server importable, which frontend-only
CI jobs cannot assume. `packages/ui` therefore consumes only the committed
`paths.d.ts` (a few thousand string literals, always present on a fresh
checkout), while the web shell consumes the full generated schema through the
same package.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Generate into `packages/ui/src/types/generated/` | No new workspace, but couples the shared UI package to backend schema generation and gives the future `@tldw/api-client` extraction no neutral leaf to sit beside. |
| Keep artifacts in the web shell; have `packages/ui` re-export them | Inverts the dependency direction: a shared package importing from a shell is the boundary violation the review flagged. |
| Commit the full ~6MB `schema.d.ts` | Repo bloat and merge-conflict churn on every backend contract change; the fingerprint drift gate already provides the CI contract. |
| Keep the hand-maintained `ClientPath` union and only add CI verification | Preserves the exact drift class (typos, missed additions) that generated types eliminate; verification stays opt-in. |

## Consequences

- `ClientPath` widens from "paths the client calls" to "paths the backend
  exposes". The lost subset documentation is accepted; the type's guarantee
  becomes "this path exists in the current contract", which is the property
  request construction actually needs. The client-calls subset remains visible
  in call sites and the `verify:openapi` MEDIA_ADD schema check.
- Backend contract changes that alter the path set now require regenerating
  `paths.d.ts` (`bun run generate` in the package) in the same PR; the
  existing fingerprint gate in `backend-required.yml` (repointed to the
  package) plus a new committed-artifact staleness check enforce this.
- Frontend-only CI jobs never need Python: they typecheck against the
  committed `paths.d.ts`.
- Stage 2+ of TASK-13526 (request-core unification, one error model,
  `TldwApiClient` domain split) build on this package; no second contract
  source may be introduced (future codegen targets this package).

## Follow-up

- Stage 2: web shell `lib/api/*` migrates onto the shared request core; one
  `ApiError` type.
- Stage 3: split `TldwApiClient` along its existing domain modules, removing
  the `store/connection.tsx` dynamic-import workaround.
- Update `apps/DEVELOPMENT.md` import-alias documentation for
  `@tldw/api-types`.
