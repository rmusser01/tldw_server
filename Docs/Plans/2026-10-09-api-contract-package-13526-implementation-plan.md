# Implementation Plan: Shared API contract package (TASK-13526 Stage 1)

Branch: `codex/api-client-unification-13526` (off `origin/dev` @ 4d6a85c4c6)
Worktree: `.worktrees/api-client-unification`
ADR check: **required — ADR-065** (`Docs/ADR/065-frontend-api-contract-package.md`), because this establishes a durable rule: one neutral workspace package owns the frontend API contract artifacts, and no second contract source may be introduced.

## Stage 1a: Create `apps/packages/api-types`
**Goal**: Neutral `@tldw/api-types` package owning generation, fingerprint, and path union.
**Changes**:
1. `package.json` (name `@tldw/api-types`, private, types/exports → `./index.d.ts`), `tsconfig.json`, `.gitignore` (`schema.d.ts`, `openapi.json`), README.
2. `scripts/generate-api-types.mjs` — moved from `apps/tldw-frontend/scripts/`, output redirected into the package: `openapi.json` + `schema.d.ts` (gitignored) + `paths.d.ts` (committed path-literal union derived from the JSON keys) + `openapi.fingerprint.json` (committed, moved from the web shell).
3. `index.d.ts` — re-exports the generated `schema` types (rich, web-shell consumption) and the `ApiPath` union from `paths.d.ts`.
4. `paths.d.ts` generation: read `openapi.json` keys, emit `export type ApiPath = "/api/..." | ...` sorted.
**Success Criteria**: `bun run generate` in the package produces schema.d.ts + paths.d.ts + fingerprint; `bun install` links the workspace; typecheck of a scratch import works.
**Tests**: unit test for the paths-union emitter (fixture openapi.json → expected union text, sorted, deduped).
**Status**: Not Started

## Stage 1b: Migrate the web shell
**Goal**: Web shell consumes the package; old artifacts delegate.
**Changes**:
1. `apps/tldw-frontend/lib/api/generated/schema.d.ts` → `export * from "@tldw/api-types"` (all `lib/api/*` imports keep working).
2. `apps/tldw-frontend/package.json` `generate:api-types` delegates to the package script; dependency `@tldw/api-types: workspace:*`.
3. Move `lib/api/openapi.fingerprint.json` → package (git mv); update readers: `.github/workflows/backend-required.yml` `--check` path, `Makefile` `openapi-fingerprint`/`openapi-drift-check` targets, `lib/api/README.md` note.
4. tsconfig paths: ensure `@tldw/api-types` resolves (workspace symlink should suffice; add alias only if the transpilePackages/paths setup requires it).
**Success Criteria**: web vitest suites that touch lib/api pass; `bun run generate:api-types` from the web shell works end to end; fingerprint check command passes against the moved file.
**Tests**: existing lib/api tests; run the backend-required fingerprint command locally (`python Helper_Scripts/export_openapi_schema.py --check <new path>`).
**Status**: Not Started

## Stage 1c: Derive ClientPath in packages/ui
**Goal**: Kill the hand-maintained union.
**Changes**:
1. `services/tldw/openapi-guard.ts`: `import type { ApiPath } from "@tldw/api-types"`; `export type ClientPath = ApiPath`; delete the ~360-entry manual union; keep `ReplacePathParams`/runtime helpers; update the header comment (manual-maintenance language → regeneration language).
2. `packages/ui/package.json`: `@tldw/api-types: workspace:*`; verify vitest + tsc resolve it (bun workspace symlink; alias fallback `@tldw/api-types` → package if needed).
3. `extension/scripts/verify-openapi-client-paths.mjs`: check #1 (ClientPath ⊆ spec) is now structural — keep the script running (it must still parse the derived form or be reduced to the MEDIA_ADD check); update its parsing if it regexes the union list.
**Success Criteria**: packages/ui `tsc --noEmit`-equivalent green (verify via its typecheck path), relevant service tests green, extension `bun run verify:openapi` green.
**Tests**: new type-level guard test (known good path accepted; `"/api/v1/definitely/not/real"` rejected) following the `watchlists-static-guard.typecheck.test.ts` pattern; existing openapi-guard/request-core tests.
**Status**: Not Started

## Stage 1d: Staleness gate + docs
**Changes**: `scripts/check-paths-artifact.mjs` in the package (regenerate paths.d.ts content in-memory from the committed fingerprint's source-of-truth flow is not possible without Python — instead the gate = `export_openapi_schema.py --check` (existing, repointed) + a lint that `paths.d.ts` is sorted/deduped/non-empty); update `apps/DEVELOPMENT.md` alias table + `lib/api/README.md`.
**Success Criteria**: CI contract tests over workflows pass (run `tldw_Server_API/tests/CI` locally for the workflow-snapshot tests).
**Status**: Not Started

## Stage 2 (separate PR): request-core unification + one ApiError
## Stage 3 (separate PR): TldwApiClient domain split (removes connection.tsx workaround)
Deferred items 3-13 of TASK-13526 remain tracked on the umbrella task.

## Explicitly deferred from this PR
Dependency-major alignment, vitest consolidation, package split of ui, auth/primitives dedupe, store splits, fonts/locales, ChatPane virtualization (owned by the sibling W1 program), composer leaf, parser predicate split, providers force bypass, exports-map bun fix.
