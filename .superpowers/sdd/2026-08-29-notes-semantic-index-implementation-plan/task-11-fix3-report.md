# TASK-13134 Task 11 Fix Round 3 Report

## Status

Implementation and verification are complete for this bounded review-fix round. Task 11 remains pending a clean re-review. TASK-13134 remains In Progress, with Tasks 12-13 outstanding.

Commit: `fix: align Notes semantic capability contracts (TASK-13134)` (this commit).

## Finding Resolutions

### 1. Active generation capability binding

- Added one shared compatibility-binding predicate used by both the public status API and semantic projector.
- A complete generation/config binding is usable only when a non-null current capability compatibility hash matches the persisted hash.
- Current capability resolution without a compatibility hash preserves the projector's existing fail-safe behavior.
- Drift retains `stale_configuration`, sets `active_generation_usable=false`, suppresses stale generation counts, and prevents the semantic-edge toggle.
- Added a backend-produced capability/status fixture consumed by the client parser and inspector integration test.

### 2. Canonical endpoint origin

- Added `canonical_semantic_endpoint_origin` as the sole origin normalizer for capability disclosure, public schema validation, persisted config validation, pending worker configuration, credential authority, and publication fences.
- Canonical output is limited to `scheme://host[:port]`, preserves explicit ports, brackets/compresses IPv6, lowercases hosts, converts Unicode domains to IDNA/punycode, and strips credentials/path/query/fragment from disclosure input.
- Added IPv6 and IDN flows through capability resolution, Pydantic response validation, client validation, persisted configuration, and production pending worker request construction.
- Canonical persisted/public/worker boundaries reject noncanonical or invalid values.

### 3. Nullable unavailable endpoint disclosure

- Public Pydantic and TypeScript capability contracts now allow `endpoint_display=null` only when indexing is unavailable with a stable non-null reason.
- Available capabilities still require a bounded, nonblank, canonical endpoint.
- Consent completeness requires the endpoint; unavailable disclosure omits the endpoint row safely.
- Existing enabled indexes with endpoint-unavailable capability reach typed status and expose Delete only, with no Enable, Rebuild, or Renew action.

### 4. Backlog status

- Corrected TASK-13134 notes and final summary to supersede prior Task 11 completion/review claims.
- TASK-13134 remains In Progress; this round is pending clean re-review; Tasks 12-13 remain outstanding.

### Adjacent fixture repair

- Updated four worker capability test doubles with the already-required `vector_backend` field. Production backend-binding checks were not weakened.

## TDD Evidence

### RED

Backend:

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=line --show-capture=no tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_jobs.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_endpoints.py -k "capability_origin_passes or missing_endpoint_only or suppresses_generation_when_current or matches_projector_when_current or api_produces_capability_drift or endpoint_unavailable_existing or capability_origin_reaches or unavailable_endpoint_capability"
```

Result: 10 failed, 1 passed, 127 deselected. Failures covered capability-hash drift exposing stale counts, nullable endpoint schema rejection, IPv6/IDN worker authority, and typed endpoint-unavailable API output. The one pass was the projector fail-safe control for a missing current compatibility hash.

Frontend:

```text
bunx vitest run ../packages/ui/src/services/tldw/__tests__/note-semantic-index.test.ts ../packages/ui/src/components/Notes/__tests__/NotesGraphInspector.semantic.integration.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphInspector.semantic.test.tsx
```

Result: 2 failed, 71 passed. The client rejected a typed unavailable null endpoint, and the inspector called `.trim()` on the null endpoint.

### GREEN

Focused backend selector: 11 passed, 127 deselected.

Focused frontend service/hook/inspector set: 83 passed across 4 files.

Final backend:

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=line --show-capture=no tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_embeddings.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_store.py tldw_Server_API/tests/Notes_Graph/unit/test_semantic_projector.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_publication.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_jobs.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_endpoints.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_route_order.py tldw_Server_API/tests/Services/test_notes_semantic_workers.py
```

Result: 306 passed, 2 warnings in 94.92s. This includes the exact seven-route and route-order contracts.

Final frontend:

```text
bunx vitest run ../packages/ui/src/services/tldw/__tests__/note-graph-suggestions.test.ts ../packages/ui/src/services/tldw/__tests__/note-semantic-index.test.ts ../packages/ui/src/components/Notes/__tests__/NotesGraphCanvas.graph-view.test.tsx ../packages/ui/src/components/Notes/__tests__/useNotesGraphAuthorityScope.test.tsx ../packages/ui/src/components/Notes/__tests__/useNotesGraphWorkspace.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.axe.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphToolbar.i18n.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphInspector.semantic.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphInspector.suggestions.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphRelationshipsView.accessibility.test.tsx ../packages/ui/src/components/Notes/__tests__/useNotesGraphSuggestions.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.responsive.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphInspector.semantic.integration.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.loading-i18n.test.tsx ../packages/ui/src/components/Notes/__tests__/NotesGraphWorkspace.view-mode.test.tsx ../packages/ui/src/i18n/__tests__/notes-semantic-fallback.test.ts
```

Result: 226 passed across 16 files.

## Static And Security Verification

- Python runtime: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python --version` -> Python 3.11.13.
- Extension TypeScript: `bun run compile` -> passed (`tsc --noEmit -p tsconfig.compile.json`).
- Ruff: exact touched backend production/tests -> all checks passed.
- Bandit: exact touched backend production -> zero findings; report `/tmp/bandit_TASK-13134_task11_fix3.json`.
- Locale duplicate check: passed.
- Locale coverage check: passed.
- Canonical Prettier check: five touched frontend/JSON files passed using `apps/extension/.prettierrc.cjs`.
- Integration test formatting: passed with the same formatting options and import plugin disabled after the plugin crashed three times on its JSON import.
- `git diff --check`: passed before report finalization and rerun after all tracked updates.

## Files

Production backend:

- `tldw_Server_API/app/core/Notes_Graph/semantic_endpoint.py`
- `tldw_Server_API/app/core/Notes_Graph/semantic_capabilities.py`
- `tldw_Server_API/app/core/Notes_Graph/semantic_embeddings.py`
- `tldw_Server_API/app/core/Notes_Graph/semantic_publication.py`
- `tldw_Server_API/app/core/Notes_Graph/semantic_api.py`
- `tldw_Server_API/app/core/Notes_Graph/semantic_projector.py`
- `tldw_Server_API/app/core/DB_Management/chacha/note_semantic_store.py`
- `tldw_Server_API/app/api/v1/schemas/notes_semantic_index.py`

Frontend:

- `apps/packages/ui/src/services/note-semantic-index.ts`
- `apps/packages/ui/src/components/Notes/NotesGraphInspector.tsx`

Tests and fixture:

- `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py`
- `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_jobs.py`
- `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_endpoints.py`
- `tldw_Server_API/tests/Services/test_notes_semantic_workers.py`
- `apps/packages/ui/src/services/tldw/__tests__/note-semantic-index.test.ts`
- `apps/packages/ui/src/components/Notes/__tests__/NotesGraphInspector.semantic.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesGraphInspector.semantic.integration.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/fixtures/semantic-capability-drift-api.json`

Tracking:

- `.superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/progress.md`
- `.superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/task-11-fix3-report.md`
- `backlog/tasks/task-13134 - Implement-Notes-embedding-index-and-semantic-graph-edges.md`

## Concerns

- No live PostgreSQL/pgvector or live embedding-provider run was performed; this round changed no schema or backend-specific vector operations.
- Repository-wide frontend `bun run typecheck` remains red on unrelated existing E2E/certification typing failures and pre-existing `NotesSemanticRun.error_code` assignments outside this diff. The extension compile that consumes the shared package passes.
- `bun run check:i18n:glossary` cannot start because the tracked script references absent `apps/extension/scripts/i18n-glossary.json`; duplicate and coverage checks pass.
- `@trivago/prettier-plugin-sort-imports` crashes inside Prettier 3.8.1 on the integration test JSON import after three attempts. The file passes Prettier with the repository's same style options and the plugin disabled; all other touched frontend files pass the canonical config.
