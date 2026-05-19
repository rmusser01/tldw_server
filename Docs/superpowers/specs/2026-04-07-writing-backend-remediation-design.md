# Writing Backend Remediation Design

Date: 2026-04-07
Topic: Remediation of confirmed backend Writing review findings in `tldw_server`

## Goal

Implement a single backend Writing remediation batch that:

- fixes all confirmed review findings in the Writing and manuscript backend surface
- adds the agreed regression and hardening tests around those fixes
- preserves existing architecture patterns by enforcing invariants at the helper or DB boundary where they belong
- avoids unrelated refactors and frontend scope expansion

## Scope

This remediation covers the backend Writing surface centered on:

- `tldw_Server_API/app/api/v1/endpoints/writing.py`
- `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- `tldw_Server_API/app/api/v1/schemas/writing_manuscript_schemas.py`
- `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- `tldw_Server_API/app/core/Writing/manuscript_analysis.py`
- related tests under `tldw_Server_API/tests/Writing/`

The work includes correctness fixes for manuscript ownership and visibility rules, delete cascades, analysis invalidation, PATCH null-clearing behavior, project-analysis parity, snapshot import atomicity, and structured analysis response parsing. It also includes the agreed regression coverage for snapshot replace rollback, wordcloud error contracts, explicit null clearing, stale-version conflicts, and analysis route validation.

## Non-Goals

This remediation does not include:

- frontend Writing Playground changes under `apps/`
- unrelated Writing feature additions
- broad endpoint or helper refactors outside the reviewed issue set
- schema redesign or new storage models unless a narrow helper adjustment requires it
- changing list-route semantics from empty results to `404` where current routes already return empty lists

## Findings Addressed

This design addresses the previously confirmed findings:

- missing same-project guards for chapter parenting and world-info reparenting
- non-atomic snapshot import in `mode="replace"`
- deleted-project child visibility leaks
- incomplete scene cascade during part soft delete
- inconsistent runtime limit and provider or model override validation on project-level analysis routes
- missing aggregate analysis invalidation when scene inputs change
- broken explicit `null` clearing for nullable manuscript PATCH fields

It also implements the agreed improvements:

- rollback regression coverage for snapshot replace failures
- negative tests for explicit `null`, blank names, empty PATCH payloads, and missing `expected-version`
- regression tests for cross-project reparenting, deleted-project child visibility, part-delete scene cascade, stale-version reorder conflicts, and project-analysis validation parity
- wordcloud `404` and failed-job response coverage
- structured non-string `message.content` parsing coverage

## Approaches Considered

### 1. Boundary-first remediation

Fix invariants in `ManuscriptDBHelper` and related persistence helpers, then update endpoints only where HTTP contract behavior must preserve caller intent.

Strengths:

- strongest defense against future callers bypassing endpoint validation
- matches the existing helper-centric manuscript design
- keeps data-integrity rules near the storage boundary

Weaknesses:

- requires careful coordination between helper changes and endpoint contract handling

### 2. Endpoint-first patching

Patch only the HTTP handlers and keep DB helpers mostly unchanged.

Strengths:

- smaller immediate code diff in some paths
- quicker for purely request-shaping bugs

Weaknesses:

- leaves persistence invariants fragmented
- increases risk of helper misuse from future routes or internal callers

### 3. Small refactor plus remediation

Add new shared utilities for patch parsing, analysis execution, snapshot import, and ownership validation before fixing behavior.

Strengths:

- potentially cleaner final structure

Weaknesses:

- expands scope and regression risk beyond the confirmed review findings
- slows delivery of concrete bug fixes

## Recommended Approach

Use the boundary-first remediation.

Execution shape:

1. move ownership, visibility, cascade, and stale-analysis invariants into `ManuscriptDBHelper`
2. update endpoint handlers to preserve explicit field presence, reuse shared analysis validation, and reject trimmed-empty names consistently
3. make snapshot replace import atomic with one outer transaction in `writing.py`
4. harden analysis response parsing and lock the resulting behavior with targeted tests

This fixes the confirmed bugs without turning the remediation into a general cleanup project.

## Design

### 1. Responsibility Boundaries

The remediation will keep clear ownership of behavior:

- `ManuscriptDBHelper` is the source of truth for manuscript ownership rules, project visibility boundaries, delete cascades, and analysis invalidation triggered by manuscript mutations
- `writing_manuscripts.py` is responsible for HTTP-layer contract behavior, including field-presence-aware PATCH parsing, trimmed-empty request rejection, and analysis-route dependency parity
- `writing.py` keeps snapshot import orchestration, but `mode="replace"` will run inside one outer `db.transaction()` so nested helper calls share the same commit or rollback boundary
- `manuscript_analysis.py` owns response-content extraction and will accept common structured `message.content` shapes before JSON parsing

### 2. Ownership and Visibility Invariants

The manuscript helper layer will enforce same-project and active-parent rules for the reviewed problem paths:

- `create_chapter` will validate `part_id` against the target `project_id`
- `update_chapter` will fetch the current chapter project and validate any new `part_id` against that same project
- `reorder_items(... entity_type="chapter" ...)` will validate `part_id` reparent targets against the provided `project_id`
- `update_world_info` will validate any `parent_id` against the world-info row's project, matching the existing create-time behavior

Project-keyed manuscript list helpers will require the project row to remain active before returning descendants. This applies to the reviewed helper paths that currently list project children directly by `project_id`, including parts, chapters, characters, relationships, world-info, plot lines, and plot holes. This closes the confirmed deleted-project enumeration class while preserving current list-route behavior: callers still get list responses, but deleted projects no longer expose descendants through those project-scoped list endpoints. Broader descendant-by-ID fetch behavior is not expanded in this batch unless it is already covered by an existing route-level project existence check, such as the current structure endpoint.

### 3. Delete Cascades and Analysis Invalidation

`soft_delete_part` will capture the affected chapter IDs before marking those chapters deleted, then delete scenes using that captured set. This fixes the current ordering bug where the scene cascade misses just-deleted chapters.

Scene mutations that change aggregate text membership will mark all affected analyses stale:

- scene create: mark chapter and project analyses stale
- scene content update: mark scene analyses stale as today, plus chapter and project analyses when content or content structure changes
- scene delete: mark chapter and project analyses stale

The design does not attempt to add wider invalidation for unrelated manuscript entities in this batch; it only covers the confirmed stale-analysis gaps tied to scene inputs.

### 4. HTTP Contract Remediation

`writing_manuscripts.py` will switch reviewed PATCH handlers from value-based parsing to field-presence-aware parsing so explicit JSON `null` can clear nullable fields.

The target behavior is:

- omitted field: leave unchanged
- present with `null` on genuinely nullable scalar or link fields: clear the field
- present with `null` on required or non-nullable identifier fields such as titles or names: reject with `400`
- present with `null` on collection-backed fields that currently round-trip as objects or arrays, such as `settings`, `custom_fields`, `properties`, or `tags`: normalize to the empty container shape rather than storing JSON `null`
- present with a value: update normally
- present but trimmed to empty for required text identifiers like titles or names: reject with `400`
- empty effective PATCH payload after field processing: reject with `400`

This applies to the reviewed manuscript endpoints where the current `is not None` or `exclude_none=True` logic loses explicit `null` intent. The design preserves the current optimistic-locking requirement through the existing `expected-version` header and adds regression coverage for missing-header failures instead of relaxing that contract.

### 5. Project Analysis Route Parity

The project-level plot-hole and consistency endpoints will use the same runtime rate-limit enforcement and provider or model override validation path already used by scene and chapter analysis routes.

The resulting project-analysis behavior will match scene and chapter analysis for:

- `429` runtime rate-limit denial
- unknown provider override rejection
- unknown model override rejection

This is a behavior alignment change, not a new feature.

### 6. Snapshot Replace Atomicity

`import_writing_snapshot(... mode="replace" ...)` will execute under one outer database transaction.

Within that transaction the endpoint will:

1. soft-delete existing sessions, templates, and themes
2. restore or insert incoming items
3. update existing live items where merge semantics require it
4. commit only after the full replace succeeds

If any helper call or validation step fails mid-import, the outer transaction will roll the entire replace back so the pre-import dataset remains intact.

To make that true across both SQLite and backend-managed Postgres transactions, any snapshot-import helper path that currently issues unconditional inner commits must be rewritten to participate in the shared transaction boundary instead of calling `execute_query(..., commit=True)` from inside the replace flow. In the current Writing code this especially applies to the soft-deleted session restore path.

The design keeps the current endpoint-level import contract and does not introduce a new DB facade just for snapshot import.

### 7. Structured Analysis Response Parsing

`_extract_content()` in `manuscript_analysis.py` will accept the common structured response patterns that can appear in `choices[0].message.content`, especially list-based content blocks with embedded text fragments.

The goal is limited:

- extract a single text string suitable for the existing markdown-fence stripping and JSON parsing flow
- preserve current handling for raw strings and simple OpenAI-style dict responses
- return a best-effort string instead of a Python repr for common structured content shapes

This is a parser hardening change, not a new provider integration layer.

### 8. Wordcloud Contract Coverage

The reviewed wordcloud route behavior is largely acceptable, but the agreed improvements require explicit regression coverage for:

- `404` when a wordcloud ID is unknown
- stable failed-job retrieval behavior once a job row is marked failed

The design does not require a large wordcloud implementation refactor unless the tests expose a concrete gap while fixing the covered behaviors.

## File Plan

Primary implementation files to modify:

- `tldw_Server_API/app/core/DB_Management/ManuscriptDB.py`
- `tldw_Server_API/app/api/v1/endpoints/writing_manuscripts.py`
- `tldw_Server_API/app/api/v1/endpoints/writing.py`
- `tldw_Server_API/app/core/Writing/manuscript_analysis.py`

Primary test files to modify:

- `tldw_Server_API/tests/Writing/test_writing_endpoint_integration.py`
- `tldw_Server_API/tests/Writing/test_manuscript_endpoint_integration.py`
- `tldw_Server_API/tests/Writing/test_manuscript_phase2_integration.py`
- `tldw_Server_API/tests/Writing/test_manuscript_analysis_integration.py`
- `tldw_Server_API/tests/Writing/test_manuscript_analysis_service.py`
- `tldw_Server_API/tests/Writing/test_manuscript_db.py`
- `tldw_Server_API/tests/Writing/test_manuscript_world_plot_db.py`
- `tldw_Server_API/tests/Writing/test_manuscript_schema_contract.py` only if field-presence/null contract coverage belongs there after implementation

`ChaChaNotes_DB.py` is in scope for transaction participation and wordcloud helpers, but this design expects snapshot atomicity to be achieved from `writing.py` using the existing transaction support rather than by broad changes to generic DB primitives.

## Testing Strategy

The remediation will add or update targeted tests for:

- snapshot replace rollback on mid-import failure
- wordcloud `404` and failed-job retrieval behavior
- cross-project chapter create, update, and reorder reparent rejection
- cross-project world-info reparent rejection
- deleted-project child visibility suppression
- deleted-project suppression across the other project-scoped manuscript list routes covered by `test_manuscript_phase2_integration.py`
- part soft-delete scene cascade
- project-analysis runtime `429` and override validation parity
- chapter and project analysis staleness after scene create, update, and delete
- explicit `null` clearing for nullable manuscript PATCH fields
- explicit `null` normalization for collection-backed PATCH fields that should remain object or array shaped in responses
- blank trimmed-name rejection on relevant create, update, and import paths
- empty PATCH payload rejection
- missing `expected-version` header failures where the route contract requires it
- stale-version reorder conflict handling
- structured list-based analysis content extraction

Verification will remain targeted to the touched Writing backend slices plus Bandit on the touched backend paths.

## Risks and Tradeoffs

- Joining child-list queries against active projects changes behavior for deleted projects from "live descendants leak through" to "descendants disappear from list routes". This is intentional and aligns with soft-delete expectations.
- Field-presence-aware PATCH parsing slightly increases handler complexity, but it is necessary to make the schema-advertised nullable contract truthful.
- Snapshot replace atomicity relies on the existing nested transaction behavior in `CharactersRAGDB`; the tests must prove the outer transaction correctly rolls back nested helper writes on failure.
- Aggregate stale-analysis invalidation will increase the number of rows marked stale during scene mutations, but that cost is acceptable relative to returning analyses that no longer match current inputs.

## Success Criteria

This design is successful if the remediation:

- closes every confirmed review finding without introducing broader API drift
- adds regression coverage for every agreed improvement area
- keeps ownership and integrity rules at the helper or DB boundary where they can protect future callers
- preserves existing behavior where the review did not identify a defect
- yields a small, testable set of touched files and targeted verification commands for implementation
