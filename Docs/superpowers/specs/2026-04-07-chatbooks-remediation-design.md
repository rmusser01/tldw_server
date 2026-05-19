# Chatbooks Remediation Design

## Summary

This spec covers implementation work to address the confirmed Chatbooks review
findings on current `dev`, not the older review branch snapshot.

The remediation has four goals:

1. Fix the confirmed runtime defects in Chatbooks job lifecycle, preview
   handling, import cleanup, and helper validation.
2. Move the API to the stricter caller-visible contracts chosen during design,
   then align schemas, OpenAPI, and docs to those live contracts.
3. Add focused regression tests for every confirmed finding.
4. Address the Stage 4 maintainability findings in a bounded way by tightening
   false-confidence tests and splitting the worst Chatbooks test hotspot, while
   avoiding a broad runtime refactor.

## Scope

In scope:

- Runtime and helper fixes in:
  - [`tldw_Server_API/app/core/Chatbooks/chatbook_service.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Chatbooks/chatbook_service.py)
  - [`tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Chatbooks/chatbook_validators.py)
  - [`tldw_Server_API/app/core/Chatbooks/jobs_adapter.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Chatbooks/jobs_adapter.py)
  - [`tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py)
- API/schema contract fixes in:
  - [`tldw_Server_API/app/api/v1/endpoints/chatbooks.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/endpoints/chatbooks.py)
  - [`tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py)
- Static contract/doc alignment in:
  - [`Docs/API-related/chatbook_openapi.yaml`](/Users/appledev/Documents/GitHub/tldw_server/Docs/API-related/chatbook_openapi.yaml)
  - [`Docs/Schemas/chatbooks_manifest_v1.json`](/Users/appledev/Documents/GitHub/tldw_server/Docs/Schemas/chatbooks_manifest_v1.json)
  - [`tldw_Server_API/app/core/Chatbooks/README.md`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Chatbooks/README.md)
  - [`Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md`](/Users/appledev/Documents/GitHub/tldw_server/Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md)
  - [`Docs/Product/Chatbooks_PRD.md`](/Users/appledev/Documents/GitHub/tldw_server/Docs/Product/Chatbooks_PRD.md)
- Regression and maintainability-focused test work in:
  - [`tldw_Server_API/tests/Chatbooks/`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/Chatbooks)
  - [`tldw_Server_API/tests/integration/test_chatbook_integration.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/integration/test_chatbook_integration.py)
  - Chatbooks E2E/server-E2E/frontend smoke tests only where they need direct
    contract updates or targeted hardening

Out of scope:

- Splitting `chatbook_service.py` or `endpoints/chatbooks.py` into smaller
  runtime modules in this pass
- Redesigning Chatbooks product scope beyond the reviewed findings
- Reworking unrelated review modules currently in progress on `dev`
- Changing remove-job semantics to be narrower than current service behavior
- Preserving the `200 + {"error": ...}` preview behavior

## Locked Decisions

- Implementation targets current `dev` state, but should execute in a fresh
  isolated worktree created from current `dev` because the main checkout is
  already dirty with unrelated work.
- `CANCELLED` becomes a protected terminal state during Chatbooks/core Jobs
  reconciliation.
- Preview failures use normal HTTP error signaling:
  - `400` for invalid archive paths, unsafe archives, missing/invalid manifests,
    and other caller-caused preview failures
  - `500` only for unexpected server-side failures
- Async continuation export is not a public feature in this pass.
  The public contract should stop advertising unsupported async continuation.
  If a caller still submits `async_mode=true`, the route should return a caller
  error (`400`), not `500`.
- Sync export responses should standardize on the current persisted-job shape:
  successful sync export returns `job_id` plus `download_url`, not raw
  `file_path`.
- Sync continuation export should match the same sync export response shape.
  `file_path` should be removed from the live sync export contract rather than
  kept as a parallel deprecated success field.
- Sync import should populate `imported_items` instead of silently leaving the
  field unimplemented.
  `imported_items` should count successfully materialized items by content type;
  renamed imports count as imported, skipped items do not, and unsupported or
  unrequested types should be absent rather than emitted as zero-value entries.
  The counts should come from the import operation result itself, not from a
  best-effort post-import database recount.
- Remove-job semantics stay broad for safe terminal states:
  - exports: `completed`, `cancelled`, `failed`, `expired`
  - imports: `completed`, `cancelled`, `failed`
- Helper-level chatbook validation should use the same hardened ZIP-member
  checks as the main archive validator.
- Stage 4 maintainability work is required, but bounded:
  split the worst test hotspot and tighten weak/fake-confidence tests; do not
  attempt a sweeping runtime architecture refactor.

## Approaches Considered

### Recommended: Focused Remediation

Fix the confirmed runtime defects first, tighten the API to the stricter
contracts, align docs/schemas to the resulting runtime behavior, and add
targeted regression tests for each finding. Address maintainability findings
through bounded test-structure cleanup instead of runtime-module decomposition.

Pros:

- Addresses all confirmed findings.
- Keeps risk centered on the reviewed behaviors.
- Improves test quality without turning the pass into a general rewrite.

Cons:

- Still touches many files across runtime, contracts, and tests.
- Leaves the large runtime files in place for now.

### Alternative: Broad Runtime Refactor + Remediation

Do everything in the recommended approach and also decompose the large runtime
Chatbooks modules now.

Pros:

- Cleaner module boundaries immediately.

Cons:

- High merge risk on current `dev`.
- Much easier to introduce unrelated regressions.
- Not required to close the reviewed findings.

### Rejected: Runtime-Only Patch Set

Fix only runtime behavior defects and add narrow tests, leaving most contract
artifacts and Stage 4 maintainability issues largely intact.

Reason rejected:

- Would leave several confirmed contract findings open.
- Would not satisfy the user’s request to address all findings.

## Design

### 1. Job Lifecycle and Cancellation Readback

The highest-severity runtime defect is cancelled-job instability on get/list
reads.

Runtime changes:

- Update [`jobs_adapter.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Chatbooks/jobs_adapter.py)
  so `CANCELLED` is treated as a protected terminal state, alongside
  `COMPLETED` and `FAILED`, for both export and import reconciliation.
- Keep the existing mapping behavior for non-terminal core Jobs rows, but stop
  remapping a Chatbooks row once it has reached `CANCELLED`.

Regression coverage:

- Add deterministic get/list-after-cancel tests that set up a local Chatbooks
  row plus a disagreeing core Jobs row and assert that the read path still
  reports `cancelled`.
- Tighten existing cancellation tests so they do not accept nearly every
  post-cancel state.

### 2. Import Cleanup and Helper Validation Hardening

Two backend-helper defects need direct fixes in the service layer.

Tokenized import cleanup:

- `_try_delete_import_file(...)` in
  [`chatbook_service.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Chatbooks/chatbook_service.py)
  should resolve tokenized `temp/...` and `import/...` paths through the same
  import archive path resolver used when imports are created.
- Cleanup should preserve the current containment guarantees while making the
  tokenized path case actually deletable.

Helper validation hardening:

- `validate_chatbook(...)` and `validate_chatbook_file(...)` should route
  through the hardened ZIP validation path in
  [`chatbook_validators.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/app/core/Chatbooks/chatbook_validators.py)
  before trusting `manifest.json`.
- This keeps helper behavior aligned with the main import/preview validator
  rules for traversal members, symlinks, suspicious archives, and other unsafe
  ZIP contents.

Regression coverage:

- Add service-level tests for tokenized orphan cleanup.
- Add helper-level tests that fail if malicious archive members are accepted via
  `validate_chatbook_file(...)`.
- Add API-level malicious archive tests for `/import` and `/preview`, so
  archive-safety coverage is no longer validator-only.

### 3. Stricter API Contracts

The route behavior should move to the stricter caller-visible contracts chosen
in design.

Preview:

- Map caller-caused preview failures to `400`.
- Reserve `500` for unexpected preview exceptions only.
- Stop returning successful `200` responses with an `error` payload.

Continuation export:

- Remove public async continuation support from the request/schema/endpoint
  contract unless a temporary compatibility layer is required.
- Unsupported async continuation must return `400`, not `500`.

Sync export/continuation:

- Standardize successful sync export responses on persisted-job semantics:
  `success`, `message`, `job_id`, `download_url`.
- Make `/export/continue` match the same sync response shape.
- Remove `file_path` from the live sync export schema/OpenAPI contract rather
  than keeping both success shapes in parallel.

Sync import:

- Populate `imported_items` with real counts from the sync import result.
- Keep `warnings` as a first-class sync response field.
- Count only successfully materialized imported items by content type.
  Renamed items count as imports; skipped items do not. Unsupported and
  unrequested types should be omitted from the map rather than represented by
  zero counts.
- Produce the counts from the import operation result itself so the response is
  deterministic and does not depend on a follow-up database recount.

Remove-job routes:

- Align endpoint docstrings, response text, and tests with the broader allowed
  terminal states already chosen in design.

Regression coverage:

- Add focused endpoint tests for preview error mapping, sync export/continuation
  response consistency, sync import counts, unsupported continuation mode, and
  remove-job success for failed/expired terminal states.

### 4. Static Contract and Documentation Alignment

After runtime behavior and schemas are corrected, update the static artifacts to
match the live contract.

Manifest schema:

- Replace or regenerate
  [`chatbooks_manifest_v1.json`](/Users/appledev/Documents/GitHub/tldw_server/Docs/Schemas/chatbooks_manifest_v1.json)
  from the actual `ChatbookManifest` shape.
- The canonical schema must validate real exports produced by the current
  implementation.

OpenAPI:

- Align request/response shapes, defaults, required fields, status codes, and
  route inventory with the stricter live API.
- This includes import request form shape, conflict options, preview semantics,
  sync export/import payloads, continuation export, and any corrected schema
  field details.

Implementation docs and PRD:

- Update the README and code guide so they no longer describe stale request
  shapes or unsupported runtime behavior.
- Update the PRD only where it currently states present-tense behavior that will
  still be false after the runtime fixes.
- Preserve planned-gap language where capabilities remain intentionally unbuilt.

Regression coverage:

- Add a manifest contract test that validates a real export against the updated
  canonical schema.
- Add explicit route/schema contract tests for the corrected live behaviors
  that previously drifted from OpenAPI, at minimum:
  - import multipart field shape versus the old `options` JSON contract
  - sync export response shape
  - sync import response shape including `warnings` and `imported_items`
  - preview failure HTTP semantics
  - absence of public async continuation support

### 5. Maintainability and Test-Structure Cleanup

The review findings require bounded test hardening, not just new tests.

Required cleanup:

- Replace false-confidence assertions in the current preview, signed-URL, and
  cancellation tests so they fail when the risky branch is not exercised.
- Split the worst Chatbooks test hotspot,
  [`test_chatbook_service.py`](/Users/appledev/Documents/GitHub/tldw_server/tldw_Server_API/tests/Chatbooks/test_chatbook_service.py),
  into smaller files by responsibility.
  Minimum split target:
  - job lifecycle
  - preview/import safety
  - continuation export
  - cleanup/import-path handling
- Tighten the weak integration assertions that directly intersect these reviewed
  defects, especially:
  - swallowed setup failures
  - “no error occurred” style assertions
  - compatibility branches that accept success without checking side effects

Explicitly not required in this pass:

- Full rewrite of Chatbooks E2E coverage
- Runtime-module decomposition
- Broad cosmetic test cleanup unrelated to reviewed findings

## Testing Strategy

This work should be implemented with TDD.

Required failing tests first:

- cancelled export/import job remains `cancelled` on get/list despite a lagging
  core Jobs row
- tokenized import orphan cleanup deletes `temp/...` and `import/...` paths
- `validate_chatbook_file(...)` rejects malicious archive members through the
  hardened validator path
- `/preview` returns `400` for invalid/unsafe archives and does not return
  `200 + error`
- `/export/continue` no longer advertises or accepts unsupported async
  continuation and returns `400` if `async_mode=true` is still submitted
- sync `/export` and sync `/export/continue` return the same persisted-job
  response shape
- sync import returns populated `imported_items`
- remove-job routes accept the chosen failed/expired terminal states
- real export manifest validates against the updated canonical schema
- malicious archives are rejected through live `/import` and `/preview`

Test categories:

- Unit tests for jobs adapter, helper validation, cleanup, and schema helpers
- Endpoint tests for preview/export/import/remove/continue contract behavior
- Contract tests for manifest schema and OpenAPI-aligned route behavior where
  feasible
- Focused integration hardening only for the reviewed weak spots

Validation expectations:

- Run the narrowest pytest slices while implementing.
- Run the touched Chatbooks scope before claiming completion.
- Run Bandit on the touched Chatbooks/runtime/docs test scope before finishing,
  per repo guidance.

## Risks and Controls

Risk: current `dev` contains unrelated in-progress work outside Chatbooks.

Control:

- implement in a fresh worktree from current `dev`
- stage and commit only Chatbooks remediation files

Risk: tightening preview and continuation contracts may break callers or tests
that accidentally relied on the looser behavior.

Control:

- update schemas/OpenAPI/docs in the same pass
- replace permissive tests with explicit contract assertions

Risk: the Chatbooks test surface is already large and partially unstable.

Control:

- prefer deterministic unit/API regression tests over broader async/E2E checks
- only tighten integration tests where they directly mask reviewed defects

## Success Criteria

This remediation is complete when all of the following are true:

- every confirmed runtime and contract finding from the Chatbooks synthesis is
  either fixed in code or resolved by aligning the stricter live contract
- regression tests exist for each fixed finding
- false-confidence tests identified in Stage 4 are tightened in the touched
  areas
- the manifest schema and OpenAPI file match the live Chatbooks behavior
- Chatbooks docs no longer contradict the stricter runtime contract in the
  reviewed areas
- the implementation can be validated on current `dev` without reverting any
  unrelated user work
