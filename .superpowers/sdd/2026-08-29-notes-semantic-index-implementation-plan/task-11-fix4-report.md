# TASK-13134 Task 11 Fix Round 4 Report

## Status

This bounded correction is implemented and verified. Task 11 remains pending a clean re-review. TASK-13134 remains In Progress, with Tasks 12-13 outstanding.

Commit: `fix: match Notes semantic runtime origins (TASK-13134)` (this commit).

## Root Cause And Fix

`canonical_semantic_endpoint_origin` used Python's legacy built-in IDNA codec. That mapped `faß.de` to `fass.de`, while both browser WHATWG parsing and the actual httpx request URL target `xn--fa-hia.de`.

The shared helper now takes the normalized ASCII host from the direct `httpx.URL.raw_host` dependency. It still takes the explicit port from `urllib.parse.urlsplit`, because httpx intentionally removes explicit default ports. Existing origin-only disclosure, credential/path/query/fragment redaction, scheme validation, IPv4 formatting, and compressed/bracketed IPv6 formatting remain unchanged. Invalid URL and host normalization errors fail closed.

No route, schema, dependency, provider control, Task 12, or Task 13 behavior changed.

## TDD Evidence

Runtime: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python --version` -> Python 3.11.13.

### RED

```text
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q --tb=line --show-capture=no tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py tldw_Server_API/tests/Notes_Graph/integration/test_semantic_jobs.py tldw_Server_API/tests/Embeddings_isolated/test_notes_semantic_policy.py -k 'capability_origin_passes_public_schema_and_pending_worker_authority or capability_origin_reaches_persisted_worker_pending_config or executor_origin_matches_httpx_idna_runtime_target'
```

Result: 3 failed, 4 passed, 125 deselected. The new `faß.de:443` cases failed as expected: capability/schema disclosure and persisted worker authority produced `https://fass.de:443`, while executor admission raised `endpoint_origin_mismatch`. Existing IPv6 and `bücher.example` controls passed.

A second focused RED for malformed hosts selected two cases. Result: 2 failed, 52 deselected because httpx percent-escaped a space-bearing host and a pre-escaped host instead of rejecting them. After the narrow percent-bearing `raw_host` guard, the same selector passed 2 tests with 52 deselected.

Frontend WHATWG control before the backend fix:

```text
bunx vitest run src/services/tldw/__tests__/note-semantic-index.test.ts -t 'matches a canonical IDNA origin to the WHATWG runtime host'
```

Result: 1 passed, 24 skipped. The existing frontend correctly parsed the runtime host as `xn--fa-hia.de` and accepted the canonical explicit-port disclosure. The first invocation from the extension Vitest root found no tests; rerunning from `apps/packages/ui` used the correct project root.

### GREEN

The final exact backend selector passed 9 tests with 125 deselected after the helper change, covering both `https://xn--fa-hia.de` and the explicit-default-port `https://xn--fa-hia.de:443` origin.

Final affected backend command covered capability, schema, store, pending execution, publication, endpoint, route-order, worker, and native adapter policy boundaries across ten files. Result: 328 passed, 2 warnings in 94.66s.

Final Task 11 frontend command covered the semantic client plus the prior 16-file graph/inspector set. Result: 227 passed across 16 files in 6.67s.

## Static And Security Verification

- Ruff lint: all touched Python production/tests passed.
- Ruff formatting: touched production helper passed.
- Canonical Prettier: touched frontend test passed with `apps/extension/.prettierrc.cjs`.
- Extension TypeScript: `bun run compile` passed.
- Bandit: scanned 36 production lines with 0 findings, 0 errors, and 0 skipped tests; report `/tmp/bandit_TASK-13134_task11_fix4.json`.
- `git diff --check`: passed before tracking updates and rerun after finalization.

## Files

Production:

- `tldw_Server_API/app/core/Notes_Graph/semantic_endpoint.py`

Tests:

- `tldw_Server_API/tests/Notes_Graph/unit/test_semantic_capabilities.py`
- `tldw_Server_API/tests/Notes_Graph/integration/test_semantic_jobs.py`
- `tldw_Server_API/tests/Embeddings_isolated/test_notes_semantic_policy.py`
- `apps/packages/ui/src/services/tldw/__tests__/note-semantic-index.test.ts`

Tracking:

- `.superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/progress.md`
- `.superpowers/sdd/2026-08-29-notes-semantic-index-implementation-plan/task-11-fix4-report.md`
- `backlog/tasks/task-13134 - Implement-Notes-embedding-index-and-semantic-graph-edges.md`

## Risks

- No live provider request was sent. The executor regression verifies the exact base URL passed to the adapter and resolves its runtime host through the same `httpx.URL` primitive used by request execution.
- No live PostgreSQL/pgvector run or repository-wide test suite was performed because this correction changes no schema, persistence backend, or vector operation. The complete affected Task 11 backend/frontend sets passed.
- Task 11 is not claimed complete; fix round 4 remains pending clean re-review, and Tasks 12-13 remain outstanding.
