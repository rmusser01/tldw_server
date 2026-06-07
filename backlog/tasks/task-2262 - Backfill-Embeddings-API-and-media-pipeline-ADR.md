---
id: TASK-2262
title: Backfill Embeddings API and media pipeline ADR
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-05 01:45'
labels:
  - docs
  - process
  - adr
  - embeddings
dependencies:
  - TASK-2261
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Backfill a bounded Embeddings ADR from TASK-2261 evidence. Scope the accepted decision to OpenAI-compatible embeddings request/response semantics, provider resolution and allowlist safeguards, optional adapter-registry routing with legacy provider-config fallback, cache/batching/circuit-breaker reliability controls, and media embedding pipeline ownership where core Jobs owns the durable root status record while Redis Streams owns stage delivery. Keep billing/quota behavior, local provider URL policy, pgvector/Chroma backend evolution, and legacy Jobs worker details as explicit caveats unless separately confirmed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create the next accepted ADR under `Docs/ADR/` using the standard ADR template and TASK-2261 evidence.
- [x] #2 Keep accepted claims scoped to OpenAI-compatible API semantics, provider resolution/allowlist safeguards, optional adapter routing, cache/batching/circuit-breaker reliability controls, and Jobs-root/Redis-stage media pipeline ownership.
- [x] #3 Update `Docs/ADR/README.md`, the INV-032 inventory row/default disposition, and the Embeddings README backlink after ADR creation.
- [x] #4 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Started TASK-2262 in isolated worktree .worktrees/backfill-embeddings-api-adr from origin/dev. Planned scope: create ADR-022 from TASK-2261 evidence, update ADR index, update INV-032 inventory disposition, add Embeddings README backlink, then record docs-only verification and Bandit applicability.

Created ADR-022 and updated ADR index, INV-032 inventory/default disposition, and the Embeddings README backlink. Verification: `git diff --check` passed; scoped file/reference check using `git grep` passed for ADR-022, TASK-2262, INV-032, the ADR path, the completed inventory disposition, and the README backlink; focused Embeddings pytest suite passed with 34 passed and 3 warnings using `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py tldw_Server_API/tests/Embeddings/test_request_batching.py tldw_Server_API/tests/Embeddings/test_embeddings_jobs_adapter.py tldw_Server_API/tests/Embeddings/test_embeddings_redis_worker.py`. Removed two pytest-generated untracked Watchlists template files before staging. Bandit: skipped because touched files are Markdown documentation and Backlog task metadata only; no Python/code paths were changed. Known blockers/skips: none beyond docs-only Bandit non-applicability.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Backfilled INV-032 as ADR-022 for the bounded Embeddings API and media pipeline decision. The ADR captures OpenAI-compatible request/response semantics, provider resolution and allowlist safeguards, optional adapter-registry routing with legacy fallback, endpoint cache/batching/circuit-breaker reliability controls, and Jobs-root/Redis-stage media pipeline ownership, with billing/accounting, local provider URL policy, vector-store backend evolution, broader cache architecture, and legacy Jobs worker removal left as separate decisions. Updated the ADR index, inventory disposition, and Embeddings README backlink. Verification passed for diff hygiene, scoped references, and the focused Embeddings pytest suite; Bandit is not applicable for the docs-only touched scope.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
