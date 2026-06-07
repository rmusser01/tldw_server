---
id: TASK-2261
title: Confirm Embeddings ADR candidate for backfill
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-05 01:24'
labels:
  - docs
  - process
  - adr
  - embeddings
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Confirm whether INV-032 from Docs/ADR/inventory/2026-06-03-decision-inventory.md is current and bounded enough for ADR backfill. Verify tldw_Server_API/app/core/Embeddings/README.md and representative code/tests for OpenAI-compatible embedding API safeguards, provider auto-detect/adapters, cache/batching/circuit breaker behavior, Redis Streams worker ownership, Jobs status/billing ownership, caveats, and any scope that should remain inventory-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create an Embeddings confirmation audit under `Docs/ADR/inventory/` using current `origin/dev` evidence.
- [x] #2 Classify `INV-032` as ready for bounded ADR backfill, needing code/doc alignment, or inventory-only, with explicit caveats.
- [x] #3 Update the decision inventory only if the confirmation result changes the tracked next action.
- [x] #4 Create a follow-up Backlog task only if the candidate is ready for ADR backfill.
- [x] #5 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Created `Docs/ADR/inventory/2026-06-04-embeddings-confirmation-audit.md`.
- Classified INV-032 as current governing and ready for one bounded Embeddings ADR backfill.
- Scoped the future ADR to OpenAI-compatible request/response semantics, provider resolution and allowlist safeguards, optional adapter-registry routing with legacy provider-config/direct execution fallback, endpoint cache/batching/circuit-breaker controls, and Jobs-root/Redis-stage media embedding pipeline ownership.
- Documented caveats for billing/accounting scope, local provider URL policy, Chroma/pgvector storage evolution, endpoint TTL cache versus broader cache modules, and the legacy Jobs worker.
- Updated `Docs/ADR/inventory/2026-06-03-decision-inventory.md` to record TASK-2261 confirmation and queue TASK-2262 for the accepted ADR backfill.
- Created TASK-2262 as the follow-up ADR implementation task.
- Verification:
  - `git diff --check` passed.
  - Scoped reference grep passed for TASK-2261, TASK-2262, INV-032, the Embeddings confirmation audit, expected ADR-022 path, and Bandit references in the touched files.
  - `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py tldw_Server_API/tests/Embeddings/test_request_batching.py tldw_Server_API/tests/Embeddings/test_embeddings_jobs_adapter.py tldw_Server_API/tests/Embeddings/test_embeddings_redis_worker.py` passed: 34 passed, 3 warnings.
  - Removed test-generated untracked Watchlists template artifacts before staging.
- Bandit: skipped because this task touched only Markdown documentation and Backlog task metadata; no Python/code paths were changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Confirmed INV-032 as ready for a bounded Embeddings ADR backfill. The confirmation audit captures current code/test evidence and caveats, the inventory now points to TASK-2261/TASK-2262, and TASK-2262 is queued for the actual ADR. Verification passed for diff hygiene, scoped references, and the focused Embeddings pytest suite. Bandit is not applicable because the touched scope is Markdown documentation and Backlog task metadata only.
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
