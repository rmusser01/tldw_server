---
id: TASK-9927
title: Fix validated Embeddings module review findings
status: Done
assignee: []
created_date: '2026-06-23 18:48'
updated_date: '2026-06-23 18:52'
labels:
  - embeddings
  - security
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and address validated current-code findings from the Embeddings module review. Scope includes job artifact path confinement, Chroma persistent-client fallback behavior, Chroma dimension mismatch handling, Redis enqueue failure handling, DLQ encryption fail-closed behavior, sensitive log redaction, and inactive sharding/request-signing helper quarantine.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Job artifact paths cannot escape the per-user embeddings artifact directory.
- [x] #2 Persistent Chroma client initialization fails closed unless stub mode or fallback is explicitly enabled.
- [x] #3 Embedding dimension mismatches no longer delete existing collections implicitly.
- [x] #4 Redis enqueue infrastructure errors fail the root job instead of leaving it orphaned.
- [x] #5 Configured DLQ encryption never silently degrades to base64-only encoding.
- [x] #6 Provider error and vector-search logs do not include sensitive payload text.
- [x] #7 Inactive sharding/request-signing modules are quarantined from runtime use.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression tests that reproduce each validated hot-path issue.
2. Implement the minimal code changes needed to make those tests pass.
3. Run focused tests for Embeddings/ChromaDB paths.
4. Run Bandit on touched Embeddings paths.
5. Update task notes/final summary with verification evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Validated findings and implemented focused fixes:
- Confined embeddings job artifact identifiers and payload-provided artifact paths to the per-user artifact directory.
- Made Redis idempotency infrastructure failures raise so the adapter fails the root job instead of orphaning it.
- Made persistent Chroma fallback fail closed by default and made dimension mismatches non-destructive.
- Removed sensitive provider response bodies and vector-search query text from logs.
- Made configured DLQ encryption fail closed when AES-GCM is unavailable or a plaintext fallback is returned.
- Gated inactive sharding and request-signing singleton helpers behind explicit configuration, including request-signing API-key manager key-file configuration.

Verification:
- Red runs observed expected regression failures before implementation, including artifact escape, Chroma fallback/deletion, Redis orphaning, DLQ downgrade, log leakage, sharding/request-signing singleton defaults, and request-signing API-key default generation.
- source .venv/bin/activate && python -m pytest --confcutdir=tldw_Server_API/tests/Embeddings_isolated -q tldw_Server_API/tests/Embeddings_isolated/test_review_findings_hardening.py tldw_Server_API/tests/Embeddings_isolated/test_request_signing_nonce_security.py => 11 passed, 14 warnings.
- source .venv/bin/activate && python -m pytest --confcutdir=tldw_Server_API/tests/ChromaDB -q tldw_Server_API/tests/ChromaDB/unit/test_chromadb_dimensions_and_list.py => 17 passed, 4 warnings.
- source .venv/bin/activate && python -m compileall -q <touched files> => passed.
- source .venv/bin/activate && python -m bandit -r <touched production files> -f json -o /tmp/bandit_embeddings_review_findings_9927.json => 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed Embeddings review remediation. Hot-path security/reliability findings are fixed with regression coverage; inactive sharding/request-signing helper risks are quarantined behind explicit configuration. Focused tests, compileall, and Bandit passed on the touched scope.
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
