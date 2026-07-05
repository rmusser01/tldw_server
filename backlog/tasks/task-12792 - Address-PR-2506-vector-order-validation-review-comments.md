---
id: TASK-12792
title: Address PR 2506 vector order validation review comments
status: Done
labels:
- review
- vector-stores
- pgvector
modified_files:
- tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_list_vectors.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR 2506 review comments about test helper imports, docstrings, and type hints for vector listing metadata order validation coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 _FakeAdapterInvalidMetadataOrder imports InvalidMetadataOrderKeyError directly from core.exceptions.
- [x] #2 New fake adapter and regression tests include docstrings and explicit type annotations.
- [x] #3 Focused tests and Bandit verification are rerun before pushing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated PR 2506 review follow-up tests to use the direct InvalidMetadataOrderKeyError import, add docstrings and type annotations for _FakeAdapterInvalidMetadataOrder and the new regression tests, and annotate only the newly added pytest assertions for Bandit B101. Verification: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_vector_store_list_vectors.py tldw_Server_API/tests/VectorStores/unit -q -> 29 passed, 5 warnings. git diff --check -> exit 0. Bandit on tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py -> exit 0, 0 findings. Bandit on the touched test file still exits 1 for the existing low-severity B101 pytest assert baseline, with 60 total B101 results and no findings on the newly added test lines.
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
