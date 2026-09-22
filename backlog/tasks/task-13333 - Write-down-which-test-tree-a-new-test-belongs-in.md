---
id: TASK-13333
title: Write down which test tree a new test belongs in
status: To Do
assignee: []
created_date: '2026-09-22 04:58'
labels:
  - tests
  - docs
dependencies: []
references:
  - 'tldw_Server_API/tests/RAG/conftest.py:141'
  - .github/workflows/ci.yml
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The _NEW suffix carries no consistent meaning and the per-module verdicts genuinely differ, so this is one documentation task, not a merge:
- RAG (47 vs 143): REAL PROBLEM. The conftests protect different things and pytest scopes each to its own directory. The only sqlite/postgres-parametrized fixture is tests/RAG/conftest.py:141 with 2 consumers, so the 143-file tree is effectively SQLite-only.
- Chat (100 vs 58): NOT duplication. One shared basename, unit-vs-integration, zero overlapping test names, both sharded in CI. tests/Chat is 2x more active yet CI labels it chat-legacy.
- TTS (56 vs 93): complementary suites (security/sanitization vs public-contract) nobody wrote down. Coverage PARTITIONS rather than overlaps. Two production shim classes exist solely to satisfy TTS_NEW and are registered as the real adapters.
- AuthNZ (5 trees, 393 files): ci.yml slices them ALPHABETICALLY, so the axis is shard wall-clock, not domain.
- Ingestion (192 files, 21 trees): no discoverable axis; the declared media_processing marker has one use and pipeline has zero.

Minimum useful step: a written rule. Today CI is the only thing that labels the split, and for Chat it mislabels.

Source: synthesis F33
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A short documented rule states where a new test goes per module
- [ ] #2 CI shard names stop calling the more active Chat tree legacy
- [ ] #3 RAG dual-backend fixture reaches the RAG_NEW tree or the split is resolved
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
