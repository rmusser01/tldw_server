---
id: TASK-13333
title: Write down which test tree a new test belongs in
status: Done
assignee: []
created_date: '2026-09-22 04:58'
updated_date: '2026-09-23 23:45'
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
- [x] #1 A short documented rule states where a new test goes per module
- [x] #2 CI shard names stop calling the more active Chat tree legacy
- [x] #3 RAG dual-backend fixture reaches the RAG_NEW tree or the split is resolved
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Commit 09a591d369. AC1: tldw_Server_API/tests/README.md has a new section, 'Where Does a New Test Go?', with a per-module table: Chat vs Chat_NEW, RAG vs RAG_NEW, TTS vs TTS_NEW, the five AuthNZ trees, and Ingestion/Media. Ingestion has no axis, so the rule there is to add to the tree that already imports the module under test. The section also states that _NEW is not 'preferred', that no tree is legacy, and that CI shard names balance wall-clock time and do not name domains. AC2: in .github/workflows/ci.yml, chat-legacy-{integration,unit-a-l,unit-m-z} are renamed to chat-{integration,unit-a-l,unit-m-z} in all 5 matrices (15 lines). The YAML parses. No condition or ruleset refers to the old names; the dev ruleset requires only the aggregate *-required checks, per Docs/Development/CI_REQUIRED_GATES.md and gh api rulesets. rag-legacy is kept: RAG/ is not documented as legacy, and the rule now says what it holds. AC3: RAG_NEW/conftest.py re-exports dual_backend_env (and DualBackendEnv) from tests/RAG/conftest.py. Verified with 'pytest --fixtures tldw_Server_API/tests/RAG_NEW/unit/...', which lists dual_backend_env from tests/RAG/conftest.py:142, and with --co over RAG+RAG_NEW (2071 collected, no errors). Before/after runs of RAG, RAG_NEW and the rest are recorded under TASK-13327 (same run). Bandit: not applicable, since only docs, CI YAML and a test conftest changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Documented a per-module placement rule for new tests in tests/README.md. Renamed the chat-legacy-* CI shards to chat-*, since that tree is the active one, not legacy. Made the only SQLite/Postgres RAG fixture (dual_backend_env) reachable from RAG_NEW via a conftest re-export. No trees were merged: per the task, the split is kept and written down.
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
