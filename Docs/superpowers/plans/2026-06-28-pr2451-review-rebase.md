# PR 2451 Review Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR 2451 onto current `dev` and address every active PR review comment.

**Architecture:** Keep the existing Embeddings hardening approach. Add regression coverage beside the affected modules, then make the smallest production changes needed to fail closed without broad refactors.

**Tech Stack:** FastAPI backend, Python, pytest, ChromaDB, Redis embeddings enqueue path, Backlog.md.

---

## Stage 1: Rebase And Review Inventory
**Goal**: Work from the current `dev` base and enumerate active PR feedback.
**Success Criteria**: Branch rebased, active review threads identified, CI guard status understood.
**Tests**: `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`
**Status**: Complete

## Stage 2: Regression Tests
**Goal**: Add red tests for each still-actionable review issue.
**Success Criteria**: Tests fail for Chroma string booleans, Kanban enqueue failure state, artifact path type rejection, singleton removal expectations, and DLQ cleanup coverage.
**Tests**: Targeted pytest node ids for the new tests.
**Status**: Complete

## Stage 3: Production Fixes
**Goal**: Implement minimal fixes that satisfy the tests and review comments.
**Success Criteria**: String booleans parse correctly, malformed artifact paths raise non-retryable `EmbeddingsJobError`, Kanban root jobs are failed on enqueue errors, request signing/sharding no longer use modified singleton accessors, and DLQ encryption metadata is written without redundant branching.
**Tests**: Targeted pytest files pass.
**Status**: Complete

## Stage 4: Verification And PR Cleanup
**Goal**: Verify touched scope, record Backlog evidence, push the rebased branch, and resolve/comment on PR threads.
**Success Criteria**: Targeted tests, compileall, shard coverage guard, and Bandit pass or have documented environment blockers; branch is pushed to PR 2451; active review threads are addressed.
**Tests**: `pytest` targeted files, `python -m compileall` on touched modules, `python -m bandit -r` on touched production paths.
**Status**: Complete

Verification completed locally:
- `python -m pytest tldw_Server_API/tests/ChromaDB/unit/test_chromadb_dimensions_and_list.py tldw_Server_API/tests/Embeddings_isolated/test_review_findings_hardening.py tldw_Server_API/tests/Embeddings_isolated/test_request_signing_nonce_security.py tldw_Server_API/tests/kanban/test_kanban_vector_search.py -q`
- `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`
- `python -m compileall tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py tldw_Server_API/app/core/DB_Management/kanban_vector_search.py tldw_Server_API/app/core/Embeddings/services/jobs_worker.py tldw_Server_API/app/core/Embeddings/request_signing.py tldw_Server_API/app/core/Embeddings/sharding.py tldw_Server_API/app/core/Embeddings/dlq_crypto.py`
- `python -m bandit -r tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py tldw_Server_API/app/core/DB_Management/kanban_vector_search.py tldw_Server_API/app/core/Embeddings/services/jobs_worker.py tldw_Server_API/app/core/Embeddings/request_signing.py tldw_Server_API/app/core/Embeddings/sharding.py tldw_Server_API/app/core/Embeddings/dlq_crypto.py -f json -o /tmp/bandit_pr2451.json`
- `git diff --check`
