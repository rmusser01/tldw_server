# Task 6 Implementer Report

## Status

DONE. Task 6 freezes authoritative owner source scope and permits only fully verified, media-only retrieval evidence. Task 7 remains responsible for API orchestration and generation.

Starting commit: `e09af9a232a3a606cf2117f684bff73e9c8d0d60`

Task commit: `feat(sharing): scope shared retrieval to canonical sources` (this report is included in that commit)

## Implementation

- Added frozen `SharedSourceSnapshotItem`, `SharedSourceSnapshot`, `VerifiedSharedEvidence`, retrieval policy, and `SharedWorkspaceChatService` dataclasses.
- Resolved `all` and `include` from authoritative owner workspace rows, with exact validation for empty, duplicate, unknown, nonqueryable, and over-cap scopes.
- Rebuilt frozen scopes by exact canonical source ID. Include snapshots ignore unrelated source changes; frozen `all` retries cannot expand to newly added sources.
- Read owner media with deleted/trash visibility and captured media ID, UUID, content hash, and readiness class in the in-memory snapshot and canonical hash input only.
- Deduplicated shared media reads and retrieval IDs. Evidence for shared media is assigned to the lexicographically smallest selected canonical source ID.
- Added one immutable retrieval policy that pins every security-sensitive `unified_rag_pipeline` parameter to owner media-only, retrieval-only operation. No caller request, plan, profile, metadata, or arbitrary kwargs are accepted.
- Added a full-signature sentinel so every future pipeline parameter requires explicit pinning or inert review.
- Rejected the complete retrieval result on pipeline errors, generated content, cache/external/generation metadata, invalid provenance, invalid or out-of-scope media IDs, or missing canonical mappings.
- Deduplicated verified chunks before assigning deterministic `E1` through `E20` labels and enforced per-field and aggregate evidence bounds.

## TDD RED

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py -q --timeout=60
```

Result: collection error because `tldw_Server_API.app.core.Sharing.shared_workspace_chat_service` did not exist (`ModuleNotFoundError`).

## GREEN Verification

Focused Task 6 suite:

```text
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py -q --timeout=60
```

Result: `35 passed, 4 warnings in 6.75s` on the final post-format rerun.

Bounded Task 4/5/6 sharing and access regressions:

```text
TLDW_TEST_NO_DOCKER=1 python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py tldw_Server_API/tests/Sharing/test_shared_workspace_repo.py tldw_Server_API/tests/Workspaces/test_workspace_job_status.py tldw_Server_API/tests/Workspaces/test_workspace_source_preview.py tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py tldw_Server_API/tests/Sharing/test_shared_workspace_chat_security.py tldw_Server_API/tests/Sharing/test_sharing_endpoints.py tldw_Server_API/tests/Sharing/test_cross_user_access.py -q -n 4 --dist=loadfile --timeout=60 -o log_cli=false
```

Result: `199 passed, 30 warnings in 63.61s`.

Ruff over both touched Python files: passed.

Bandit:

```text
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py -f json -o /tmp/bandit_task_12020_40_task6.json
```

Result: zero findings, zero errors, zero skipped tests, 917 production LOC scanned.

`git diff --check`: passed.

## Files

Production:

- `tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py`

Tests:

- `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py`

Tracking/reporting:

- `backlog/tasks/task-12020.40 - Bind-recipient-shared-workspace-sources-and-chat-to-the-canonical-share.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/progress.md`
- `.superpowers/sdd/2026-08-21-recipient-shared-research-workspace-data-plane/task-6-implementer-report.md`

## Self-Review

- Snapshot hashes use canonical JSON over sorted source IDs and only source ID, owner media ID, media UUID, content hash, and readiness class. Owner media IDs remain in memory and are never added to recipient receipts or public output.
- Snapshot revalidation detects canonical source removal/remap, UUID or hash change, deletion, trash, and readiness loss while preserving frozen include/all membership semantics.
- Retrieval receives explicit nonempty selected media IDs, owner media storage and namespace, and no notes/chat database. Cache, profiles, provider-backed transforms, reranking, adaptive reruns, generation, history, external fallback, structured/image/video features, and streaming are disabled.
- Every returned document is validated before retention, including documents after the 20-evidence limit. One invalid document rejects the complete result.
- No Task 5 route or schema behavior changed. The two unrelated untracked watchlist templates were not touched or staged.

## PostgreSQL State

Task 6 changed no PostgreSQL schema, migration, policy, fixture, or query. PostgreSQL was not started or touched; the service is exercised with deterministic owner-store doubles and existing sharing/access regressions.

## Residual Risks

The policy sentinel is intentionally coupled to the current `unified_rag_pipeline` signature and will fail when that signature changes until each new parameter is reviewed. Task 7 must revalidate the frozen snapshot at its orchestration boundary and must consume only `VerifiedSharedEvidence` for generation and citations.
