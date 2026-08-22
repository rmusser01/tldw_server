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

## Fix Round 1

Reviewed head: `53665b9a979608964a2066d2a63f1ab064fc3c00`

### Changes

- Changed `_serialize_result_document` so authoritative top-level source provenance survives serialization and overrides conflicting metadata while dict-backed documents without a top-level source retain general compatibility.
- Required exact submitted query, empty expansion/errors, strict `generated_answer is None`, `cache_hit is False`, empty derived outputs, and allowlisted media-only metadata on actual `UnifiedRAGResponse` results.
- Pinned `min_score`, `chunk_type_filter`, `ocr_confidence_threshold`, and the active-path `timeout_seconds`; added an AST sentinel for recognized literal hidden `kwargs` reads and an explicit reviewed-absent set.
- Reloaded current authoritative workspace-source rows for canonical source ID/title mapping. Same-media aliases use the lexicographically smallest selected source ID and its bounded title; RAG titles are ignored and titles remain outside the snapshot hash.
- Enforced bounded canonical chunk identities, exact nonnegative locators, full-document validation before E20 retention, and media/chunk deduplication that rejects conflicting content, score, or locators. Contentless records are still fully validated and participate in conflict detection.
- Enforced exact stored source IDs, media UUIDs, and content hashes without strip normalization; malformed `all` rows now return a disclosure-safe service error.
- Raised evidence retention to 4,000 characters per item and 48,000 aggregate while retaining deterministic `E1` through `E20` labels.

### TDD Evidence

RED results, captured before each production correction:

```text
serializer: 3 failed, 1 passed
response/policy: 28 failed, 3 passed
snapshot/title: 9 failed
identity/capacity: 15 failed
active retrieval timeout: 1 failed
contentless malformed locator self-review: 1 failed
```

Final focused command:

```text
TLDW_TEST_NO_DOCKER=1 .venv/bin/python -m pytest tldw_Server_API/tests/RAG/test_unified_pipeline_document_serialization.py tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py -q --timeout=60 -o log_cli=false
```

Result: `94 passed, 4 warnings in 7.60s`.

Final bounded Task 4/5/6 regression matrix used the serializer, Task 6 retrieval, access/repository, workspace status/preview, recipient endpoints/security, sharing endpoints, and cross-user access files with `-n 4 --dist=loadfile --timeout=60`.

Result: `258 passed, 30 warnings in 196.70s`.

Ruff passed all four touched Python files. Bandit scanned both touched production files and reported zero findings across 9,003 LOC. `git diff --check` passed.

### Files

- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- `tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py`
- `tldw_Server_API/tests/RAG/test_unified_pipeline_document_serialization.py`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py`
- Task 6 backlog, progress, and implementer report tracking.

### Self-Review And Residual Risk

The shared caller still accepts no caller-owned plan/profile/metadata/kwargs and passes only explicit owner media scope. Every returned document, including contentless records and records after E20, is validated before evidence can escape. The serializer change is intentionally general and covered for dict-backed and production `Document` objects. The policy signature and hidden-kwargs sentinels intentionally require explicit review when the unified pipeline grows.

Task 7 still owns API orchestration, receipt persistence, generation, and persisted citation quote budgeting. No Task 5 route/schema behavior changed, and the unrelated watchlist templates were not touched or staged.

### PostgreSQL State

PostgreSQL was not started or touched. Fix Round 1 changed no PostgreSQL schema, migration, RLS policy, fixture, or query.

## Fix Round 2

Reviewed head: `f741f881e486540b43df4e5efd57f6ca718a42ca`

### Changes

- Added a default-on `include_retrieval_diagnostics` unified-pipeline gate. Shared retrieval pins it off, so `profile_resolution`, `source_status`, and `why_these_sources` cannot enter its response metadata while ordinary callers retain the existing default diagnostics.
- Exercised `retrieve_verified_evidence` through the real unified pipeline and a controlled media retriever. The integration exposed and fixed owner namespace reconstruction by passing the owner user ID as a server-owned dynamic input alongside the explicit owner media database, namespace, and IDs.
- Classified invalid source-row media IDs and malformed exact media UUID/content hashes as storage-shape errors. `all` maps them to `shared_workspace_unavailable`, including mixed rows, while initial `include` and revalidation preserve their established mappings.
- Replaced the hidden-kwargs key-set-only analysis with a scope-aware AST validator. Every load of the outer pipeline `kwargs` must be the direct receiver of an approved literal-key `get`/`pop`/`setdefault` call or a constant-string subscript; aliasing, dynamic keys, iteration, membership, truthiness, unpacking, and indirect method access fail the sentinel, while nested functions with their own `kwargs` are ignored.
- Restored serializer `setdefault` precedence for all non-source metadata identities and locators. Only top-level `source` is authoritative; shared retrieval still rejects preserved chunk identity conflicts in the actual serialized response shape.

### TDD Evidence

RED results captured before each production correction:

```text
real pipeline diagnostics: 1 failed
real owner namespace integration: 1 failed, 2 passed
malformed all-mode identities: 14 failed, 6 passed
outer kwargs analyzer fixtures: 8 failed, 1 passed
serializer and shared actual-shape regressions: 3 failed
```

Focused GREEN command:

```text
TLDW_TEST_NO_DOCKER=1 .venv/bin/python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py tldw_Server_API/tests/RAG/test_unified_pipeline_document_serialization.py -q --timeout=60 -o log_cli=false
```

Result: `126 passed, 4 warnings in 6.85s`.

Focused Task 4 and Task 5 regressions:

```text
TLDW_TEST_NO_DOCKER=1 .venv/bin/python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py -q --timeout=60 -o log_cli=false
TLDW_TEST_NO_DOCKER=1 .venv/bin/python -m pytest tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py tldw_Server_API/tests/Sharing/test_shared_workspace_chat_security.py -q --timeout=60 -o log_cli=false
```

Results: `9 passed, 2 warnings in 6.62s`; `40 passed, 4 warnings in 7.71s`.

The established 290-item Task 4/5/6 xdist matrix scheduled all tests and reached `pytest_sessionfinish` without reporting an assertion failure. Its `xdist`/`execnet` node teardown exceeded the prior cleanup-latency profile and was interrupted once with exit 130. No aggregate rerun was made, and no pytest/xdist worker remained.

Ruff passed all four touched Python files. Bandit reported zero findings and zero errors across both touched production files (9,006 LOC, zero skips). `git diff --check` passed.

### Files

- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- `tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py`
- `tldw_Server_API/tests/RAG/test_unified_pipeline_document_serialization.py`
- `tldw_Server_API/tests/Sharing/test_shared_workspace_chat_retrieval.py`
- Task 6 backlog, progress, and implementer report tracking.

### Self-Review And Residual Risk

The diagnostics switch defaults to the prior ordinary-client behavior and gates only response diagnostics, not retrieval. The shared policy owns every dynamic value and still accepts no caller plan, profile, metadata, or arbitrary kwargs. Source scope remains explicit owner-only media; serializer compatibility does not weaken shared validation because authoritative source and top-level/metadata identity conflicts are checked after serialization.

Task 7 remains responsible for API orchestration, generation, receipts, and persisted citations. The policy signature and hidden-kwargs sentinels intentionally require review when the unified pipeline changes. The aggregate teardown issue remains repository test-harness cleanup behavior; focused Task 4, Task 5, Task 6, real-pipeline, and serializer acceptance targets are conclusive.

### PostgreSQL State

PostgreSQL was not started or touched. Fix Round 2 changed no PostgreSQL schema, migration, RLS policy, fixture, or query.
