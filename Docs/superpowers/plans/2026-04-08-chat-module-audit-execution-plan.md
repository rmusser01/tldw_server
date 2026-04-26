# Chat Module Audit Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved Chat audit, gather targeted test evidence, and deliver a severity-ranked review covering the scoped Chat routes and shared `core/Chat` code.

**Architecture:** The audit proceeds in fixed route groups. Each group freezes its route and test inventory first, performs static inspection second, runs targeted tests only where they validate ambiguous or risky behavior third, and records findings immediately in one review artifact so evidence and conclusions stay linked. No production code changes are planned during this execution unless the user explicitly pivots from review to remediation.

**Tech Stack:** Python/FastAPI source inspection, `rg`, `sed`, `pytest`, repo-local markdown docs under `Docs/superpowers/*`

---

## File Map

- Create: `Docs/superpowers/reviews/2026-04-08-chat-module-review.md` - canonical audit artifact with scope, evidence log, route inventory, findings, open questions, and coverage gaps.
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_loop.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_grammars.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_workflows.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_service.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_orchestrator.py`
- Inspect: `tldw_Server_API/app/core/Chat/streaming_utils.py`
- Inspect: `tldw_Server_API/app/core/Chat/request_queue.py`
- Inspect: `tldw_Server_API/app/core/Chat/rate_limiter.py`
- Inspect: `tldw_Server_API/app/core/Chat/document_generator.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_loop_store.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_loop_approval.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_loop_engine.py`
- Inspect: `tldw_Server_API/app/core/Chat/validate_dictionary.py`
- Inspect: `tldw_Server_API/app/core/Chat/README.md`
- Inspect: `tldw_Server_API/app/core/Chat/orchestrator/provider_resolution.py`
- Inspect: `tldw_Server_API/app/core/Chat/orchestrator/request_validation.py`
- Inspect: `tldw_Server_API/app/core/Chat/orchestrator/stream_execution.py`
- Test: `tldw_Server_API/tests/Chat`
- Test: `tldw_Server_API/tests/Chat_NEW`
- Test: `tldw_Server_API/tests/Chat_Workflows`
- Test: `tldw_Server_API/tests/Streaming`
- Test: `tldw_Server_API/tests/e2e/test_workspace_chat_scope.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_chat_workflows_deps.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_chat_workflows_permissions.py`

### Task 1: Freeze Inventory And Create The Audit Artifact

**Files:**
- Create: `Docs/superpowers/reviews/2026-04-08-chat-module-review.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_loop.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_grammars.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_workflows.py`
- Test: `tldw_Server_API/tests/Chat`
- Test: `tldw_Server_API/tests/Chat_NEW`
- Test: `tldw_Server_API/tests/Chat_Workflows`
- Test: `tldw_Server_API/tests/Streaming`

- [ ] **Step 1: Create the review scaffold**

```markdown
# Chat Module Audit Findings

## Scope
- Spec: `Docs/superpowers/specs/2026-04-08-chat-module-review-design.md`
- In scope: `chat.py`, `chat_documents.py`, `chat_loop.py`, `chat_grammars.py`, `chat_dictionaries.py`, `chat_workflows.py`, shared `core/Chat/*`
- Out of scope: Character Chat, Chatbooks, unrelated provider internals

## Evidence Log
| Group | Routes | Static review | Targeted tests | Result | Notes |
| --- | --- | --- | --- | --- | --- |

## Route Inventory

## Critical
_None._

## High
_None._

## Medium
_None._

## Low
_None._

## Open Questions

## Coverage Gaps
```

- [ ] **Step 2: Freeze the route inventory**

Run:

```bash
rg -n "@router\\.(get|post|patch|delete)\\(" \
  tldw_Server_API/app/api/v1/endpoints/chat.py \
  tldw_Server_API/app/api/v1/endpoints/chat_documents.py \
  tldw_Server_API/app/api/v1/endpoints/chat_loop.py \
  tldw_Server_API/app/api/v1/endpoints/chat_grammars.py \
  tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py \
  tldw_Server_API/app/api/v1/endpoints/chat_workflows.py
```

Expected: a concrete decorator list for every scoped endpoint file. Copy the route groupings into the `Route Inventory` section before reading implementation details.

- [ ] **Step 3: Freeze the test inventory**

Run:

```bash
rg --files tldw_Server_API/tests | rg '(^|/)(Chat|Chat_NEW|Chat_Workflows|Streaming)/|workspace_chat_scope|chat_workflows_(deps|permissions)'
```

Expected: a bounded evidence list for the scoped review. Map each relevant file into the `Evidence Log`, and write `no direct route test found` for scoped routes that have no direct evidence file.

- [ ] **Step 4: Verify the low-traffic routes are not skipped**

Run:

```bash
rg -n "async def list_chat_commands|async def validate_chat_dictionary|async def create_chat_completion|async def get_chat_queue_status|async def get_chat_queue_activity|async def list_chat_conversations|async def get_chat_conversation|async def update_chat_conversation|async def get_conversation_tree|async def create_conversation_share_link|async def resolve_conversation_share_token|async def save_chat_knowledge|async def get_chat_analytics|async def persist_rag_context|async def get_rag_context|async def get_messages_with_rag_context|async def get_conversation_citations" \
  tldw_Server_API/app/api/v1/endpoints/chat.py
```

Expected: every major `chat.py` route family appears in `Route Inventory` and has either a mapped test file or an explicit `coverage gap candidate` note.

- [ ] **Step 5: Save the artifact before deeper review**

Run:

```bash
git diff -- Docs/superpowers/reviews/2026-04-08-chat-module-review.md
```

Expected: the diff shows only the scaffold, route inventory, and evidence-log seed entries.

### Task 2: Audit Main Chat Completion, Queue, And Streaming Behavior

**Files:**
- Modify: `Docs/superpowers/reviews/2026-04-08-chat-module-review.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_service.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_orchestrator.py`
- Inspect: `tldw_Server_API/app/core/Chat/streaming_utils.py`
- Inspect: `tldw_Server_API/app/core/Chat/request_queue.py`
- Inspect: `tldw_Server_API/app/core/Chat/rate_limiter.py`
- Inspect: `tldw_Server_API/app/core/Chat/orchestrator/provider_resolution.py`
- Inspect: `tldw_Server_API/app/core/Chat/orchestrator/request_validation.py`
- Inspect: `tldw_Server_API/app/core/Chat/orchestrator/stream_execution.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_call_params.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_normalization.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_request_queue.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_queue_future.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_queue_estimate.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_queue_status_endpoint.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_queue_activity_endpoint.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py`

- [ ] **Step 1: Read the command, validation, completion, and queue entry points**

Run:

```bash
rg -n "async def list_chat_commands|async def validate_chat_dictionary|async def create_chat_completion|async def get_chat_queue_status|async def get_chat_queue_activity" \
  tldw_Server_API/app/api/v1/endpoints/chat.py
```

Expected: exact entry-point line numbers for the scoped route group. Read each function body plus the helper calls it delegates to before writing any findings.

- [ ] **Step 2: Read the shared core files behind those routes**

Run:

```bash
sed -n '1,260p' tldw_Server_API/app/core/Chat/chat_service.py
sed -n '1,260p' tldw_Server_API/app/core/Chat/chat_orchestrator.py
sed -n '1,260p' tldw_Server_API/app/core/Chat/streaming_utils.py
sed -n '1,260p' tldw_Server_API/app/core/Chat/request_queue.py
sed -n '1,260p' tldw_Server_API/app/core/Chat/rate_limiter.py
sed -n '1,220p' tldw_Server_API/app/core/Chat/orchestrator/provider_resolution.py
sed -n '1,220p' tldw_Server_API/app/core/Chat/orchestrator/request_validation.py
sed -n '1,220p' tldw_Server_API/app/core/Chat/orchestrator/stream_execution.py
```

Expected: enough implementation context to judge auth flow, provider resolution, queue backpressure, streaming lifecycle, and error mapping. If a finding depends on code below these ranges, extend the read around the exact symbol before recording it.

- [ ] **Step 3: Read the matching tests before running them**

Run:

```bash
sed -n '1,220p' tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py
sed -n '1,220p' tldw_Server_API/tests/Chat/unit/test_chat_service_call_params.py
sed -n '1,240p' tldw_Server_API/tests/Chat/unit/test_request_queue.py
sed -n '1,240p' tldw_Server_API/tests/Chat/unit/test_streaming_utils.py
sed -n '1,220p' tldw_Server_API/tests/Chat/integration/test_chat_endpoint.py
sed -n '1,220p' tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py
sed -n '1,220p' tldw_Server_API/tests/Chat_NEW/unit/test_queue_status_endpoint.py
sed -n '1,220p' tldw_Server_API/tests/Chat_NEW/unit/test_queue_activity_endpoint.py
```

Expected: a clear view of what the current tests actually assert, not just their filenames.

- [ ] **Step 4: Run the targeted unit tests for this route group**

Run:

```bash
source .venv/bin/activate
TEST_MODE=1 python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_call_params.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_normalization.py \
  tldw_Server_API/tests/Chat/unit/test_request_queue.py \
  tldw_Server_API/tests/Chat/unit/test_streaming_utils.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_queue_future.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_queue_estimate.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_queue_status_endpoint.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_queue_activity_endpoint.py \
  -v
```

Expected: tests collect and run. Record pass, fail, flaky, or blocked status in `Evidence Log`, and classify any failures as product defect, coverage gap, or environment issue.

- [ ] **Step 5: Run the targeted integration tests for this route group**

Run:

```bash
source .venv/bin/activate
TEST_MODE=1 python -m pytest \
  tldw_Server_API/tests/Chat/integration/test_chat_endpoint.py \
  tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py \
  tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py \
  -v
```

Expected: integration coverage for request normalization, routing, and streaming behavior. Record the outcome exactly; do not treat a pass as proof that the code path is safe if static review still shows a defect.

- [ ] **Step 6: Record route-group findings immediately**

Add findings to `Docs/superpowers/reviews/2026-04-08-chat-module-review.md` using this template:

```markdown
### High: Streaming branch can leave queue state inconsistent on early disconnect
- Files: `tldw_Server_API/app/api/v1/endpoints/chat.py:4471`, `tldw_Server_API/app/core/Chat/request_queue.py:1`
- Why it matters: A primary chat path can leak state or mis-report active work under disconnect pressure.
- Evidence: Static review of queue commit callback and disconnect path; targeted tests in `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py`
- Recommendation: Tighten queue cleanup invariants and add an assertion around disconnect-driven cleanup.
```

- [ ] **Step 7: Promote missing evidence to explicit coverage gaps**

Run:

```bash
rg -n "queue|stream|routing|disconnect" \
  tldw_Server_API/tests/Chat \
  tldw_Server_API/tests/Chat_NEW \
  tldw_Server_API/tests/Streaming
```

Expected: enough evidence to say either `covered` or `coverage gap`. Do not leave risky behavior implied but unclassified.

### Task 3: Audit Conversations, Share Links, Knowledge Save, Analytics, And RAG Context Routes

**Files:**
- Modify: `Docs/superpowers/reviews/2026-04-08-chat-module-review.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_history.py`
- Inspect: `tldw_Server_API/app/core/Chat/conversation_enrichment.py`
- Inspect: `tldw_Server_API/app/core/Chat/message_utils.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_conversations_scope.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_share_links_api.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_knowledge_save.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_conversations_tree_analytics.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_knowledge_save.py`
- Test: `tldw_Server_API/tests/e2e/test_workspace_chat_scope.py`

- [ ] **Step 1: Read the conversation, sharing, knowledge, and RAG-context entry points**

Run:

```bash
rg -n "async def save_chat_knowledge|async def list_chat_conversations|async def get_chat_conversation|async def update_chat_conversation|async def get_conversation_tree|async def create_conversation_share_link|async def list_conversation_share_links|async def revoke_conversation_share_link|async def resolve_conversation_share_token|async def get_chat_analytics|async def persist_rag_context|async def get_rag_context|async def get_messages_with_rag_context|async def get_conversation_citations" \
  tldw_Server_API/app/api/v1/endpoints/chat.py
```

Expected: exact line numbers for every scoped route in this group. Read each implementation before reviewing the supporting helpers.

- [ ] **Step 2: Read the shared helpers behind conversation and knowledge routes**

Run:

```bash
sed -n '1,240p' tldw_Server_API/app/core/Chat/chat_history.py
sed -n '1,240p' tldw_Server_API/app/core/Chat/conversation_enrichment.py
sed -n '1,240p' tldw_Server_API/app/core/Chat/message_utils.py
```

Expected: enough context to evaluate ownership checks, persisted metadata shape, share-token handling, and enrichment side effects.

- [ ] **Step 3: Read the direct tests and note missing route coverage**

Run:

```bash
sed -n '1,240p' tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py
sed -n '1,220p' tldw_Server_API/tests/Chat/unit/test_chat_conversations_scope.py
sed -n '1,220p' tldw_Server_API/tests/Chat/unit/test_chat_share_links_api.py
sed -n '1,220p' tldw_Server_API/tests/Chat/unit/test_chat_knowledge_save.py
sed -n '1,220p' tldw_Server_API/tests/Chat_NEW/integration/test_chat_conversations_tree_analytics.py
sed -n '1,220p' tldw_Server_API/tests/Chat_NEW/integration/test_chat_knowledge_save.py
sed -n '1,220p' tldw_Server_API/tests/e2e/test_workspace_chat_scope.py
rg -n "persist_rag_context|get_rag_context|get_messages_with_rag_context|get_conversation_citations" \
  tldw_Server_API/tests/Chat \
  tldw_Server_API/tests/Chat_NEW \
  tldw_Server_API/tests/e2e
```

Expected: either direct tests for the route or a concrete `no direct route test found` note in `Coverage Gaps`, especially for RAG-context and citations endpoints.

- [ ] **Step 4: Run the targeted tests for this route group**

Run:

```bash
source .venv/bin/activate
TEST_MODE=1 python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py \
  tldw_Server_API/tests/Chat/unit/test_chat_conversations_scope.py \
  tldw_Server_API/tests/Chat/unit/test_chat_share_links_api.py \
  tldw_Server_API/tests/Chat/unit/test_chat_knowledge_save.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_conversations_tree_analytics.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_knowledge_save.py \
  tldw_Server_API/tests/e2e/test_workspace_chat_scope.py \
  -v
```

Expected: direct evidence for conversation ownership, share-link handling, knowledge-save rollback behavior, analytics/tree routes, and workspace scoping. Record exact outcomes in the `Evidence Log`.

- [ ] **Step 5: Record findings and explicit no-test gaps**

Add new findings to `Docs/superpowers/reviews/2026-04-08-chat-module-review.md`, and add a `Coverage Gaps` item for any scoped route that still lacks a direct assertion file after the `rg` check in Step 3.

### Task 4: Audit Documents, Loop, Grammars, Dictionaries, And Workflows

**Files:**
- Modify: `Docs/superpowers/reviews/2026-04-08-chat-module-review.md`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_loop.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_grammars.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`
- Inspect: `tldw_Server_API/app/api/v1/endpoints/chat_workflows.py`
- Inspect: `tldw_Server_API/app/core/Chat/document_generator.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_loop_store.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_loop_approval.py`
- Inspect: `tldw_Server_API/app/core/Chat/chat_loop_engine.py`
- Inspect: `tldw_Server_API/app/core/Chat/validate_dictionary.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_document_generator.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_grammar_endpoints.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_workflows.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_dictionary_validate_endpoint.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_loop_endpoints.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_loop_dual_emit_compat.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_llamacpp_extensions_api.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_chat_loop_store.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_chat_loop_store_compaction.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_chat_loop_engine.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_chat_loop_approval.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_dictionary_validator.py`
- Test: `tldw_Server_API/tests/Chat_Workflows/test_chat_workflows_api.py`
- Test: `tldw_Server_API/tests/Chat_Workflows/test_chat_workflows_service.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_chat_workflows_deps.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_chat_workflows_permissions.py`

- [ ] **Step 1: Read the five adjacent endpoint modules**

Run:

```bash
sed -n '1,260p' tldw_Server_API/app/api/v1/endpoints/chat_documents.py
sed -n '1,220p' tldw_Server_API/app/api/v1/endpoints/chat_loop.py
sed -n '1,220p' tldw_Server_API/app/api/v1/endpoints/chat_grammars.py
sed -n '1,260p' tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py
sed -n '1,260p' tldw_Server_API/app/api/v1/endpoints/chat_workflows.py
```

Expected: a route-level view of auth, ownership, exception handling, streaming, and persistence behavior for each adjacent surface.

- [ ] **Step 2: Read the supporting core modules**

Run:

```bash
sed -n '1,260p' tldw_Server_API/app/core/Chat/document_generator.py
sed -n '1,220p' tldw_Server_API/app/core/Chat/chat_loop_store.py
sed -n '1,220p' tldw_Server_API/app/core/Chat/chat_loop_approval.py
sed -n '1,260p' tldw_Server_API/app/core/Chat/chat_loop_engine.py
sed -n '1,220p' tldw_Server_API/app/core/Chat/validate_dictionary.py
```

Expected: enough context to judge state lifetime, approval flow, document-generation contract handling, and dictionary validation logic.

- [ ] **Step 3: Read the direct tests before running them**

Run:

```bash
sed -n '1,220p' tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py
sed -n '1,220p' tldw_Server_API/tests/Chat/unit/test_chat_grammar_endpoints.py
sed -n '1,220p' tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py
sed -n '1,220p' tldw_Server_API/tests/Chat/unit/test_chat_workflows.py
sed -n '1,220p' tldw_Server_API/tests/Chat_NEW/integration/test_chat_loop_endpoints.py
sed -n '1,220p' tldw_Server_API/tests/Chat_NEW/integration/test_chat_loop_dual_emit_compat.py
sed -n '1,220p' tldw_Server_API/tests/Chat_NEW/integration/test_chat_dictionary_validate_endpoint.py
sed -n '1,220p' tldw_Server_API/tests/Chat_Workflows/test_chat_workflows_api.py
sed -n '1,220p' tldw_Server_API/tests/AuthNZ_Unit/test_chat_workflows_permissions.py
```

Expected: a concrete understanding of what the current tests cover for each adjacent surface, including auth wiring and compatibility expectations.

- [ ] **Step 4: Run the targeted tests for documents, grammars, and dictionaries**

Run:

```bash
source .venv/bin/activate
TEST_MODE=1 python -m pytest \
  tldw_Server_API/tests/Chat/integration/test_document_generation_endpoints.py \
  tldw_Server_API/tests/Chat/unit/test_document_generator.py \
  tldw_Server_API/tests/Chat/unit/test_chat_grammar_endpoints.py \
  tldw_Server_API/tests/Chat/unit/test_chat_dictionary_endpoints.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_dictionary_validate_endpoint.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_llamacpp_extensions_api.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_dictionary_validator.py \
  -v
```

Expected: route-level evidence for document-generation contracts, saved grammar behavior, dictionary CRUD/validation, and llama.cpp grammar bridging.

- [ ] **Step 5: Run the targeted tests for loop and workflows**

Run:

```bash
source .venv/bin/activate
TEST_MODE=1 python -m pytest \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_loop_endpoints.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_loop_dual_emit_compat.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_chat_loop_store.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_chat_loop_store_compaction.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_chat_loop_engine.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_chat_loop_approval.py \
  tldw_Server_API/tests/Chat/unit/test_chat_workflows.py \
  tldw_Server_API/tests/Chat_Workflows/test_chat_workflows_api.py \
  tldw_Server_API/tests/Chat_Workflows/test_chat_workflows_service.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_chat_workflows_deps.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_chat_workflows_permissions.py \
  -v
```

Expected: evidence for in-memory loop state behavior, approval flow, workflow route correctness, and permission wiring.

- [ ] **Step 6: Record findings for each adjacent surface separately**

Expected: `Docs/superpowers/reviews/2026-04-08-chat-module-review.md` contains distinct findings or `no issue found in this pass` notes for documents, loop, grammars, dictionaries, and workflows. Do not collapse these into one generic section.

### Task 5: Synthesize Cross-Cutting Risks And Prepare Delivery

**Files:**
- Modify: `Docs/superpowers/reviews/2026-04-08-chat-module-review.md`
- Inspect: `Docs/superpowers/specs/2026-04-08-chat-module-review-design.md`
- Inspect: `Docs/superpowers/plans/2026-04-08-chat-module-audit-execution-plan.md`

- [ ] **Step 1: Deduplicate and normalize the findings**

Review `Docs/superpowers/reviews/2026-04-08-chat-module-review.md` and merge duplicate findings that point to the same root cause. Keep the strongest file/line references and preserve route-specific evidence in the bullet text.

- [ ] **Step 2: Re-apply the severity rubric from the spec**

Run:

```bash
sed -n '94,126p' Docs/superpowers/specs/2026-04-08-chat-module-review-design.md
```

Expected: every finding in the review doc still matches the agreed `Critical` / `High` / `Medium` / `Low` rules.

- [ ] **Step 3: Verify evidence completeness**

Run:

```bash
rg -n "^## Evidence Log|^## Route Inventory|^## Critical|^## High|^## Medium|^## Low|^## Open Questions|^## Coverage Gaps" \
  Docs/superpowers/reviews/2026-04-08-chat-module-review.md
```

Expected: the review artifact contains all required sections. Every route group from Task 1 should have an `Evidence Log` row and either a finding, a `no issue found in this pass` note, or a `coverage gap` note.

- [ ] **Step 4: Prepare the user-facing review summary**

Copy the final findings into this structure for the terminal response:

```markdown
## Findings
### High: ...
- Files: `path:line`
- Risk: ...
- Evidence: ...

## Open Questions
- ...

## Coverage Gaps
- ...
```

Expected: the final terminal response can be written directly from the review doc without re-reading the whole codebase.

- [ ] **Step 5: Leave remediation as a separate follow-on**

Add this line to the bottom of `Docs/superpowers/reviews/2026-04-08-chat-module-review.md`:

```markdown
Remediation is intentionally out of scope for this audit. If the user wants fixes, create a separate remediation plan from the finalized findings set.
```

Expected: the review remains an audit artifact, not an implicit refactor plan.
