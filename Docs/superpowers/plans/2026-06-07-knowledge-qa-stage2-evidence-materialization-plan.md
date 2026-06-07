# Knowledge QA Stage 2 Evidence Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every Knowledge QA source row has inspectable evidence or a specific unavailable reason, plus stable source, chunk, and excerpt identifiers.

**Architecture:** Add backend response contract fields through the RAG result mapper and expose them through existing frontend `RagResult` rendering. Keep this stage focused on materializing evidence; citation validity enforcement happens in TASK-2279.4.

**Tech Stack:** Python, FastAPI/Pydantic, RAG response mapping, TypeScript, React, Pytest, Vitest.

**Backlog Task:** TASK-2279.3

---

## Boundaries

- Do not enforce citation validity in this stage.
- Do not change search ranking semantics except to preserve evidence metadata.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Modify: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/result_model.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/response_mapping.py`
- Create: `tldw_Server_API/tests/RAG/test_knowledge_evidence_materialization.py`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/types.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SourceCard.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SourceList.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SourceViewerModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/evidence/EvidenceRail.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SearchDetailsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceCard.behavior.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SourceList.viewer.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SearchDetailsPanel.test.tsx`

## Task 1: Add Backend Evidence Contract Tests

- [ ] **Step 1: Write failing response mapper tests**

Create `test_knowledge_evidence_materialization.py`:

```python
from tldw_Server_API.app.core.RAG.rag_service.response_mapping import _normalize_documents


def test_normalize_documents_preserves_knowledge_evidence_fields():
    doc = {
        "id": "media:42:chunk:7",
        "content": "Visible matched excerpt.",
        "metadata": {
            "title": "Grounded QA checklist",
            "source_type": "media_db",
            "source_id": "42",
            "chunk_id": "7",
            "evidence_origin": "local_library",
            "source_status": "searched",
            "unavailable_reason": None,
        },
        "score": 0.91,
    }

    [normalized] = _normalize_documents([doc])

    assert normalized["id"] == "media:42:chunk:7"
    assert normalized["content"] == "Visible matched excerpt."
    assert normalized["metadata"]["source_id"] == "42"
    assert normalized["metadata"]["chunk_id"] == "7"
    assert normalized["metadata"]["evidence_origin"] == "local_library"
    assert normalized["metadata"]["source_status"] == "searched"
```

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_knowledge_evidence_materialization.py -v
```

Expected: fail until mapping preserves all fields or handles non-dict document shapes consistently.

- [ ] **Step 2: Add unavailable reason test**

Add a test where `content` is empty and metadata contains `unavailable_reason="deleted_or_unavailable"`. Expected response still includes the unavailable reason.

## Task 2: Extend Backend Mapping Safely

- [ ] **Step 1: Update schema examples**

In `rag_schemas_unified.py`, update `UnifiedRAGResponse` examples to show documents with:

- `metadata.source_id`
- `metadata.source_type`
- `metadata.chunk_id`
- `metadata.evidence_origin`
- `metadata.source_status`
- `metadata.unavailable_reason`

- [ ] **Step 2: Normalize evidence fields**

In `response_mapping.py`, keep `_normalize_documents()` backwards compatible but ensure dict, attr, and wrapped `.document` shapes preserve evidence metadata.

- [ ] **Step 3: Run backend tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_knowledge_evidence_materialization.py tldw_Server_API/tests/RAG/test_unified_schema_and_pipeline.py -v
```

Expected: all tests pass.

## Task 3: Add Frontend Evidence Types And Rendering

- [ ] **Step 1: Write failing frontend tests**

Update `SourceCard.behavior.test.tsx` and `SourceList.viewer.test.tsx` to assert:

- matched excerpt is visible when full content is unavailable
- specific unavailable reason is visible
- source id and chunk id are available as data attributes or stable open targets
- web fallback origin is labeled when present

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/SourceCard.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/SourceList.viewer.test.tsx
```

Expected: fail before UI support.

- [ ] **Step 2: Extend `RagResult`**

Modify `types.ts`:

```ts
export type EvidenceOrigin = "local_library" | "web_fallback" | "mixed" | "unknown_origin"

export type RagResult = {
  // existing fields
  sourceId?: string
  sourceType?: string
  chunkId?: string
  evidenceOrigin?: EvidenceOrigin
  sourceStatus?: string
  unavailableReason?: string | null
}
```

- [ ] **Step 3: Render evidence details**

Update `SourceCard.tsx`, `SourceList.tsx`, `SourceViewerModal.tsx`, `EvidenceRail.tsx`, and `SearchDetailsPanel.tsx` to prefer:

1. excerpt
2. content/text/chunk
3. unavailable reason with recovery copy

Never render only `Full source content is unavailable` when an excerpt or reason exists.

## Task 4: Verify

- [ ] **Step 1: Run focused frontend tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/SourceCard.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/SourceList.viewer.test.tsx src/components/Option/KnowledgeQA/__tests__/SearchDetailsPanel.test.tsx
```

- [ ] **Step 2: Run backend Bandit on touched paths**

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py tldw_Server_API/app/core/RAG/rag_service/result_model.py tldw_Server_API/app/core/RAG/rag_service/response_mapping.py -f json -o /tmp/bandit_knowledge_qa_evidence.json
```

Expected: no new findings in touched code.

- [ ] **Step 3: Run diff hygiene and commit**

```bash
git diff --check -- tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py tldw_Server_API/app/core/RAG/rag_service apps/packages/ui/src/components/Option/KnowledgeQA
git add tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py tldw_Server_API/app/core/RAG/rag_service apps/packages/ui/src/components/Option/KnowledgeQA tldw_Server_API/tests/RAG/test_knowledge_evidence_materialization.py "backlog/tasks/task-2279.3 - Materialize-Knowledge-QA-evidence-excerpts-and-source-identifiers.md"
git commit -m "feat: materialize knowledge qa evidence"
```
