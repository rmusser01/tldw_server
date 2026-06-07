# Knowledge QA Stage 1B Citation Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce Knowledge QA citation validity, weak-evidence abstention, degraded-answer classification, and web-fallback origin labeling.

**Architecture:** Add a backend trust classifier around RAG response mapping and mirror its fields in frontend normalization. A normal answer may be `cited_answer` only when citations map to returned sources with inspectable evidence. Weak, empty, uncitable, unavailable, or zero-relevance evidence must abstain or degrade.

**Tech Stack:** Python, FastAPI/Pydantic, RAG response mapping, TypeScript, React, Pytest, Vitest.

**Backlog Task:** TASK-2278.4

---

## Boundaries

- Depends on TASK-2278.3 evidence materialization.
- Do not require semantic claim-to-source adjudication in this release slice.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Create: `tldw_Server_API/app/core/RAG/rag_service/trust_contracts.py`
- Create: `tldw_Server_API/tests/RAG/test_knowledge_trust_contracts.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/response_mapping.py`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/trustState.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/panels/NoResultsRecovery.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/trustState.test.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.source-status.test.tsx`

## Task 1: Add Backend Trust Contract Tests

- [ ] **Step 1: Write failing tests**

Create `test_knowledge_trust_contracts.py`:

```python
from tldw_Server_API.app.core.RAG.rag_service.trust_contracts import classify_knowledge_answer_trust


def test_uncited_answer_is_degraded():
    trust = classify_knowledge_answer_trust(
        answer="Unsupported answer.",
        documents=[{"id": "doc-1", "content": "Evidence", "metadata": {"source_status": "searched"}}],
        citations=[],
        web_fallback_used=False,
    )
    assert trust["state"] == "uncited_degraded_answer"


def test_cited_answer_requires_inspectable_evidence():
    trust = classify_knowledge_answer_trust(
        answer="Grounded answer [1].",
        documents=[{"id": "doc-1", "content": "", "metadata": {"source_status": "unavailable"}}],
        citations=[{"index": 1, "document_id": "doc-1"}],
        web_fallback_used=False,
    )
    assert trust["state"] == "no_answer_insufficient_evidence"
    assert trust["reason_codes"] == ["missing_inspectable_evidence"]
```

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_knowledge_trust_contracts.py -v
```

Expected: fail because trust classifier does not exist.

- [ ] **Step 2: Add web fallback origin test**

Assert a response with `web_fallback_used=True` returns `evidence_origin="web_fallback"` or `mixed` and never labels external evidence as local library evidence.

## Task 2: Implement Backend Trust Classifier

- [ ] **Step 1: Create `trust_contracts.py`**

Implement a small deterministic classifier:

```python
def classify_knowledge_answer_trust(*, answer, documents, citations, web_fallback_used):
    if not documents and not web_fallback_used:
        return {"state": "no_results", "reason_codes": ["no_evidence"], "evidence_origin": "local_library"}
    if _has_weak_or_missing_evidence(documents):
        return {"state": "no_answer_insufficient_evidence", "reason_codes": ["missing_inspectable_evidence"], "evidence_origin": _origin(documents, web_fallback_used)}
    if answer and not citations:
        return {"state": "uncited_degraded_answer", "reason_codes": ["missing_citations"], "evidence_origin": _origin(documents, web_fallback_used)}
    if answer and _citations_map_to_documents(citations, documents):
        return {"state": "cited_answer", "reason_codes": [], "evidence_origin": _origin(documents, web_fallback_used)}
    return {"state": "unknown_trust", "reason_codes": ["unclassified"], "evidence_origin": "unknown_origin"}
```

Keep helper functions private and deterministic.

- [ ] **Step 2: Attach trust metadata in response mapping**

Modify `response_mapping.py` to add metadata such as:

```python
metadata["knowledge_trust"] = {
    "state": trust["state"],
    "reason_codes": trust["reason_codes"],
    "evidence_origin": trust["evidence_origin"],
}
```

Do not remove existing fields.

- [ ] **Step 3: Update schema examples**

In `UnifiedRAGResponse`, document `metadata.knowledge_trust`.

## Task 3: Mirror Trust Metadata In Frontend

- [ ] **Step 1: Extend `trustState.ts` tests**

Add tests that backend `metadata.knowledge_trust.state` takes precedence over local heuristic classification when present, except transport failure and extension sync failure still win.

- [ ] **Step 2: Parse backend trust metadata**

Update `KnowledgeQAProvider.tsx` and `trustState.ts` to read the backend trust metadata from RAG responses.

- [ ] **Step 3: Render weak-evidence reasons**

Update `AnswerPanel.tsx` and `NoResultsRecovery.tsx` so `missing_citations`, `missing_inspectable_evidence`, `low_relevance`, and `web_fallback_used` have visible but concise recovery copy.

## Task 4: Verify

- [ ] **Step 1: Run backend tests**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/RAG/test_knowledge_trust_contracts.py tldw_Server_API/tests/RAG/test_knowledge_evidence_materialization.py -v
```

- [ ] **Step 2: Run frontend tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/KnowledgeQA/__tests__/trustState.test.ts src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx src/components/Option/KnowledgeQA/__tests__/NoResultsRecovery.source-status.test.tsx
```

- [ ] **Step 3: Run Bandit**

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/RAG/rag_service/trust_contracts.py tldw_Server_API/app/core/RAG/rag_service/response_mapping.py -f json -o /tmp/bandit_knowledge_qa_trust.json
```

- [ ] **Step 4: Commit**

```bash
git add tldw_Server_API/app/core/RAG/rag_service/trust_contracts.py tldw_Server_API/app/core/RAG/rag_service/response_mapping.py tldw_Server_API/app/api/v1/schemas/rag_schemas_unified.py tldw_Server_API/tests/RAG/test_knowledge_trust_contracts.py apps/packages/ui/src/components/Option/KnowledgeQA "backlog/tasks/task-2278.4 - Enforce-Knowledge-QA-citation-validity-and-abstention.md"
git commit -m "feat: enforce knowledge qa trust contract"
```
