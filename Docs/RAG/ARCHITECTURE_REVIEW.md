# RAG Architecture Review

## RAG Review Record

### Thin Endpoint Follow-Up

- Standard RAG evidence coordination now occurs inside `app/core/RAG/rag_service/unified_pipeline.py`.
- `app/api/v1/endpoints/rag_unified.py` remains an HTTP adapter: validation, dependency resolution, delegation, response mapping, and streaming framing.
- Current-`dev` RAG fixes were reconciled before this refactor: analytics backend bootstrap, shared outbound policy handling, structure-DB fallback, and rerank debug snapshot gating.
