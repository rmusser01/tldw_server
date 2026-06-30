# RAG Architecture Review

## RAG Review Record

### Thin Endpoint Follow-Up

- Standard RAG evidence coordination now occurs inside `app/core/RAG/rag_service/unified_pipeline.py`.
- `app/api/v1/endpoints/rag_unified.py` remains an HTTP adapter: validation, dependency resolution, delegation, response mapping, and streaming framing.
- Current-`dev` RAG fixes were reconciled before this refactor: analytics backend bootstrap, shared outbound policy handling, structure-DB fallback, and rerank debug snapshot gating.

### Follow-Up Verification Record

- Phase 0 reconciled current `dev` RAG fixes before refactoring.
- Phase 1 moved standard evidence coordination and canonical contract threading into core.
- Phase 2 moved agentic config/toolbox/structure-DB fallback ownership into `agentic_execution.py`.
- Phase 3 moved streaming event generation into `streaming_executor.py`.
- Endpoint responsibilities are now validation, dependency resolution, core delegation, response mapping, and HTTP stream framing.
