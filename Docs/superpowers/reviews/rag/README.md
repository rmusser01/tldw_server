# RAG Review Ledger

This directory holds the staged architecture and maintainability review for the RAG subsystem.

## Stage Order

1. `2026-04-07-stage1-architecture-survey-and-inventory.md`
2. `2026-04-07-stage2-unified-pipeline-orchestration.md`
3. `2026-04-07-stage3-api-schema-and-request-boundaries.md`
4. `2026-04-07-stage4-retrieval-boundaries-and-data-sources.md`
5. `2026-04-07-stage5-reranking-and-post-retrieval-composition.md`
6. `2026-04-07-stage6-test-gaps-and-synthesis.md`

## Canonical Paths

- [Stage 1](./2026-04-07-stage1-architecture-survey-and-inventory.md)
- [Stage 2](./2026-04-07-stage2-unified-pipeline-orchestration.md)
- [Stage 3](./2026-04-07-stage3-api-schema-and-request-boundaries.md)
- [Stage 4](./2026-04-07-stage4-retrieval-boundaries-and-data-sources.md)
- [Stage 5](./2026-04-07-stage5-reranking-and-post-retrieval-composition.md)
- [Stage 6](./2026-04-07-stage6-test-gaps-and-synthesis.md)

## Review Rules

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with a rolling summary.
- Keep the stage files as the durable review ledger for this audit.
