# RAG Review Ledger

This directory holds the staged architecture and maintainability review for the RAG subsystem.

## Stage Order

1. `2026-04-07-stage1-architecture-survey-and-inventory.md`
2. `2026-04-07-stage2-unified-pipeline-orchestration.md`
3. `2026-04-07-stage3-api-schema-and-request-boundaries.md`
4. `2026-04-07-stage4-retrieval-boundaries-and-data-sources.md`
5. `2026-04-07-stage5-reranking-and-post-retrieval-composition.md`
6. `2026-04-07-stage6-test-gaps-and-synthesis.md`

### 2026-09-21 extension pass

The 2026-04-07 stages above run the three-axis Codeslop rubric with performance excluded. The 2026-09-21 pass extends them under the five-axis rubric (duplication, encapsulation, sequential coupling, correctness, efficiency). It does not supersede or edit the 2026-04-07 files; it re-verifies them and adds the two axes they did not carry.

7. `2026-09-21-stage1-reconciliation-and-inventory.md`
8. `2026-09-21-stage2-efficiency-and-correctness.md`
9. `2026-09-21-stage3-duplication-and-test-topology.md`

## Canonical Paths

- [Stage 1](./2026-04-07-stage1-architecture-survey-and-inventory.md)
- [Stage 2](./2026-04-07-stage2-unified-pipeline-orchestration.md)
- [Stage 3](./2026-04-07-stage3-api-schema-and-request-boundaries.md)
- [Stage 4](./2026-04-07-stage4-retrieval-boundaries-and-data-sources.md)
- [Stage 5](./2026-04-07-stage5-reranking-and-post-retrieval-composition.md)
- [Stage 6](./2026-04-07-stage6-test-gaps-and-synthesis.md)
- [2026-09-21 Stage 1 — reconciliation and inventory](./2026-09-21-stage1-reconciliation-and-inventory.md)
- [2026-09-21 Stage 2 — efficiency and correctness](./2026-09-21-stage2-efficiency-and-correctness.md)
- [2026-09-21 Stage 3 — duplication and test topology](./2026-09-21-stage3-duplication-and-test-topology.md)
- Sidecars: [`2026-09-21-stage1-hotspot-sizes.txt`](./2026-09-21-stage1-hotspot-sizes.txt), [`2026-09-21-stage1-churn-baseline.txt`](./2026-09-21-stage1-churn-baseline.txt)

The prior-findings reconciliation (which 2026-04-07 findings are still live, which are addressed) lives in `2026-09-21-stage1-reconciliation-and-inventory.md`.

## Review Rules

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with a rolling summary.
- Keep the stage files as the durable review ledger for this audit.
