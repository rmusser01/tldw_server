# Evaluations Module Audit — 2026-09-21

Module: `tldw_Server_API/app/core/Evaluations/` (70 `.py` files, 35,805 LOC)
Slug: `evaluations`
Baseline: working tree at branch `fix/chat-composer-question-mark`, HEAD `ea956eb1ed`.
Reviewer posture: **read-only for source**. No source file was edited during this audit.

## Relationship to the prior ledger

A previous review of this area lives at [`../evals-module/README.md`](../evals-module/README.md)
(single file, 708 lines, 8 slices). **That ledger is not superseded and was not modified.**
It covered the auth surface, run lifecycle, persistence path validation, CRUD/benchmark/dataset
endpoints, webhook ownership and embeddings-A/B idempotency — largely from the API side inward.

This audit is deliberately disjoint from it: it is a **maintainability + correctness audit driven by
size x churn**, focused on the axes the prior ledger did not work — duplication / utility
consolidation, scoring and aggregation math, batch-workload efficiency, and drift against the four
binding Evaluations ADRs. Every finding here is a new discovery; no seed cluster from the shared
briefing mapped to this module.

Reconciliation of the prior ledger's 16 findings (still-live vs already-addressed) is in
[`2026-09-21-stage4-synthesis.md`](2026-09-21-stage4-synthesis.md).

## Binding ADRs checked before asserting anything

- `Docs/ADR/012-evaluations-resource-id-prefixes.md` — `eval_` / `run_` / `dataset_` prefixes.
- `Docs/ADR/013-evaluations-deletion-lifecycle.md` — soft-delete evaluations, hard-delete datasets.
- `Docs/ADR/014-evaluations-openai-compatible-schemas.md` — separate request/response schemas,
  `object` field, **Unix `created` timestamps**, list wrappers.
- `Docs/ADR/015-evaluations-existing-evaluator-integration.md` — runner/service own orchestration;
  dedicated evaluator modules own scoring.

No finding in this ledger contradicts any of the four. Two findings assert **drift from** ADR-014
(EVAL-001) and ADR-015 (EVAL-004) and say so explicitly.

## Four rules for this ledger

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

The generic six-stage arc was **collapsed to four** because stages 3 and 4 of the template
(API/schema boundaries, data-source boundaries) were already worked in depth by
[`../evals-module/`](../evals-module/README.md); repeating them here would have padded the ledger
rather than added signal.

| Stage | File | Subject |
| --- | --- | --- |
| 1 | [`2026-09-21-stage1-inventory-and-evaluator-family.md`](2026-09-21-stage1-inventory-and-evaluator-family.md) | Size x churn inventory, test inventory by import-grep, the evaluator behaviour matrix |
| 2 | [`2026-09-21-stage2-scoring-and-aggregation.md`](2026-09-21-stage2-scoring-and-aggregation.md) | Scoring normalization, aggregation math, threshold/pass-rate correctness |
| 3 | [`2026-09-21-stage3-shared-helpers-and-efficiency.md`](2026-09-21-stage3-shared-helpers-and-efficiency.md) | Duplicated helpers, persistence/layering, batch-workload cost drivers |
| 4 | [`2026-09-21-stage4-synthesis.md`](2026-09-21-stage4-synthesis.md) | Prior-findings reconciliation, priority order, proposed Backlog tasks, not-covered list |

## Machine-generated sidecars

- `2026-09-21-stage1-source-inventory.txt` — every `.py` under the module with LOC, descending.
- `2026-09-21-stage1-churn-baseline.txt` — 12-month commit counts per file.
- `2026-09-21-stage1-test-inventory.txt` — the 109 test files reached by
  `grep -rl "core\.Evaluations" tldw_Server_API/tests`.

## Environment limitation affecting validation

The local venv used for this audit is missing `sklearn` and `hypothesis`, both of which are
**declared core dependencies** in `pyproject.toml` (`scikit-learn>=1.3.0` at :87, `hypothesis` at
:61). Eight test modules therefore fail to collect locally. This is an environment gap, **not** a
code defect, and is recorded as such in each stage's `## Validation Commands`. Where a claim could
not be executed locally it is labelled by import-grep reachability, never as measured coverage.
