# Review ledger — `tldw_Server_API/app/core/Ingestion_Media_Processing/`

Module slug: `ingestion-media-processing`. Audit date: 2026-09-21.
Scope: 82 Python files, 55,842 LOC. Read-only audit; no source file was modified.

## Rules for this ledger

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with
  a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

| Stage | File | Subject |
| --- | --- | --- |
| 1 | `2026-09-21-stage1-inventory-and-survey.md` | Size/churn inventory, test location by import-grep, ADR constraints |
| 2 | `2026-09-21-stage2-ocr-backend-family.md` | The nine OCR backends: coercion, JSON salvage, temp-file lifetime, structural clones |
| 3 | `2026-09-21-stage3-error-policy-and-boundaries.md` | Per-file exception allowlists, HTTP/cancellation swallowing, core→api layering, result-envelope schema |
| 4 | `2026-09-21-stage4-efficiency-tests-synthesis.md` | Cost drivers, test-tree scatter, synthesis and proposed tasks |

Stages 4 and 5 of the standard RAG arc (data-source boundaries, composition) are collapsed into
stage 3 and stage 4: this module has one data-source boundary (`core/DB_Management/media_db`, reached
only through `persistence.py`) and no separate composition layer worth its own stage.

## Machine-generated sidecars

- `2026-09-21-stage1-source-inventory.txt` — `wc -l` per source file, descending.
- `2026-09-21-stage1-churn-baseline.txt` — commits per file, last 12 months.
- `2026-09-21-stage1-test-inventory.txt` — the 192 test files that import
  `core.Ingestion_Media_Processing`, found by import-grep (never by path).
- `2026-09-21-stage2-truthy-divergence.txt` — every truthy-coercion site in the module, grouped by the
  exact character set it accepts.
- `2026-09-21-stage3-exception-tuples.txt` — the 25 per-file `_*_NONCRITICAL_EXCEPTIONS` tuples, as
  written and reduced to their minimal (subclass-free) sets.

## Canonical links

- `Docs/Architecture.md` — operative layering rule ("Clients → FastAPI endpoints → Core domain
  services → Databases"; "no raw SQL in endpoints").
- `Docs/ADR/024-deepseek-ocr-local-transformers-backend.md` — binding on the `deepseek` OCR backend.
- `Docs/ADR/022-embeddings-api-and-media-pipeline.md` — binding on media-embedding pipeline ownership.
- `Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md` — binding on download/URL paths.
- Adjacent ledgers already covering neighbouring code, not re-reported here:
  `Docs/superpowers/reviews/web-scraping/`, `Docs/superpowers/reviews/web-scraping-ingest/`,
  `Docs/superpowers/reviews/db-management/`.
