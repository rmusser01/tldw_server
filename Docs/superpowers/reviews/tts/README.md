# TTS module review ledger — 2026-09-21

Module under review: `tldw_Server_API/app/core/TTS/` (72 files, 42,067 LOC).
Slug: `tts`. Part of the repo-wide core-module duplication + correctness audit dated 2026-09-21.

## Rules for this ledger

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

The standard six-stage arc was collapsed to three substantive stages. TTS has no data-source
boundary of its own (no DB tables, no raw SQL — verified, see stage 3) and its API/schema
boundary is small enough to fold into stage 3, so stages 3-6 of the generic arc would have been
padding.

| Stage | File | Subject |
| --- | --- | --- |
| 1 | `2026-09-21-stage1-architecture-survey.md` | Size × churn map, module layout, ADR constraints, what is and is not in scope |
| 2 | `2026-09-21-stage2-adapter-family.md` | The 25-adapter family: shared concerns, which copy is correct, where they disagree |
| 3 | `2026-09-21-stage3-orchestration-boundaries-tests.md` | `tts_service_v2` orchestration, API/schema boundary, efficiency, test-tree split, synthesis |

## Machine-generated inventories (sidecars)

- `2026-09-21-stage1-source-inventory.txt` — every `.py` under the module with LOC
- `2026-09-21-stage1-churn-baseline.txt` — 12-month commit counts per file
- `2026-09-21-stage2-adapter-matrix.txt` — per-adapter shared-concern matrix (concerns A–H)

## Canonical links

- Binding ADR for this module: `Docs/ADR/011-audio-api-semantics.md` (model-first routing,
  structured streaming errors by default, `return_download_link` non-streaming only,
  cooldown-driven adapter init retry). Every finding here was checked against it.
- Operative architecture: `Docs/Architecture.md`.
- Module README: `tldw_Server_API/app/core/TTS/README.md`.
- Decomposition template for god modules: the shipped `core/DB_Management/media_db/` package split.
