# Review ledger — `tldw_Server_API/app/core/LLM_Calls/` (slug: `llm-calls`)

Audit date: 2026-09-21. Module size: 21,841 LOC across 64 `.py` files (see
`2026-09-21-stage1-source-inventory.txt`). Read-only maintainability + correctness
audit; no source file was modified.

## Four rules for this ledger

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with
  a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

| Stage | File | Covers |
| --- | --- | --- |
| 1 | `2026-09-21-stage1-architecture-and-inventory.md` | Module shape, churn, test reachability, transport layering survey |
| 2 | `2026-09-21-stage2-provider-adapter-family.md` | The 16-adapter family: transport, streaming, timeout/base-url, feature flags |
| 3 | `2026-09-21-stage3-error-and-token-boundaries.md` | Error→HTTP status mapping (C8), tiktoken fallback (C6), scalar coercion (C2) |

Stages 4–6 of the canonical RAG arc are collapsed into stages 1–3: `LLM_Calls` owns no
data-source boundary of its own (it holds no DB access — confirmed, zero `sqlite3`/`DB_Management`
imports), its API/schema boundary is a single consumer (`Chat/chat_service.py` plus four direct
registry callers), and composition and test-gap analysis are recorded inline in each stage's
`## Tests Reviewed` section rather than padded into separate files.

## Canonical links

- `Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md` — binding on routing, overrides, SSE contract
- `Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md` — binding on outbound egress
- `Docs/ADR/030-configured-local-llm-egress-policy.md` — binding on configured-local endpoints
- `tldw_Server_API/app/core/LLM_Calls/README.md` — module README (exists; ADR-025 was backfilled from it)
- `Docs/Architecture.md` — operative layering document

## Machine-generated sidecars

- `2026-09-21-stage1-source-inventory.txt` — every file with LOC
- `2026-09-21-stage1-churn-baseline.txt` — 12-month commit counts per file
- `2026-09-21-stage1-test-inventory.txt` — the 192 test files reaching this module by import-grep
- `2026-09-21-stage2-behavior-matrix.txt` — per-adapter transport / streaming / timeout / truthy matrices
