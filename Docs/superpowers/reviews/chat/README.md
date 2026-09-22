# Review ledger — `tldw_Server_API/app/core/Chat/`

Module slug: `chat`. Audit date: 2026-09-21. Scope: 46 `.py` files, 25,698 LOC.
Part of the repo-wide core-module maintainability + correctness audit.
This review was **read-only**: no source file under `tldw_Server_API/` was modified.

## Four rules this ledger follows

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with
  a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

The canonical six-stage arc was collapsed to three substantive stages; the API/schema-boundary and
data-source-boundary stages are folded into stage 3 because `core/Chat` has exactly one API boundary
(`api/v1/endpoints/chat.py`) and one storage boundary (`CharactersRAGDB` / `chacha`), and padding them
apart would have produced two thin files instead of one complete one.

| Stage | File | Covers |
| --- | --- | --- |
| 1 | `2026-09-21-stage1-architecture-inventory.md` | Module map, size x churn, recent defect history, what is live vs dead |
| 2 | `2026-09-21-stage2-orchestration-streaming.md` | `chat_orchestrator.py`, `chat_service.py`, `streaming_utils.py` — correctness, duplication, efficiency |
| 3 | `2026-09-21-stage3-boundaries-data-tests.md` | core→api layering, storage ownership, authorization context, test-tree assessment, synthesis |

## Machine-generated inventories (`.txt` sidecars)

- `2026-09-21-stage1-source-inventory.txt`
- `2026-09-21-stage1-churn-baseline.txt`
- `2026-09-21-stage2-sse-inline-sites.txt`
- `2026-09-21-stage3-test-inventory.txt`

## Binding documents checked before asserting

- `Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md` — adapter registry is the integration
  boundary; streams normalize to OpenAI-style `data: ...` chunks terminated by one `[DONE]`; ADR
  explicitly leaves "sync/async provider call paths are unified" open for a future decision.
  Finding `chat-4` is consistent with, not contrary to, that ADR.
- `Docs/ADR/009-quick-chat-docs-assistant-modes.md` — Quick Chat modal/UI contract. No finding in this
  ledger touches it.
- `Docs/Architecture.md` — "keep storage access centralized via `core/DB_Management/`", "no raw SQL in
  endpoints". Basis for finding `chat-5`.
- `CONTRIBUTING.md` — `tldw_Server_API/app/api/v1/**` is owner-only; findings touching it are labelled.

## Finding index

Findings are numbered `chat-1` … `chat-14` and are ordered by severity in
`2026-09-21-stage3-boundaries-data-tests.md#synthesis`. Each finding lives in the stage where its
evidence was gathered.
