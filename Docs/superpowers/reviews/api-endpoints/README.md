# api-endpoints review ledger

Audit slug: `api-endpoints`. Target: `tldw_Server_API/app/api/v1/endpoints/` — 269 Python files,
247,943 LOC. Date: 2026-09-21. Read-only audit; no source file was modified.

**Everything recommended in this ledger is owner-only work.** `CONTRIBUTING.md` forbids non-owner PRs
from modifying `tldw_Server_API/app/api/v1/**`. Every finding is labelled `owner-only: yes` unless its
fix genuinely lands outside `api/v1/`.

## Rules for this ledger

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with
  a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

The canonical six-stage arc was collapsed to three substantive stages. Stages 2 and 4 of the standard
arc (core orchestration, data-source boundaries) have no separate subject here — the endpoints layer
*is* the boundary — so their content is folded into stage 2 and stage 3 below rather than padded out.

| Stage | File | Subject |
| --- | --- | --- |
| 1 | `2026-09-21-stage1-inventory-and-layering.md` | Size x churn inventory, raw-SQL layering violation, god-module shape |
| 2 | `2026-09-21-stage2-duplication-clusters.md` | Discord/Slack clones, datetime, truncation, error mapping, cursors, pagination |
| 3 | `2026-09-21-stage3-correctness-efficiency-tests.md` | Latent bugs, efficiency cost drivers, test reachability, synthesis |

Machine-generated inventories live in `.txt` sidecars next to these files:

- `2026-09-21-stage1-churn-baseline.txt`
- `2026-09-21-stage1-hotspot-sizes.txt`
- `2026-09-21-stage1-raw-sql-inventory.txt`
- `2026-09-21-stage2-c9-clone-measurements.txt`
- `2026-09-21-stage3-test-inventory.txt`

## Inherited from adjacent ledgers (not re-reported here)

Seven adjacent ledgers were read in full before this audit began. What each already owns:

- **`api-pagination/`** (2026-04-25) — whole-repo pagination scan: 411 pagination-like query params,
  173 schema classes, six coexisting param dialects, a canonical metadata contract, and a proposed
  `schemas/pagination.py`. It owns the *contract rule* ("`page` aliases are one-based and convert to
  zero-based offset") but **never enumerates call sites**. `watchlists` appears twice as inventory only
  and has no findings of its own. Finding 13 below is filed as the missing per-site enumeration, not as
  a rediscovery of the contract.
- **`api-response-envelope/`** (2026-04-25) — 2,188 route decorators classified; success/error envelope
  contract; `main.py`'s `{"detail": ...}` behaviour. Owns the *convergence design* for response and
  error shape; never enumerates endpoints that hand-build error dicts. Finding 9 is filed as the
  enumeration, cross-referenced to their contract.
- **`auth-dependencies/`** (2026-04-25) — 14 auth-dependency patterns counted across 210 modules; 99
  modules on legacy user deps; 57 with manual admin checks. Explicitly pre-empts "duplicate auth checks
  are redundant" (defence in depth is intentional). No auth-dependency-duplication finding is raised
  here. Their caveat is honoured: static counts are a migration map, not proof of misconfiguration.
- **`characters-backend/`** (2026-03-23, rebaselined 2026-04-15) — owns `characters_endpoint.py`,
  `character_chat_sessions.py`, `character_messages.py`. Live findings there: ChaChaNotes bootstrap
  503 poisoning, world-book `character_id` mislabelling, avatar-less version history, non-atomic
  message-cap checks, hybrid-search `total` from the prefetch window. Finding 8 below is on a different
  code path (persona preview truncation, `:5651`/`:5936`) not covered by that review.
- **`moderation-backend/`** (2026-04-07) — owns `moderation.py` and the moderation call sites in
  `chat.py` only. Its `chat.py` findings (governance `chat_type` propagation at `:2996`, `:3852`) are
  distinct from findings 1, 5, 6 and 15 here.
- **`web-scraping/`** (2026-04-16) — zero endpoint files; outbound-policy migration record only.
  Nothing inherited.
- **`shared-api-client/`** (2026-04-17) — frontend only (`apps/packages/ui/src/services/tldw/`).
  Nothing inherited.

## Scope not covered

See the "Not Covered" section at the end of
`2026-09-21-stage3-correctness-efficiency-tests.md`.
