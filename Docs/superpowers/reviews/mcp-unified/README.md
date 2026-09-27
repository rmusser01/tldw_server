# MCP Unified — maintainability + correctness review ledger

Module: `tldw_Server_API/app/core/MCP_unified/` (141,588 LOC across 253 `.py` files,
of which 90,042 LOC / 147 files are the in-app `tests/` subtree).
Slug: `mcp-unified`. Audit date: 2026-09-21. Reviewer stance: **read-only** — no source
file under `tldw_Server_API/` was modified for this review.

## Rules for this ledger

- Write findings before suggested actions in every stage file.
- Label uncertain items as probable risks or assumptions instead of confirmed defects.
- Keep later-stage summaries pointed back to the stage files; do not replace the per-stage record with a rolling summary.
- Keep the stage files as the durable review ledger for this audit.

## Stage order

| Stage | File | Covers |
| --- | --- | --- |
| 1 | `2026-09-21-stage1-architecture-and-inventory.md` | Size × churn survey, package topology, ADR constraints, what was prioritised and why |
| 2 | `2026-09-21-stage2-module-contract-and-sanitization.md` | `BaseModule` contract, the `sanitize_input` override cluster (C10), filesystem/glob correctness |
| 3 | `2026-09-21-stage3-authorization-and-config-boundaries.md` | `_is_admin` divergence, per-user DB attribution, settings/truthy coercion (C2), caching |
| 4 | `2026-09-21-stage4-tests-ci-and-package-boundary.md` | Split test trees, CI shard gap, the `apps/mcp-unified` extraction boundary, C1 |

Stages for "API/schema boundaries" and "data-source boundaries" in the standard arc are
**collapsed into stages 3 and 4**: this module has exactly one `core/ -> app/api/` import
(`modules/implementations/rag_module.py` -> `api/v1/schemas/rag_schemas_unified`, the mild
schema-only shape the briefing explicitly de-prioritises) and no raw SQL outside
`DB_Management/`. Padding those into standalone stages would add pages and no findings.

## Machine-generated sidecars

- `2026-09-21-stage1-source-inventory.txt` — non-test source files by LOC
- `2026-09-21-stage1-churn-baseline.txt` — 12-month commits per file
- `2026-09-21-stage1-test-inventory.txt` — both test trees, classified by import target

## Binding constraints checked before asserting

- `Docs/ADR/033-mcp-unified-stdio-contract-hardening.md` (Accepted) — supersedes ADR-032.
  Owns the strict stdio surface, revision profiles, bounded limits, integrity-protected
  pagination cursors, and the `GatewayCoreRuntime` boundary. **All of that lives in
  `apps/mcp-unified/src/mcp_unified/`, not in this module.** No finding here contradicts it;
  findings 4 and 12 sit alongside it (test placement, not protocol semantics).
- `Docs/ADR/042-browser-transport-admission-and-attestation.md` (Accepted) — browser
  transport admission. `browser_cdp_module.py` is in scope only for its settings accessors;
  its admission logic was not challenged.
- `CONTRIBUTING.md` owner-only paths: `tldw_Server_API/app/api/v1/**` and
  `tldw_Server_API/app/main.py` are owner-only. **`apps/mcp-unified/**` is NOT** on that list.
  No finding below requires an owner-only change; the `owner-only` field is `no` throughout.
