# Post-merge fresh-install UAT — 2026-09-18

**Preparation in progress; no full acceptance yet.** Frozen application revision: `3cff7962721a60b768464221c1f7fe2a8b25e4d5`, the verified PR2967 merge into dev. Working branch: `codex/postmerge-uat-20260918`. The requester explicitly authorized continuing this matrix with UAT261 open after the original retained PostgreSQL TestBot returned `BEEP BOOP` without its required final period.

Each cell uses a separate source archive, fresh app configuration/data, and a new browser session. Installed dependencies, the system browser and local model services are reused; this is fresh application state, not a clean-machine installation. PostgreSQL uses the repository's official fixtures and restricted direct runtime roles. Cells execute serially. Historical results do not count toward this run.

## Workflow outcomes

These are the existing12 named journeys recovered from frontend UAT/E2E and shared integration workflows. Historical A/B/C names describe coverage tiers; no authoritative mapping to three core loops was found. Every applicable boundary needs an explicit outcome and evidence. Actual image attachment, source citations, natural multi-user expiry, and reciprocal account isolation remain required.

| Row | Journey | SQLite single | SQLite multi | PostgreSQL single | PostgreSQL multi |
|---|---|---|---|---|---|
| 1 | Fresh setup, provider discovery and first real Chat | Pending | Pending | Pending | Pending |
| 2 | Auth reload, logout/disconnect, outage recovery and natural expiry | Pending | Pending | Pending | Pending |
| 3 | Two-turn Chat, real failure/Retry, image and visibility | Pending | Pending | Pending | Pending |
| 4 | File ingestion, search, cited QA and Media-to-Chat | Pending | Pending | Pending | Pending |
| 5 | Exact Playwright Wikipedia URL and grounded QA | Pending | Pending | Pending | Pending |
| 6 | Biology Note to exactly five cards and five-card Study | Pending | Pending | Pending | Pending |
| 7 | Pirate Prompt application, real ARRR reply and reload | Pending | Pending | Pending | Pending |
| 8 | Fresh TestBot, exact reply, character transitions and reload | Pending | Pending | Pending | Pending |
| 9 | Chat to Note/backlink/card; mixed Study, early End, practice and re-rate | Pending | Pending | Pending | Pending |
| 10 | Source analysis, Multi-Item Review, changed reanalysis and failure preservation | Pending | Pending | Pending | Pending |
| 11 | Permission-aware delete, dated Trash and exact restore | Pending | Pending | Pending | Pending |
| 12 | Reciprocal Alice/Bob metadata/content/draft/job isolation | N/A: single user | Pending | N/A: single user | Pending |

## Preparation and limits

- Run ID: `postmerge-full-20260918`. Planned API/UI ports: SQLite single18800/18880; SQLite multi18801/18881; PostgreSQL single18802/18882; PostgreSQL multi18803/18883.
- Source copy and launcher origin/link checks precede profile initialization. PostgreSQL databases are held by official fixtures; application roles have no superuser or BYPASSRLS privilege.
- Run PostgreSQL single first, then PostgreSQL multi, SQLite single and SQLite multi. Preserve all failures without editing frozen application source during the matrix.
- Exact source row5 remains `https://en.wikipedia.org/wiki/Playwright_(software)`. External denial blocks dependent steps; no substitute source counts as success.
- Row3 uses the actual128px PNG at `apps/tldw-frontend/public/icon/128.png`, SHA256 `dbdcfaab8f17290258a6ad0271e1c9bdf4adf93601b5bed467080f3ec5e0c2ff`. A capability guard alone does not certify image generation or persistence.
- UAT261 is open under its unchanged exact-output criterion. Its retained-profile post-merge check does not count as this fresh matrix's row8.
- Optional audio/MCP/evaluation/watchlist/extension coverage and clean-machine dependency installation are outside these12 rows.
- Generated browser captures stay local under ignored `.tmp/` or `output/playwright/`. The [running tracker](FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md) records every identified issue.
