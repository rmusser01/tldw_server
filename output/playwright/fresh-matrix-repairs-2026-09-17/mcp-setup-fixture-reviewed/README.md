# Reviewed MCP setup and malformed-schema fixture repairs

TASK13260.193/UAT251: typed nullable setup queries and Boolean-compatible projections. Independent44focused tests pass. Native Save packs acceptance remains pending.

TASK13260.194/UAT252: one SQLite malformed-schema fixture now uses an explicitly closed fixture connection; all89 assertions and outside-function bytes preserved. Independent combined139passes/0skips includes actual PostgreSQL controls and guard negatives. Ruff4/Bandit5 existing findings unchanged; no new findings.

See author reports, source snapshots and independent review251/review252 reports for causal failures, exact commands and limitations. Native profiles are unchanged on86458; this packet makes no native acceptance claim.
