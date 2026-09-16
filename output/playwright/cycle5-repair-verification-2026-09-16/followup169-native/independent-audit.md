# UAT169 natural scheduler acceptance — PASS

Task: TASK-13260.106. Independent auditor: source013_diagnosis.
Committed HEAD at capture: 1d790537974aadcce71c78b2c5b2329c4cf6007f.

## Source and startup

The working scheduler.py, committed HEAD version and reviewed followup169 version all match SHA256 61cf62e93b392b0cf61a0dd74ddb70fce62e7f3518fa6ae7290d4b10d394441a. The reviewed fix converts the explicitly UTC PostgreSQL audit-log cutoff to the table's naive-UTC timestamp parameter. Registration uses the real scheduler's five-minute interval.

The native PGmulti log records PostgreSQL pool initialization/connection, server startup and auth scheduler registration. The process-start log is 2026-09-16 15:44:32.646 PDT (22:44:32.646Z); auth scheduler startup is 15:44:33.842 PDT (22:44:33.842Z). Root's launcher-dispatch time 22:44:24Z is distinct from these server startup entries.

## Natural scheduled executions

- 2026-09-16 15:49:33.840 → 2026-09-16 15:49:33.843 PDT; 2026-09-16T22:49:33.840Z → 2026-09-16T22:49:33.843Z; log lines 1849–1850.
- 2026-09-16 15:54:33.852 → 2026-09-16 15:54:33.854 PDT; 2026-09-16T22:54:33.852Z → 2026-09-16T22:54:33.854Z; log lines 3224–3225.

No scheduled invocation was forced by this audit. Both run-start and successful-completion entries are present for the real named authentication monitor. Importantly, the implementation catches noncritical exceptions and logs Failed to monitor auth failures, so a scheduler success alone is insufficient. I checked that explicit caught-failure text, method traceback references, named-job exceptions/misses and naive/aware timestamp errors throughout the captured whole-startup log prefix: 0 matches. No matching failure appears after the successful runs through 2026-09-16 15:56:50.473 PDT (2026-09-16T22:56:50.473Z).

## Durable evidence

sanitized-receipt.json retains only normalized event categories, exact times, original line numbers, source/review hashes and a private-log prefix digest. The private log is live and may grow: captured prefix 794618 bytes, SHA256 937a70b02ca7f111594b617e25ef66d81bf3d1f23457f3af6cdde07a5d284eb0. No raw log lines or environment/configuration/auth data were copied or printed by this audit. This sanitized packet is ready for retention.

## Limits

This establishes live natural scheduling and no matching swallowed monitor failure within the recorded observation window. It does not establish future uptime, other scheduled jobs, a live alert condition or external alert delivery. The source match is filesystem/startup evidence, not an in-memory attestation. Actual threshold, cutoff boundary and redaction behavior are covered separately by the reviewed real-PostgreSQL/SQLite tests in followup169. No browser/runtime/database/source/task mutation occurred; only this private evidence packet was written.
