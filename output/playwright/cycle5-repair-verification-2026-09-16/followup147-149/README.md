# PostgreSQL Chat, MCP and Collections startup repairs

147 corrects ChaCha keyword sequence ownership;148 replaces a SQLite-only MCP write probe and handles database failures locally;149 reuses existing column inspection for idempotent Collections backfills.

All three have independent required-PG verification:14719tests,14821tests,1497tests, each0skips. These scoped suites overlap other retained runs and are not summed. Production Bandit0; Ruff0new.

Actual normal startup on new r3 single/multi PostgreSQL profiles passes the affected boundaries: single Chat warm-up creates its default character, both MCP Media modules initialize, and both reading-digest schedulers start. Authenticated MCP media health reports healthy in both modes; digest-schedule reads200 in both. Source checkpoint records exact bytes, private profile roots and no application test flags/SQLite auth fallback. The first private r3 launcher generation had a path typo and exited before database creation; corrected launchers used the original repository.

Authenticated Chat/Notes reads then expose separate UAT150: cached dependency health sends SQLite PRAGMA to PostgreSQL. The failed HTTP results and sanitized database messages are retained, not discarded. Both r3 APIs were stopped after evidence; profiles/holders remain. Startup-only logs intentionally end at completed startup; full later redacted traces remain private for150 diagnosis. No browser/full-UAT or DB-level tenant-isolation pass is claimed.
