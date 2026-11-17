# Authoritative quota fixture contracts — TASK-13622

Quota-switch tests now explicitly declare empty native retained usage for their synthetic user. The production native ledger read remains unchanged and fail-closed. Four controls require unavailable native usage to raise with quota enforcement on or off, through both plain and combined checks.

The deprecated-email characterization harness now supplies its storage service dependency for exactly synthetic user7. It returns the characterized quota5120 and usage0, independent of the fake email transaction. The matrix asserts one lookup on successful updates and none on rejected paths, while retaining all auth, response, deprecation, email-write and commit-exit expectations.

The original reproduction was4failed27passed. A patch initially placed in the v1 caller was moved to the correct deprecated-email harness before final verification. Both complete suites now pass35/35; nearby quota resolver and profile error mapping pass24/24. Black and Ruff pass. Touched-test Bandit excludes intentional test assertions (B101); the two B608 static SQL-query matcher findings also exist in the committed baseline, with zero new findings. Normal hooks are recorded in the task. Independent bounded review is clear.

ADR assessment: no new ADR; these test dependencies match current authoritative read contracts. No production auth, quota, database or transaction behavior was changed. Docker/PostgreSQL physical qualification, remaining whole-PR failures, final review, rebase and merge remain separate.

PR: https://github.com/rmusser01/tldw_server/pull/3229
