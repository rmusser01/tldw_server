# UAT262 test-fixture cleanup supplement

This supplement preserves the original UAT262 production approval.

## Scope review

The production repository SHA remains `302af943a627fc98fd994b3f194cd394c206851375afb68a7b413a9bb3fc8e8f`; no production, authority, scope, SQL-guard, or route behavior changed. The only correction in the maintained PostgreSQL readback fixture replaces the literal `password_hash="x"` with `uuid.uuid4().hex`.

This removes the new Bandit B106 test-fixture finding without suppression. It is a generated value used only as a hash-shaped test argument and does not weaken the virtual-key assertions or add a credential.

## Evidence review

- Final fixture SHA-256: `01cf5b01a614e6a68760c955846b792c2bbc195ab5d5d89a9db7d7865a4afe6b`.
- Affected official disposable-PostgreSQL readback: exit 0, **1 passed, 4 warnings**; redacted log SHA-256 `adb48a76aa54844e2faf42a4856572b85ac7c1bc9bff0960cfc847b1620026e0`.
- Ruff JSON is empty, SHA-256 `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`.
- Full Bandit JSON records 29 test-assert B101 findings and **zero B106**, with no suppressions; SHA-256 `b2ceacc1bbaaf853650d30347a505bd510e395d04b26b9cc45d07c5e8228811f`.

The report correctly withdraws the unretained SQLite failure claim and does not treat the author’s unreceipted 44-test/static output as acceptance evidence. The earlier independent review’s retained 2 PostgreSQL and 96 adjacent passes remain the acceptance basis.

## Verdict

**APPROVE the test-only cleanup.** It removes the newly introduced scanner finding while preserving the approved real PostgreSQL regression coverage.
