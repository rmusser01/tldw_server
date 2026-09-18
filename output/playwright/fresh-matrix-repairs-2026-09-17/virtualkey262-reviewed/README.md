# Verified PostgreSQL virtual-key SQL repair (UAT262)

Two existing PostgreSQL INSERT forms now separate positional parameters with ordinary whitespace. The SQL guard remains unchanged and fail-closed. Query columns, parameter order, casts, transaction and audit semantics are unchanged.

Baseline278 with original auth modules reproduces the failure, independently attributed in the sibling authscope257259-reviewed packet. Independent verification: original org/team caller and real text/JSONB stored-key readback2passed; full adjacent authentication96passed; direct guard2passed; zero skips. Only the maintained readback fixture later changed to generate its test password hash, resolving root-detected BanditB106. That affected PG test passed again; scoped review approved it. Root Ruff is clean and Bandit production has no findings; final test findings are only B101 assertions.

Author unretained44-test/static claims are not used as acceptance evidence. The original summary is preserved, and its unsupported SQLite failure claim is explicitly withdrawn in the correction report. Root full Bandit beforecleanup found B106 despite the original all-B101 claim; the corrected full scan has no B106 and no suppression. Failed runner/test-setup attempts remain documented.

The fixture-database runner uses the same owned cluster and official fixtures with a disposable label-hashed database name; it does not create a replacement cluster. Fixtures provision/clean up. Private configs, raw runtime logs and credentials are excluded. Receipt references outside this packet remain hash-only local evidence; no standalone replay guarantee. This finding came from actual integration tests, so no native browser-key workflow is claimed or required. Full UAT is not accepted.
