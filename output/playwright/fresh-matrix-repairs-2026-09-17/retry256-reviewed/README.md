# Reviewed ordinary Chat retry identity repair (UAT256)

A new question identical to the last answered question could receive a model-selection error before persistence, then fail Retry with an answered-turn conflict. The bounded repair permits a distinct valid client ID at that completed tail, retaining conservative rejection for identical, missing or malformed identities. Pending/error-tail and attachment guards remain unchanged; no global historical idempotency contract is introduced.

Causal RED reached the original 409 on SQLite and real PostgreSQL. The first PG attempts were blocked by sandbox reachability, not fixture unavailability; authorized fixture runs corrected that evidence. Final root combined causal/control verification passed 10 tests, zero skips, 6 environment/test warnings. Independent review passed 8 dual-backend controls; author adjacent checks passed 117 retry controls, 31 image controls and the separate official PG image snapshot.

Production Ruff, compile, Bandit and source diff checks passed. Test-file I001 is unchanged baseline; all 128 test Bandit findings are assertion B101. Full logs are hash-only when omitted; bounded receipts and verification reports are retained. Native original failure/Retry/canonical reload is still required. No full matrix acceptance is claimed.

The initial review incorrectly said the report/diff were missing; ROUND1 corrects that path lookup and approves the malformed-identity coverage follow-up. Original review and reports are preserved. Final production hash 3cec9d7ed4c6cf30701341d7d4a5dfc0ed833c401591f32e78ed859aee835721, test hash 56be81875f970c642a2cff7cc608b8f961e467dcf0765744d13464b9068d7355.

All payloads preserve original bytes, using gzip for large or terminal-whitespace-bearing text. Known credential variants/JWT patterns are scanned before writing. Private profiles/configurations/raw logs are excluded.
