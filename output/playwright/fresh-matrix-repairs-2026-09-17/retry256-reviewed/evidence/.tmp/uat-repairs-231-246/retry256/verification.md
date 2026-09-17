# UAT256 Retry Identity Repair Verification

## Causal RED

- Command (exit 1):
  `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-red-20260917a node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_explicit_retry_persists_new_identical_question_after_pre_persistence_model_rejection -q --tb=short`
- SQLite failed at the intended completed-tail guard: `409 This turn already has an answer` after the new, distinct client ID received the strict unavailable-model 400 before persistence.
- PostgreSQL did not reach test setup because the mandatory fixture reported `Postgres required ... but not reachable`.

## Final GREEN Evidence

- Command (exit 0):
  `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-sqlite-20260917a node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs 'tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_explicit_retry_persists_new_identical_question_after_pre_persistence_model_rejection[sqlite]' -q --tb=short`
  Result: `1 passed`.
- Command (exit 0):
  `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-adjacent-20260917a node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py tldw_Server_API/tests/Chat/unit/test_failed_retry_provider_order.py -q --tb=short`
  Result: `117 passed`.
- Command (exit 0):
  `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-images-20260917a node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py -k 'not postgres_strict_image_snapshot' -q --tb=short`
  Result: `31 passed, 1 deselected`.
- Dual-backend command after the repair (exit 1) confirmed the SQLite parameter passed, but PostgreSQL remained unreachable before setup:
  `TLDW_UAT_EVIDENCE_LABEL=retry256-green-20260917b ... test_explicit_retry_persists_new_identical_question_after_pre_persistence_model_rejection -q --tb=short`
  Result: `1 passed, 1 error` (`Postgres required ... but not reachable`).

## Static and Security

- `python -m ruff check tldw_Server_API/app/core/Chat/chat_service.py`: exit 0.
- `python -m compileall -q tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py`: exit 0.
- `python -m bandit -r tldw_Server_API/app/core/Chat/chat_service.py -f json -o /tmp/bandit_retry256_production.json`: exit 0; no findings.
- Full touched-scope Bandit report contains 123 `B101` findings, all in the pytest integration file; it contains no production finding.
- Ruff reports `I001` for the integration test module. The same `I001` occurs on the prepared baseline copy, so this is unchanged baseline debt.
- `git diff --check`: exit 0.

## Source Hashes

Prepared baseline `6f6983b0620aae1f0892c6b0d3ae3bebfc105e02`:

- `chat_service.py`: `02b32a2562e0942b494d6b8f440bb81b184f28f2b86aaaad3983af6b1cdd8c7a`
- `test_persona_backed_chat_conversations.py`: `51713f49453ab549682f9e9543d6f13596365a1ca32392e8941058eedbb5aaa5`

Repaired source:

- `chat_service.py`: `3cec9d7ed4c6cf30701341d7d4a5dfc0ed833c401591f32e78ed859aee835721`
- `test_persona_backed_chat_conversations.py`: `874f846ccbdb2cd7204f7748e6ec1d08f9a3741a84cf9726022a43150853bc9c`

## Correction: Official PostgreSQL Fixture

The earlier mandatory-runner PostgreSQL setup errors were sandbox network isolation, not an unavailable fixture. The following commands used the authorized elevated connection path without changing fixture configuration, DSNs, Docker state, or test guards.

- Causal PostgreSQL RED baseline overlay (exit 1): restored the repaired source unconditionally after temporarily overlaying prepared revision `6f6983b0620aae1f0892c6b0d3ae3bebfc105e02` for the PostgreSQL parameter only. The real PostgreSQL backend initialized and the test failed at the expected `409 This turn already has an answer` guard. Receipt: `retry256-pg-red-20260917a.redacted.log`.
- Final dual-backend GREEN (exit 0):
  `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-dual-green-20260917a node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_explicit_retry_persists_new_identical_question_after_pre_persistence_model_rejection -q --tb=short`
  Result: `2 passed` (SQLite and PostgreSQL).
- PostgreSQL attachment control (exit 0):
  `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-pg-image-20260917a node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py::test_postgres_strict_image_snapshot -q --tb=short`
  Result: `1 passed`.
- Post-overlay source hash reconfirmed: `chat_service.py` is `3cec9d7ed4c6cf30701341d7d4a5dfc0ed833c401591f32e78ed859aee835721`.

## Review Round 1: Conservative Invalid-ID Controls

- Added an actual endpoint, dual-backend parameterized control for four answered-tail identities: matching valid IDs, missing request ID, malformed request ID (`bad id!`), and malformed persisted legacy ID (`bad id!`). Each asserts 409, unchanged canonical rows, and no retry provider dispatch.
- Command (exit 0):
  `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-round1-controls-20260917a node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_explicit_retry_rejects_answered_tail_without_matching_valid_identity -q --tb=short`
  Result: `8 passed` (four identities on SQLite and PostgreSQL).
- Command (exit 0):
  `source .venv/bin/activate && TLDW_UAT_EVIDENCE_LABEL=retry256-round1-green-20260917a node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_explicit_retry_persists_new_identical_question_after_pre_persistence_model_rejection tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py::test_explicit_retry_rejects_answered_tail_without_matching_valid_identity -q --tb=short`
  Result: `10 passed`.
- Repeated static verification: production Ruff and compile exit 0; production Bandit exit 0 with no findings; full test-file Bandit exit 1 reports only 128 test-only `B101` assertions. `git diff --check` exit 0.
- Current hashes: `chat_service.py` `3cec9d7ed4c6cf30701341d7d4a5dfc0ed833c401591f32e78ed859aee835721`; `test_persona_backed_chat_conversations.py` `56be81875f970c642a2cff7cc608b8f961e467dcf0765744d13464b9068d7355`.
