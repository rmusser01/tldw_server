# UAT253 MediaFiles PostgreSQL binding repair

TASK13260.195; two owned source/test paths are frozen in source-freeze.json. Parent owns tracking, integration and native acceptance.

## Cause and minimal change

The real PG server reported syntax near ':' (character43) for the original-file lookup. MediaFilesRepository used SQLite :name placeholders throughout. The existing shared preparation contract converts positional ? to psycopg %s; mapping parameters are already expected to use psycopg's named syntax. The repository therefore sent literal colon placeholders to PostgreSQL. The rich-detail service always queries original-file availability, so even a valid ingested plaintext source with no registered original returned500. This was not evidence of lost source text or a Boolean schema failure.

All existing repository statements now use the established positional ?/tuple contract. The insert binds deleted=False, matching the actual PostgreSQL BOOLEAN and accepted SQLite boolean binding. No global translator, schema, authorization, storage or runtime policy changed. Query failures still propagate; a successful absent-file query returnsNone.

## Verification

Exact commands: commands.json. Required official PostgreSQL fixtures only; no held native DB mutations.

- Private original reproduction:3 failures, zero skips, matching absent original lookup, actual rich-detail service and registration.
- Permanent causal RED:4 failures /6 passes, zero skips. All fiveSQLite cases and the PG error-propagation control passed; PG real detail/CRUD controls failed on colon binding.
- Focused GREEN:10 passed, zero skips.
- Final frozen GREEN:93 passed /8 warnings across6 suites, zero skips, clean exit0. This includes actual endpoint response construction/schema/ETag with real SQLite and PG service/repository reads, all existing MediaFiles cases, endpoint error/sanitization and request-scope guard controls.
- New lifecycle controls preserve reupload-latest ordering, exact quoted/colon/question-mark input values, file-type ordering, include-deleted behavior, single and bulk hard/soft deletion, other-media rows, retained shared references and rollback under a caller transaction. Rich detail preserves source text without a registered original and reports availability after registration.
- Ruff:2 pre-existing production findings (I001,UP037), zero added; new test clean. Initial new test import-order finding was corrected before final frozen93 run.
- Production Bandit:0 findings/0 errors. Whole touched scan reports only B101 test assertions; these are test assertions, not production security findings. Scoped diff whitespace check passes. No compiler run needed for this Python-only unit.

## Limits

The endpoint control calls the actual handler with a resolved fixture DB/current principal; it is not a live HTTP authentication test. Cross-media deletion/reference tests and existing request-scope regressions preserve established boundaries; this patch adds no RLS or permission mechanism. Actual targeted native acceptance remains pending. No archive, browser, provider, runtime, task, staging or commit changes were made by the author.
