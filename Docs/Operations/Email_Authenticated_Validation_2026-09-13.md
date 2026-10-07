# Authenticated Email Validation — 2026-09-13

Tracking: TASK-13255. Starting revision: `1600d9b8c8`.

## Scope and outcome

Synthetic API keys now exercise real AuthNZ credential validation and production
per-user SQLite database routing for email search/detail. The tests create users
through `AuthnzUsersRepo` and keys through `APIKeyManager`; neither authentication
nor the Media DB FastAPI dependency is overridden. This extends the earlier
upload/parser validation, whose harness substituted those dependencies.

The new module is
`tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_authenticated_access.py`.
It uses temporary AuthNZ, Media and audit storage, intercepted socket/DNS and HTTP
calls, and guards on summarization and embedding dispatch. Guard attempts are
recorded and checked after teardown, including errors caught by application code.
The combined runner also guards network calls during test collection.

## Covered behavior

- Missing credentials and invalid keys return 401 for search and detail.
- Interleaved and concurrent users see only their own messages. The test checks
  distinct physical Media DB paths and deliberately overlapping numeric message IDs.
- A foreign message ID absent from the requesting user's DB returns 404. A foreign
  cursor returns 400; an injected tenant query parameter cannot change key ownership.
- Revoking a previously accepted key prevents further reads.
- Configured single-user keys work through `X-API-KEY` and Bearer headers, while an
  incorrect key is rejected. Multi-user API keys are validated through stored keys.
- The authenticated `/media/search` operator bridge returns the requesting user's
  messages. Disabling email search returns 404 for email search/detail; Gmail source
  APIs remain disabled.
- A separate case reloads the actual main app through the existing isolation helper
  and exercises its route registration and test-mode request middleware. It validates auth,
  own-message search, foreign detail rejection and the media-search bridge.

## Verification

The final combined run passed **65 tests**, with eight environment/dependency
warnings and **zero outbound socket/DNS attempts**. It includes all ten new cases
alongside the existing offline upload, email endpoint, cursor endpoint and
media-search compatibility suites. No production-code changes were required.
Independent review reported no blockers and separately passed nine cases, leaving
the main-app case to the primary run. Counts overlap and should not be summed.

Ruff lint and format checks pass. Bandit reports zero findings on the new test
module with B101 excluded for intentional pytest assertions. Local logs:
`/tmp/email_auth_combined_13255.log`, `/tmp/email_main_auth_13255.log`,
`/tmp/email_auth_review_13255.log`, and `/tmp/bandit_email_auth_13255.json`.

```bash
source .venv/bin/activate
export TEST_MODE=true AUTO_DOWNLOAD_MODELS=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_authenticated_access.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_search_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_search_cursor_endpoint.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_search_request_model.py \
  tldw_Server_API/tests/MediaIngestion_NEW/integration/test_email_offline_ingestion.py -q
```

The fixture explicitly sets `TESTING=false` and `EVALS_HEAVY_ADMIN_ONLY=true` to
disable the known evaluation auth bypass. Negative credential tests are required
alongside successful reads. Redis is disabled; no connector worker is started.
All credentials and messages are generated synthetic test data.

## Remaining limits

`ASGITransport` does not execute application lifespan. Main-app routing and its
test-mode middleware are covered. The main app omits security headers, HTTP metrics,
usage logging and access logging in test mode. Startup, worker lifecycle, listening sockets, reverse
proxies and deployed configuration are not certified. The tests do not exercise
password/JWT login or authenticated upload quota/billing enforcement. Upload
parsing/persistence retains its earlier separate synthetic integration evidence.

TASK-13256 subsequently added authenticated upload and organization storage-quota
tests. See `Docs/Operations/Email_Authenticated_Upload_Validation_2026-09-25.md`
for that narrower, newer evidence and its remaining billing/deployment limits.

Multi-user SQLite is used only as an isolated test configuration. Production
multi-user deployments require PostgreSQL; PostgreSQL RLS/migration/runtime parity
remains unverified here. Native PST/OST and the 1M-message performance gate also
remain open. Optional live Gmail is deferred and is not a core release dependency.
No personal mailbox or personal email was accessed, and no LLM was called.
