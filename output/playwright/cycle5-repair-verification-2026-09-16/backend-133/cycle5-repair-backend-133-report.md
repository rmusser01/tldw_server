# UAT133 / TASK13260.72 — frozen implementation

## Change
An explicit failed Retry that has passed existing saved-tail, content/image, overlap, residual-user and client-correlation checks moves its exact historical user row to the current-turn position. The historical row retains its already validated text/images and Character transformation. All other historical rows retain requested ordering. The move introduces no database write or provider-visible app-only metadata. Existing out-of-window/current-turn behavior and continuation prefill remain unchanged.

Production: `tldw_Server_API/app/core/Chat/chat_service.py`, six added lines. Tests: new `Chat/unit/test_failed_retry_provider_order.py` plus four actual endpoint cases in existing `Chat/integration/test_persona_backed_chat_conversations.py`.

## Red → green evidence
- Final provider-body chain: actual context assembly → actual prompt templating → research no-op → call builder → adapter request → CustomOpenAI payload. Controlled saved-history fixture; no transport/inference. Corrected RED **15 failed / 39 passed** (`-red-corrected.log`), all failures descending last-turn placement. Initial run also exposed fixture assumptions: public schema excludes limit0 and selected ASC-prefix windows retain prior residual handling; those were corrected without production changes and retained in `-red.log`.
- Actual endpoint + actual SQLite conversation: two successful pairs → provider502 → failed Retry, both stream and nonstream, ASC/DESC. Transport response controlled. Corrected RED **2 failed / 2 passed** (`-endpoint-red-corrected.log`). Initial fixture count omitted the saved system row; `-endpoint-red.log` retains that fixture error separately.
- Final focused provider suite **56 passed** (`-final-focused.log`), including two extra controls proving distinct equal-text turns remain distinct and ordinary repeat sends still write a new user.
- Broader verification: **185 passed / 1 skipped**, 186 collected (`-green.log`). This run collected the original54 provider cases; final focused56 adds the two equal-text controls. The skipped node is `tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py::test_postgres_strict_image_snapshot`, dependent on official `pg_database_config`. The original run did not request `-rs`, so its exact dynamic fixture reason was not printed; source permits unreachable server or temporary DB-creation failure. **PostgreSQL acceptance is unresolved, not waived**; root is preparing the official fixture, and the required rerun will use `TLDW_TEST_POSTGRES_REQUIRED=1` plus `-rs` to prohibit silent skip.

## Boundaries covered
Neutral/persona/Character, ASC/DESC, full client history/user-only, limits1/3/20 and internal service-only0, unchanged residual-user409, exact canonical identity, no additional saved user, literal neutral/persona text, existing Character substitution, image MIME/bytes, saved error filtering, and equal-content historic/current turns. Limited ascending full-client context can already persist a residual assistant outside its selected prefix; tests preserve this existing behavior rather than broadening overlap semantics. Public API minimum history limit remains1.

## Static checks and limits
Ruff: production + new test0; existing integration file retains exactly one pre-existing I001 at import line1 (baseline `-test-ruff-before.json`, final `-ruff-after.json`). Production Bandit0 (`-bandit.json`); all three files AST-parse; git diff --check passes. Local log scan for JWT/provider-key patterns0 (`-scan.json`), not an end-to-end audit of arbitrary secrets.

Exact code/test hashes: `-code-manifest.json`; final owned manifest includes official task72. No browser/runtime/config/inference/frontend/staging/commit. Native acceptance and independent review remain pending.

## Commands
From repo root after `source .venv/bin/activate`:

```sh
python -m pytest tldw_Server_API/tests/Chat/unit/test_failed_retry_provider_order.py -q
python -m pytest tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py -k dispatches_latest -q
python -m pytest tldw_Server_API/tests/Chat/unit/test_failed_retry_provider_order.py tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py -q
python -m ruff check tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_failed_retry_provider_order.py tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py --output-format json
python -m bandit tldw_Server_API/app/core/Chat/chat_service.py -f json
```

## Required PostgreSQL follow-up
Root restored the official PostgreSQL18.6 fixture. Bounded backend32 and AuthNZ2 checks passed with REQUIRED=1/NO_DOCKER=1 and zero skips. The two formerly skipped nodes now execute. The bootstrap test assertion defect was corrected separately under TASK13260.75.1; see /private/tmp/cycle5-postgres-136-report.md for exact commands, scope and redacted evidence. This resolves the recorded fixture-execution gap for these controls, not native full-workflow acceptance. Product code for this unit remains unchanged.
