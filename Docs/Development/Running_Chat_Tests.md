# Running Chat Tests

This guide summarizes how to run the Chat module tests, the normalized folder structure, and key environment flags.

## Structure
- Unit tests: `tldw_Server_API/tests/Chat/unit/`
- Integration tests: `tldw_Server_API/tests/Chat/integration/`
- Shared fixtures: `tldw_Server_API/tests/Chat/conftest.py`, `tldw_Server_API/tests/Chat/test_fixtures.py`

## Quick Commands
- All Chat tests (unit + integration):
  - `python -m pytest tldw_Server_API/tests/Chat -m "unit or integration" -v`
- Unit-only:
  - `python -m pytest tldw_Server_API/tests/Chat/unit -m unit -v`
- Integration-only:
  - `python -m pytest tldw_Server_API/tests/Chat/integration -m integration -v`

## Commercial Provider Tests (Opt-In)
- These are disabled by default to avoid network calls. Enable with env flags and real keys:
  - `export RUN_COMMERCIAL_CHAT_TESTS=true`
  - `export OPENAI_API_KEY="<real-openai-key>"` (and others as needed)
- Run only commercial integration tests (Chat scope):
  - `python -m pytest tldw_Server_API/tests/Chat -m "integration and external_api" -v`
- Target a specific templating test:
  - `python -m pytest tldw_Server_API/tests/Chat/integration/test_chat_completions_integration.py::test_commercial_provider_with_template_and_char_data_openai_integration -v`

Notes:
- Key loading precedence (high → low): environment variables → `.env`/`.ENV` in project root or `tldw_Server_API/Config_Files/` → `Config_Files/config.txt` `[API]` entries.
- Tests skip automatically when `RUN_COMMERCIAL_CHAT_TESTS` is not set or a usable key is missing.
- Network access is required for real commercial tests.

## Key Sanity Check (No Secrets Printed)
```python
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys
keys = get_api_keys()
k = keys.get('openai') or ''
print({'openai_present': bool(k), 'length': len(k), 'masked': (k[:4]+'...'+k[-4:]) if k else None})
```

## Streaming Tests
- Streaming tests use an async HTTPX client fixture and Server-Sent Events normalization.
- TestClient’s SSE limitations are worked around where needed; some streaming scenarios may be marked `@pytest.mark.skip` in integration tests.

## Character Chat Real-Backend E2E
- First-class Character Chat UI work must include a real-backend browser run when the change affects character selection, streaming, persistence, or resume behavior.
- Runbook: `Docs/Development/Character_Chat_Real_Backend_E2E.md`.
- The release-signoff path must verify `/api/v1/chats/{chat_id}/complete-v2` with `include_character_context: true`; frontend-only route interception is not sufficient evidence.

## Tips
- Single-user mode: ensure `API_BEARER` is not set in the environment; tests expect `X-API-KEY`/`Token` headers per mode.
- For deterministic limits during CI, you may set `TEST_MODE=true` when applicable.

See also: `Docs/Development/Chat_Tests_Consolidation.md` for the rationale and what moved.
