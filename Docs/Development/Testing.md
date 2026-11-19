Testing Architecture and Fixture Plugins
---------------------------------------

This repository uses pytest plugin modules to provide shared fixtures in a
scalable, low‑coupling way. Suite‑level `conftest.py` files stay lean and focus
on markers, CLI options, and lightweight environment overrides.

Plugin Layout
-------------

- `tldw_Server_API/tests/_plugins/e2e_fixtures.py`
  - Exposes end‑to‑end fixtures (API clients, credentials, trackers).
- `tldw_Server_API/tests/_plugins/e2e_state_fixtures.py`
  - Session/test run state helpers used by e2e suites (shared state, teardown coordination).
- `tldw_Server_API/tests/_plugins/chat_fixtures.py`
  - Chat helpers and authenticated client wrappers (pre‑existing).
- `tldw_Server_API/tests/_plugins/media_fixtures.py`
  - Temporary media file fixtures (text/PDF/audio) with auto‑cleanup.

Aggregator
----------

Each test suite declares plugins explicitly via `pytest_plugins` in its local
`conftest.py`. Example from `tldw_Server_API/tests/e2e/conftest.py`:

```
pytest_plugins = [
    "tldw_Server_API.tests._plugins.e2e_fixtures",
    "tldw_Server_API.tests._plugins.e2e_state_fixtures",
    "tldw_Server_API.tests._plugins.chat_fixtures",
    "tldw_Server_API.tests._plugins.media_fixtures",
]
```

Note: If adopting pytest>=8 constraints about plugin discovery in nested
`conftest.py` files, prefer registering these at a top‑level `conftest.py`
instead. Tests already running with nested registration may continue to do so.

Fixture Contracts (Types)
-------------------------

- `api_client() -> APIClient`
  - Returns a sync client that talks to the running API (`E2E_TEST_BASE_URL`) or
    an in‑process ASGI client when `E2E_INPROCESS=1`.
  - Honors single‑user mode by setting `X-API-KEY` when available.

- `authenticated_client(api_client, test_user_credentials) -> APIClient`
  - Single‑user: injects API key via `X-API-KEY` and returns the client.
  - Multi‑user: registers the test user if needed, logs in, and sets
    `Authorization: Bearer <token>`.

- `test_user_credentials() -> dict[str, str]`
  - Example: `{"username": "e2e_test_user_<ts>", "email": ..., "password": ...}`.

- `data_tracker()`
  - Tracks created resources during tests (media IDs, prompt IDs, file paths) and
    ensures temporary files are cleaned up at session end.

- Media fixtures (optional helpers)
  - `test_text_file() -> str`
  - `test_pdf_file() -> str`
  - `test_audio_file() -> str`
  - Each yields a temporary file path and handles cleanup automatically.

Design Rules
------------

- Keep heavy work out of module import time; do it inside fixture bodies.
- Use `__all__` in plugin modules to make the exported surface explicit.
- Prefer composition: re‑use helpers from `tldw_Server_API.tests.e2e.fixtures` to
  avoid duplication and drift.

Offline By Default
------------------

Every test runs inside `tldw_Server_API/tests/helpers/offline_network.block_live_network()`
through an autouse fixture in `tldw_Server_API/tests/conftest.py` (TASK-15486). A DNS
lookup or connect to a non-loopback host raises, and the test errors at teardown with
`live network attempted: [(kind, host), ...]`, so a failure the code under test swallowed
still fails the test. Stub the download, provider or DNS call instead.

- Loopback is decided by address: `127.0.0.0/8`, `::1`, IPv4-mapped loopback and `0.0.0.0`,
  plus the names `localhost`, `testserver`, the machine's own hostname and the host of
  `TEST_DATABASE_URL` / `TEST_DB_HOST`. The names are allowed at DNS only; a connect to the address
  they resolve to is judged by address, so a hostname that resolves off loopback is blocked (a
  false positive, never a leak), and a `TEST_DATABASE_URL` naming a remote numeric address is allowed
  by design (the developer's own database).
- A numeric host's lookup passes (nothing leaves the machine); its connect is still blocked.
- The teardown message lists each attempt with its thread name and a stack tail, so an attempt made by
  a thread another test left running is attributable.
- Opt out with `@pytest.mark.live_network`; `external_api`, `local_llm_service`,
  `provider_inventory`, `requires_model` and `stt_golden` opt out too, as does every test under
  `tests/e2e/`, `tests/server_e2e_tests/` and `tests/frontend_e2e/` (they drive a live server).
- A test that still reaches the network and is not yet stubbed is listed by node id in
  `tldw_Server_API/tests/live_network_baseline.txt`; `tests/lint/test_live_network_baseline_ratchet.py`
  keeps that list from growing. Stub the seam (`offline_network.stub_host_resolution` for the egress
  policy's resolver, `allow_egress_hosts` for CI's `WORKFLOWS_EGRESS_ALLOWLIST`, the shared downloader's
  `_create_async_client` with an `httpx.MockTransport`) and delete the line.
- Not covered: session- and module-scoped fixture setup (the fixture is function-scoped), the
  `app/core/MCP_unified/tests` tree (its own conftest), and anything that bypasses Python's
  `socket.socket`: `_socket.socket.connect` on the C class, UDP `sendto`, `gethostbyname_ex` /
  `gethostbyaddr`, and resolvers outside Python (libpq, c-ares, grpc).

CI Guard
--------

The CI workflow includes a fast guard that verifies core fixtures are discoverable:

```
pytest --fixtures -q tldw_Server_API/tests/e2e | rg -q '^test_user_credentials$'
```

It fails the build if `test_user_credentials` is not registered, catching
regressions where a plugin stops exposing required fixtures.
