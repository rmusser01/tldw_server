# Local provider 401/403/429 chat answers, and offline custom OpenAI adapter tests: Implementation Plan

> For agentic workers: use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

> **Review folded in (r2).** The independent review (Ready with fixes, 0 blocker, 3 important, 5 minor) is folded in: Task 3 line range (34-51); Task 2 RED is `6 failed, 45 passed, 6 errors`; Task 3 GREEN is `6 passed` and the notes say 6, 6, 8; measurements reworded; the endpoint test is now written in Task 3 (Step 1b) so all RED runs precede the fixes; ruff scoped to new and changed files and Bandit via `uvx bandit -ll`; two Review Focus lines added (blocker coverage gaps, WebUI `providerUnconfigured` path); follow-up 2 now names the tests. The previous version is `plan_fu2_provider_errors.r1.md`.

**Goal:** (1) A local provider's (llama.cpp, Ollama, Kobold, ooba, tabbyapi, vLLM, aphrodite, local-llm) 401, 403 or 429 reply gets the same chat response a remote provider's gets, streamed and not; a 404 keeps its response. (2) The six custom OpenAI adapter error cases pass with every non-loopback DNS lookup and connect blocked, and are proven to make none.

**Architecture:**
- TASK-13883: `_raise_sanitized_local_provider_error` (non-stream, and the Kobold handler) builds `ChatAuthenticationError` / `ChatRateLimitError` for 401/403 and 429 instead of `ChatBadRequestError`. The stream generator in `_chat_with_openai_compatible_local_server` keeps its in-band error frame but gives a 401/403 frame the code `provider_authentication_failed`. The chat endpoint already maps those types and that code for remote providers, so no endpoint change.
- TASK-13884: the six cases resolve `api.openai.com` because `config.txt` sets `custom_openai_api_ip` and `custom_openai2_api_ip` to `https://api.openai.com/v1` and the configured custom slots read that server-owned endpoint (the request-level `base_url` the tests pass is stripped by `_sanitize_request`). A test helper blocks live network and records attempts. A fixture pins both slots to a loopback URL through the documented `CUSTOM_OPENAI_API_IP` / `CUSTOM_OPENAI2_API_IP` env keys, so the real egress check still runs, against a loopback origin.

**Tech Stack:** Python, pytest, httpx, FastAPI TestClient, backlog-py.

**Spec:** `backlog/tasks/task-13883 - Chat-a-local-provider-s-401-403-or-429-reply-answers-400-Invalid-request.md`, `backlog/tasks/task-13884 - Tests-custom-OpenAI-adapter-error-tests-resolve-api.openai.com-through-the-real-egress-check.md` (both filed by PR #3289 / TASK-13865).

**Measured on:** `origin/dev` at `e06eeec68c` (Merge pull request #3300), in an isolated worktree. Python `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`, `TLDW_TEST_NO_DOCKER=1`.

## Measurements (all with every non-loopback DNS lookup and connect blocked and recorded; for the six-case runs the only host recorded is `api.openai.com`)

TASK-13884, on `e06eeec68c`:
- The two files' `custom_openai` / `CustomOpenAIAdapter*` cases (6 tests) fail with `EgressPolicyError('Host could not be resolved')`. The only lookup attempted is `api.openai.com`.
- The URL the adapter requests, for both `CustomOpenAIAdapter` and `CustomOpenAIAdapter2`: `https://api.openai.com/v1/chat/completions`, scope `ConfiguredEndpointScope(https, api.openai.com, 443)`. Source: `Config_Files/config.txt:1037` and `:1047`, read by `resolve_trusted_provider_endpoint` (`provider_config_resolution.py:92-101`).
- With `CUSTOM_OPENAI_API_IP=http://127.0.0.1:9/v1` and `CUSTOM_OPENAI2_API_IP=http://127.0.0.1:9/v1` exported, the same six tests pass and the blocker records no attempt.
- Whole of `tests/LLM_Adapters/unit` plus `tests/LLM_Calls` (1451 tests) under the blocker: 6 failed (these six), 1443 passed, 2 skipped. Non-loopback hosts recorded: `api.openai.com` (the six), `openaipublic.blob.core.windows.net` and `slot-2.example`. The blocker does not record loopback names. See Follow-ups.

TASK-13883, `/api/v1/chat/completions`, `api_provider: llama.cpp`, transport stubbed on `LlamaCppAdapter.http_fetcher` / `http_streamer` (no socket opened):

| Upstream | Not streamed (dev) | Streamed (dev) | Remote provider, not streamed | Remote, streamed |
|---|---|---|---|---|
| 400 | 400 `"Invalid request."` | 502 `provider_unavailable` | n/a | n/a |
| 401 | 400 `"Invalid request."` | 502 `provider_unavailable` | 502 `provider_authentication_failed` | 502 `provider_authentication_failed` |
| 403 | 400 `"Invalid request."` | 502 `provider_unavailable` | 502 `provider_authentication_failed` | 502 `provider_authentication_failed` |
| 404 | 400 `"Invalid request."` | 502 `provider_unavailable` | n/a | n/a |
| 429 | 400 `"Invalid request."` | 502 `provider_unavailable` | 429 `"Rate limit exceeded. Please retry."` | 502 `provider_unavailable` |
| 500 | 502 `"The chat service provider is currently unavailable."` | 502 `provider_unavailable` | n/a | n/a |

Remote column: an `OpenAIAdapter` whose `chat` / `stream` raises `build_sanitized_chat_error("openai", status_code=N)`, which is what `ChatProvider._raise_sanitized_provider_failure` raises for an upstream N.

Two facts that shape the fix:
- A remote provider's streamed 429 already answers 502 `provider_unavailable` (there is no rate-limit stream code in `PROVIDER_STREAM_ERROR_MESSAGES`, `streaming_utils.py:352-364`). So AC #2's "the response a remote provider's rate limit gets, streamed" is what local streaming already gives. Only the non-stream 429 and the stream 401/403 change. The streamed 429 gets a pin test, not a code change.
- Every streamed local failure is swallowed into one in-band frame (`local_adapters.py:508-509`, TASK-13373 design), so a typed exception is not raised there. The fix keeps that design: the frame's `code` is what `normalize_provider_stream_error` reads (`candidate.get("code")`).

Dry run of the fix (diff kept at `<scratchpad>/fu2_dryrun.diff`, then reverted): the table above becomes: 401/403 not streamed and streamed 502 `provider_authentication_failed`; 429 not streamed 429 `"Rate limit exceeded. Please retry."`; 429 streamed, 404, 400, 500 unchanged. These equal the remote column. Of 304 tests in `LLM_Calls/test_local_http_error_mapping.py`, `test_local_streaming_contract.py`, `test_summarization_adapter.py`, `test_provider_adapter_runtime_boundary.py`, `LLM_Adapters/unit/test_authoritative_local_adapter_snapshot.py`, `test_local_adapter_merge.py`, exactly 3 fail: `test_local_openai_rejection_keeps_the_upstream_status[401|403|429]`, which assert the old behaviour and are replaced in Task 2.

## Global Constraints

- Test first: every fix has a test that fails on `e06eeec68c` for the reason measured above.
- No live network, ever. Every test and probe here runs under `block_live_network()` (Task 1) or the equivalent pytest plugin; stubs replace provider transports. A dry run of an earlier plan once made a real request to api.openai.com, so the reviewer and executor must not run any step without the blocker in place.
- How the tests prove no live DNS query happens: the `no_live_network` fixture patches `socket.getaddrinfo`, `socket.gethostbyname`, `socket.socket.connect` and `connect_ex`; any non-loopback call raises (`gaierror` / `OSError`) and is appended to a list; the fixture asserts the list is empty at teardown, so a call whose exception a library swallowed still fails the test. `test_no_live_network_fixture.py` proves the blocker itself works (blocks, records, allows loopback, restores). The RED run of Task 2 shows the blocker catching the exact leak (`api.openai.com`).
- Public error text stays bounded: no upstream body, URL or exception text in any new error (`raise_detached_error`, existing messages).
- Do not touch `chat.py`, `streaming_utils.py` or `exceptions.py`.
- `local_adapters.py` and `Summarization_General_Lib.py` are not in `Helper_Scripts/ci/scope_predicate_baseline.txt` (0 matches), so line shifts there need no baseline regeneration. Re-check at execution with `grep -c "local_adapters.py\|Summarization_General_Lib" Helper_Scripts/ci/scope_predicate_baseline.txt`.
- Backlog: only through backlog-py (ADR-059), `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never git stash.

## Review Focus

Input classes and failure modes the tests do not exercise:
- A local server that answers 401/403/429 with a non-JSON or huge body: the mapping reads only the status, and `upstream_quota_exhausted` is deliberately not called for local 429 (a local server has no billing quota), so `ChatRateLimitError.quota_exhausted` stays False. Confirm nobody needs it.
- Callers that branch on `ChatBadRequestError` to mean "do not retry", now receiving `ChatRateLimitError` or `ChatAuthenticationError` from local adapters: `git grep -n "ChatRateLimitError\|ChatAuthenticationError" -- tldw_Server_API/app` for retry, fallback, circuit-breaker or health-marking code that would now treat a local 429 as retryable or a local 401 as a bad credential (BYOK invalidation, `_is_auth_401_error` OAuth retry at `chat.py:5416` only applies to provider `openai`). The measured endpoint answers above are the only paths exercised.
- Other endpoints that call local adapters (character chat, prompt studio, RAG generation, evaluations, flashcards, quizzes): they map `ChatAuthenticationError` / `ChatRateLimitError` already for remote providers; only `test_summarization_adapter.py`'s local cases (still green) were run. The reviewer should skim `flashcards.py:2539` and `quizzes.py:857` ("The chat provider rejected the request") for a local 401 now falling into a different branch.
- The 408/498 branch kept in `_summary_error_code`; the removed 401/403/429 branches there are dead after this change (local was the only source of those statuses on a `ChatBadRequestError`). `test_a_local_provider_reply_reaches_the_summary_code` guards the result.
- Mid-stream failures (after output): they carry no HTTP status, so they stay `provider_unavailable`. Only a pre-output 401/403 changes.
- `local-llm` and the six other handlers share `_chat_with_openai_compatible_local_server`; only `llama.cpp` is exercised end to end at the endpoint. Kobold has its own non-stream call to `_raise_sanitized_local_provider_error` (line 1101) and is covered by the unit test only through the shared function, not by a Kobold-specific test.
- The loopback pin uses port 9 (discard) and nothing listens: a test that ever reached a real `connect` would be blocked by the blocker, and fails at teardown.
- Blocker coverage gaps (none affects the six cases, whose pinned slots do no lookup): it patches `socket.getaddrinfo`, `gethostbyname`, `socket.socket.connect` and `connect_ex` globally, so executor-thread lookups (`egress._getaddrinfo_with_timeout`) are recorded. It does not cover `gethostbyname_ex`, `getnameinfo`, `from socket import getaddrinfo` copies, third-party resolvers (dnspython, aiodns) or `_socket` used directly. An executor thread abandoned by the egress timeout path that calls the resolver after the context exits is neither recorded nor blocked.
- WebUI consumers of `provider_authentication_failed` (`apps/tldw-frontend/lib/api.ts:47,129`, `provider-error-contract.ts`) map it to the fixed text "The selected provider credentials could not be authenticated."; remote providers already emit it, so nothing needs a change. `apps/packages/ui/src/hooks/chat/useCharacterChatMode.ts:198` classifies it as `providerUnconfigured`, so a local server's 401 now sends the user down the "set up a provider" recovery path. That text is slightly off for a local server but arguably right (the server needs a key); a small UX consequence of AC #1, not a blocker. Nothing in `apps/` keys on a local stream frame's code.
- Developer machines with `CUSTOM_OPENAI_API_IP` set in the environment: the fixture overrides it for the test, restored after.

## PR split, branch, commits

One PR. The tasks are independent, but both are small and test-only or near test-only, and neither holds up the other, so two PRs would only double the serial merge-queue wait.
- Branch: `fix/local-provider-auth-rate-errors-offline-adapter-tests`, from `origin/dev`, PR base `dev`.
- Commit 1: `test(llm): custom OpenAI adapter error tests run with live network blocked (TASK-13884)`: Task 1 and Task 2.
- Commit 2: `fix(chat): a local provider's 401, 403 and 429 answer like a remote provider's (TASK-13883)`: Tasks 3 to 5.
- Commit 3: `docs(backlog): close TASK-13883 and TASK-13884`, plus the follow-up tasks (Task 6).
- Each commit is green on its own; RED states are working-tree only.

## Task 1: The network blocker and its self-test (TASK-13884)

**Files:**
- Create: `tldw_Server_API/tests/helpers/offline_network.py`
- Create: `tldw_Server_API/tests/LLM_Adapters/unit/test_no_live_network_fixture.py`
- Modify: `tldw_Server_API/tests/LLM_Adapters/conftest.py` (add two fixtures after the existing imports)

**Interfaces:**
- `block_live_network() -> ContextManager[list[tuple[str, str]]]`: yields the attempts list, entries `("dns" | "connect", host)`.
- Fixture `no_live_network`: yields that list, asserts it is empty at teardown.
- Fixture `loopback_custom_openai_endpoints`: pins slots 1 and 2.

- [ ] **Step 1: Write the self-test (RED: the module does not exist)**

```python
"""The blocker that proves a test makes no live DNS query or connect."""

from __future__ import annotations

import socke

import pytes

from tldw_Server_API.tests.helpers.offline_network import block_live_network


@pytest.mark.uni
def test_a_non_loopback_lookup_is_blocked_and_recorded():
    with block_live_network() as attempts:
        with pytest.raises(socket.gaierror):
            socket.getaddrinfo("api.openai.com", 443)
        with pytest.raises(socket.gaierror):
            socket.gethostbyname("api.openai.com")
    assert attempts == [("dns", "api.openai.com"), ("dns", "api.openai.com")]


@pytest.mark.uni
def test_a_non_loopback_connect_is_blocked_before_any_packet():
    with block_live_network() as attempts:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(OSError):
                sock.connect(("93.184.216.34", 80))
            with pytest.raises(OSError):
                sock.connect_ex(("93.184.216.34", 80))
        finally:
            sock.close()
    assert attempts == [("connect", "93.184.216.34"), ("connect", "93.184.216.34")]


@pytest.mark.uni
def test_loopback_is_allowed_and_not_recorded():
    with block_live_network() as attempts:
        assert socket.getaddrinfo("127.0.0.1", 80)
    assert attempts == []


@pytest.mark.uni
def test_the_real_functions_are_restored():
    before = (socket.getaddrinfo, socket.gethostbyname, socket.socket.connect, socket.socket.connect_ex)
    with block_live_network():
        assert socket.getaddrinfo is not before[0]
    after = (socket.getaddrinfo, socket.gethostbyname, socket.socket.connect, socket.socket.connect_ex)
    assert after == before
```

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_no_live_network_fixture.py`
Expected: collection error `ModuleNotFoundError: No module named 'tldw_Server_API.tests.helpers.offline_network'`.

- [ ] **Step 2: Write the helper (GREEN)**

```python
"""Block every non-loopback DNS lookup and connect, and record each attempt.

A call whose exception a library swallows still shows up in the returned list, so a tes
can assert that no live network was touched.
"""

from __future__ import annotations

import socke
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

_LOOPBACK = frozenset({"localhost", "127.0.0.1", "::1", "testserver"})


def _host(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    return str(value).lower()


@contextmanager
def block_live_network() -> Iterator[list[tuple[str, str]]]:
    attempts: list[tuple[str, str]] = []
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_connect = socket.socket.connec
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        if _host(host) in _LOOPBACK:
            return real_getaddrinfo(host, *args, **kwargs)
        attempts.append(("dns", _host(host)))
        raise socket.gaierror(socket.EAI_NONAME, "live DNS lookup blocked by the test")

    def gethostbyname(host):
        if _host(host) in _LOOPBACK:
            return real_gethostbyname(host)
        attempts.append(("dns", _host(host)))
        raise socket.gaierror(socket.EAI_NONAME, "live DNS lookup blocked by the test")

    def guarded(real):
        def connect(self, address, *args, **kwargs):
            # A str address is an AF_UNIX path; only (host, port) tuples can leave the machine.
            if isinstance(address, tuple) and _host(address[0]) not in _LOOPBACK:
                attempts.append(("connect", _host(address[0])))
                raise OSError("live connect blocked by the test")
            return real(self, address, *args, **kwargs)

        return connec

    with (
        mock.patch.object(socket, "getaddrinfo", getaddrinfo),
        mock.patch.object(socket, "gethostbyname", gethostbyname),
        mock.patch.object(socket.socket, "connect", guarded(real_connect)),
        mock.patch.object(socket.socket, "connect_ex", guarded(real_connect_ex)),
    ):
        yield attempts
```

- [ ] **Step 3: Add the fixtures to `tldw_Server_API/tests/LLM_Adapters/conftest.py`** (after the `from tldw_Server_API.tests.helpers.app_main_state import ...` line)

```python
from tldw_Server_API.tests.helpers.offline_network import block_live_network


@pytest.fixture
def no_live_network():
    """Block every non-loopback DNS lookup and connect; fail the test if any was attempted."""
    with block_live_network() as attempts:
        yield attempts
    assert attempts == [], f"live network attempted: {attempts}"


@pytest.fixture
def loopback_custom_openai_endpoints(monkeypatch):
    """Point both configured custom OpenAI slots at a loopback origin.

    config.txt defaults both slots to https://api.openai.com/v1, and the adapters read the
    server-owned endpoint, not the request's base_url. The egress check still runs, agains
    127.0.0.1; nothing listens on port 9 and the tests stub the transport.
    """
    for key in ("CUSTOM_OPENAI_API_IP", "CUSTOM_OPENAI2_API_IP"):
        monkeypatch.setenv(key, "http://127.0.0.1:9/v1")
```

- [ ] **Step 4: Run the self-test.** Same command as Step 1. Expected: `4 passed`.

## Task 2: Run the six cases offline (TASK-13884)

**Files:**
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_midstream_error_raises.py`
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_stream_error_normalization.py`

- [ ] **Step 1 (RED): block the network in both files.** Add directly under each file's imports:

```python
pytestmark = pytest.mark.usefixtures("no_live_network")
```

Run: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_midstream_error_raises.py tldw_Server_API/tests/LLM_Adapters/unit/test_adapter_stream_error_normalization.py`
Expected on `e06eeec68c`: `6 failed, 45 passed, 6 errors` (the 6 errors are the fixture's teardown assertion on the same six tests, not a broken run); the six are the `CustomOpenAIAdapter`, `CustomOpenAIAdapter2` and `custom_openai` 400/401/429/500 cases, failing with `EgressPolicyError('Host could not be resolved')`, plus the fixture's teardown assertion naming `('dns', 'api.openai.com')`. Record `6 failed, 45 passed, 6 errors` in the task notes.

- [ ] **Step 2 (GREEN): pin the slots.** Change the line to:

```python
pytestmark = pytest.mark.usefixtures("no_live_network", "loopback_custom_openai_endpoints")
```

Run the same command. Expected: `51 passed`, and no teardown error.

- [ ] **Step 3: Run the broader neighbours.**
`TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit tldw_Server_API/tests/LLM_Calls`
Expected: no failures (baseline on `e06eeec68c` without the changes: the same 6 fail only when a blocker is on; with live DNS they pass).

- [ ] **Step 4: Commit 1** (`git add tldw_Server_API/tests/helpers/offline_network.py tldw_Server_API/tests/LLM_Adapters`).

## Task 3: Non-stream mapping, unit and endpoint tests (TASK-13883)

**Files:**
- Modify: `tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py`
- Create: `tldw_Server_API/tests/LLM_Adapters/integration/test_adapters_chat_endpoint_local_provider_status.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`

- [ ] **Step 1 (RED): replace `test_local_openai_rejection_keeps_the_upstream_status` including its two decorators (lines 34-51: `@pytest.mark.unit`, the `@pytest.mark.parametrize("status_code", ...)` line, and the function) with:**

```python
@pytest.mark.uni
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [(401, ChatAuthenticationError), (403, ChatAuthenticationError), (429, ChatRateLimitError)],
)
def test_local_openai_auth_and_rate_limit_replies_get_the_remote_error_types(status_code, error_type):
    error = _make_httpx_error(status_code)

    def _fetch(**_kwargs):
        raise error

    with pytest.raises(error_type) as exc:
        _chat_with_openai_compatible_local_server(
            api_base_url="http://localhost:8000",
            model_name="local-model",
            input_data=[{"role": "user", "content": "ping"}],
            http_fetcher=_fetch,
        )
    assert exc.value.status_code == status_code
    assert exc.value.upstream_status_code == status_code
    assert "failure" not in exc.value.message
    assert exc.value.__cause__ is None


@pytest.mark.uni
def test_local_openai_unpulled_model_stays_a_rejected_request():
    error = _make_httpx_error(404)

    def _fetch(**_kwargs):
        raise error

    with pytest.raises(ChatBadRequestError) as exc:
        _chat_with_openai_compatible_local_server(
            api_base_url="http://localhost:8000",
            model_name="local-model",
            input_data=[{"role": "user", "content": "ping"}],
            http_fetcher=_fetch,
        )
    assert exc.value.status_code == 400
    assert exc.value.message == "The local model provider rejected the request."
    assert exc.value.upstream_status_code == 404
```

Change the import line to `from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError, ChatBadRequestError, ChatProviderError, ChatRateLimitError`.

- [ ] **Step 1b (RED): write the endpoint test.** Create `tldw_Server_API/tests/LLM_Adapters/integration/test_adapters_chat_endpoint_local_provider_status.py` (transport stubbed on the adapter class, blocker on; expected values are the remote column measured above):

```python
"""A local provider's upstream 401, 403 and 429 get the chat answer a remote provider's get.

The transport is stubbed on the adapter class, no socket is opened, and the blocker fails
the test if a non-loopback DNS lookup or connect is attempted.
"""

from __future__ import annotations

import httpx
import pytes

# Ensure chat fixtures (client/auth) are registered as pytest fixtures
from tldw_Server_API.tests._plugins import chat_fixtures as _chat_pl  # noqa: F401

AUTH = (502, "provider_authentication_failed")
UNAVAILABLE = (502, "provider_unavailable")


@pytest.fixture(autouse=True)
def _env(monkeypatch, no_live_network):
    monkeypatch.setenv("STREAMS_UNIFIED", "1")
    monkeypatch.setenv("LOGURU_LEVEL", "ERROR")
    monkeypatch.delenv("TEST_MODE", raising=False)


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    response = httpx.Response(status, request=request, content=b'{"error":{"message":"upstream secret"}}')
    return httpx.HTTPStatusError("upstream secret", request=request, response=response)


class _Response:
    def __init__(self, status: int):
        self._status = status

    def raise_for_status(self):
        raise _status_error(self._status)

    def close(self):
        pass


class _StreamContext:
    def __init__(self, status: int):
        self._response = _Response(status)

    def __enter__(self):
        return self._response

    def __exit__(self, *_exc):
        return False


def _post(authenticated_client, monkeypatch, status: int, stream: bool):
    from tldw_Server_API.app.core.LLM_Calls.providers import local_adapters as local_mod

    monkeypatch.setattr(local_mod.LlamaCppAdapter, "http_fetcher", staticmethod(lambda **_k: _Response(status)))
    monkeypatch.setattr(
        local_mod.LlamaCppAdapter, "http_streamer", staticmethod(lambda **_k: _StreamContext(status))
    )
    return authenticated_client.post(
        "/api/v1/chat/completions",
        json={
            "api_provider": "llama.cpp",
            "model": "local-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        },
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, AUTH), (403, AUTH), (429, (429, None)), (404, (400, None))],
    ids=["unauthorized", "forbidden", "rate-limit", "unpulled-model"],
)
def test_local_provider_reply_not_streamed(monkeypatch, authenticated_client, status, expected):
    response = _post(authenticated_client, monkeypatch, status, stream=False)

    assert response.status_code == expected[0], response.tex
    detail = response.json()["detail"]
    if expected[1] is not None:
        assert detail["error_code"] == expected[1]
    elif status == 429:
        assert detail == "Rate limit exceeded. Please retry."
    else:
        assert detail == "Invalid request."
    assert "upstream secret" not in response.tex


@pytest.mark.integration
@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, AUTH), (403, AUTH), (429, UNAVAILABLE), (404, UNAVAILABLE)],
    ids=["unauthorized", "forbidden", "rate-limit", "unpulled-model"],
)
def test_local_provider_reply_streamed(monkeypatch, authenticated_client, status, expected):
    response = _post(authenticated_client, monkeypatch, status, stream=True)

    assert response.status_code == expected[0], response.tex
    assert response.json()["detail"]["error_code"] == expected[1]
    assert "upstream secret" not in response.tex
```

Run all three RED files before any source edit: the unit file below, the Task 4 stream file, and this one:
`TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/integration/test_adapters_chat_endpoint_local_provider_status.py` (about 30 s; each case boots the app).
Expected on `e06eeec68c`: `5 failed, 3 passed` (not streamed 401, 403, 429; streamed 401, 403 fail; streamed 429 and both 404s pass). Record it in the notes.

Run: `TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py`
Expected on `e06eeec68c`: `3 failed` (401, 403, 429: `ChatBadRequestError` raised, `DID NOT RAISE` the expected type), 404 test passes.

- [ ] **Step 2 (GREEN): `local_adapters.py`.**

Imports (lines 11-15) become:

```python
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatAuthenticationError,
    ChatBadRequestError,
    ChatConfigurationError,
    ChatProviderError,
    ChatRateLimitError,
)
```

In `_raise_sanitized_local_provider_error`, replace the block that builds the `ChatBadRequestError` (currently lines 211-223) with:

```python
    if isinstance(exc, ChatBadRequestError) or (
        isinstance(status, int) and 400 <= status < 500
    ):
        # The types a remote provider's 401, 403 and 429 get, so chat answers them alike.
        if status in (401, 403):
            error = ChatAuthenticationError(provider=provider, status_code=status)
        elif status == 429:
            error = ChatRateLimitError(provider=provider)
        else:
            error = ChatBadRequestError(
                provider=provider,
                message="The local model provider rejected the request.",
            )
        if status is not None:
            # As ChatProvider._raise_sanitized_provider_failure.
            error.upstream_status_code = status
        raise_detached_error(error)
```

Run the unit-file command. Expected: `6 passed` (the file's `400` and `5xx` tests, the three auth/rate-limit params, and the 404 test).

## Task 4: Stream frame code (TASK-13883)

**Files:**
- Modify: `tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`

- [ ] **Step 1 (RED): append to the test file** (add `import httpx` at the top):

```python
class _RejectingResponse(DummyResponse):
    def __init__(self, status_code: int):
        super().__init__([])
        request = httpx.Request("POST", "http://fake.local/v1/chat/completions")
        self._error = httpx.HTTPStatusError(
            "upstream body must not reach the frame",
            request=request,
            response=httpx.Response(status_code, request=request),
        )

    def raise_for_status(self) -> None:
        raise self._error


@pytest.mark.uni
@pytest.mark.parametrize(
    ("status_code", "code"),
    [
        (401, "provider_authentication_failed"),
        (403, "provider_authentication_failed"),
        (429, "provider_unavailable"),
        (404, "provider_unavailable"),
        (500, "provider_unavailable"),
    ],
)
def test_local_stream_error_frame_code_follows_the_upstream_status(status_code, code):
    def streamer(**_kwargs):
        return DummyStreamContext(_RejectingResponse(status_code))

    chunks = list(
        _chat_with_openai_compatible_local_server(
            api_base_url="http://fake.local",
            model_name="local-model",
            input_data=[{"role": "user", "content": "Ping"}],
            streaming=True,
            provider_name="TestLocalProvider",
            http_streamer=streamer,
            configured_endpoint_scope=ConfiguredEndpointScope.from_url("http://fake.local"),
        )
    )

    assert len(chunks) == 2
    frame = json.loads(chunks[0][len("data: ") : -2])
    assert frame["error"]["code"] == code
    assert "upstream body" not in chunks[0]
    assert chunks[1] == "data: [DONE]\n\n"
```

Run: `TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py`
Expected on `e06eeec68c`: `2 failed` (401, 403: code is `provider_unavailable`), 429/404/500 pass (they pin the unchanged behaviour).

- [ ] **Step 2 (GREEN): `local_adapters.py`.**

Add under `_LOCAL_PROVIDER_UNAVAILABLE_MESSAGE`:

```python
_LOCAL_PROVIDER_AUTH_MESSAGE = "The local model provider rejected the credentials."
```

Replace `_local_provider_stream_error_frame`:

```python
def _local_provider_stream_error_frame(provider_name: Any, status: int | None = None) -> str:
    """Build the canonical bounded SSE error for local-provider failures.

    A 401 or 403 reply gets the code a remote provider's stream gets for one; every other
    failure, a 429 included, is ``provider_unavailable``, as it is for a remote provider.
    """

    provider = _bounded_local_provider_name(provider_name)
    auth_failed = status in (401, 403)
    return sse_data(
        {
            "error": {
                "code": "provider_authentication_failed" if auth_failed else "provider_unavailable",
                "message": (
                    _LOCAL_PROVIDER_AUTH_MESSAGE if auth_failed else _LOCAL_PROVIDER_UNAVAILABLE_MESSAGE
                ),
                "type": f"{provider}_stream_error",
            }
        }
    )
```

In `stream_generator` (about line 468): add `error_status = None` after `error_pending = False`; in the `except Exception as exc:` branch, after `error_pending = True`, add `error_status = get_http_status_from_exception(exc)`; change the yield to `yield _local_provider_stream_error_frame(safe_provider_name, error_status)`.

Run the Step 1 command. Expected: `6 passed` (the existing test plus five).

## Task 5: Endpoint answers match the remote ones; summary cleanup (TASK-13883)

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py` (the endpoint test file was created in Task 3) (lines 169-181)

- [ ] **Step 1 (RED, already written and run in Task 3 Step 1b):** the endpoint file is authored and shown failing before any fix is applied; nothing to do here until Step 2.
- [ ] **Step 2 (GREEN): with Tasks 3 and 4 applied, run the file.** Expected: `8 passed`, and no teardown error from `no_live_network`.

- [ ] **Step 3: remove the now-dead summary branches.** In `Summarization_General_Lib.py`, replace the `ChatBadRequestError` branch (lines 169-181) with:

```python
    if isinstance(exc, ChatBadRequestError):
        upstream_status = getattr(exc, "upstream_status_code", None)
        # 408 (OpenRouter timeout) and 498 (Groq flex capacity) are transient 4xx replies.
        if upstream_status in {408, 498}:
            return "provider_failure"
        # The provider refused this request (an unknown model, a prompt over its context);
        # sending it again gets the same answer.
        return "request_rejected"
```

A 401, 403 or 429 reaches `ChatAuthenticationError` / `ChatRateLimitError` above it now. Run:
`TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py -k "local_provider_reply"`
Expected: `7 passed` per provider param (14 total): authentication for 401/403, rate_limit for 429, `provider_failure` for 408 and 500, `request_rejected` for 404 and 400, unchanged.

- [ ] **Step 4: Regression sweep (focused, no full suite).**
`TLDW_TEST_NO_DOCKER=1 <python> -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls tldw_Server_API/tests/LLM_Adapters tldw_Server_API/tests/Chat/unit/test_authoritative_adapter_translation.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_validation_boundary.py`
Expected: no failures. If one appears, run it on a clean `origin/dev` detached worktree first and record the baseline before blaming this change (Postgres and property tests skip without their env).
Also run Bandit on the touched source, from uvx as the implementer brief says (it is not installed in the venv): `uvx bandit -ll tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`; record the result, or the skip if uvx cannot run. Lint only the new and changed files: `<python> -m ruff check` on the two source files, the three new test files, `tests/helpers/offline_network.py`, `tests/LLM_Adapters/conftest.py` and the changed LLM_Calls test files, not the repo.

- [ ] **Step 5: Commit 2** (`git add` the two source files and the three test files by path).

## Task 6: Backlog close-out and follow-ups

Compute the ids at execution under the id lock (AGENTS.md "Task ids collide" command); write the next free ids below as TASK-<N>. Run from the repo root with `PYTHONPATH=tools/backlog-py/src <python> -m backlog_py --cwd . task ...`.

- [ ] **Step 1: TASK-13884.**
  - `task edit TASK-13884 -s "In Progress"` at start of work.
  - `--append-notes "RED (e06eeec68c): with no_live_network on both files, 6 failed, 45 passed, 6 errors (teardown assertion): the custom_openai and CustomOpenAIAdapter* cases fail EgressPolicyError('Host could not be resolved'); the only lookup attempted is api.openai.com (config.txt custom_openai_api_ip / custom_openai2_api_ip, read as the server-owned endpoint; the request base_url is stripped)."`
  - `--append-notes "GREEN: with loopback_custom_openai_endpoints pinning CUSTOM_OPENAI_API_IP and CUSTOM_OPENAI2_API_IP to http://127.0.0.1:9/v1, 51 passed and the fixture records no attempt. test_no_live_network_fixture.py (4 tests) proves the blocker blocks and records dns and connect, allows loopback, and restores. LLM_Adapters/unit + LLM_Calls: no failures."`
  - `--check-ac 1`, `--check-dod 1 --check-dod 2 --check-dod 4 --check-dod 6`; DoD #3 (documentation): no doc describes these tests, record "n/a" in the notes and check it. Bandit: test-only change, skip documented; check DoD #4 with that note.
  - `--final-summary "<what changed: helper, two fixtures, six cases now offline, how no live DNS is proven>"`, `--check-dod 5`, `-s Done`.
- [ ] **Step 2: TASK-13883.**
  - `-s "In Progress"`, then notes: RED counts (unit 3 failed, stream unit 2 failed, endpoint 5 failed 3 passed), the measured before/after table, and the AC #2 finding that the streamed 429 already equals the remote streamed 429 (502 `provider_unavailable`), pinned not changed.
  - GREEN notes: counts from Tasks 3 to 5 (6, 6, 8 passed; summary local cases 14 passed; sweep result), Bandit result.
  - `--check-ac 1 --check-ac 2`, DoD checked as above (documentation: no API doc describes the local error answers; grep of `Docs/` found only plan files), final summary, `-s Done`.
  - The ADR assessment goes in the notes (below).
- [ ] **Step 3: file the follow-ups** (below) with `task create ... --id TASK-<N>`.
- [ ] **Step 4:** `git add -A backlog/tasks`, Commit 3, push the branch, open the PR to `dev`. Per the memory rules the PR needs the owner's Change summary section (ADR-004) unless the owner extends the waiver; ask. After the clean-review subagent pass, `gh variable get MERGE_QUEUE` decides rebase versus auto-merge.

## Follow-ups to file at execution

1. `TASK-<N>` Chat: a streamed provider 429 answers 502 provider_unavailable for every provider. Measured on `e06eeec68c`: the remote streamed 429 (and the groq endpoint test `test_chat_endpoint_streaming_error_groq`) answers 502 `provider_unavailable`, while the non-stream 429 answers 429. `PROVIDER_STREAM_ERROR_MESSAGES` has no rate-limit code. This is what makes AC #2 "streamed" a no-op for local. A real fix is a product choice (new stream code, status, `Retry-After`), so it is its own task. Labels: llm, chat, errors.
2. `TASK-<N>` Tests: some LLM_Adapters/unit or LLM_Calls test attempts a live DNS lookup of `openaipublic.blob.core.windows.net` (a tiktoken vocabulary download) and passes only through a fallback. Measured under the blocker with those two directories (1451 tests). The tests (identified by the review): `LLM_Adapters/unit/test_llm_models_filters.py` (5 tests) and `test_llm_providers_capabilities_merge.py` (about 9 tests), through the `/llm/models` and `/llm/providers` tokenizer resolution. Labels: tests, llm.
3. `TASK-<N>` Tests: make `no_live_network` autouse for `tests/LLM_Adapters/unit` once item 2 is fixed (it fails today on exactly those tests; `LLM_Calls/test_provider_config_resolution.py` also records `slot-2.example`, which fails closed), so the next config-default leak fails at once instead of in a DNS-less environment. Only that directory plus `LLM_Calls` were measured; other directories were not.
4. Not filed: a Kobold-specific endpoint test (Review Focus). Covered by the shared function; file only if the reviewer asks.

## ADR assessmen

No new ADR. The change maps local failures onto existing public error types and an existing stream error code; it adds no durable architecture rule. Governing: ADR-025 (LLM provider adapter routing and overrides) concerns routing and overrides, not error typing, and is unaffected; ADR-059 governs the backlog edits. The Chat error vocabulary (`ChatAuthenticationError`, `ChatRateLimitError`, `provider_authentication_failed`) already exists and is reused unchanged. Re-run `grep -n -i "error" Docs/ADR/README.md` at execution in case a newer ADR landed.

## Decisions for the owner

None. Both tasks' ACs are settled by the code: AC #1 and #2 ask for the remote answer, and the remote answer is measured. The one product-shaped finding (a streamed 429 loses its 429 for every provider) is out of this plan's scope and is filed as a follow-up, not decided here.
