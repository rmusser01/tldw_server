"""Regression guard for TASK-13313 -- the production session object must be the tested one.

`chat_calls.create_session_with_retries` branched on the test environment and returned a
different class in each case, and said so in its own docstring:

    if _os.getenv("PYTEST_CURRENT_TEST"):
        return _legacy_create_session_with_retries(total=1)
    return _SessionShim(...)

PYTEST_CURRENT_TEST is set for every test, so `_SessionShim` -- the object every
production call gets -- was never constructed by the suite. Any defect introduced into
it was undetectable by CI. The providers routed through it are Cohere, Moonshot and Zai,
plus the two legacy OpenAI embeddings call sites in chat_calls itself.

The branch was not load-bearing. Both classes end at the same
`http_client.fetch(method="POST", ...)`; `_RetrySession` merely passes a dedicated
`client=`, and for streaming `_SessionShim` delegates to `_RetrySession` anyway. Tests
that need to control the session already monkeypatch `create_session_with_retries`
itself, which is unaffected by which class the unpatched function would have returned.
`test_provider_unsafe_post_no_retry.py` goes further and re-points it at the
`http_helpers` version explicitly -- the suite was already working around this gate.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.LLM_Calls import chat_calls

pytestmark = pytest.mark.unit


def test_the_factory_returns_the_production_object_under_pytest() -> None:
    session = chat_calls.create_session_with_retries(total=1)

    assert isinstance(session, chat_calls._SessionShim), (
        "create_session_with_retries handed back the legacy facade because "
        "PYTEST_CURRENT_TEST is set. The production object is then never constructed by "
        f"any test in the suite. Got {type(session).__name__}."
    )


def test_non_streaming_post_goes_through_the_central_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shim's whole purpose: egress policy and TLS pinning via http_client.fetch."""
    seen: dict[str, object] = {}

    def _fake_fetch(**kwargs):
        seen.update(kwargs)

        class _Resp:
            status_code = 200

            def json(self):
                return {"ok": True}

            def raise_for_status(self):
                return None

        return _Resp()

    monkeypatch.setattr(chat_calls, "fetch", _fake_fetch)

    session = chat_calls.create_session_with_retries(total=1)
    response = session.post(
        "https://provider.invalid/v1/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "m"},
        timeout=30,
    )

    assert response.status_code == 200
    assert seen["method"] == "POST"
    assert seen["url"] == "https://provider.invalid/v1/chat/completions"
    assert seen["json"] == {"model": "m"}
    assert seen["timeout"] == 30


def test_provider_posts_are_single_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pinned because the docstring commits to it: no idempotency contract exists.

    The `total=` argument is deliberately ignored; a future edit that starts honouring
    it would silently begin replaying non-idempotent provider POSTs.
    """
    seen: dict[str, object] = {}

    def _fake_fetch(**kwargs):
        seen.update(kwargs)

        class _Resp:
            status_code = 200

        return _Resp()

    monkeypatch.setattr(chat_calls, "fetch", _fake_fetch)

    chat_calls.create_session_with_retries(total=5).post(
        "https://provider.invalid/v1/chat/completions", json={}
    )

    assert seen["retry"].attempts == 1, (
        f"provider POST would be retried {seen['retry'].attempts} times; these requests "
        "are not idempotent"
    )


def test_streaming_still_uses_the_legacy_facade_for_iter_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the shim delegates streaming, which is why removing the branch is safe."""
    built: list[object] = []

    class _FakeLegacy:
        def post(self, url, *, headers=None, json=None, stream=False, timeout=None):
            built.append(("post", stream))
            return object()

    monkeypatch.setattr(
        chat_calls, "_legacy_create_session_with_retries", lambda **kw: _FakeLegacy()
    )

    chat_calls.create_session_with_retries(total=1).post(
        "https://provider.invalid/v1/chat/completions", json={}, stream=True
    )

    assert built == [("post", True)], (
        "streaming no longer delegates to the legacy facade, so iter_lines semantics "
        "are no longer preserved"
    )


def test_monkeypatching_the_factory_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam the suite actually uses, asserted so removing the branch cannot break it."""
    sentinel = object()
    monkeypatch.setattr(chat_calls, "create_session_with_retries", lambda **kw: sentinel)

    assert chat_calls.create_session_with_retries(total=1) is sentinel
