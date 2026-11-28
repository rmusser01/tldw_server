"""Slides standalone-HTML and MCP model completion moderate their raw HTTP calls (TASK-13950.27 PR I, R7)."""

# ruff: noqa: F811 - fixtures are imported by name, then requested as arguments

from __future__ import annotations

import json

import httpx
import pytest

from tldw_Server_API.app.core import http_client as hc
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import (
    ModelCompletionFailure,
    ModelFailureDomain,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_transport import (  # noqa: F401 - fixtures
    Body,
    Harness,
    _envelope,
    _policy,
    _request,
    credentials,
)
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_adapter import (  # noqa: F401 - fixtures
    harness,
    invoke as mcp_invoke,
)
from tldw_Server_API.tests.Slides.test_standalone_html_provider import (  # noqa: F401 - fixtures
    DOCUMENT,
    SYSTEM_PROMPT,
    _call,
    _config,
    _response,
    _response_body,
    _target,
    provider_module,
    standalone_egress_env,
)

pytestmark = pytest.mark.integration

ADAPTERS = {"openai": "openai_official_chat_v1", "anthropic": "anthropic_official_messages_v1"}


@pytest.fixture
def standalone_provider_egress(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "api.openai.com,api.anthropic.com")


@pytest.fixture
def evaluator_spy(monkeypatch):
    calls: list[str] = []
    for name in ("evaluate_text", "redact_text", "redact_text_with_count"):
        real = getattr(PolicyEvaluator, name)

        def spy(self, *a, _real=real, _name=name, **k):
            calls.append(_name)
            return _real(self, *a, **k)

        monkeypatch.setattr(PolicyEvaluator, name, spy)
    return calls


@pytest.fixture(autouse=True)
def mcp_network(monkeypatch):
    """DNS and egress mocked as the MCP transport's own tests do; requests stay native HTTPX on a MockTransport."""
    monkeypatch.delenv("HTTP_CERT_PINS", raising=False)

    async def validate(url, **kwargs):
        return None

    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)


# ---- Slides standalone HTML -----------------------------------------------------------------------------------


def _slides_transport(provider: str, text: str = DOCUMENT):
    posts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        response, _ = _response(_response_body(provider, text))
        return response

    return httpx.MockTransport(handler), posts


async def _slides(module, provider: str, *, text: str = DOCUMENT, user_content: str = "source text"):
    target = _target(ADAPTERS[provider])
    transport, posts = _slides_transport(provider, text)
    try:
        return await _call(module, _config(target), transport, user_content=user_content), posts
    except module.StandaloneHtmlProviderError as exc:
        return exc, posts


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_slides_standalone_input_block_no_post(provider, provider_module, moderation_hooks):
    outcome, posts = await _slides(provider_module, provider, user_content="forbidden words")

    assert isinstance(outcome, provider_module.StandaloneHtmlProviderError)
    assert outcome.code == "llm_policy_refused"
    assert posts == []


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_slides_standalone_input_redact_changes_the_posted_text(provider, provider_module, moderation_hooks):
    outcome, posts = await _slides(provider_module, provider, user_content="my secret plan")

    assert outcome == DOCUMENT.encode("utf-8")
    sent = json.loads(posts[0].content)["messages"]
    assert sent[-1]["content"] == "my [X] plan"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_slides_standalone_output_redact(provider, provider_module, moderation_hooks):
    document = DOCUMENT.replace("Deck</section>", "the secret deck</section>")

    outcome, _ = await _slides(provider_module, provider, text=document)

    assert outcome == document.replace("secret", "[X]").encode("utf-8")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_slides_standalone_output_block_is_a_refusal(provider, provider_module, moderation_hooks):
    document = DOCUMENT.replace("Deck</section>", "a forbidden deck</section>")

    outcome, _ = await _slides(provider_module, provider, text=document)

    assert isinstance(outcome, provider_module.StandaloneHtmlProviderError)
    assert outcome.code == "llm_policy_refused"

# The worker's classification of `llm_policy_refused` as terminal on its first attempt is asserted in
# tests/Slides/test_standalone_html_generation_jobs.py::test_provider_terminal_retention_starts_when_failure_occurs.


async def test_slides_standalone_policy_resolution_failure_is_a_terminal_refusal(provider_module, make_service, monkeypatch):
    """A policy that cannot be resolved fails closed as `llm_policy_refused`, not as a retried worker error."""
    import tldw_Server_API.app.core.Moderation.gateway_hooks as hooks

    make_service(enabled=True)
    monkeypatch.setattr(hooks, "resolve_policy", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("store down")))
    target = _target(ADAPTERS["openai"])

    with pytest.raises(provider_module.StandaloneHtmlProviderError) as text_error:
        provider_module._moderate_text("a generated document")
    with pytest.raises(provider_module.StandaloneHtmlProviderError) as request_error:
        provider_module._moderate_request(target, {"messages": [{"role": "user", "content": "x"}]})

    assert text_error.value.code == request_error.value.code == "llm_policy_refused"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_raw_http_global_off_zero_calls(provider, provider_module, make_service, evaluator_spy):
    make_service(enabled=False)

    outcome, posts = await _slides(provider_module, provider, user_content="forbidden secret words")

    assert outcome == DOCUMENT.encode("utf-8")
    assert json.loads(posts[0].content)["messages"][-1]["content"] == "forbidden secret words"
    assert evaluator_spy == []


def test_raw_http_off_returns_the_identical_objects(provider_module, make_service, evaluator_spy):
    make_service(enabled=False)
    target = _target(ADAPTERS["openai"])
    payload = {"model": "m", "messages": [{"role": "user", "content": "forbidden"}]}

    assert provider_module._moderate_request(target, payload) is payload
    assert provider_module._moderate_text("forbidden secret") == "forbidden secret"
    assert evaluator_spy == []


# ---- MCP model completion ---------------------------------------------------------------------------------------


def _refusal_code(error: ModelCompletionFailure) -> tuple[str, ModelFailureDomain]:
    return error.code, error.domain


async def test_mcp_model_completion_input_block_and_output_redact(mcp_network, credentials, moderation_hooks):
    harness = Harness(body=Body([json.dumps(_envelope("the secret answer")).encode()]))
    transport = harness.transport()

    with pytest.raises(ModelCompletionFailure) as blocked:
        await transport.complete(_request(user_prompt="forbidden words"), await credentials())
    assert _refusal_code(blocked.value) == ("model_policy_refused", ModelFailureDomain.REQUEST)
    assert harness.requests == []

    received: list[str] = []
    result = await transport.complete(
        _request(user_prompt="my secret plan"), await credentials(), on_receipt=received.append
    )
    assert json.loads(harness.requests[0].content)["messages"][-1]["content"] == "my [X] plan"
    assert result.content == "the [X] answer"
    assert received == ["the [X] answer"]


async def test_mcp_model_completion_output_block_is_a_refusal(mcp_network, credentials, moderation_hooks):
    harness = Harness(body=Body([json.dumps(_envelope("a forbidden answer")).encode()]))
    received: list[str] = []

    with pytest.raises(ModelCompletionFailure) as blocked:
        await harness.transport().complete(_request(), await credentials(), on_receipt=received.append)

    assert _refusal_code(blocked.value) == ("model_policy_refused", ModelFailureDomain.REQUEST)
    assert received == [], "the blocked text is not cached as the receipt"


async def test_mcp_model_completion_global_off_zero_calls(mcp_network, credentials, make_service, evaluator_spy):
    make_service(enabled=False)
    harness = Harness(body=Body([json.dumps(_envelope("forbidden secret answer")).encode()]))

    result = await harness.transport().complete(_request(user_prompt="forbidden secret"), await credentials())

    assert result.content == "forbidden secret answer"
    assert json.loads(harness.requests[0].content)["messages"][-1]["content"] == "forbidden secret"
    assert evaluator_spy == []


async def test_mcp_model_completion_moderates_for_the_identitys_user(harness, moderation_hooks):
    """The transport's moderation runs as the MCP invocation's user, so per-user overrides and Guardian apply."""
    from tldw_Server_API.app.core.Moderation.gateway_hooks import moderate_output_text_for_surface
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

    seen: list[int | None] = []
    moderated: list[tuple[str, str]] = []
    resolved: list[object] = []
    original_resolve = moderation_hooks.get_effective_policy

    def resolving(user_id, *, surface):
        resolved.append(user_id)
        return original_resolve(user_id, surface=surface)

    moderation_hooks.get_effective_policy = resolving
    original = harness.transport.complete

    async def spy(*args, **kwargs):
        seen.append(current_llm_user_id())
        # The real moderator, in the transport's task context: the policy is resolved for the identity's user.
        moderated.append(
            moderate_output_text_for_surface("a forbidden reply", surface="mcp.model_completion", user_id=current_llm_user_id())
        )
        return await original(*args, **kwargs)

    harness.transport.complete = spy
    result = await mcp_invoke(harness, harness.build())

    assert result.content == "valid content"
    assert seen == [harness.identity.user_id]
    assert moderated == [("Output violates moderation policy", "block")]
    assert resolved == [str(harness.identity.user_id)]
