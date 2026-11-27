"""The direct provider-adapter surfaces reach the gateway and moderate (TASK-13950.27 PR J).

Evaluations (judge calls exempt as ``internal.eval_judge``, generation as ``evaluations``), data tables, the
explainer, document and audio insights, web search, note titles, and summarization, which stays a direct adapter
call and is moderated through the gateway's public hook runners.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

from tldw_Server_API.app.core.exceptions import DataTablesJobError, EgressPolicyError, LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as sgl
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.tests.Media.test_document_insights_service_prompt import context  # noqa: F401  (fixture)
from tldw_Server_API.tests.Moderation.conftest import (
    DONE,
    PROVIDER,
    StubRegistry,
    finish_frame,
    openai_frame,
    reply,
    usage_frame,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[4]
HOSTILE = "internal.byok_probe"  # an exempt surface: were it honored, moderation would be skipped
REFUSED_REPLY = {
    **reply("x"),
    "choices": [
        {"index": 0, "finish_reason": "content_filter",
         "message": {"role": "assistant", "content": "Output violates moderation policy"}}
    ],
}
MODEL = {"v": "m"}  # the R6 validation test swaps in a model the stubbed inventory lists


@pytest.fixture
def spy(stub_llm_registry):
    """A test pre-hook recording ``call.surface`` for every gateway call."""
    surfaces: list[str | None] = []
    remove = gateway.add_policy_hooks(pre_call=lambda call: surfaces.append(call.surface))
    yield surfaces
    remove()


@pytest.fixture(autouse=True)
def provider(monkeypatch, stub_llm_registry):
    """The stub registry for the gateway and for summarization's direct adapter lookup."""
    for name in ("DEFAULT_MODEL_E2E_FAKE", "DEFAULT_MODEL_OPENAI"):
        monkeypatch.setenv(name, "m")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(sgl, "get_registry", lambda: stub_llm_registry)
    monkeypatch.setitem(MODEL, "v", "m")
    return stub_llm_registry


def _user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


# ---- one driver per site: runs the real site once and raises LLMPolicyRefusal for a refusal ---------------------


async def _eval_runner(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Evaluations import eval_runner

    return await asyncio.to_thread(
        eval_runner._call_adapter_text,
        api_endpoint=provider, messages_payload=_user(text), model=MODEL["v"], api_key="k", **extra,
    )


async def _ms_g_eval(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Evaluations import ms_g_eval

    return await asyncio.to_thread(
        ms_g_eval._call_adapter_text,
        api_endpoint=provider, messages_payload=_user(text), model=MODEL["v"], api_key="k", **extra,
    )


async def _wordbench(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Evaluations.wordbench_runner import WordBenchRunner

    return await WordBenchRunner(api_name=provider, api_key="k", model=MODEL["v"])._call_provider(
        provider, {"prompt": text, "logprobs": 3, **extra}
    )


async def _eval_cli(module: str, text: str, provider: str, **extra: Any) -> Any:
    import importlib

    helper = importlib.import_module(f"tldw_Server_API.app.core.Evaluations.cli.{module}")._call_adapter_text
    return await asyncio.to_thread(
        helper, api_endpoint=provider, messages_payload=_user(text), model=MODEL["v"], api_key="k",
        llm_surface="evaluations", **extra,
    )


async def _evals_cli(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    return await _eval_cli("evals_cli", text, provider, **extra)


async def _evals_cli_enhanced(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    return await _eval_cli("evals_cli_enhanced", text, provider, **extra)


async def _data_tables(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Data_Tables import jobs_worker

    jm = SimpleNamespace(update_job_progress=lambda *a, **k: None)
    with mock.patch.object(jobs_worker, "get_default_provider", lambda: provider), \
            mock.patch.object(jobs_worker, "resolve_provider_api_key", lambda *a, **k: ("k", None)):
        try:
            return await jobs_worker._generate_table_payload(jm, 1, MODEL["v"], text)
        except DataTablesJobError as exc:
            if str(exc) == "llm_policy_refused":
                assert exc.retryable is False
                raise LLMPolicyRefusal("refused") from None
            raise


async def _explainer(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Explainer.jobs import make_configured_explainer_generator

    env = {"EXPLAINER_GENERATOR_ENABLED": "1", "EXPLAINER_GENERATOR_PROVIDER": provider,
           "EXPLAINER_GENERATOR_MODEL": MODEL["v"], **extra}
    with mock.patch.dict(os.environ, env):
        generator = make_configured_explainer_generator()
    return await generator(SimpleNamespace(as_messages=lambda: _user(text)))


def _insights(provider: str = PROVIDER) -> Any:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Insights import (
        LiveInsightSettings,
        LiveMeetingInsights,
    )

    frames: list[dict[str, Any]] = []

    async def send_json(payload: dict[str, Any]) -> None:
        frames.append(payload)

    insights = LiveMeetingInsights(
        SimpleNamespace(send_json=send_json), LiveInsightSettings(enabled=True, provider=provider, model=MODEL["v"])
    )
    insights.frames = frames  # type: ignore[attr-defined]
    return insights


async def _audio_insights(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    return await _insights(provider)._call_llm(text, stage="live")


async def _web_apis(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs

    return await asyncio.to_thread(WebSearch_APIs.analyze_question, text, provider)


async def _web_search(text: str, provider: str = PROVIDER, **extra: Any) -> Any:
    from tldw_Server_API.app.core.WebSearch import Web_Search

    return await asyncio.to_thread(Web_Search.analyze_question, text, provider)


async def _note_title(text: str, provider: str = "openai", **extra: Any) -> Any:
    from tldw_Server_API.app.core.Writing.note_title import TitleGenOptions, _try_generate_title_llm

    return await asyncio.to_thread(
        _try_generate_title_llm, text, TitleGenOptions(), system_message="s", title_instruction="Title"
    )


# site -> (surface, driver, text reaches a user message, answer is parsed or returned)
SITES: dict[str, tuple[str, Callable[..., Awaitable[Any]], bool, bool]] = {
    "eval_runner": ("evaluations", _eval_runner, True, True),
    "wordbench": ("evaluations", _wordbench, True, True),
    "evals_cli": ("evaluations", _evals_cli, True, True),
    "evals_cli_enhanced": ("evaluations", _evals_cli_enhanced, True, True),
    "data_tables": ("data_tables.generation", _data_tables, True, True),
    "explainer": ("explainer", _explainer, True, True),
    "audio_insights": ("audio.insights", _audio_insights, True, True),
    # both put the question in the system prompt (the user message is fixed), so only the answer is moderated
    "web_apis": ("web.search", _web_apis, False, True),
    "web_search": ("web.search", _web_search, False, True),
    "note_title": ("writing.note_title", _note_title, True, True),
}


@pytest.mark.parametrize("site", [*SITES, "ms_g_eval"])
async def test_surface_reaches_gateway(site, spy, provider):
    surface, driver = ("internal.eval_judge", _ms_g_eval) if site == "ms_g_eval" else SITES[site][:2]
    with contextlib.suppress(Exception):  # the stub's answer is not each site's expected shape
        await driver("hello there")
    assert spy and set(spy) == {surface}
    assert provider.requests, "the call reached the provider stub"


@pytest.mark.parametrize("site", [name for name, spec in SITES.items() if spec[2]])
async def test_input_block_refuses_before_the_provider(site, moderation_hooks, provider):
    with pytest.raises(LLMPolicyRefusal):
        await SITES[site][1]("forbidden words")
    assert provider.requests == []


@pytest.mark.parametrize("site", [name for name, spec in SITES.items() if spec[3]])
async def test_output_block_refuses_and_is_not_parsed(site, moderation_hooks, provider):
    provider.reply = REFUSED_REPLY
    with pytest.raises(LLMPolicyRefusal):
        await SITES[site][1]("hello there")
    assert len(provider.requests) == 1, "a refused answer is never retried or re-asked"


@pytest.mark.parametrize("surface_site", ["eval_runner", "web_apis", "web_search"])
async def test_a_refusal_in_a_text_helper_is_not_a_provider_failure_or_empty_text(surface_site, moderation_hooks, provider):
    """The helpers return text; a refused answer must not come back as the refusal sentence or the response repr."""
    provider.reply = REFUSED_REPLY
    with pytest.raises(LLMPolicyRefusal):
        await SITES[surface_site][1]("hello there")


# ---- evaluations: the judge is exempt, the evaluated model is not ----------------------------------------------


async def test_eval_judge_is_exempt_and_generation_is_moderated(moderation_hooks, provider):
    assert await _ms_g_eval("forbidden words")  # the judge's text is neither blocked nor redacted
    assert len(provider.requests) == 1
    provider.requests.clear()
    with pytest.raises(LLMPolicyRefusal):
        await _eval_runner("forbidden words")
    assert provider.requests == []


async def test_a_judge_output_is_not_redacted(moderation_hooks, provider):
    provider.reply = reply("the secret reply")
    assert await _ms_g_eval("hello") == "the secret reply"
    assert await _eval_runner("hello") == "the [X] reply"


@pytest.mark.parametrize("what", ["input", "output"])
async def test_a_refused_generation_is_a_sample_error_never_a_score(what, moderation_hooks, provider, tmp_path):
    from tldw_Server_API.app.core.Evaluations.eval_runner import EvaluationRunner

    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    spec = {"api_name": PROVIDER, "model": "m", "api_key": "k", "allowed_labels": ["A", "B"]}
    question = "forbidden words" if what == "input" else "Which one?"
    provider.reply = REFUSED_REPLY if what == "output" else reply("A")
    results = [
        await runner._eval_label_choice({"input": {"question": question}, "expected": "A"}, spec, {"config": {}}, "s1"),
        await runner._eval_nli_factcheck(
            {"input": {"claim": question, "evidence": "e"}, "expected": "SUPPORTED"}, spec, {"config": {}}, "s2"
        ),
    ]
    for result in results:
        assert set(result) == {"sample_id", "error"}, result
    if what == "input":
        assert provider.requests == []


async def test_wordbench_refusal_is_an_error_result_not_an_analysis(moderation_hooks, provider):
    from tldw_Server_API.app.core.Evaluations.wordbench_runner import WordBenchRunner

    provider.reply = REFUSED_REPLY
    result = await WordBenchRunner(api_name=PROVIDER, api_key="k", model="m").analyze_prompt("hello")
    assert result["error"] and result["analysis"] is None and result["logprobs"] is None


EVALS_CLI_DIR = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Evaluations" / "cli"


@pytest.mark.parametrize("module", ["evals_cli", "evals_cli_enhanced"])
def test_the_eval_clis_name_a_judge_or_generation_surface_at_every_call(module):
    """The CLIs are operator tools and register no hooks; each call still names its surface, a literal."""
    tree = ast.parse((EVALS_CLI_DIR / f"{module}.py").read_text())
    surfaces = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_call_adapter_text":
            keyword = {k.arg: k.value for k in node.keywords}.get("llm_surface")
            assert isinstance(keyword, ast.Constant), f"{module}:{node.lineno} passes no literal llm_surface"
            surfaces.append(keyword.value)
    assert sorted(surfaces) == ["evaluations", "internal.eval_judge"]


@pytest.mark.parametrize("module", ["evals_cli", "evals_cli_enhanced"])
async def test_the_eval_cli_helpers_reach_the_gateway(module, spy, provider):
    import importlib

    helper = importlib.import_module(
        f"tldw_Server_API.app.core.Evaluations.cli.{module}"
    )._call_adapter_text
    for surface in ("internal.eval_judge", "evaluations"):
        spy.clear()
        await asyncio.to_thread(
            helper, api_endpoint=PROVIDER, messages_payload=_user("hi"), model="m", api_key="k", llm_surface=surface
        )
        assert spy == [surface]


# ---- the other surfaces' own refusal handling ------------------------------------------------------------------


async def test_data_table_refusal_fails_the_job_without_a_retry(moderation_hooks, provider):
    """A refused generation is a non-retryable DataTablesJobError, so the table is marked failed, not re-queued."""
    from tldw_Server_API.app.core.Data_Tables import jobs_worker

    jm = SimpleNamespace(update_job_progress=lambda *a, **k: None)
    with mock.patch.object(jobs_worker, "get_default_provider", lambda: PROVIDER), \
            mock.patch.object(jobs_worker, "resolve_provider_api_key", lambda *a, **k: ("k", None)):
        with pytest.raises(DataTablesJobError) as exc:
            await jobs_worker._generate_table_payload(jm, 1, "m", "forbidden words")
    assert (str(exc.value), exc.value.retryable) == ("llm_policy_refused", False)


async def test_explainer_refusal_carries_its_failure_code(moderation_hooks, provider):
    provider.reply = REFUSED_REPLY
    with pytest.raises(LLMPolicyRefusal) as exc:
        await _explainer("hello")
    assert (exc.value.failure_code, exc.value.retryable) == ("llm_policy_refused", False)


async def test_audio_insights_refusal_is_an_error_frame_with_its_code(moderation_hooks, provider):
    insights = _insights()
    await insights._generate_and_send([], stage="final", transcript_override="forbidden words")
    assert provider.requests == []
    assert [(f["type"], f.get("code")) for f in insights.frames] == [("insight_error", "llm_policy_refused")]


async def test_document_insights_input_block_is_a_400(moderation_hooks, provider, context):  # noqa: F811
    context.content = "forbidden words"
    response = context.client.post("/api/v1/media/7/insights", json={"force": True})
    assert response.status_code == 400, response.text
    assert provider.requests == []


async def test_document_insights_output_block_is_a_400_and_not_cached(moderation_hooks, provider, context):  # noqa: F811
    provider.reply = REFUSED_REPLY
    response = context.client.post("/api/v1/media/7/insights", json={"force": True})
    assert response.status_code == 400, response.text
    assert context.cache == {}


async def test_document_insights_reaches_gateway(spy, provider, context):  # noqa: F811
    context.client.post("/api/v1/media/7/insights", json={"force": True})
    assert spy == ["media.document_insights"]


async def test_note_title_input_block_propagates(moderation_hooks, provider):
    """The helper ends in a broad except that fell back to the heuristic; the refusal passes through."""
    with pytest.raises(LLMPolicyRefusal):
        await _note_title("forbidden words")
    assert provider.requests == []


# ---- R6: the gateway's model validation runs for each moved site ------------------------------------------------


@pytest.fixture
def listed_model(monkeypatch):
    """The gateway's inventory lookup lists ``listed-model`` for every provider; returns the providers it was asked."""
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils
    from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

    asked: list[str] = []

    def discover(name, _key, **_kwargs):
        asked.append(name)
        return ModelDiscoveryResult("ready", ("listed-model",))

    monkeypatch.setattr(gateway, "discover_provider_models", discover)
    monkeypatch.setitem(MODEL, "v", "listed-model")
    monkeypatch.setenv("DEFAULT_MODEL_OPENAI", "listed-model")
    monkeypatch.setattr(adapter_utils, "resolve_provider_model", lambda *_a: "listed-model")  # over the configured model
    monkeypatch.setattr(  # web search's import-time config snapshot has no provider endpoint under pytest
        "tldw_Server_API.app.core.WebSearch.Web_Search.loaded_config_data",
        {"openai_api": {"api_base_url": "https://api.openai.com/v1"}},
    )
    return asked


@pytest.mark.provider_inventory  # keeps the gateway's real model validation instead of the conftest stub
@pytest.mark.parametrize("site", [*SITES, "ms_g_eval"])
async def test_the_gateway_model_validation_runs_for_each_moved_site(site, listed_model, provider):
    """R6: the model-inventory check the gateway adds runs for the moved site, and passes for a listed model."""
    driver = _ms_g_eval if site == "ms_g_eval" else SITES[site][1]
    with contextlib.suppress(Exception):  # the stub's answer is not each site's expected shape
        await driver("hello there", provider="openai") if site != "note_title" else await driver("hello there")
    assert "openai" in listed_model
    assert provider.requests and provider.requests[0]["model"] == "listed-model"


# ---- R10: no user- or config-controlled mapping sets the surface -----------------------------------------------


@pytest.mark.parametrize(
    ("surface", "driver", "extra"),
    [
        ("evaluations", _eval_runner, {"llm_surface": HOSTILE}),
        ("internal.eval_judge", _ms_g_eval, {"llm_surface": HOSTILE}),
        ("evaluations", _wordbench, {"llm_surface": HOSTILE}),
    ],
    ids=["eval_runner", "ms_g_eval", "wordbench"],
)
async def test_a_smuggled_llm_surface_is_ignored(surface, driver, extra, spy, provider):
    with contextlib.suppress(Exception):
        await driver("hello there", **extra)
    assert spy == [surface]


async def test_a_smuggled_llm_surface_does_not_unmoderate_a_generation_call(moderation_hooks, provider):
    """A surface key in the helper's extra kwargs would otherwise reach the gateway and skip the policy."""
    with pytest.raises(LLMPolicyRefusal):
        await _eval_runner("forbidden words", llm_surface=HOSTILE)
    assert provider.requests == []


async def test_a_smuggled_llm_surface_in_the_audio_adapter_kwargs_is_dropped(spy, moderation_hooks, provider):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Insights import LiveMeetingInsights

    with pytest.raises(LLMPolicyRefusal):
        await asyncio.to_thread(
            LiveMeetingInsights._adapter_chat_call,
            api_endpoint=PROVIDER, messages_payload=_user("forbidden words"), model="m", temp=0.1, max_tokens=5,
            llm_surface=HOSTILE,
        )
    assert spy == ["audio.insights"] and provider.requests == []


@pytest.mark.parametrize("module", ["WebSearch.Web_Search", "Web_Scraping.WebSearch_APIs"])
async def test_a_smuggled_llm_surface_through_the_web_summarize_wrapper_is_moderated(module, moderation_hooks, provider):
    import importlib

    mod = importlib.import_module(f"tldw_Server_API.app.core.{module}")
    with pytest.raises(LLMPolicyRefusal):
        await asyncio.to_thread(
            mod.summarize, input_data="forbidden words", custom_prompt_arg=None, api_name=PROVIDER, api_key="k",
            model_override="m", app_config={}, input_is_literal_text=True, llm_surface=HOSTILE,
        )
    assert provider.requests == []


@pytest.mark.parametrize("module", ["WebSearch.Web_Search", "Web_Scraping.WebSearch_APIs"])
def test_the_web_search_summarize_wrapper_cannot_be_told_its_surface(module, monkeypatch):
    import importlib

    mod = importlib.import_module(f"tldw_Server_API.app.core.{module}")
    seen: dict[str, Any] = {}
    monkeypatch.setattr(mod, "analyze", lambda **kwargs: seen.update(kwargs) or "ok")
    assert mod.summarize(input_data="x", custom_prompt_arg=None, api_name=PROVIDER, llm_surface=HOSTILE) == "ok"
    assert seen["llm_surface"] == "web.search"


# ---- summarization: direct by design, moderated through the gateway's hook runners ------------------------------


def _analyze(text: str = "hello there", *, streaming: bool = False, **extra: Any) -> Any:
    return sgl.analyze(
        PROVIDER, text, None, api_key="k", model_override="m", app_config={}, input_is_literal_text=True,
        streaming=streaming, **extra,
    )


def _drain(result: Any) -> str:
    return "".join(result) if not isinstance(result, str) else result


def test_summarization_imports_no_moderation():
    """Layering: core/LLM_Calls never imports core/Moderation; a fresh interpreter proves the import graph."""
    code = (
        "import sys\n"
        "import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib\n"
        "bad = sorted(m for m in sys.modules if m.startswith('tldw_Server_API.app.core.Moderation'))\n"
        "assert not bad, bad\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "TEST_MODE": "true", "DISABLE_HEAVY_STARTUP": "1"}
    done = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr[-800:]


async def test_summarization_reaches_the_gateway_hooks_with_its_surface(spy, provider):
    _analyze()
    _analyze(llm_surface="rag.auxiliary")
    assert spy == ["media.summarization", "rag.auxiliary"]


@pytest.mark.parametrize("streaming", [False, True], ids=["chat", "stream"])
@pytest.mark.parametrize("raise_on_error", [False, True])
async def test_analyze_input_block_and_output_redact(streaming, raise_on_error, moderation_hooks, provider):
    with pytest.raises(LLMPolicyRefusal) as exc:  # not wrapped as a SummaryProviderError, even when it raises typed
        _drain(_analyze("forbidden words", streaming=streaming, raise_on_error=raise_on_error))
    assert exc.value.detail == "Input violates moderation policy"
    assert provider.requests == []

    provider.reply = reply("the secret reply")
    provider.items = [openai_frame("the secret reply"), finish_frame(), usage_frame(), DONE]
    assert _drain(_analyze(streaming=streaming, raise_on_error=raise_on_error)) == "the [X] reply"


@pytest.mark.parametrize("streaming", [False, True], ids=["chat", "stream"])
async def test_analyze_output_block_is_a_refusal_not_a_summary(streaming, moderation_hooks, provider):
    provider.reply = REFUSED_REPLY
    provider.items = [openai_frame("forbidden words in the reply"), finish_frame(), DONE]
    with pytest.raises(LLMPolicyRefusal):
        _drain(_analyze(streaming=streaming, raise_on_error=True))
    assert len(provider.requests) == 1


async def test_analyze_with_moderation_off_is_unchanged(provider):
    provider.reply = reply("the secret reply")
    assert _analyze() == "the secret reply"


async def test_analyze_still_refuses_disabled_model_and_keeps_error_codes(moderation_hooks, provider, monkeypatch):
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )

    set_llm_provider_overrides_cache_for_tests({PROVIDER: LLMProviderOverride(provider=PROVIDER, denied_models=["m"])})
    with pytest.raises(sgl.SummaryProviderError) as refused:
        _analyze(raise_on_error=True)
    assert refused.value.code == "configuration"
    assert provider.requests == []
    set_llm_provider_overrides_cache_for_tests({})

    for reason in ("dns_changed", "tls_pin_error"):
        def blocked(self, request, *, timeout=None, _reason=reason):
            raise EgressPolicyError("blocked", reason_code=_reason)

        monkeypatch.setattr(StubRegistry, "chat", blocked)
        with pytest.raises(sgl.SummaryProviderError) as failed:
            _analyze(raise_on_error=True)
        assert failed.value.code == "provider_failure", reason  # transient: stays retryable


# ---- analyze() importers: a refusal reaches the caller's refusal path, not a swallowed default -----------------


async def test_workflow_summarize_step_block_fails_step(moderation_hooks, provider):
    from tldw_Server_API.app.core.Workflows.adapters.content.summarize import run_summarize_adapter

    ctx = {"user_id": "1", "workflow_id": "w", "step_run_id": "s"}
    with pytest.raises(LLMPolicyRefusal):
        await run_summarize_adapter({"text": "forbidden words", "api_name": PROVIDER}, ctx)
    assert provider.requests == []


async def test_workflow_summarize_step_reaches_gateway(spy, provider):
    from tldw_Server_API.app.core.Workflows.adapters.content.summarize import run_summarize_adapter

    await run_summarize_adapter({"text": "hello", "api_name": PROVIDER}, {"user_id": "1", "workflow_id": "w", "step_run_id": "s"})
    assert spy == ["workflows.step"]


async def test_media_ingest_summary_block_is_recorded_not_retried(moderation_hooks, provider, tmp_path):
    """A blocked chunk analysis is a recorded warning on the item, which keeps its content; nothing is re-asked."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Plaintext.Plaintext_Files import process_document_content

    source = tmp_path / "doc.txt"
    source.write_text("forbidden words in the document", encoding="utf-8")
    result = await asyncio.to_thread(
        process_document_content,
        doc_path=source, perform_chunking=False, chunk_options=None, perform_analysis=True, summarize_recursively=False,
        api_name=PROVIDER, api_key=None, custom_prompt=None, system_prompt=None, title_override=None,
        author_override=None, keywords=None, base_dir=tmp_path,
    )
    assert "forbidden words" in result["content"]
    assert any("moderation policy" in warning for warning in result["warnings"]), result["warnings"]
    assert provider.requests == []


def test_rag_gap_analysis_names_the_auxiliary_surface():
    """The RAG gap-analysis call passes ``rag.auxiliary``, not the summarization default."""
    tree = ast.parse((REPO_ROOT / "tldw_Server_API/app/core/RAG/rag_service/pipeline/augment.py").read_text())
    surfaces = [
        {k.arg: k.value for k in node.keywords}.get("llm_surface")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "llm_analyze"
    ]
    assert [getattr(s, "value", None) for s in surfaces] == ["rag.auxiliary"]


@pytest.mark.provider_inventory
async def test_document_insights_runs_the_gateway_model_validation(listed_model, provider, context):  # noqa: F811
    """R6: the model-inventory check runs for the endpoint, and passes for a listed model."""
    context.client.post("/api/v1/media/7/insights", json={"force": True, "model": "listed-model"})
    assert "openai" in listed_model
    assert provider.requests and provider.requests[0]["model"] == "listed-model"


# ---- fix round 1 -------------------------------------------------------------------------------------------------


def test_ms_g_eval_resolved_credentials_without_a_handle_or_key_do_not_fall_back_to_the_server_key(provider):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Evaluations import ms_g_eval

    with pytest.raises(ChatConfigurationError):
        ms_g_eval._call_adapter_text(
            api_endpoint="openai", messages_payload=_user("hi"), model="m", api_key=None, credentials_resolved=True
        )
    assert provider.requests == []


async def test_qa3_refused_prediction_is_a_sample_error_not_an_aborted_run_or_a_label(moderation_hooks, provider, tmp_path):
    from tldw_Server_API.app.core.Evaluations.unified_evaluation_service import UnifiedEvaluationService

    service = UnifiedEvaluationService(db_path=str(tmp_path / "e.db"), enable_webhooks=False)

    async def store(**_kwargs: Any) -> str:
        return "eval-1"

    service._store_evaluation_result = store  # type: ignore[method-assign]
    provider.reply = reply("SUPPORTED")
    items = [{"id": 1, "question": "forbidden words", "label": "SUPPORTED"}, {"id": 2, "question": "fine", "label": "SUPPORTED"}]
    out = await service.evaluate_qa3(items, generate_predictions=True, api_name=PROVIDER)
    results = out["results"]["results"]
    assert results[0]["error"] == "llm_policy_refused" and "pred" not in results[0]
    assert results[1]["pred"] == "SUPPORTED"
    assert out["results"]["accuracy"] == 1.0  # the refused sample is in no average


async def test_xml_import_records_a_refused_summary_without_storing_it(moderation_hooks, provider, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import XML_Ingestion_Lib as xml

    stored: dict[str, Any] = {}

    class _Repo:
        def add_media_with_keywords(self, **kwargs: Any) -> str:
            stored.update(kwargs)
            return "ok"

    class _Ctx:
        def __enter__(self):
            return object()

        def __exit__(self, *_a):
            return False

    monkeypatch.setattr(xml, "managed_media_database", lambda **_k: _Ctx())
    monkeypatch.setattr(xml, "get_media_repository", lambda _db: _Repo())
    monkeypatch.setattr(xml, "analyze", lambda *_a, **_k: (_ for _ in ()).throw(LLMPolicyRefusal("Input violates moderation policy")))
    source = tmp_path / "d.xml"
    source.write_text("<root><a>forbidden words</a></root>", encoding="utf-8")
    message = await asyncio.to_thread(
        xml.import_xml_handler, str(source), "t", "a", "", None, None, True, PROVIDER, "k"
    )
    assert stored.get("summary") == "No summary provided"
    assert "refused by moderation policy" in message


# ---- RAG optional stages: a refusal is "unverified/skipped", never a score or a heuristic grade -------------------


async def _refusing(*_a: Any, **_k: Any) -> Any:
    raise LLMPolicyRefusal("Input violates moderation policy")


@pytest.mark.parametrize("propagate_errors", [False, True])
async def test_faithfulness_refusal_is_unscored_not_zero(propagate_errors):
    from tldw_Server_API.app.core.RAG.rag_service.faithfulness import FaithfulnessEvaluator

    class _LLM:
        async def generate(self, prompt: str) -> str:
            raise LLMPolicyRefusal("Input violates moderation policy")

    result = await FaithfulnessEvaluator(_LLM(), propagate_errors=propagate_errors).evaluate_detailed("An answer.", "Some context.")
    assert result.score is None
    assert result.to_dict()["failure_code"] == "llm_policy_refused" and result.to_dict()["verification_available"] is False


async def test_faithfulness_refused_verification_is_not_an_unsupported_claim():
    from tldw_Server_API.app.core.RAG.rag_service.faithfulness import FaithfulnessEvaluator

    calls = []

    class _LLM:
        async def generate(self, prompt: str) -> str:
            calls.append(prompt)
            if len(calls) == 1:
                return '["Paris is the capital of France"]'
            raise LLMPolicyRefusal("Input violates moderation policy")

    result = await FaithfulnessEvaluator(_LLM()).evaluate_detailed("An answer.", "Some context.")
    assert result.score is None and result.claims == []


async def test_document_grader_refusal_is_a_labelled_no_signal_fallback_not_a_grade():
    from types import SimpleNamespace

    from tldw_Server_API.app.core.RAG.rag_service.document_grader import DocumentGrader

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise LLMPolicyRefusal("Input violates moderation policy")

    grader = DocumentGrader(analyze_fn=refuse)
    graded = await grader.grade_document("q", SimpleNamespace(id="d1", content="text", score=0.9))
    assert graded.method == "score_fallback"  # the retrieval score stands; no LLM verdict
    assert graded.metadata == {"error": "llm_policy_refused", "verification_available": False}


async def test_knowledge_strips_refusal_skips_the_stage_and_keeps_the_documents():
    from tldw_Server_API.app.core.RAG.rag_service.knowledge_strips import KnowledgeStripsProcessor
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

    def refuse(*_a: Any, **_k: Any) -> Any:
        raise LLMPolicyRefusal("Input violates moderation policy")

    docs = [Document(id="d1", content="Some content about the topic. " * 30, source=DataSource.MEDIA_DB, score=0.5, metadata={})]
    result = await KnowledgeStripsProcessor(analyze_fn=refuse, llm_provider=PROVIDER, llm_model="m").process("topic", docs)
    assert result.documents == docs and result.strips == []
    assert result.metadata["failure_code"] == "llm_policy_refused" and result.metadata["verification_available"] is False
