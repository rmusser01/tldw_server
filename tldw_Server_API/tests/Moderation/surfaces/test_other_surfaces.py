"""The remaining gateway surfaces name their surface and a refusal propagates (TASK-13950.27 PR I)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import adapter_registry, gateway
from tldw_Server_API.tests.Moderation.conftest import PROVIDER, StubRegistry, reply

pytestmark = pytest.mark.integration

REFUSED_REPLY = {
    **reply("x"),
    "choices": [
        {"index": 0, "finish_reason": "content_filter",
         "message": {"role": "assistant", "content": "Output violates moderation policy"}}
    ],
}
HOSTILE = "internal.byok_probe"  # an exempt surface: were it honored, moderation would be skipped


@pytest.fixture
def spy(stub_llm_registry):
    """A test pre-hook recording ``call.surface`` for every gateway call."""
    surfaces: list[str | None] = []
    remove = gateway.add_policy_hooks(pre_call=lambda call: surfaces.append(call.surface))
    yield surfaces
    remove()


# ---- one driver per site ---------------------------------------------------------------------------------------
# A driver runs the real site once with ``text`` in the user message and raises LLMPolicyRefusal when the site
# reports a refusal. Sites that turn a refusal into their own result (chunking) raise it here.


def _definition(extra: dict[str, Any] | None = None):
    from tldw_Server_API.tests.Notifications.test_automation_executors import _definition

    return _definition({"provider": PROVIDER, "model": "m", **(extra or {})})


async def _automation(text: str, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Scheduled_Tasks.automation_executors import _run_generation

    return await _run_generation(_definition(extra), text)


async def _automation_envelope(text: str) -> Any:
    from tldw_Server_API.app.core.Scheduled_Tasks.automation_executors import _envelope_llm_call

    return await _envelope_llm_call(
        messages=[{"role": "user", "content": text}], api_provider=PROVIDER, model="m", llm_surface=HOSTILE
    )


async def _chunking(text: str) -> Any:
    from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import (
        AutoChunkBoundaryAssistantRequest,
        ChatAutoChunkBoundaryAssistant,
        _Availability,
    )

    assistant = ChatAutoChunkBoundaryAssistant(config_loader=dict)

    async def available(_request: Any) -> Any:
        return _Availability(True, provider=PROVIDER, model="m", api_key="k", app_config={})

    assistant._check_availability = available  # type: ignore[method-assign]
    result = await assistant.refine(
        AutoChunkBoundaryAssistantRequest(chunk_options={}, chunking_plan={}, extracted_text=text)
    )
    if result.fallback_reason == "ai_assist_policy_refused":
        raise LLMPolicyRefusal("refused")
    return result


async def _claims_service(text: str) -> Any:
    from tldw_Server_API.app.core.Claims_Extraction import claims_service

    return await asyncio.to_thread(
        claims_service._fva_claims_analyze_call, PROVIDER, text, None, "k", None, model_override="m"
    )


async def _research_verification(text: str) -> Any:
    """The claims-verification call of Research Workspace artifacts (PR G left it unnamed)."""
    from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import _default_analyze_fn

    return await asyncio.to_thread(_default_analyze_fn, PROVIDER, text, None, "k", None, model_override="m")


async def _ingestion_claims(text: str) -> Any:
    from tldw_Server_API.app.core.Claims_Extraction.ingestion_claims import extract_claims_for_chunks

    return await asyncio.to_thread(
        extract_claims_for_chunks,
        [{"text": text, "metadata": {"chunk_index": 0}}], extractor_mode="openai", max_per_chunk=2,
    )


def _prepared(text: str) -> Any:
    from tldw_Server_API.app.core.Notes_Graph.suggestion_capabilities import SuggestionCapabilityLimits
    from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint, split_evidence_windows
    from tldw_Server_API.app.core.Notes_Graph.suggestion_generation import build_generation_request
    from tldw_Server_API.app.core.Notes_Graph.suggestion_retrieval import RetrievalResult

    title = "A title"
    retrieval = RetrievalResult(
        source_note_id="source-note", source_fingerprint=content_fingerprint(title, text),
        source_windows=split_evidence_windows(note_id="source-note", title=title, content=text, max_windows=4, max_code_points=480),
        terms=("private", "source"), candidates=(), tag_catalog=(), backend_overfetch_count=0,
        excluded_oversized_candidate_count=0, projection_fresh=True, estimated_input_tokens=10,
    )
    return build_generation_request(
        retrieval=retrieval, source_title=title, source_content=text, limits=SuggestionCapabilityLimits()
    )


async def _notes_graph(text: str) -> Any:
    from tldw_Server_API.app.core.Notes_Graph.suggestion_generation import (
        GenerationProvider,
        SuggestionGenerationError,
        generate_suggestions_once,
    )

    provider = GenerationProvider(
        adapter="openai", model="gpt-test", api_key="k",
        app_config={"openai_api": {"api_base_url": "https://api.openai.com/v1"}},
        provider_capabilities={"supports_json_schema": True}, endpoint_url="https://api.openai.com/v1",
    )
    try:
        return await generate_suggestions_once(prepared=_prepared(text), provider=provider)
    except SuggestionGenerationError as exc:
        if exc.code == "notes_graph_provider_policy_refused":
            raise LLMPolicyRefusal("refused") from None
        raise


async def _rpg(text: str) -> Any:
    from tldw_Server_API.app.core.RPG.rules.answering import ChatRulesAnswerGenerator, RulesAnswerOptions
    from tldw_Server_API.tests.RPG.test_rpg_rules_answering import _item

    return await ChatRulesAnswerGenerator().generate(
        query=text, evidence=[_item()], options=RulesAnswerOptions(provider=PROVIDER, model="m")
    )


def _skill_context() -> Any:
    return SimpleNamespace(
        default_provider=PROVIDER, default_model="m", tool_executor=SimpleNamespace(), tool_definitions=[],
        user_id=1, client_id="c", max_tool_calls=1, api_key="k", app_config=None, available_tools=None,
    )


async def _skills(text: str) -> Any:
    from tldw_Server_API.app.core.Skills.skill_executor import SkillExecutor

    return await SkillExecutor()._execute_forked("s", text, [], None, _skill_context())


def _slides(text: str) -> Any:
    from tldw_Server_API.app.core.Slides.slides_generator import SlidesGenerator

    return SlidesGenerator().generate_from_text(
        source_text=text, title_hint=None, provider=PROVIDER, model="m", api_key="k", temperature=None,
        max_tokens=None, max_source_tokens=None, max_source_chars=None, enable_chunking=False,
        chunk_size_tokens=None, summary_tokens=None,
    )


async def _slides_driver(text: str) -> Any:
    return await asyncio.to_thread(_slides, text)


async def _web(text: str) -> Any:
    from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
    from tldw_Server_API.app.core.Web_Scraping.extraction.strategies.llm import call_llm_provider

    return await asyncio.to_thread(
        call_llm_provider,
        provider=PROVIDER, settings={"model": "m", "api_key": "k"}, messages=[{"role": "user", "content": text}],
        app_config={}, dependencies=build_default_dependencies(), stage="llm_extraction", url="https://example.test/",
    )


async def _manuscript_analysis(text: str, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Writing.manuscript_analysis import analyze_pacing

    result = await analyze_pacing(text, provider=PROVIDER, model="m", **extra)
    return result


async def _manuscript_job(text: str, **extra: Any) -> Any:
    from tldw_Server_API.app.core.Writing.manuscript_annotation_jobs import process_scene_annotation_review_job

    db = SimpleNamespace(get_scene=lambda _id: {"project_id": "p", "version": 1, "content_plain": text})
    return await process_scene_annotation_review_job(
        manuscript_db=db,
        job_payload={"project_id": "p", "scene_id": "s", "scene_version": 1, "provider": PROVIDER, "model": "m", **extra},
    )


async def _manuscript_selected(text: str) -> Any:
    from tldw_Server_API.app.core.Writing.manuscript_annotations import create_selected_text_review_annotation

    return await create_selected_text_review_annotation(
        manuscript_db=SimpleNamespace(), scene={"content_plain": text, "project_id": "p"}, scene_id="s",
        provider=PROVIDER, model="m", scene_version=1, start=0, end=1, selected_text=text[:1] or "x",
        category_hints=[], instruction=None,
    )


async def _vn_turn(text: str) -> Any:
    from tldw_Server_API.app.core.VN_Play.adapters import ChatVNPlayTurnAdapter

    context = SimpleNamespace(
        input_payload={"input_text": text}, scene_state=None, recent_events=[],
        session=SimpleNamespace(mode="freeform", settings={"provider": PROVIDER, "model": "m"}, id=1),
    )
    return await ChatVNPlayTurnAdapter().generate_turn(context)


async def _vn_generation(text: str, **extra: Any) -> Any:
    from tldw_Server_API.app.core.VN_Play.adapters import (
        ScriptedVNGenerationAdapter,
        VNGenerationAdapterError,
        VNGenerationCallRequest,
    )

    request = VNGenerationCallRequest(
        profile_snapshot={"definition": {"provider": PROVIDER, "model": "m", **extra}},
        messages=[{"role": "user", "content": text}], output_schema="s",
        usage_context={"vn_session_id": "v", **extra},
    )
    try:
        return await ScriptedVNGenerationAdapter().generate(request)
    except VNGenerationAdapterError as exc:
        if exc.public_error_code == "llm_policy_refused":
            raise LLMPolicyRefusal("refused") from None
        raise


# site -> (surface, driver, text reaches a user message, answer is parsed or returned)
SITES: dict[str, tuple[str, Callable[..., Awaitable[Any]], bool, bool]] = {
    "automation": ("scheduled_tasks.automation", _automation, True, True),
    "automation_envelope": ("scheduled_tasks.automation", _automation_envelope, True, False),
    "chunking": ("chunking.boundaries", _chunking, True, True),
    "claims_service": ("claims.extraction", _claims_service, True, True),
    "research_verification": ("claims.extraction", _research_verification, True, True),
    "ingestion_claims": ("claims.extraction", _ingestion_claims, True, True),
    "notes_graph": ("notes.graph_suggestions", _notes_graph, True, True),
    "rpg": ("rpg.rules", _rpg, True, True),
    "skills": ("skills.execution", _skills, False, True),  # the skill text is a system message: not moderated
    "slides": ("slides.generation", _slides_driver, True, True),
    "web": ("web.extraction", _web, True, True),
    "manuscript_analysis": ("writing.manuscript", _manuscript_analysis, True, True),
    "manuscript_job": ("writing.manuscript", _manuscript_job, True, True),
    "manuscript_selected": ("writing.manuscript", _manuscript_selected, True, True),
    "vn_turn": ("vn_play", _vn_turn, True, True),
    "vn_generation": ("vn_play", _vn_generation, True, True),
}


@pytest.fixture(autouse=True)
def provider(monkeypatch, stub_llm_registry):
    """The stub registry, with the real transport policy Notes graph requires; keys for sites that resolve their own."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    class _Registry(StubRegistry):
        def get_audited_call_policy_transport(self, name):
            return adapter_registry.get_registry().get_audited_call_policy_transport(name)

    registry = _Registry()
    monkeypatch.setattr(gateway, "_get_llm_registry", lambda: registry)
    return registry


@pytest.mark.parametrize("site", SITES)
async def test_surface_reaches_gateway(site, spy, provider):
    surface, driver, _, _ = SITES[site]
    with contextlib.suppress(Exception):  # the stub's answer is not each site's expected shape
        await driver("hello there")
    assert spy and set(spy) == {surface}
    assert provider.requests, "the call reached the provider stub"


@pytest.mark.parametrize("site", [name for name, spec in SITES.items() if spec[2]])
async def test_input_block_refuses_before_the_provider(site, moderation_hooks, provider):
    _, driver, _, _ = SITES[site]
    with pytest.raises(LLMPolicyRefusal):
        await driver("forbidden words")
    assert provider.requests == []


@pytest.mark.parametrize("site", [name for name, spec in SITES.items() if spec[3]])
async def test_output_block_refuses_and_is_not_parsed(site, moderation_hooks, provider):
    _, driver, _, _ = SITES[site]
    provider.reply = REFUSED_REPLY
    with pytest.raises(LLMPolicyRefusal):
        await driver("hello there")
    assert len(provider.requests) == 1, "a refused answer is never retried or re-asked"


async def test_claims_extraction_block_propagates_and_is_not_retried(moderation_hooks, provider):
    """Ingestion claims: a refusal is not the heuristic fallback, and the provider is asked once."""
    provider.reply = REFUSED_REPLY
    with pytest.raises(LLMPolicyRefusal):
        await _ingestion_claims("Alpha fact sentence for the claim. Beta fact sentence for the claim.")
    assert len(provider.requests) == 1


async def test_claims_engine_extraction_block_propagates(moderation_hooks, provider):
    """The engine's analyze path (`_fva_claims_analyze_call`) lets the refusal out of the engine's broad excepts."""
    from tldw_Server_API.app.core.Claims_Extraction import claims_service
    from tldw_Server_API.app.core.Claims_Extraction.claims_engine import LLMBasedClaimExtractor

    provider.reply = REFUSED_REPLY
    with pytest.raises(LLMPolicyRefusal):
        await LLMBasedClaimExtractor(claims_service._fva_claims_analyze_call).extract(
            "Alpha fact sentence. Beta fact sentence.", max_claims=2
        )
    assert len(provider.requests) == 1


async def test_slides_generation_output_redact(moderation_hooks, provider):
    provider.reply = reply(json.dumps({"title": "T", "slides": [{"title": "A", "content": "the secret reply"}]}))
    deck = await asyncio.to_thread(_slides, "hello there")
    assert deck["slides"][0]["content"] == "the [X] reply"


@pytest.mark.provider_inventory  # keeps the gateway's real model validation instead of the conftest stub
async def test_ingestion_claims_runs_the_gateway_model_validation(monkeypatch, provider):
    """R6: the model-inventory check the gateway adds now runs for ingestion claims, and passes for a listed model."""
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils
    from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

    asked: list[str] = []

    def discover(name, _key, **_kwargs):
        asked.append(name)
        return ModelDiscoveryResult("ready", ("listed-model",))

    monkeypatch.setattr(gateway, "discover_provider_models", discover)
    monkeypatch.setattr(adapter_utils, "resolve_provider_model", lambda *_a: "listed-model")

    await _ingestion_claims("Alpha fact sentence for the claim. Beta fact sentence for the claim.")

    assert asked == ["openai"]
    assert provider.requests[0]["model"] == "listed-model"


# ---- R10: no user- or config-controlled mapping sets the surface -----------------------------------------------


@pytest.mark.parametrize(
    ("surface", "driver"),
    [
        ("scheduled_tasks.automation", lambda: _automation("forbidden words", llm_surface=HOSTILE, call_args={"llm_surface": HOSTILE})),
        ("scheduled_tasks.automation", lambda: _automation_envelope("forbidden words")),
        ("writing.manuscript", lambda: _manuscript_job("forbidden words", llm_surface=HOSTILE)),
        ("vn_play", lambda: _vn_generation("forbidden words", llm_surface=HOSTILE)),
    ],
    ids=["automation", "automation_envelope", "manuscript_job", "vn_generation"],
)
async def test_a_smuggled_llm_surface_is_ignored_and_the_call_is_moderated(surface, driver, spy, moderation_hooks, provider):
    with pytest.raises(LLMPolicyRefusal):
        await driver()
    assert spy == [surface]
    assert provider.requests == []


async def test_a_skill_cannot_name_its_surface(spy, moderation_hooks, provider):
    """A skill supplies its own model, tools and text; a surface key in it is not read."""
    from tldw_Server_API.app.core.Skills.skill_executor import SkillExecutor

    skill = {"name": "s", "content": "do it", "context": "fork", "model": "m", "llm_surface": HOSTILE}
    with contextlib.suppress(Exception):
        await SkillExecutor().execute(skill, "", _skill_context())
    assert spy == ["skills.execution"]


# ---- refusal handling of the sites that turn a refusal into their own result ----------------------------------


async def test_chunking_refusal_is_a_labelled_fallback_not_a_provider_error(moderation_hooks, provider):
    from tldw_Server_API.app.core.Chunking.auto_boundary_assistant import (
        AutoChunkBoundaryAssistantRequest,
        ChatAutoChunkBoundaryAssistant,
        _Availability,
    )

    assistant = ChatAutoChunkBoundaryAssistant(config_loader=dict)

    async def available(_request: Any) -> Any:
        return _Availability(True, provider=PROVIDER, model="m", api_key="k", app_config={})

    assistant._check_availability = available  # type: ignore[method-assign]
    result = await assistant.refine(
        AutoChunkBoundaryAssistantRequest(chunk_options={}, chunking_plan={}, extracted_text="forbidden words")
    )
    assert (result.used_llm, result.fallback_reason) == (False, "ai_assist_policy_refused")
    assert provider.requests == []


async def test_writing_annotation_job_refusal_is_terminal_with_its_own_code(moderation_hooks, provider):
    from tldw_Server_API.app.core.exceptions import WritingAnnotationReviewJobError
    from tldw_Server_API.app.services.writing_annotation_review_jobs_worker import handle_writing_annotation_review_job

    db = SimpleNamespace(get_scene=lambda _id: {"project_id": "p", "version": 1, "content_plain": "forbidden words"})
    job = {
        "job_type": "writing_scene_annotation_review",
        "payload": {"project_id": "p", "scene_id": "s", "scene_version": 1, "provider": PROVIDER, "model": "m", "max_comments": 3},
    }
    from tldw_Server_API.app.core.Writing.manuscript_annotation_jobs import WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE

    job["job_type"] = WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE
    with pytest.raises(WritingAnnotationReviewJobError) as exc:
        await handle_writing_annotation_review_job(job, manuscript_db=db)
    assert (exc.value.retryable, exc.value.failure_code) == (False, "llm_policy_refused")
    assert provider.requests == []


async def test_manuscript_analysis_refusal_is_not_stored_as_an_analysis(moderation_hooks, provider):
    """The analysis helpers end in a broad except that stored ``{"error": ...}``; the refusal passes through."""
    with pytest.raises(LLMPolicyRefusal):
        await _manuscript_analysis("forbidden words")


async def test_web_extraction_refusal_is_not_retried(moderation_hooks, provider):
    with pytest.raises(LLMPolicyRefusal):
        await _web("forbidden words")
    assert provider.requests == []
