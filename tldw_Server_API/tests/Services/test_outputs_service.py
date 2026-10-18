from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization
from tldw_Server_API.app.services import outputs_service


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ingest_output_to_media_db_uses_media_repository_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_db = object()

    class _FakeRepo:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def add_media_with_keywords(self, **kwargs):
            self.calls.append(kwargs)
            return 88, "media-uuid", "stored"

    fake_repo = _FakeRepo()
    seen_db: list[object] = []

    def _fake_get_media_repository(db):
        seen_db.append(db)
        return fake_repo

    monkeypatch.setattr(outputs_service, "get_media_repository", _fake_get_media_repository, raising=False)

    media_id = await outputs_service._ingest_output_to_media_db(
        media_db=media_db,
        output_id=17,
        title="Weekly Briefing",
        content="Rendered body",
        output_type="briefing",
        output_format="md",
        storage_path="weekly.md",
        template_id=9,
        run_id=33,
        item_ids=[1, 2],
        tags=["watchlist", "briefing"],
        variant_of=5,
    )

    assert media_id == 88
    assert seen_db == [media_db]
    assert len(fake_repo.calls) == 1
    payload = fake_repo.calls[0]
    assert payload["url"] == "output://17"
    assert payload["title"] == "Weekly Briefing"
    assert payload["media_type"] == "output_briefing"
    assert payload["content"] == "Rendered body"
    assert payload["keywords"] == ["watchlist", "briefing"]
    assert payload["transcription_model"] == "output"
    assert payload["overwrite"] is False
    assert payload["ingestion_date"]
    assert json.loads(str(payload["safe_metadata"])) == {
        "output_id": 17,
        "output_type": "briefing",
        "output_format": "md",
        "storage_path": "weekly.md",
        "template_id": 9,
        "run_id": 33,
        "item_ids": [1, 2],
        "variant_of": 5,
    }


@pytest.mark.unit
def test_infer_output_tts_provider_from_model_omnivoice() -> None:
    assert outputs_service._infer_output_tts_provider_from_model("omnivoice") == "omnivoice"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_tts_generation_defaults_preserves_explicit_omnivoice() -> None:
    resolved_model, resolved_voice, resolved_speed = await outputs_service._resolve_tts_generation_defaults(
        tts_model="omnivoice",
        tts_voice=None,
        template_row=SimpleNamespace(
            metadata_json=json.dumps(
                {
                    "tts_default_model": outputs_service.DEFAULT_KITTEN_TTS_MODEL,
                    "tts_default_voice": outputs_service.DEFAULT_KITTEN_TTS_VOICE,
                    "tts_default_speed": 1.25,
                }
            )
        ),
    )

    assert resolved_model == "omnivoice"
    assert resolved_voice == "auto"
    assert resolved_speed == 1.25


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_tts_generation_defaults_template_selected_omnivoice_forces_auto_voice() -> None:
    resolved_model, resolved_voice, resolved_speed = await outputs_service._resolve_tts_generation_defaults(
        tts_model=None,
        tts_voice=None,
        template_row=SimpleNamespace(
            metadata_json=json.dumps(
                {
                    "tts_default_model": "omnivoice",
                    "tts_default_voice": "legacy-template-voice",
                    "tts_default_speed": 0.9,
                }
            )
        ),
    )

    assert resolved_model == "omnivoice"
    assert resolved_voice == "auto"
    assert resolved_speed == 0.9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_tts_generation_defaults_template_selected_omnivoice_preserves_explicit_voice() -> None:
    resolved_model, resolved_voice, resolved_speed = await outputs_service._resolve_tts_generation_defaults(
        tts_model=None,
        tts_voice="caller-selected",
        template_row=SimpleNamespace(
            metadata_json=json.dumps(
                {
                    "tts_default_model": "omnivoice",
                    "tts_default_voice": "legacy-template-voice",
                    "tts_default_speed": 0.9,
                }
            )
        ),
    )

    assert resolved_model == "omnivoice"
    assert resolved_voice == "caller-selected"
    assert resolved_speed == 0.9


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resolve_tts_generation_defaults_omnivoice_blank_voice_falls_back_to_auto() -> None:
    resolved_model, resolved_voice, resolved_speed = await outputs_service._resolve_tts_generation_defaults(
        tts_model="omnivoice",
        tts_voice="   ",
        template_row=SimpleNamespace(metadata_json=None),
    )

    assert resolved_model == "omnivoice"
    assert resolved_voice == outputs_service.DEFAULT_OMNIVOICE_TTS_VOICE
    assert resolved_speed is None


# --- LLM summaries through the real analyze() (TASK-13636) ---


class _Adapter:
    """The provider adapter below analyze(): records each request and replies or raises."""

    def __init__(self) -> None:
        self.reply = "Model summary."
        self.error: BaseException | None = None
        self.requests: list[dict] = []

    def chat(self, request, timeout=None):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": self.reply}}]}

    def sent(self) -> str:
        """The user message of the only request."""
        assert len(self.requests) == 1
        return self.requests[0]["messages"][0]["content"]


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> _Adapter:
    fake = _Adapter()
    monkeypatch.setattr(summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _provider: fake))
    return fake


def _refusal() -> LLMQuotaExceededError:
    return LLMQuotaExceededError(
        category="llm_tokens_month", current=12, limit=10, message="LLM token quota exceeded", retry_after=60
    )


_LLM = {"api_name": "openai", "model_override": "test-model"}


@pytest.mark.unit
def test_summarize_single_article_returns_the_model_summary_and_sends_the_text_once(adapter: _Adapter) -> None:
    assert outputs_service._summarize_single_article("Cedar article body.", **_LLM) == "Model summary."
    assert adapter.sent().count("Cedar article body.") == 1
    assert adapter.requests[0]["model"] == "test-model"
    assert adapter.requests[0]["system_message"] == outputs_service._OUTPUTS_SYSTEM_PROMPT


@pytest.mark.unit
def test_summarize_text_block_returns_the_model_summary_and_sends_the_text_once(adapter: _Adapter) -> None:
    assert outputs_service._summarize_text_block("- Cedar story", **_LLM) == "Model summary."
    assert adapter.sent().count("- Cedar story") == 1
    assert adapter.requests[0]["system_message"] == outputs_service._OUTPUTS_SYSTEM_PROMPT


@pytest.mark.unit
def test_summarize_single_article_sends_the_text_with_a_custom_prompt_that_has_no_placeholder(
    adapter: _Adapter,
) -> None:
    outputs_service._summarize_single_article("Cedar article body.", custom_prompt="Summarize in French.", **_LLM)
    assert "Summarize in French." in adapter.sent()
    assert adapter.sent().count("Cedar article body.") == 1


@pytest.mark.unit
def test_summarize_single_article_returns_nothing_when_the_provider_fails(adapter: _Adapter) -> None:
    # An excerpt of the article would render as if the model wrote it (TASK-13642).
    adapter.reply = ""  # no usable analysis: a provider failure
    assert outputs_service._summarize_single_article("Cedar article body.", **_LLM) == ""


@pytest.mark.unit
async def test_summarize_items_for_output_leaves_no_summary_when_the_provider_fails(adapter: _Adapter) -> None:
    adapter.reply = ""
    items = await outputs_service.summarize_items_for_output([{"id": 1, "content": "Cedar article body."}], **_LLM)
    assert items[0]["llm_summary"] == ""


@pytest.mark.unit
@pytest.mark.parametrize("helper", ["_summarize_single_article", "_summarize_text_block"])
def test_a_summary_that_starts_with_error_is_still_the_summary(adapter: _Adapter, helper: str) -> None:
    # Provider failures raise (raise_on_error), so a reply is the model's text, whatever it starts with.
    adapter.reply = "Error: budgets ran out, the article says, so the bridge work stopped."
    assert getattr(outputs_service, helper)("Cedar article body.", **_LLM) == adapter.reply


@pytest.mark.unit
def test_summarize_text_block_returns_nothing_when_analyze_returns_an_error(adapter: _Adapter) -> None:
    adapter.reply = ""
    assert outputs_service._summarize_text_block("- Cedar story", **_LLM) == ""


@pytest.mark.unit
def test_summarize_single_article_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
    adapter.error = _refusal()
    with pytest.raises(LLMQuotaExceededError):
        outputs_service._summarize_single_article("Cedar article body.", **_LLM)


@pytest.mark.unit
def test_summarize_text_block_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
    adapter.error = _refusal()
    with pytest.raises(LLMQuotaExceededError):
        outputs_service._summarize_text_block("- Cedar story", **_LLM)


@pytest.mark.unit
async def test_summarize_items_for_output_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
    adapter.error = _refusal()
    with pytest.raises(LLMQuotaExceededError):
        await outputs_service.summarize_items_for_output([{"id": 1, "content": "Cedar article body."}], **_LLM)


@pytest.mark.unit
async def test_generate_group_summaries_sends_text_with_braces_once(adapter: _Adapter) -> None:
    groups = [{"name": "Rust", "items": [{"title": "Rust {macros} and {}"}]}]
    result = await outputs_service.generate_group_summaries(groups, **_LLM)
    assert result[0]["summary"] == "Model summary."
    assert adapter.sent().count("Rust {macros} and {}") == 1


@pytest.mark.unit
async def test_generate_group_summaries_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
    adapter.error = _refusal()
    with pytest.raises(LLMQuotaExceededError):
        await outputs_service.generate_group_summaries([{"name": "g", "items": [{"title": "Cedar"}]}], **_LLM)


@pytest.mark.unit
async def test_generate_briefing_summary_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
    adapter.error = _refusal()
    with pytest.raises(LLMQuotaExceededError):
        await outputs_service.generate_briefing_summary([{"title": "Cedar"}], **_LLM)


@pytest.mark.unit
async def test_classify_items_by_topic_lets_a_quota_refusal_through(adapter: _Adapter) -> None:
    adapter.error = _refusal()
    with pytest.raises(LLMQuotaExceededError):
        await outputs_service.classify_items_by_topic([{"title": "Cedar"}], **_LLM)
