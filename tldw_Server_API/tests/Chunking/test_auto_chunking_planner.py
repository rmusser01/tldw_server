from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.Chunking.auto_planner import (
    AutoChunkingPlan,
    AutoChunkingProfile,
    AutoChunkingRequest,
    merge_profiles,
    plan_auto_chunking,
    plan_auto_chunking_request,
    profile_from_source,
    profile_from_text,
)

pytestmark = pytest.mark.unit


def test_planner_returns_no_auto_plan_when_chunking_is_disabled_legacy_or_manual():
    disabled = plan_auto_chunking(
        perform_chunking=False,
        chunking_mode="auto",
        media_type="document",
    )
    legacy = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode=None,
        media_type="document",
    )
    manual = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="manual",
        media_type="document",
    )

    assert disabled.chunk_options is None
    assert disabled.chunking_plan is None
    assert legacy.chunk_options is None
    assert legacy.chunking_plan is None
    assert manual.chunk_options is None
    assert manual.chunking_plan is None


def test_document_with_headings_prefers_structure_aware_and_goal_sizing():
    profile = AutoChunkingProfile(
        media_type="pdf",
        source_name="research-paper.pdf",
        text_length=25_000,
        has_headings=True,
        has_tables=True,
    )

    qa = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="pdf",
        goal="qa_search",
        profile=profile,
        template_name="academic_pdf",
    )
    navigation = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="pdf",
        goal="navigation_summary",
        profile=profile,
        template_name="academic_pdf",
    )

    assert qa.chunk_options == {
        "method": "structure_aware",
        "max_size": 700,
        "overlap": 140,
        "adaptive": False,
        "multi_level": False,
        "language": None,
    }
    assert qa.chunking_plan["template_name"] == "academic_pdf"
    assert "outline" in qa.chunking_plan["derived_views"]
    assert navigation.chunk_options["method"] == "structure_aware"
    assert navigation.chunk_options["max_size"] > qa.chunk_options["max_size"]
    assert navigation.chunking_plan["goal"] == "navigation_summary"
    json.dumps(navigation.chunking_plan)


def test_unstructured_document_falls_back_to_sentences_when_semantic_unavailable():
    decision = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="document",
        goal="balanced",
        profile=AutoChunkingProfile(media_type="document", text_length=9_000),
        semantic_available=False,
    )

    assert decision.chunk_options["method"] == "sentences"
    assert "semantic_unavailable" in decision.chunking_plan["fallback_reason"]
    assert decision.chunking_plan["used_llm"] is False


def test_audio_video_profiles_use_sentence_chunks_and_time_views():
    decision = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="video",
        goal="balanced",
        profile=AutoChunkingProfile(
            media_type="video",
            text_length=18_000,
            has_timecodes=True,
            has_speaker_labels=True,
            language="en",
        ),
    )

    assert decision.chunk_options["method"] == "sentences"
    assert decision.chunk_options["language"] == "en"
    assert "time_ranges" in decision.chunking_plan["derived_views"]
    assert "speaker_segments" in decision.chunking_plan["derived_views"]


def test_ebook_email_and_web_article_rules():
    ebook = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="ebook",
        profile=AutoChunkingProfile(media_type="ebook", has_chapters=True),
    )
    email = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="email",
        profile=AutoChunkingProfile(media_type="email", text_length=5_000),
    )
    web = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="web",
        profile=AutoChunkingProfile(
            media_type="web",
            source_name="https://example.test/post",
            has_headings=True,
            has_lists=True,
        ),
    )

    assert ebook.chunk_options["method"] == "ebook_chapters"
    assert "chapter_outline" in ebook.chunking_plan["derived_views"]
    assert email.chunk_options["method"] == "sentences"
    assert email.chunk_options["max_size"] == 1000
    assert email.chunk_options["overlap"] == 150
    assert "message_boundaries" in email.chunking_plan["derived_views"]
    assert web.chunk_options["method"] == "structure_aware"
    assert "section_titles" in web.chunking_plan["derived_views"]


def test_ai_and_template_fallback_reasons_are_recorded_without_llm_calls():
    decision = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="document",
        profile=AutoChunkingProfile(media_type="document", text_length=3_000),
        requested_llm=True,
        llm_available=False,
        template_status="error",
        template_error="classifier unavailable",
    )

    assert decision.chunking_plan["used_llm"] is False
    assert "ai_assist_unavailable" in decision.chunking_plan["fallback_reason"]
    assert "template_error" in decision.chunking_plan["fallback_reason"]
    assert "classifier unavailable" in decision.chunking_plan["rationale"]


def test_ai_assist_opt_in_without_adapter_preserves_deterministic_plan():
    profile = AutoChunkingProfile(
        media_type="document",
        text_length=24_000,
        has_headings=True,
        has_tables=True,
    )
    baseline = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="document",
        goal="qa_search",
        profile=profile,
        requested_llm=False,
        llm_available=False,
    )
    requested_without_adapter = plan_auto_chunking(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="document",
        goal="qa_search",
        profile=profile,
        requested_llm=True,
        llm_available=False,
    )

    assert requested_without_adapter.chunk_options == baseline.chunk_options
    assert requested_without_adapter.chunking_plan["method"] == baseline.chunking_plan["method"]
    assert requested_without_adapter.chunking_plan["max_size"] == baseline.chunking_plan["max_size"]
    assert requested_without_adapter.chunking_plan["overlap"] == baseline.chunking_plan["overlap"]
    assert requested_without_adapter.chunking_plan["used_llm"] is False
    assert "ai_assist_unavailable" in requested_without_adapter.chunking_plan["fallback_reason"]
    assert "no boundary adapter is available" in requested_without_adapter.chunking_plan["rationale"]


def test_profile_builders_detect_text_signals_and_merge_source_hints():
    source = profile_from_source(
        media_type="document",
        filename="report.md",
        title="Quarterly Report",
        language="en",
    )
    text = profile_from_text(
        "# Intro\n\n- first item\n- second item\n\n| A | B |\n| - | - |\n\n"
        "Speaker 1: hello\n00:01:03 topic shift\n\nChapter 2\n"
    )
    merged = merge_profiles(source, text)

    assert merged.media_type == "document"
    assert merged.source_name == "report.md"
    assert merged.title == "Quarterly Report"
    assert merged.language == "en"
    assert merged.has_headings is True
    assert merged.has_lists is True
    assert merged.has_tables is True
    assert merged.has_speaker_labels is True
    assert merged.has_timecodes is True
    assert merged.has_chapters is True


def test_request_and_plan_types_are_serializable_and_drive_planning():
    request = AutoChunkingRequest(
        perform_chunking=True,
        chunking_mode="auto",
        media_type="document",
        goal="balanced",
        profile=AutoChunkingProfile(media_type="document", has_headings=True),
        template_name="manual",
    )

    decision = plan_auto_chunking_request(request)
    plan = AutoChunkingPlan.from_metadata(decision.chunking_plan)

    assert decision.chunk_options["method"] == "structure_aware"
    assert plan.goal == "balanced"
    assert plan.template_name == "manual"
    assert plan.to_metadata() == decision.chunking_plan
    json.dumps(plan.to_metadata())
