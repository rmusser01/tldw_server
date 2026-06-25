"""Tests for Jobs-backed manuscript scene annotation review helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper
from tldw_Server_API.app.core.Writing.manuscript_annotation_jobs import (
    WRITING_JOBS_DOMAIN,
    WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE,
    build_scene_annotation_review_job_payload,
    enqueue_scene_annotation_review_job,
    process_scene_annotation_review_job,
    writing_annotation_review_jobs_queue,
)


pytestmark = pytest.mark.unit


class _RecordingJobs:
    def __init__(self) -> None:
        self.created_jobs: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        for row in self.created_jobs:
            if row.get("idempotency_key") == kwargs.get("idempotency_key"):
                return row
        job_id = len(self.created_jobs) + 200
        row = {"id": job_id, "uuid": f"job-{job_id}", "status": "queued", **kwargs}
        self.created_jobs.append(row)
        return row


@pytest.fixture()
def db(tmp_path: Path) -> CharactersRAGDB:
    chacha = CharactersRAGDB(str(tmp_path / "writing-annotation-review-jobs.db"), client_id="review-jobs-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


@pytest.fixture()
def manuscript(db: CharactersRAGDB) -> tuple[ManuscriptDBHelper, dict[str, Any]]:
    helper = ManuscriptDBHelper(db)
    project_id = helper.create_project("Queued Review Novel")
    chapter_id = helper.create_chapter(project_id, "Chapter 1")
    scene_id = helper.create_scene(
        chapter_id,
        project_id,
        title="Opening",
        content_plain="Alpha beta gamma. Alpha beta delta. Omega closes.",
    )
    scene = helper.get_scene(scene_id)
    assert scene is not None
    return helper, {"project_id": project_id, "chapter_id": chapter_id, "scene": scene}


def _mock_llm_response(payload: object) -> dict[str, Any]:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def test_review_jobs_queue_defaults_and_respects_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WRITING_ANNOTATION_REVIEW_JOBS_QUEUE", raising=False)
    assert writing_annotation_review_jobs_queue() == "default"

    monkeypatch.setenv("WRITING_ANNOTATION_REVIEW_JOBS_QUEUE", " writing-ai ")
    assert writing_annotation_review_jobs_queue() == "writing-ai"

    monkeypatch.setenv("WRITING_ANNOTATION_REVIEW_JOBS_QUEUE", "   ")
    assert writing_annotation_review_jobs_queue() == "default"


def test_review_job_payload_is_sanitized_and_contains_review_settings() -> None:
    payload = build_scene_annotation_review_job_payload(
        project_id="project-1",
        scene_id="scene-1",
        scene_version=3,
        provider="openai",
        model="gpt-4o-mini",
        max_comments=4,
        category_filters=["clarity", "pacing"],
        review_focus="Check scene-level tension.",
    )

    assert payload == {
        "project_id": "project-1",
        "scene_id": "scene-1",
        "scene_version": 3,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_comments": 4,
        "category_filters": ["clarity", "pacing"],
        "review_focus": "Check scene-level tension.",
    }
    forbidden_payload_keys = {
        "owner_user_id",
        "user_id",
        "scene_text",
        "content_plain",
        "selected_text",
        "annotation_body",
        "suggested_fix",
        "raw_model_output",
    }
    assert forbidden_payload_keys.isdisjoint(payload)


def test_enqueue_review_job_uses_writing_domain_owner_metadata_and_stable_idempotency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WRITING_ANNOTATION_REVIEW_JOBS_QUEUE", "writing-review")
    jobs = _RecordingJobs()

    first = enqueue_scene_annotation_review_job(
        job_manager=jobs,
        owner_user_id="42",
        project_id="project-1",
        scene_id="scene-1",
        scene_version=7,
        provider="openai",
        model="gpt-4o-mini",
        max_comments=5,
        category_filters=["clarity"],
        review_focus="Focus on promises and payoff.",
    )
    second = enqueue_scene_annotation_review_job(
        job_manager=jobs,
        owner_user_id="42",
        project_id="project-1",
        scene_id="scene-1",
        scene_version=7,
        provider="openai",
        model="gpt-4o-mini",
        max_comments=5,
        category_filters=["clarity"],
        review_focus="Focus on promises and payoff.",
    )
    changed_focus = enqueue_scene_annotation_review_job(
        job_manager=jobs,
        owner_user_id="42",
        project_id="project-1",
        scene_id="scene-1",
        scene_version=7,
        provider="openai",
        model="gpt-4o-mini",
        max_comments=5,
        category_filters=["clarity"],
        review_focus="Focus only on pacing.",
    )

    created = jobs.created_jobs[0]
    assert first["id"] == second["id"]
    assert changed_focus["id"] != first["id"]
    assert created["domain"] == WRITING_JOBS_DOMAIN
    assert created["queue"] == "writing-review"
    assert created["job_type"] == WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE
    assert created["owner_user_id"] == "42"
    assert "owner_user_id" not in created["payload"]
    assert "scene-1" in created["idempotency_key"]
    assert "v7" in created["idempotency_key"]
    assert "openai" in created["idempotency_key"]
    assert "gpt-4o-mini" in created["idempotency_key"]
    assert "5" in created["idempotency_key"]


@pytest.mark.asyncio
async def test_process_scene_review_job_creates_bounded_anchored_annotations(
    manuscript: tuple[ManuscriptDBHelper, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper, ids = manuscript
    scene = ids["scene"]
    from tldw_Server_API.app.core.Chat import chat_service

    monkeypatch.setattr(
        chat_service,
        "perform_chat_api_call_async",
        AsyncMock(
            return_value=_mock_llm_response(
                {
                    "annotations": [
                        {
                            "category": "clarity",
                            "quote": "gamma",
                            "body": "Clarify why this image matters.",
                            "suggested_fix": "Tie the image to the choice.",
                        },
                        {
                            "category": "pacing",
                            "quote": "Omega",
                            "body": "This can wait until later.",
                        },
                    ]
                }
            )
        ),
    )

    result = await process_scene_annotation_review_job(
        manuscript_db=helper,
        job_payload={
            "project_id": ids["project_id"],
            "scene_id": scene["id"],
            "scene_version": scene["version"],
            "provider": "openai",
            "model": "gpt-4o-mini",
            "max_comments": 1,
            "category_filters": ["clarity", "pacing"],
            "review_focus": "Scene-level review.",
        },
    )

    assert len(result["created_annotation_ids"]) == 1
    assert result["diagnostics"] == []
    annotation = helper.get_annotation(result["created_annotation_ids"][0])
    assert annotation is not None
    assert annotation["source"] == "ai_scene_review"
    assert annotation["category"] == "clarity"
    assert annotation["selected_text"] == "gamma"
    assert annotation["anchor_start"] == scene["content_plain"].index("gamma")


@pytest.mark.asyncio
async def test_process_scene_review_job_suppresses_duplicate_open_annotations(
    manuscript: tuple[ManuscriptDBHelper, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper, ids = manuscript
    scene = ids["scene"]
    existing_start = scene["content_plain"].index("gamma")
    helper.create_annotation(
        project_id=ids["project_id"],
        target_type="scene",
        target_id=scene["id"],
        category="clarity",
        source="ai_scene_review",
        body="Clarify why this image matters.",
        scene_version=scene["version"],
        anchor_start=existing_start,
        anchor_end=existing_start + len("gamma"),
        selected_text="gamma",
    )
    from tldw_Server_API.app.core.Chat import chat_service

    monkeypatch.setattr(
        chat_service,
        "perform_chat_api_call_async",
        AsyncMock(
            return_value=_mock_llm_response(
                {
                    "annotations": [
                        {
                            "category": "clarity",
                            "quote": "gamma",
                            "body": "Clarify why this image matters.",
                        },
                        {
                            "category": "pacing",
                            "quote": "Omega",
                            "body": "Let this beat land more cleanly.",
                        },
                    ]
                }
            )
        ),
    )

    result = await process_scene_annotation_review_job(
        manuscript_db=helper,
        job_payload={
            "project_id": ids["project_id"],
            "scene_id": scene["id"],
            "scene_version": scene["version"],
            "provider": "openai",
            "model": "gpt-4o-mini",
            "max_comments": 5,
            "category_filters": [],
        },
    )

    assert len(result["created_annotation_ids"]) == 1
    annotation = helper.get_annotation(result["created_annotation_ids"][0])
    assert annotation is not None
    assert annotation["category"] == "pacing"
    assert annotation["selected_text"] == "Omega"


@pytest.mark.asyncio
async def test_process_scene_review_job_reports_stale_scene_version_without_llm_call(
    manuscript: tuple[ManuscriptDBHelper, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper, ids = manuscript
    scene = ids["scene"]
    from tldw_Server_API.app.core.Chat import chat_service

    llm_call = AsyncMock()
    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", llm_call)

    result = await process_scene_annotation_review_job(
        manuscript_db=helper,
        job_payload={
            "project_id": ids["project_id"],
            "scene_id": scene["id"],
            "scene_version": scene["version"] + 1,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "max_comments": 3,
            "category_filters": [],
        },
    )

    assert result["created_annotation_ids"] == []
    assert result["diagnostics"] == [
        {
            "code": "scene_version_mismatch",
            "message": "Scene changed before annotation review started.",
        }
    ]
    llm_call.assert_not_awaited()
