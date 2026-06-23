"""Unit tests for Audio Studio API schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.audio_studio_schemas import (
    AudioStudioClipUpsert,
    AudioStudioExportCreate,
    AudioStudioGenerationCreate,
    AudioStudioProjectArchiveRequest,
    AudioStudioProjectCreate,
    AudioStudioProjectUpdate,
    AudioStudioRenderCreate,
    AudioStudioResourceKind,
    AudioStudioSectionUpsert,
    AudioStudioTrackUpsert,
    AudioStudioWorkflow,
)


pytestmark = pytest.mark.unit


def test_workflow_and_resource_enums_are_stable_strings() -> None:
    assert [workflow.value for workflow in AudioStudioWorkflow] == [
        "narration",
        "podcast",
        "briefing",
        "music",
    ]
    assert [kind.value for kind in AudioStudioResourceKind] == [
        "section",
        "track",
        "clip",
        "artifact",
        "render",
        "export",
    ]
    assert isinstance(AudioStudioWorkflow.NARRATION.value, str)


@pytest.mark.parametrize("workflow", ["narration", "podcast", "briefing", "music"])
def test_project_create_accepts_first_class_workflows(workflow: str) -> None:
    payload = AudioStudioProjectCreate(title=f"{workflow} project", workflow=workflow)

    assert payload.workflow.value == workflow
    assert payload.title == f"{workflow} project"


def test_project_update_requires_base_revision_id() -> None:
    with pytest.raises(ValidationError, match="base_revision_id"):
        AudioStudioProjectUpdate(title="Renamed")

    payload = AudioStudioProjectUpdate(title="Renamed", base_revision_id="rev_001")
    assert payload.base_revision_id == "rev_001"


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (AudioStudioSectionUpsert, {"title": "Intro", "body_text": "Hello"}),
        (AudioStudioTrackUpsert, {"name": "Narration", "kind": "speech"}),
        (AudioStudioClipUpsert, {"title": "Clip", "track_id": "trk_1", "clip_type": "speech"}),
    ],
)
def test_resource_upserts_require_base_revision_id(model, payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="base_revision_id"):
        model(**payload)

    assert model(**payload, base_revision_id="rev_001").base_revision_id == "rev_001"


@pytest.mark.parametrize("model", [AudioStudioGenerationCreate, AudioStudioRenderCreate, AudioStudioExportCreate])
def test_job_create_requests_require_bounded_idempotency_key(model) -> None:
    valid_payload = {
        "idempotency_key": "client-key-123456",
        "target_resource_kind": "section",
        "target_resource_id": "sec_001",
        "target_revision_id": "rev_001",
    }
    if model is AudioStudioGenerationCreate:
        valid_payload.update({"kind": "speech", "provider": "tts"})
    elif model is AudioStudioRenderCreate:
        valid_payload.update({"render_type": "preview_mix"})
    else:
        valid_payload.update({"export_type": "package"})

    assert model(**valid_payload).idempotency_key == "client-key-123456"

    with pytest.raises(ValidationError, match="at least 16"):
        model(**{**valid_payload, "idempotency_key": "short"})

    with pytest.raises(ValidationError, match="at most 200"):
        model(**{**valid_payload, "idempotency_key": "x" * 201})


@pytest.mark.parametrize("field_name", ["provider", "options"])
def test_generation_payload_rejects_secrets_and_external_urls(field_name: str) -> None:
    payload = {
        "kind": "speech",
        "provider": "tts",
        "target_resource_kind": "section",
        "target_resource_id": "sec_001",
        "target_revision_id": "rev_001",
        "idempotency_key": "client-key-123456",
        field_name: {"external_url": "https://provider.example.invalid", "api_key": "secret"},
    }

    with pytest.raises(ValidationError, match="external_url|secret"):
        AudioStudioGenerationCreate(**payload)


def test_nested_payload_rejects_secret_like_keys() -> None:
    with pytest.raises(ValidationError, match="secret"):
        AudioStudioRenderCreate(
            render_type="preview_mix",
            target_resource_kind="render",
            target_resource_id="rnd_001",
            target_revision_id="rev_001",
            idempotency_key="client-key-123456",
            settings={"safe": {"token": "abc"}},
        )


@pytest.mark.parametrize(
    "provider_payload",
    [
        {"access_token": "secret"},
        {"oauth": {"refresh-token": "secret"}},
        {"nested": [{"private_key": "secret"}]},
        {"credentials": "secret"},
        {"clientCredential": "secret"},
    ],
)
def test_generation_payload_rejects_common_credential_key_variants(provider_payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="secret|credential|external_url"):
        AudioStudioGenerationCreate(
            kind="speech",
            provider=provider_payload,
            target_resource_kind="section",
            target_resource_id="sec_001",
            target_revision_id="rev_001",
            idempotency_key="client-key-123456",
        )


def test_generation_payload_allows_harmless_tokenizer_key() -> None:
    payload = AudioStudioGenerationCreate(
        kind="speech",
        provider={"tokenizer": "cl100k_base"},
        target_resource_kind="section",
        target_resource_id="sec_001",
        target_revision_id="rev_001",
        idempotency_key="client-key-123456",
    )

    assert payload.provider == {"tokenizer": "cl100k_base"}


def test_project_archive_request_requires_base_revision_id() -> None:
    with pytest.raises(ValidationError, match="base_revision_id"):
        AudioStudioProjectArchiveRequest()

    assert AudioStudioProjectArchiveRequest(base_revision_id="rev_001").base_revision_id == "rev_001"
