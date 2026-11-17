from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    ArtifactVerificationResult,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.Research_Workspace.artifact_generation import (
    ResearchWorkspaceArtifactVerificationError,
    generate_research_workspace_artifact,
)
from tldw_Server_API.app.api.v1.schemas.research_workspace_artifacts import (
    ResearchWorkspaceArtifactGenerateRequest,
)


def test_research_workspace_artifact_request_accepts_at_most_fifty_media_ids():
    request = ResearchWorkspaceArtifactGenerateRequest(
        artifact_type="mindmap",
        media_ids=list(range(1, 51)),
        model="test-model",
    )

    assert len(request.media_ids) == 50


def test_research_workspace_artifact_request_rejects_more_than_fifty_media_ids():
    with pytest.raises(ValidationError):
        ResearchWorkspaceArtifactGenerateRequest(
            artifact_type="mindmap",
            media_ids=list(range(1, 52)),
            model="test-model",
        )


@pytest.mark.asyncio
async def test_generate_research_workspace_artifact_uses_claims_verifier_override():
    captured: dict[str, object] = {}

    async def fake_chat(**kwargs):
        captured["chat"] = kwargs
        return "| Metric | Value |\n| --- | --- |\n| Retention | 82 percent |"

    async def fake_verify(**kwargs):
        captured["verify"] = kwargs
        return ArtifactVerificationResult(
            verdict="grounded",
            report={"total_claims": 1, "claims": [{"status": "verified"}]},
            unit_results=[],
            metadata={
                "generation_provider": kwargs["generation_provider"],
                "generation_model": kwargs["generation_model"],
                "verification_provider": kwargs["verification_provider"],
                "verification_model": kwargs["verification_model"],
                "verification_llm_is_default": False,
            },
        )

    result = await generate_research_workspace_artifact(
        artifact_type="data_table",
        source_documents=[
            Document(
                id="media:101",
                content="Project Falcon improved retention from 64 percent to 82 percent.",
                metadata={"source_type": "media", "source_id": "101", "title": "Falcon"},
            )
        ],
        generation_provider="llamacpp",
        generation_model="generation-model",
        verification_provider="openrouter",
        verification_model="claims-model",
        temperature=0.2,
        top_p=0.9,
        max_tokens=900,
        chat_fn=fake_chat,
        verify_fn=fake_verify,
    )

    verify_kwargs = captured["verify"]
    assert isinstance(verify_kwargs, dict)
    assert verify_kwargs["generation_provider"] == "llamacpp"
    assert verify_kwargs["generation_model"] == "generation-model"
    assert verify_kwargs["verification_provider"] == "openrouter"
    assert verify_kwargs["verification_model"] == "claims-model"
    assert result["claim_verification"]["metadata"]["verification_provider"] == "openrouter"


@pytest.mark.asyncio
async def test_generate_research_workspace_artifact_rejects_failed_verification():
    async def fake_chat(**_kwargs):
        return "Project Falcon retention improved to 90 percent."

    async def fake_verify(**_kwargs):
        return ArtifactVerificationResult(
            verdict="failed",
            report={"total_claims": 1, "claims": [{"status": "numerical_error"}]},
            unit_results=[],
            metadata={"verification_provider": "llamacpp"},
        )

    with pytest.raises(ResearchWorkspaceArtifactVerificationError) as exc_info:
        await generate_research_workspace_artifact(
            artifact_type="audio_overview",
            source_documents=[
                Document(
                    id="media:101",
                    content="Project Falcon improved retention from 64 percent to 82 percent.",
                    metadata={"source_type": "media", "source_id": "101", "title": "Falcon"},
                )
            ],
            generation_provider="llamacpp",
            generation_model="generation-model",
            verification_provider=None,
            verification_model=None,
            temperature=0.2,
            top_p=0.9,
            max_tokens=900,
            chat_fn=fake_chat,
            verify_fn=fake_verify,
        )

    assert exc_info.value.claim_verification["verdict"] == "failed"


@pytest.mark.asyncio
async def test_generate_research_workspace_artifact_rejects_needs_revision_verification():
    async def fake_chat(**_kwargs):
        return "Project Falcon retention improved to 82 percent."

    async def fake_verify(**_kwargs):
        return ArtifactVerificationResult(
            verdict="needs_revision",
            report={"total_claims": 1, "claims": [{"status": "unverified"}]},
            unit_results=[],
            metadata={"reason": "no_claims", "verification_provider": "llamacpp"},
        )

    with pytest.raises(ResearchWorkspaceArtifactVerificationError) as exc_info:
        await generate_research_workspace_artifact(
            artifact_type="audio_overview",
            source_documents=[
                Document(
                    id="media:101",
                    content="Project Falcon improved retention from 64 percent to 82 percent.",
                    metadata={"source_type": "media", "source_id": "101", "title": "Falcon"},
                )
            ],
            generation_provider="llamacpp",
            generation_model="generation-model",
            verification_provider=None,
            verification_model=None,
            temperature=0.2,
            top_p=0.9,
            max_tokens=900,
            chat_fn=fake_chat,
            verify_fn=fake_verify,
        )

    assert exc_info.value.claim_verification["verdict"] == "needs_revision"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"choices": [{"message": {"content": None, "reasoning_content": "private-thought"}}]},
        {"choices": [{"message": {"content": "", "reasoning_content": "private-thought"}}]},
        {"choices": [{"message": {"content": {"provider_metadata": "private-thought"}}}]},
        {"choices": [{"message": {"content": 42}}]},
        {"choices": [], "provider_metadata": "private-thought"},
        {"choices": [{"message": None}]},
        {"choices": "malformed-provider-envelope"},
        {"choices": {"0": {"message": {"content": "unexpected text"}}}},
    ],
    ids=[
        "reasoning-only",
        "empty-final",
        "object-final",
        "numeric-final",
        "no-choice",
        "null-message",
        "malformed-choices",
        "mapping-choices",
    ],
)
async def test_artifact_rejects_responses_without_final_text_before_verification(response: object):
    async def fake_chat(**_kwargs):
        return response

    async def unexpected_verify(**_kwargs):
        raise AssertionError("Provider envelopes and reasoning must never reach verification")

    with pytest.raises(ValueError, match="final text") as error:
        await generate_research_workspace_artifact(
            artifact_type="audio_overview",
            source_documents=[Document(id="media:1", content="The telescope is named Selene.", metadata={})],
            generation_provider="llamacpp",
            generation_model="controlled-model",
            verification_provider=None,
            verification_model=None,
            temperature=0.2,
            top_p=0.9,
            max_tokens=4096,
            chat_fn=fake_chat,
            verify_fn=unexpected_verify,
        )
    assert "private-thought" not in str(error.value)  # nosec B101 -- A failed provider response must not leak its reasoning or metadata through the public error.


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_audio_prompt_keeps_short_sources_short_and_verifies_only_final_text(structured: bool):
    content = "The telescope is named Selene."
    captured: dict[str, object] = {}

    async def fake_chat(**kwargs):
        captured["chat"] = kwargs
        return (
            {"choices": [{"message": {"content": content, "reasoning_content": "private-thought"}}]}
            if structured
            else content
        )

    async def fake_verify(**kwargs):
        captured["verify"] = kwargs
        return ArtifactVerificationResult(verdict="grounded", report={}, unit_results=[], metadata={})

    result = await generate_research_workspace_artifact(
        artifact_type="audio_overview",
        source_documents=[Document(id="media:1", content=content, metadata={})],
        generation_provider="llamacpp",
        generation_model="controlled-model",
        verification_provider=None,
        verification_model=None,
        temperature=0.2,
        top_p=0.9,
        max_tokens=4096,
        chat_fn=fake_chat,
        verify_fn=fake_verify,
    )
    assert result["content"] == content  # nosec B101 -- Preserve the actual final answer for both supported response shapes.
    assert all(unit.text == content for unit in captured["verify"]["units"])  # nosec B101 -- Reasoning is excluded from the real verification-unit input.
    prompt = captured["chat"]["messages_payload"][0]["content"].lower()
    assert "shorter" in prompt and "do not pad" in prompt  # nosec B101 -- The audio request must explicitly permit a source-proportionate script instead of inventing a minimum length.
