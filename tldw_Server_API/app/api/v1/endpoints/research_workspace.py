"""Capability and generation endpoints for the Research Workspace WebUI surface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.research_workspace_artifacts import (
    ResearchWorkspaceArtifactGenerateRequest,
    ResearchWorkspaceArtifactGenerateResponse,
)
from tldw_Server_API.app.api.v1.schemas.research_workspace_capabilities import (
    ResearchWorkspaceCapabilitiesResponse,
)
from tldw_Server_API.app.api.v1.schemas.research_workspace_citations import (
    CitationSupportReport,
    ResearchWorkspaceCitationVerifyRequest,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ
from tldw_Server_API.app.core.RAG.rag_service.types import Document
from tldw_Server_API.app.core.Research_Workspace.artifact_generation import (
    ResearchWorkspaceArtifactVerificationError,
    generate_research_workspace_artifact,
)
import tldw_Server_API.app.core.Research_Workspace.capabilities as _capabilities

router = APIRouter(prefix="/research-workspace", tags=["research-workspace"])


@router.get(
    "/capabilities",
    response_model=ResearchWorkspaceCapabilitiesResponse,
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("research_workspace.capabilities")),
    ],
)
async def research_workspace_capabilities(
    current_user: User = Depends(get_request_user),
) -> ResearchWorkspaceCapabilitiesResponse:
    """Return user-safe Research Workspace capability readiness."""
    return await _capabilities.collect_research_workspace_capabilities(user_id=current_user.id)


def _media_text(row: dict[str, Any]) -> str:
    for key in ("content", "text", "transcription", "summary"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    content = row.get("content")
    if isinstance(content, dict):
        for key in ("text", "content", "summary"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _media_row_to_document(media_id: int, row: dict[str, Any]) -> Document | None:
    text = _media_text(row)
    if not text:
        return None
    title = str(row.get("title") or row.get("name") or f"Media {media_id}").strip()
    return Document(
        id=f"media:{media_id}",
        content=text,
        metadata={
            "source_type": "media",
            "source_id": str(media_id),
            "media_id": media_id,
            "title": title,
        },
    )


@router.post(
    "/artifacts/generate",
    response_model=ResearchWorkspaceArtifactGenerateResponse,
    dependencies=[
        Depends(RequirePermission(MEDIA_READ)),
        Depends(rbac_rate_limit("research_workspace.artifacts.generate")),
    ],
)
async def generate_research_workspace_artifact_endpoint(
    request: ResearchWorkspaceArtifactGenerateRequest,
    media_db: Any = Depends(get_media_db_for_user),
) -> ResearchWorkspaceArtifactGenerateResponse:
    """Generate a source-grounded Research Workspace draft artifact and verify it internally."""
    documents: list[Document] = []
    for media_id in request.media_ids:
        row = media_db.get_media_by_id(int(media_id), include_deleted=False, include_trash=False)
        if not row:
            raise HTTPException(status_code=404, detail=f"Media item {media_id} not found")
        document = _media_row_to_document(int(media_id), row)
        if document is not None:
            documents.append(document)

    if not documents:
        raise HTTPException(status_code=400, detail="Selected media did not contain usable source content")

    try:
        result = await generate_research_workspace_artifact(
            artifact_type=request.artifact_type,
            source_documents=documents,
            generation_provider=request.api_provider,
            generation_model=request.model,
            verification_provider=request.claims_verification_provider,
            verification_model=request.claims_verification_model,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
    except ResearchWorkspaceArtifactVerificationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "claim_verification_failed",
                "claimVerification": exc.claim_verification,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ResearchWorkspaceArtifactGenerateResponse.model_validate(result)


async def get_citation_analyzer(
    request: Request, current_user: User = Depends(get_request_user),
) -> AsyncIterator[Any]:
    """Own the trusted credential runtime for this bounded verification request."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.Provider_Credentials.byok_helpers import derive_trusted_credential_scope
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        capture_provider_override_call_snapshot,
    )
    from tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime import (
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
        ProviderCredentialRuntime,
    )
    from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context

    user_id, team_ids, org_ids, trusted_override = derive_trusted_credential_scope(request, current_user)
    runtime = ProviderCredentialRuntime(user_id=user_id, team_ids=team_ids, org_ids=org_ids,
        trusted_base_url_override=trusted_override, override_snapshot_resolver=capture_provider_override_call_snapshot)

    async def analyze(provider, input_data, prompt, api_key, system_message, temp=None, **kwargs):
        model = kwargs.get("model_override")
        credentials = await runtime.resolve(provider, model=model)
        with llm_call_context(user_id=user_id):
            response = await chat_service.perform_chat_api_call_async(
                api_endpoint=provider, model=model, messages_payload=[{"role": "user", "content": prompt}],
                system_message=system_message, temp=0, max_tokens=4096, streaming=False,
                **{PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: credentials},
                **({"response_format": kwargs["response_format"]} if kwargs.get("response_format") else {}),
            )
        await runtime.mark_used(credentials)
        return response

    try:
        yield analyze
    finally:
        await runtime.close()


@router.post(
    "/citations/verify", response_model=CitationSupportReport,
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("research_workspace.citations.verify"))],
)
async def verify_research_workspace_citations(
    body: ResearchWorkspaceCitationVerifyRequest,
    workspace_db: Any = Depends(get_chacha_db_for_user),
    media_db: Any = Depends(get_media_db_for_user),
    analyze: Any = Depends(get_citation_analyzer),
) -> CitationSupportReport:
    """Assess complete answer units using only their owned, exact cited passages."""
    from tldw_Server_API.app.core.Claims_Extraction.citation_support import verify_cited_text
    from tldw_Server_API.app.core.Claims_Extraction.workspace_citations import resolve_workspace_citations

    try:
        evidence = await resolve_workspace_citations(body.workspace_id,
            [item.model_dump() for item in body.evidence], workspace_db=workspace_db, media_db=media_db)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Workspace not found") from exc
    report = await verify_cited_text(body.text, evidence, provider=body.api_provider, model=body.model,
        analyze_fn=analyze, context={"workspace_id": body.workspace_id})
    return CitationSupportReport.model_validate(report)
