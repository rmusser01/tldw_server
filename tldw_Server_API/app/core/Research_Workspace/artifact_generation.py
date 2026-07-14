"""Backend Research Workspace artifact draft generation with internal Claims verification."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
from tldw_Server_API.app.core.Claims_Extraction.artifact_verification import (
    ArtifactVerificationResult,
    ArtifactVerificationUnit,
    verify_generated_artifact_against_sources,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document


ResearchWorkspaceArtifactType = Literal["audio_overview", "data_table", "mindmap"]
ChatFn = Callable[..., Awaitable[Any]]
VerifyFn = Callable[..., Awaitable[ArtifactVerificationResult]]

MAX_SOURCE_CHARS = 18000


class ResearchWorkspaceArtifactVerificationError(ValueError):
    """Raised when generated Research Workspace artifact content fails verification."""

    def __init__(self, claim_verification: dict[str, Any]):
        super().__init__("Research Workspace artifact claim verification failed")
        self.claim_verification = claim_verification


def _source_text(source_documents: Sequence[Document]) -> str:
    blocks: list[str] = []
    remaining = MAX_SOURCE_CHARS
    for index, document in enumerate(source_documents, start=1):
        content = str(document.content or "").strip()
        if not content:
            continue
        title = str(document.metadata.get("title") or document.id or f"Source {index}").strip()
        block = f"Source {index}: {title}\n{content}"
        if len(block) > remaining:
            block = block[:remaining]
        blocks.append(block)
        remaining -= len(block) + 2
        if remaining <= 0:
            break
    return "\n\n".join(blocks).strip()


def _prompt_for_artifact(artifact_type: ResearchWorkspaceArtifactType, source_text: str) -> tuple[str, str]:
    if artifact_type == "audio_overview":
        return (
            "You are a source-grounded audio script writer. Use only the provided source content. "
            "Do not invent facts. Write plain spoken prose without speaker labels, bullets, or stage directions.",
            "Create a spoken overview script (2-3 minutes when read aloud) that introduces the topic, "
            "covers the main points, and concludes with key takeaways.\n\n"
            f"Selected sources:\n{source_text}",
        )
    if artifact_type == "data_table":
        return (
            "You are a data table generator. Return ONLY a markdown table with pipe delimiters, "
            "a header row, and a separator row. Do not include commentary or code fences.",
            "Extract structured data from the provided sources and format it as a markdown table.\n\n"
            "Include key entities, important attributes and values, and supported relationships or comparisons.\n\n"
            f"Sources:\n{source_text}",
        )
    return (
        "You are a mind map generator. Return ONLY Mermaid mindmap syntax. You may wrap the result in a "
        "```mermaid code fence, but do not include commentary, explanations, or prose outside the diagram.",
        "Analyze the provided sources and create a Mermaid mindmap that captures the central theme, "
        "3-5 major branches, and the most important subtopics.\n\n"
        f"Sources:\n{source_text}",
    )


async def _default_chat_fn(**kwargs: Any) -> Any:
    return await perform_chat_api_call_async(**kwargs)


def _coerce_chat_text(response: Any) -> str:
    content = extract_response_content(response)
    if content is not None:
        return str(content).strip()
    return str(response or "").strip()


def _strip_mermaid_fence(content: str) -> str:
    text = content.strip()
    fence = re.match(r"^```(?:mermaid)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    return fence.group(1).strip() if fence else text


def _clean_mindmap_label(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^(mindmap|graph|flowchart|sequenceDiagram|stateDiagram(?:-v2)?|gantt)\b", "", text, flags=re.I)
    text = re.sub(r"^[A-Za-z0-9_-]+\s*(?:-->|---|==>|-.->)\s*", "", text)
    text = re.sub(r"^[A-Za-z0-9_-]+\s*", "", text)
    text = text.strip("[](){}<>\"'`:- ")
    text = re.sub(r"\(\((.*?)\)\)", r"\1", text)
    text = re.sub(r"\[(.*?)\]", r"\1", text)
    return " ".join(text.split()).strip()


def _table_units(content: str) -> list[ArtifactVerificationUnit]:
    lines = [line.strip() for line in content.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return [ArtifactVerificationUnit(unit_id="data_table:content", text=content, claims=[content])]
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    units: list[ArtifactVerificationUnit] = []
    for row_index, line in enumerate(lines[2:], start=1):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        pairs = [
            f"{headers[index]}: {cell}"
            for index, cell in enumerate(cells[: len(headers)])
            if index < len(headers) and cell
        ]
        text = "; ".join(pairs).strip()
        if text:
            units.append(
                ArtifactVerificationUnit(
                    unit_id=f"data_table:row:{row_index}",
                    text=text,
                    claims=[text],
                    metadata={"row_index": row_index},
                )
            )
    return units or [ArtifactVerificationUnit(unit_id="data_table:content", text=content, claims=[content])]


def _audio_units(content: str) -> list[ArtifactVerificationUnit]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
    if not paragraphs:
        paragraphs = [content.strip()]
    return [
        ArtifactVerificationUnit(
            unit_id=f"audio_script:paragraph:{index}",
            text=paragraph,
            claims=[paragraph],
            metadata={"paragraph_index": index},
        )
        for index, paragraph in enumerate(paragraphs, start=1)
        if paragraph
    ]


def _mindmap_units(content: str) -> list[ArtifactVerificationUnit]:
    mermaid = _strip_mermaid_fence(content)
    units: list[ArtifactVerificationUnit] = []
    for index, line in enumerate(mermaid.splitlines(), start=1):
        label = _clean_mindmap_label(line)
        if not label or label.lower() in {"mindmap", "graph", "flowchart"}:
            continue
        units.append(
            ArtifactVerificationUnit(
                unit_id=f"mindmap:node:{index}",
                text=label,
                claims=[label],
                metadata={"line": index},
            )
        )
    return units or [ArtifactVerificationUnit(unit_id="mindmap:content", text=mermaid, claims=[mermaid])]


def _verification_units(
    artifact_type: ResearchWorkspaceArtifactType,
    content: str,
) -> list[ArtifactVerificationUnit]:
    if artifact_type == "audio_overview":
        return _audio_units(content)
    if artifact_type == "data_table":
        return _table_units(content)
    return _mindmap_units(content)


async def generate_research_workspace_artifact(
    *,
    artifact_type: ResearchWorkspaceArtifactType,
    source_documents: list[Document],
    generation_provider: str | None,
    generation_model: str | None,
    verification_provider: str | None,
    verification_model: str | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    chat_fn: ChatFn | None = None,
    verify_fn: VerifyFn | None = None,
) -> dict[str, Any]:
    """Generate and verify a Research Workspace draft artifact before returning it."""
    source_text = _source_text(source_documents)
    if not source_text:
        raise ValueError("No usable source content was found.")

    system_message, user_prompt = _prompt_for_artifact(artifact_type, source_text)
    chat_response = await (chat_fn or _default_chat_fn)(
        api_endpoint=generation_provider,
        model=generation_model,
        messages_payload=[{"role": "user", "content": user_prompt}],
        system_message=system_message,
        temp=temperature,
        topp=top_p,
        max_tokens=max_tokens,
        streaming=False,
    )
    content = _coerce_chat_text(chat_response)
    if not content:
        raise ValueError("Generated artifact content was empty.")

    verifier = verify_fn or verify_generated_artifact_against_sources
    claim_verification = await verifier(
        artifact_type=artifact_type,
        units=_verification_units(artifact_type, content),
        source_documents=source_documents,
        generation_provider=generation_provider,
        generation_model=generation_model,
        verification_provider=verification_provider,
        verification_model=verification_model,
        generation_context={"query": f"generated {artifact_type}"},
    )
    claim_verification_payload = claim_verification.to_dict()
    if claim_verification.verdict != "grounded":
        raise ResearchWorkspaceArtifactVerificationError(claim_verification_payload)

    data: dict[str, Any] = {}
    if artifact_type == "mindmap":
        data["mermaid"] = _strip_mermaid_fence(content)

    return {
        "artifact_type": artifact_type,
        "content": content,
        "data": data,
        "claim_verification": claim_verification_payload,
    }
