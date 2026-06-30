"""Shared setup readiness lane, status, and overlay constants.

The first-run readiness flow uses these values as a small contract between the
backend setup API and the WebUI. Keep restart/admin/remote-blocked state as
overlays so lane status remains about capability readiness only.
"""

from __future__ import annotations

from typing import Any

LANE_CHAT = "chat"
LANE_EMBEDDINGS_RAG = "embeddings_rag"
LANE_SPEECH = "speech"

LANE_IDS = (
    LANE_CHAT,
    LANE_EMBEDDINGS_RAG,
    LANE_SPEECH,
)

LANE_STATUSES = (
    "not_configured",
    "previewed",
    "provisioning",
    "ready",
    "ready_with_warnings",
    "failed",
    "blocked",
    "skipped",
)

OVERLAY_IDS = (
    "restart_required",
    "requires_admin",
    "remote_setup_blocked",
    "network_unavailable",
    "downloads_disabled",
    "package_installs_disabled",
)

LANE_LABELS = {
    LANE_CHAT: "Chat",
    LANE_EMBEDDINGS_RAG: "Embeddings/RAG",
    LANE_SPEECH: "Speech",
}

LANE_PRIMARY_CAPABILITIES = {
    LANE_CHAT: "chat",
    LANE_EMBEDDINGS_RAG: "rag_search",
    LANE_SPEECH: "transcription",
}

LANE_SECONDARY_CAPABILITIES = {
    LANE_CHAT: (),
    LANE_EMBEDDINGS_RAG: (),
    LANE_SPEECH: ("tts",),
}


def build_lane_summary(
    lane_id: str,
    *,
    status: str = "not_configured",
    selection: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    blockers: list[str] | None = None,
    consequences: list[str] | None = None,
) -> dict[str, Any]:
    """Return a normalized lane summary payload for setup readiness."""

    if lane_id not in LANE_IDS:
        raise ValueError(f"Unsupported setup readiness lane: {lane_id}")
    if status not in LANE_STATUSES:
        raise ValueError(f"Unsupported setup readiness status: {status}")

    return {
        "lane_id": lane_id,
        "label": LANE_LABELS[lane_id],
        "status": status,
        "primary_capability": LANE_PRIMARY_CAPABILITIES[lane_id],
        "secondary_capabilities": list(LANE_SECONDARY_CAPABILITIES[lane_id]),
        "selection": selection or {},
        "warnings": warnings or [],
        "blockers": blockers or [],
        "consequences": consequences or [],
    }


__all__ = [
    "LANE_CHAT",
    "LANE_EMBEDDINGS_RAG",
    "LANE_IDS",
    "LANE_LABELS",
    "LANE_PRIMARY_CAPABILITIES",
    "LANE_SECONDARY_CAPABILITIES",
    "LANE_SPEECH",
    "LANE_STATUSES",
    "OVERLAY_IDS",
    "build_lane_summary",
]
