from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthorityDecision:
    action: str
    reason: str


def decide_authority(
    source_type: str,
    event_type: str,
    authority_settings: dict[str, object],
) -> AuthorityDecision:
    if source_type == "user":
        return AuthorityDecision(action="commit", reason="user_direct_commit")
    if source_type == "system":
        return AuthorityDecision(action="commit", reason="internal_system_commit")
    if source_type == "import" and authority_settings.get("import_auto_commit") is True:
        return AuthorityDecision(action="commit", reason="import_auto_commit_enabled")
    if source_type == "mcp" and authority_settings.get("mcp_auto_commit") is True:
        return AuthorityDecision(action="commit", reason="mcp_auto_commit_enabled")
    if source_type == "model" and authority_settings.get("model_auto_commit") is True:
        allowed_event_types = set(authority_settings.get("model_auto_commit_event_types") or [])
        if event_type in allowed_event_types:
            return AuthorityDecision(
                action="commit",
                reason="model_event_type_auto_commit_enabled",
            )
    return AuthorityDecision(
        action="proposal",
        reason=f"{source_type}_{event_type}_requires_review",
    )
