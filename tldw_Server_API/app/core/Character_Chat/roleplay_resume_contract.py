"""Original Resume preconditions shared by admission, generation and commit."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import RoleplayResumeFenceV1


class RoleplayResumeConflictError(ValueError):
    """A definitive no-write refusal of the original protected operation."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code

    def to_detail(self) -> dict[str, str]:
        """Expose the bounded reason, never private behavior or another owner's state."""
        return {"status": "resume_conflict", "code": self.code}


def roleplay_resume_fence(state: dict[str, Any]) -> dict[str, Any]:
    """Detach a complete validated fence from one coherent owner read."""
    return RoleplayResumeFenceV1.model_validate(
        {
            "version": 1,
            "snapshot_digest": state["behavior_snapshot"]["digest"],
            "settings_version": state["settings_version"],
            "history_version": state["history_version"],
            "tail": state["tail"],
        }
    ).model_dump(mode="json")


def require_roleplay_resume_fence(state: dict[str, Any], expected: dict[str, Any]) -> None:
    """Refuse unavailable authority or any changed component before a write."""
    if state.get("resume_eligible") is not True:
        raise RoleplayResumeConflictError("resume_unavailable")
    actual = roleplay_resume_fence(state)
    for field, code in (
        ("snapshot_digest", "snapshot_changed"),
        ("settings_version", "settings_changed"),
        ("history_version", "history_fence_changed"),
        ("tail", "history_fence_changed"),
    ):
        if actual[field] != expected[field]:
            raise RoleplayResumeConflictError(code)


def roleplay_resume_sampling_arguments(effective: dict[str, Any]) -> dict[str, Any]:
    """Preserve saved sampling or refuse before admitting an unrepresentable input."""
    from tldw_Server_API.app.core.LLM_Calls.capability_registry import (
        get_allowed_fields,
    )

    sampling = effective["sampling"]
    if effective["provider"] == "mlx":
        raise RoleplayResumeConflictError("resume_provider_unsupported")
    if (
        effective["provider"] == "openai"
        and effective["model"].lower().startswith("gpt-5")
        and sampling["top_p"] != 1.0
    ):
        raise RoleplayResumeConflictError("resume_sampling_unsupported")
    arguments = {
        "temp": sampling["temperature"],
        "top_p": sampling["top_p"],
        "stop": list(sampling["stop"]),
    }
    if "repetition_penalty" in get_allowed_fields(effective["provider"]):
        arguments["repetition_penalty"] = sampling["repetition_penalty"]
    elif sampling["repetition_penalty"] != 1.0:
        raise RoleplayResumeConflictError("resume_sampling_unsupported")
    return arguments
