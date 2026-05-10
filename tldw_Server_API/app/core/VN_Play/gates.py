"""Runtime admission gates for VN Play sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tldw_Server_API.app.core.VN_Play.constants import (
    TRUST_LEVEL_MIXED,
    TRUST_LEVEL_UNTRUSTED_IMPORT,
)
from tldw_Server_API.app.core.VN_Play.models import CharacterSafetyResult, GateResult
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import (
    LOCAL_DEFAULT_POLICY_DEFINITION,
    STRICT_HOSTED_POLICY_DEFINITION,
)
from tldw_Server_API.app.core.VN_Policy.service import evaluate_character_safety_definition

_BUILTIN_POLICY_DEFINITIONS = {
    "local_default": LOCAL_DEFAULT_POLICY_DEFINITION,
    "strict_hosted": STRICT_HOSTED_POLICY_DEFINITION,
}


def evaluate_character_safety(
    *,
    character: Mapping[str, Any],
    content_rating: str,
    settings: Mapping[str, Any],
    trust_level: str,
) -> CharacterSafetyResult:
    """Evaluate whether character safety metadata permits a VN Play session."""
    status = _effective_policy_status(
        _character_safety_status(character),
        trust_level=trust_level,
    )
    policy_profile_id, policy_definition = _resolve_policy_profile(settings)
    if policy_definition is None:
        return CharacterSafetyResult(
            allowed=False,
            status=status,
            error_code="policy_profile_unresolved",
            message="VN Play policy profile definition is not resolved.",
        )
    decision = evaluate_character_safety_definition(
        profile_definition=policy_definition,
        policy_profile_id=policy_profile_id,
        content_rating=content_rating,
        metadata_status=status,
    )
    first_reason = decision["reasons"][0] if decision["reasons"] else {}
    reason_code = first_reason.get("code")
    message = first_reason.get("message")
    if decision["decision"] == "block":
        return CharacterSafetyResult(
            allowed=False,
            status=status,
            error_code=reason_code,
            message=message,
        )
    if decision["decision"] == "warn":
        return CharacterSafetyResult(
            allowed=True,
            status=status,
            warning_code=reason_code,
            message=message,
        )
    return CharacterSafetyResult(allowed=True, status=status)


def evaluate_runtime_gates(
    *,
    characters: Iterable[Mapping[str, Any]],
    content_rating: str,
    settings: Mapping[str, Any],
    trust_level: str,
) -> GateResult:
    """Evaluate all V1 runtime gates that do not require model or DB calls."""
    warnings: list[dict[str, Any]] = []
    for character in characters:
        result = evaluate_character_safety(
            character=character,
            content_rating=content_rating,
            settings=settings,
            trust_level=trust_level,
        )
        if not result.allowed:
            return GateResult(
                allowed=False,
                warnings=tuple(warnings),
                error_code=result.error_code,
                error_message=result.message,
            )
        if result.warning_code is not None:
            warnings.append(
                {
                    "code": result.warning_code,
                    "character_id": character.get("id"),
                    "status": result.status,
                }
            )
    return GateResult(allowed=True, warnings=tuple(warnings))


def _character_safety_status(character: Mapping[str, Any]) -> str:
    metadata = character.get("safety_metadata")
    if isinstance(metadata, Mapping):
        status = metadata.get("age_status") or metadata.get("status")
        if isinstance(status, str) and status.strip():
            return _normalize_safety_status(status)

    explicit_status = character.get("age_status") or character.get("safety_status")
    if isinstance(explicit_status, str) and explicit_status.strip():
        return _normalize_safety_status(explicit_status)

    flags = {
        "is_minor": character.get("is_minor"),
        "minor": character.get("minor"),
        "is_adult": character.get("is_adult"),
        "adult": character.get("adult"),
    }
    minor_flag = _truthy_flag(flags["is_minor"]) or _truthy_flag(flags["minor"])
    adult_flag = _truthy_flag(flags["is_adult"]) or _truthy_flag(flags["adult"])
    if minor_flag and adult_flag:
        return "conflicting"
    if minor_flag:
        return "minor"
    if adult_flag:
        return "adult"

    age = character.get("age_years", character.get("age"))
    if isinstance(age, int):
        return "adult" if age >= 18 else "minor"

    return "missing"


def _normalize_safety_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"adult", "18_plus", "18plus", "of_age"}:
        return "adult"
    if normalized in {"minor", "under_18", "under18"}:
        return "minor"
    if normalized in {"missing", "unspecified", "not_provided"}:
        return "missing"
    if normalized in {"unknown", "ambiguous", "unknown_ambiguous", "unknown_or_ambiguous"}:
        return "unknown_or_ambiguous"
    if normalized in {"conflict", "conflicting"}:
        return "conflicting"
    if normalized in {"imported_untrusted", "untrusted_import", "imported_without_trusted_provenance"}:
        return "imported_untrusted"
    return "unknown_or_ambiguous"


def _effective_policy_status(status: str, *, trust_level: str) -> str:
    if (
        trust_level in {TRUST_LEVEL_UNTRUSTED_IMPORT, TRUST_LEVEL_MIXED}
        and status not in {"minor", "conflicting"}
    ):
        return "imported_untrusted"
    return status


def _resolve_policy_profile(settings: Mapping[str, Any]) -> tuple[str, Mapping[str, Any] | None]:
    raw_profile_id = settings.get("policy_profile_id") or settings.get("vn_policy_profile_id") or "local_default"
    profile_id = str(raw_profile_id).strip() or "local_default"
    if profile_id in _BUILTIN_POLICY_DEFINITIONS:
        return profile_id, _BUILTIN_POLICY_DEFINITIONS[profile_id]
    resolved_definition = (
        settings["policy_definition"]
        if "policy_definition" in settings
        else settings.get("resolved_policy_definition")
    )
    if isinstance(resolved_definition, Mapping):
        return profile_id, resolved_definition
    return profile_id, None


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False
