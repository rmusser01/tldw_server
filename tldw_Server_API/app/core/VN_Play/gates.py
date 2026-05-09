"""Runtime admission gates for VN Play sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tldw_Server_API.app.core.VN_Play.constants import (
    TRUST_LEVEL_MIXED,
    TRUST_LEVEL_UNTRUSTED_IMPORT,
)
from tldw_Server_API.app.core.VN_Play.models import CharacterSafetyResult, GateResult

_MATURE_RATINGS = {"mature", "adult", "explicit", "nsfw"}


def evaluate_character_safety(
    *,
    character: Mapping[str, Any],
    content_rating: str,
    settings: Mapping[str, Any],
    trust_level: str,
) -> CharacterSafetyResult:
    """Evaluate whether character safety metadata permits a VN Play session."""
    status = _character_safety_status(character)
    normalized_rating = str(content_rating or "").strip().lower()
    is_mature = normalized_rating in _MATURE_RATINGS

    if status == "conflicting":
        return CharacterSafetyResult(
            allowed=False,
            status=status,
            error_code="character_safety_conflicting",
            message="Character safety metadata is conflicting.",
        )

    if status == "minor" and is_mature:
        return CharacterSafetyResult(
            allowed=False,
            status=status,
            error_code="character_safety_minor_disallowed",
            message="Mature VN Play sessions require adult character metadata.",
        )

    if status == "unknown":
        if is_mature and not bool(settings.get("allow_unknown_character_safety")):
            return CharacterSafetyResult(
                allowed=False,
                status=status,
                error_code="character_safety_unknown_requires_override",
                message="Unknown character safety metadata requires an explicit override.",
            )
        if (
            trust_level in {TRUST_LEVEL_UNTRUSTED_IMPORT, TRUST_LEVEL_MIXED}
            and not bool(settings.get("allow_untrusted_character_safety"))
        ):
            return CharacterSafetyResult(
                allowed=False,
                status=status,
                error_code="character_safety_untrusted_requires_override",
                message="Imported character safety metadata requires an explicit override.",
            )
        return CharacterSafetyResult(
            allowed=True,
            status=status,
            warning_code="character_safety_unknown",
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

    return "unknown"


def _normalize_safety_status(status: str) -> str:
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"adult", "18_plus", "18plus", "of_age"}:
        return "adult"
    if normalized in {"minor", "under_18", "under18"}:
        return "minor"
    if normalized in {"unknown", "unspecified", "missing"}:
        return "unknown"
    if normalized in {"conflict", "conflicting"}:
        return "conflicting"
    return "unknown"


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False
