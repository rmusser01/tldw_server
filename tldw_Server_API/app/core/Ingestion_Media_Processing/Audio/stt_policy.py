from __future__ import annotations

import asyncio
import contextlib
import hashlib
import mimetypes
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.org_stt_settings_repo import AuthnzOrgSttSettingsRepo
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import (
    FILE_CATEGORY_STT_AUDIO,
    SOURCE_FEATURE_STT,
)
from tldw_Server_API.app.core.AuthNZ.settings import is_multi_user_mode
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_org_memberships_for_user
from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationPolicy,
    get_moderation_service,
)
from tldw_Server_API.app.core.config import get_stt_config, load_comprehensive_config
from tldw_Server_API.app.core.config_sections.stt import load_stt_config
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.services.storage_quota_service import get_storage_service


_STT_POLICY_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


@dataclass(frozen=True)
class STTPolicy:
    org_id: int | None
    delete_audio_after_success: bool
    audio_retention_hours: float
    redact_pii: bool
    allow_unredacted_partials: bool
    redact_categories: list[str]


@dataclass(frozen=True)
class STTAudioRetentionDecision:
    delete_after_success: bool
    expires_at: datetime | None


def _normalize_categories(raw_categories: list[str] | tuple[str, ...] | set[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_categories or []:
        value = str(raw).strip().lower()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return normalized


def _expand_redact_categories(raw_categories: list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    normalized = _normalize_categories(raw_categories)
    if not normalized:
        return None

    expanded: set[str] = set()
    for category in normalized:
        expanded.add(category)
        if category.startswith("pii_") and len(category) > 4:
            expanded.add(category[4:])
        elif category != "pii":
            expanded.add(f"pii_{category}")
    return expanded


def _stt_policy_from_config() -> STTPolicy:
    try:
        config = load_stt_config(load_comprehensive_config(), os.environ)
    except _STT_POLICY_EXCEPTIONS:
        config = get_stt_config()

    if isinstance(config, dict):
        delete_audio_after_success = bool(config.get("delete_audio_after_success", True))
        audio_retention_hours = float(config.get("audio_retention_hours", 0.0))
        redact_pii = bool(config.get("redact_pii", False))
        allow_unredacted_partials = bool(config.get("allow_unredacted_partials", False))
        raw_categories = config.get("redact_categories", [])
    else:
        delete_audio_after_success = bool(getattr(config, "delete_audio_after_success", True))
        audio_retention_hours = float(getattr(config, "audio_retention_hours", 0.0))
        redact_pii = bool(getattr(config, "redact_pii", False))
        allow_unredacted_partials = bool(getattr(config, "allow_unredacted_partials", False))
        raw_categories = getattr(config, "redact_categories", [])

    if audio_retention_hours <= 0:
        audio_retention_hours = 0.0
        delete_audio_after_success = True

    return STTPolicy(
        org_id=None,
        delete_audio_after_success=delete_audio_after_success,
        audio_retention_hours=audio_retention_hours,
        redact_pii=redact_pii,
        allow_unredacted_partials=allow_unredacted_partials,
        redact_categories=_normalize_categories(list(raw_categories or [])),
    )


async def _resolve_org_id(
    *,
    principal: AuthPrincipal | None,
    user_id: int | None,
) -> int | None:
    if isinstance(principal, AuthPrincipal):
        active_org_id = getattr(principal, "active_org_id", None)
        if active_org_id is not None:
            return int(active_org_id)
        org_ids = list(getattr(principal, "org_ids", []) or [])
        if len(org_ids) == 1:
            return int(org_ids[0])
        if getattr(principal, "subject", None) == "single_user":
            return None

    if not is_multi_user_mode():
        return None

    if user_id is None:
        return None

    try:
        memberships = await list_org_memberships_for_user(int(user_id))
    except _STT_POLICY_EXCEPTIONS as exc:
        logger.debug("STT policy org membership lookup failed for user_id={}: {}", user_id, exc)
        return None

    resolved_org_ids: list[int] = []
    for membership in memberships or []:
        try:
            org_id = membership.get("org_id")
        except _STT_POLICY_EXCEPTIONS:
            continue
        if org_id is not None:
            org_id_int = int(org_id)
            if org_id_int not in resolved_org_ids:
                resolved_org_ids.append(org_id_int)
    if len(resolved_org_ids) == 1:
        return resolved_org_ids[0]
    return None


async def resolve_effective_stt_policy(
    *,
    principal: AuthPrincipal | None = None,
    user_id: int | None = None,
    db: Any | None = None,
) -> STTPolicy:
    default_policy = _stt_policy_from_config()
    org_id = await _resolve_org_id(principal=principal, user_id=user_id)
    if org_id is None:
        return default_policy

    repo_db = db
    if repo_db is None:
        try:
            from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

            repo_db = await get_db_pool()
        except _STT_POLICY_EXCEPTIONS as exc:
            logger.debug("STT policy DB pool lookup failed for org_id={}: {}", org_id, exc)
            return default_policy

    try:
        repo = AuthnzOrgSttSettingsRepo(repo_db)
        row = await repo.get_settings(org_id)
    except _STT_POLICY_EXCEPTIONS as exc:
        logger.debug("STT org policy lookup failed for org_id={}: {}", org_id, exc)
        return default_policy

    if not row:
        return default_policy

    delete_audio_after_success = bool(row.get("delete_audio_after_success", default_policy.delete_audio_after_success))
    audio_retention_hours = float(row.get("audio_retention_hours", default_policy.audio_retention_hours))
    redact_pii = bool(row.get("redact_pii", default_policy.redact_pii))
    allow_unredacted_partials = bool(
        row.get("allow_unredacted_partials", default_policy.allow_unredacted_partials)
    )
    redact_categories = _normalize_categories(row.get("redact_categories", default_policy.redact_categories))

    if audio_retention_hours <= 0:
        audio_retention_hours = 0.0
        delete_audio_after_success = True

    return STTPolicy(
        org_id=org_id,
        delete_audio_after_success=delete_audio_after_success,
        audio_retention_hours=audio_retention_hours,
        redact_pii=redact_pii,
        allow_unredacted_partials=allow_unredacted_partials,
        redact_categories=redact_categories,
    )


def merge_request_overrides(
    base_policy: STTPolicy,
    *,
    delete_audio_after_success: bool | None = None,
    audio_retention_hours: float | None = None,
    redact_pii: bool | None = None,
    allow_unredacted_partials: bool | None = None,
    redact_categories: list[str] | None = None,
) -> STTPolicy:
    merged_delete = base_policy.delete_audio_after_success
    if delete_audio_after_success is not None:
        if base_policy.delete_audio_after_success and not delete_audio_after_success:
            raise ValueError("request override cannot disable delete_audio_after_success")
        merged_delete = bool(delete_audio_after_success)

    merged_retention = float(base_policy.audio_retention_hours)
    if audio_retention_hours is not None:
        requested_retention = float(audio_retention_hours)
        if requested_retention < 0:
            raise ValueError("request override retention must be non-negative")
        if base_policy.audio_retention_hours <= 0:
            if requested_retention > 0:
                raise ValueError("request override cannot increase retention above tenant policy")
        elif requested_retention > base_policy.audio_retention_hours:
            raise ValueError("request override cannot increase retention above tenant policy")
        merged_retention = requested_retention

    merged_redact_pii = base_policy.redact_pii
    if redact_pii is not None:
        if base_policy.redact_pii and not redact_pii:
            raise ValueError("request override cannot disable tenant-required redaction")
        merged_redact_pii = bool(redact_pii)

    merged_allow_unredacted_partials = base_policy.allow_unredacted_partials
    if allow_unredacted_partials is not None:
        if not base_policy.allow_unredacted_partials and allow_unredacted_partials:
            raise ValueError("request override cannot allow unredacted partials when tenant policy forbids them")
        merged_allow_unredacted_partials = bool(allow_unredacted_partials)

    merged_categories = list(base_policy.redact_categories)
    if redact_categories is not None:
        normalized_categories = _normalize_categories(redact_categories)
        if not set(base_policy.redact_categories).issubset(set(normalized_categories)):
            raise ValueError("request override cannot remove tenant-required redact categories")
        merged_categories = normalized_categories
        if normalized_categories:
            merged_redact_pii = True

    if merged_retention <= 0:
        merged_retention = 0.0
        merged_delete = True

    return STTPolicy(
        org_id=base_policy.org_id,
        delete_audio_after_success=merged_delete,
        audio_retention_hours=merged_retention,
        redact_pii=merged_redact_pii,
        allow_unredacted_partials=merged_allow_unredacted_partials,
        redact_categories=merged_categories,
    )


def apply_transcript_text_policy(
    text: str,
    *,
    policy: STTPolicy,
    is_partial: bool,
) -> str:
    if not isinstance(text, str) or not text:
        return text
    if is_partial and policy.allow_unredacted_partials:
        return text
    if not policy.redact_pii:
        return text

    try:
        moderation = get_moderation_service()
        pii_rules = moderation._load_builtin_pii_rules()
        moderation_policy = ModerationPolicy(
            enabled=True,
            input_enabled=False,
            output_enabled=True,
            input_action="warn",
            output_action="redact",
            redact_replacement="[PII]",
            per_user_overrides=False,
            block_patterns=pii_rules,
            categories_enabled=_expand_redact_categories(policy.redact_categories),
        )
        return moderation.redact_text(text, moderation_policy)
    except _STT_POLICY_EXCEPTIONS as exc:
        logger.warning("STT transcript redaction failed; suppressing text for safety: {}", exc)
        return "[redaction-unavailable]"


def apply_transcript_payload_policy(payload: dict[str, Any], *, policy: STTPolicy) -> dict[str, Any]:
    payload_type = str(payload.get("type", "")).strip().lower()
    if payload_type not in {"partial", "transcription", "full_transcript"}:
        return payload

    text = payload.get("text")
    if not isinstance(text, str):
        return payload

    updated = dict(payload)
    updated["text"] = apply_transcript_text_policy(
        text,
        policy=policy,
        is_partial=(payload_type == "partial" and not bool(payload.get("is_final"))),
    )
    segments = updated.get("segments")
    if isinstance(segments, list):
        updated["segments"] = redact_timed_segments(segments, policy=policy)
    return updated


def redact_timed_segments(
    segments: list[dict[str, Any]] | None,
    *,
    policy: STTPolicy,
) -> list[dict[str, Any]] | None:
    if not isinstance(segments, list):
        return segments

    redacted_segments: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            redacted_segments.append(segment)
            continue
        updated_segment = dict(segment)
        text = updated_segment.get("text", updated_segment.get("Text"))
        if isinstance(text, str):
            redacted_text = apply_transcript_text_policy(text, policy=policy, is_partial=False)
            if "text" in updated_segment:
                updated_segment["text"] = redacted_text
            if "Text" in updated_segment:
                updated_segment["Text"] = redacted_text
        redacted_segments.append(updated_segment)
    return redacted_segments


def build_audio_retention_decision(policy: STTPolicy) -> STTAudioRetentionDecision:
    if policy.delete_audio_after_success or policy.audio_retention_hours <= 0:
        return STTAudioRetentionDecision(delete_after_success=True, expires_at=None)
    return STTAudioRetentionDecision(
        delete_after_success=False,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=float(policy.audio_retention_hours)),
    )


def get_websocket_auth_principal(websocket: Any) -> AuthPrincipal | None:
    try:
        state = getattr(websocket, "state", None)
        principal = getattr(state, "auth_principal", None)
    except _STT_POLICY_EXCEPTIONS:
        return None
    return principal if isinstance(principal, AuthPrincipal) else None


class RedactingWebSocketProxy:
    def __init__(self, websocket: Any, *, policy: STTPolicy) -> None:
        self._websocket = websocket
        self._policy = policy

    def __getattr__(self, name: str) -> Any:
        return getattr(self._websocket, name)

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self._websocket.send_json(apply_transcript_payload_policy(payload, policy=self._policy))


def _get_stt_audio_storage_path(user_id: int, suffix: str) -> Path:
    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    date_path = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    target_dir = outputs_dir / "stt_audio" / date_path
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{datetime.now(timezone.utc).isoformat()}:{suffix}".encode("utf-8")).hexdigest()[:12]
    safe_suffix = suffix if str(suffix).startswith(".") else f".{suffix}" if suffix else ".wav"
    return target_dir / f"stt_audio_{digest}{safe_suffix}"


def _compute_file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


async def retain_stt_audio_artifact(
    *,
    user_id: int,
    source_path: str | Path,
    original_filename: str | None,
    policy: STTPolicy,
    org_id: int | None = None,
) -> dict[str, Any] | None:
    decision = build_audio_retention_decision(policy)
    if decision.delete_after_success:
        return None

    source = Path(source_path)
    if not source.exists():
        return None

    target_path = _get_stt_audio_storage_path(user_id, source.suffix or ".wav")
    await asyncio.to_thread(shutil.copy2, source, target_path)
    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    relative_path = str(target_path.relative_to(outputs_dir))
    mime_type = mimetypes.guess_type(target_path.name)[0] or "audio/wav"
    file_size = await asyncio.to_thread(lambda: int(target_path.stat().st_size))
    checksum = await asyncio.to_thread(_compute_file_checksum, target_path)
    service = await get_storage_service()

    try:
        return await service.register_generated_file(
            user_id=user_id,
            filename=target_path.name,
            storage_path=relative_path,
            file_category=FILE_CATEGORY_STT_AUDIO,
            source_feature=SOURCE_FEATURE_STT,
            file_size_bytes=file_size,
            org_id=org_id,
            original_filename=original_filename,
            mime_type=mime_type,
            checksum=checksum,
            expires_at=decision.expires_at,
            check_quota=True,
        )
    except Exception:
        with contextlib.suppress(Exception):
            target_path.unlink()
        raise


def ensure_websocket_state(websocket: Any) -> Any:
    state = getattr(websocket, "state", None)
    if state is None:
        state = SimpleNamespace()
        with contextlib.suppress(Exception):
            setattr(websocket, "state", state)
    return state
