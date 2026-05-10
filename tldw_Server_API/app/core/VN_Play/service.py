"""VN Play runtime service orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import VNProfileSnapshotRepository
from tldw_Server_API.app.core.DB_Management.VNScripts_DB import VNScriptsRepository
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Play.assets import resolve_scene_directives
from tldw_Server_API.app.core.VN_Play.branch_navigation import (
    branch_filter_ids,
    build_branch_navigation,
    filter_branch_events,
)
from tldw_Server_API.app.core.VN_Play.constants import (
    BRANCH_RESTORE_TARGET_CHOICE_POINT,
    BRANCH_RESTORE_TARGET_LATEST,
    ERROR_BRANCH_NOT_FOUND,
    ERROR_BRANCH_RESTORE_NOT_ALLOWED,
    ERROR_BRANCH_RESTORE_TARGET_UNAVAILABLE,
    ERROR_CHOICE_NOT_ALLOWED,
    ERROR_IDEMPOTENCY_KEY_CONFLICT,
    ERROR_INTERNAL_ERROR,
    ERROR_INVALID_CHOICE_ID,
    ERROR_RESTORE_ACTION_IN_PROGRESS,
    ERROR_RETRY_LAST_TURN_NOT_FAILED,
    ERROR_SCRIPT_ADVANCE_BLOCKED,
    ERROR_SCRIPT_ENDED,
    ERROR_SCRIPTED_STORY_TURN_NOT_ALLOWED,
    ERROR_STALE_SCENE_VERSION,
    ERROR_TURN_IN_PROGRESS,
    EVENT_CHOICE_SELECTED,
    EVENT_CHOICE_PRESENTED,
    EVENT_MODEL_TURN,
    EVENT_SCENE_STATE_CHANGED,
    EVENT_SESSION_CHECKPOINT_CREATED,
    EVENT_SESSION_RESTORED,
    EVENT_SESSION_STARTED,
    EVENT_TURN_COMPLETED,
    EVENT_TURN_FAILED,
    EVENT_TURN_STARTED,
    EVENT_VISUAL_DIRECTIVE_APPLIED,
    EVENT_VISUAL_DIRECTIVE_REJECTED,
    EVENT_VISUAL_DIRECTIVE_REQUESTED,
    EVENT_USER_TURN,
    EVENT_MODEL_TURN_PARSE_FAILED,
    MODE_FREEFORM,
    MODE_SCRIPTED_STORY,
    MODE_STORY,
    SESSION_ACTION_STATUS_ABANDONED,
    SESSION_ACTION_STATUS_COMPLETED,
    SESSION_ACTION_STATUS_FAILED,
    SESSION_ACTION_STATUS_PENDING,
    STORY_BRANCH_LABEL_MAX_LENGTH,
    TURN_STATUS_ABANDONED,
    TURN_STATUS_COMPLETED,
    TURN_STATUS_MODEL_CALLING,
    TURN_STATUS_MODEL_FAILED,
    TURN_STATUS_PARSE_FAILED,
    TURN_STATUS_PENDING,
)
from tldw_Server_API.app.core.VN_Policy.service import evaluate_character_safety_definition
from tldw_Server_API.app.core.VN_Play.models import SceneState, TurnResult
from tldw_Server_API.app.core.VN_Play.parser import VNPlayParseError, coerce_turn_result
from tldw_Server_API.app.core.VN_Play.state import derive_scene_state
from tldw_Server_API.app.core.VN_Scripts.service import _character_safety_status

MAX_SCRIPT_EXECUTION_STEPS = 500
SCRIPT_GENERATION_SOURCE_LITERAL = "script_literal"
_CONFLICT_REPLAY_ERROR_CODES = {
    ERROR_IDEMPOTENCY_KEY_CONFLICT,
    ERROR_RESTORE_ACTION_IN_PROGRESS,
    ERROR_STALE_SCENE_VERSION,
    ERROR_TURN_IN_PROGRESS,
}


class VNPlayError(Exception):
    """Base error raised by VN Play runtime services."""


class VNPlayNotFoundError(VNPlayError):
    """Raised when a VN Play resource cannot be found for the current owner."""


class VNPlayConflictError(VNPlayError):
    """Raised when the requested turn cannot be applied to current session state."""


class VNPlayTurnError(VNPlayError):
    """Raised when a turn attempt fails after being accepted."""


class VNPlayTurnAdapter(Protocol):
    """Adapter boundary for model-backed VN Play turn generation."""

    async def generate_turn(self, context: VNPlayTurnContext) -> Any:
        """Generate a normalized turn result for the provided context."""


@dataclass(frozen=True, slots=True)
class VNPlaySession:
    """Service-level VN Play session."""

    id: int
    owner_user_id: int
    mode: str
    title: str
    status: str
    primary_character_id: int
    additional_character_ids: list[int]
    linked_chat_id: str | None
    vn_asset_pack_id: int
    asset_manifest_version: str | None
    source_world_book_ids: list[int]
    content_rating: str
    trust_level: str
    linked_chat_mode: str
    seed: str | None
    settings: dict[str, Any]
    scene_version: int
    active_turn_request_id: int | None
    script_id: int | None = None
    script_version_id: int | None = None
    script_manifest_snapshot_id: int | None = None
    script_policy_snapshot_id: int | None = None
    script_generation_profile_snapshot_id: int | None = None
    script_position: dict[str, Any] = field(default_factory=dict)
    active_session_action_id: int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> VNPlaySession:
        return cls(
            id=int(row["id"]),
            owner_user_id=int(row["owner_user_id"]),
            mode=str(row["mode"]),
            title=str(row["title"]),
            status=str(row["status"]),
            primary_character_id=int(row["primary_character_id"]),
            additional_character_ids=[
                int(item) for item in row.get("additional_character_ids", [])
            ],
            linked_chat_id=_optional_str(row.get("linked_chat_id")),
            vn_asset_pack_id=int(row["vn_asset_pack_id"]),
            asset_manifest_version=_optional_str(row.get("asset_manifest_version")),
            source_world_book_ids=[
                int(item) for item in row.get("source_world_book_ids", [])
            ],
            content_rating=str(row["content_rating"]),
            trust_level=str(row["trust_level"]),
            linked_chat_mode=str(row["linked_chat_mode"]),
            seed=_optional_str(row.get("seed")),
            settings=dict(row.get("settings") or {}),
            script_id=_optional_int(row.get("script_id")),
            script_version_id=_optional_int(row.get("script_version_id")),
            script_manifest_snapshot_id=_optional_int(row.get("script_manifest_snapshot_id")),
            script_policy_snapshot_id=_optional_int(row.get("script_policy_snapshot_id")),
            script_generation_profile_snapshot_id=_optional_int(
                row.get("script_generation_profile_snapshot_id")
            ),
            script_position=dict(row.get("script_position") or {}),
            scene_version=int(row["scene_version"]),
            active_turn_request_id=_optional_int(row.get("active_turn_request_id")),
            active_session_action_id=_optional_int(row.get("active_session_action_id")),
            created_at=_optional_str(row.get("created_at")),
            updated_at=_optional_str(row.get("updated_at")),
        )


@dataclass(frozen=True, slots=True)
class VNPlayTurnContext:
    """Input provided to the turn adapter."""

    session: VNPlaySession
    input_payload: dict[str, Any]
    scene_state: SceneState
    recent_events: list[dict[str, Any]]
    turn_request_id: int


@dataclass(frozen=True, slots=True)
class VNPlayTurnResponse:
    """Stored response returned to clients for a submitted turn."""

    turn_request_id: int
    status: str
    scene_version: int
    events: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VNPlayTurnResponse:
        return cls(
            turn_request_id=int(payload["turn_request_id"]),
            status=str(payload["status"]),
            scene_version=int(payload["scene_version"]),
            events=_list_of_dicts(payload.get("events")),
            warnings=_list_of_dicts(payload.get("warnings")),
            error_code=(
                str(payload["error_code"])
                if payload.get("error_code") is not None
                else None
            ),
            error_message=(
                str(payload["error_message"])
                if payload.get("error_message") is not None
                else None
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "turn_request_id": self.turn_request_id,
            "status": self.status,
            "scene_version": self.scene_version,
            "events": self.events,
            "warnings": self.warnings,
        }
        if self.error_code is not None:
            payload["error_code"] = self.error_code
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        return payload


class DeterministicVNPlayTurnAdapter:
    """Deterministic adapter for tests and local smoke flows."""

    async def generate_turn(self, context: VNPlayTurnContext) -> TurnResult:
        input_text = _input_text(context.input_payload)
        narrative_text = f"Echo: {input_text}"
        return TurnResult(
            narrative_text=narrative_text,
            dialogue=[{"speaker": "Narrator", "text": narrative_text}],
            scene_updates={
                "location_key": context.scene_state.location_key or "default",
                "mood": context.scene_state.mood or "neutral",
            },
        )


class VNPlayService:
    """High-level service for VN Play sessions and turns."""

    def __init__(
        self,
        *,
        repo: VNPlayRepository,
        owner_user_id: int,
        adapter: VNPlayTurnAdapter | None = None,
    ) -> None:
        self.repo = repo
        self.owner_user_id = owner_user_id
        self.adapter = adapter or DeterministicVNPlayTurnAdapter()
        self._manifest_cache: dict[int, dict[str, Any]] = {}

    def create_session(
        self,
        *,
        mode: str,
        title: str,
        primary_character_id: int,
        vn_asset_pack_id: int,
        additional_character_ids: Sequence[int] | None = None,
        linked_chat_id: str | None = None,
        asset_manifest_version: str | None = None,
        source_world_book_ids: Sequence[int] | None = None,
        content_rating: str = "general",
        trust_level: str = "local",
        linked_chat_mode: str = "read_only_context",
        seed: str | None = None,
        settings: Mapping[str, Any] | None = None,
        script_id: int | None = None,
        script_version_id: int | None = None,
        acknowledgements: Sequence[str] | None = None,
    ) -> VNPlaySession:
        script_context = self._script_session_context(
            mode=mode,
            primary_character_id=primary_character_id,
            vn_asset_pack_id=vn_asset_pack_id,
            content_rating=content_rating,
            script_id=script_id,
            script_version_id=script_version_id,
            acknowledgements=acknowledgements,
        )
        row = self.repo.create_session(
            owner_user_id=self.owner_user_id,
            mode=mode,
            title=title,
            primary_character_id=primary_character_id,
            vn_asset_pack_id=vn_asset_pack_id,
            additional_character_ids=additional_character_ids,
            linked_chat_id=linked_chat_id,
            asset_manifest_version=asset_manifest_version,
            source_world_book_ids=source_world_book_ids,
            content_rating=content_rating,
            trust_level=trust_level,
            linked_chat_mode=linked_chat_mode,
            seed=seed,
            settings=settings,
            script_id=script_context.get("script_id"),
            script_version_id=script_context.get("script_version_id"),
            script_manifest_snapshot_id=script_context.get("script_manifest_snapshot_id"),
            script_policy_snapshot_id=script_context.get("script_policy_snapshot_id"),
            script_generation_profile_snapshot_id=script_context.get(
                "script_generation_profile_snapshot_id"
            ),
            script_position=script_context.get("script_position"),
        )
        self.repo.append_event(
            session_id=int(row["id"]),
            owner_user_id=self.owner_user_id,
            event_type=EVENT_SESSION_STARTED,
            event_payload={
                "schema_version": 1,
                "scene_version": int(row["scene_version"]),
                "mode": mode,
                "content_rating": content_rating,
            },
            source="system",
        )
        return VNPlaySession.from_row(row)

    def _script_session_context(
        self,
        *,
        mode: str,
        primary_character_id: int,
        vn_asset_pack_id: int,
        content_rating: str,
        script_id: int | None,
        script_version_id: int | None,
        acknowledgements: Sequence[str] | None,
    ) -> dict[str, Any]:
        """Resolve immutable script metadata to pin on a scripted session."""
        if mode != MODE_SCRIPTED_STORY:
            if script_id is not None or script_version_id is not None:
                raise ValueError("script_fields_require_scripted_story")
            return {}
        if script_id is None or script_version_id is None:
            raise ValueError("script_version_required")

        script_repo = VNScriptsRepository.initialized(self.repo.db)
        script = script_repo.get_script(
            int(script_id),
            owner_user_id=self.owner_user_id,
        )
        if script is None:
            raise ValueError("script_not_found")
        version = script_repo.get_version(
            int(script_id),
            int(script_version_id),
            owner_user_id=self.owner_user_id,
        )
        if version is None:
            raise ValueError("script_version_not_found")
        if int(script["primary_asset_pack_id"]) != int(version["asset_pack_id"]):
            raise ValueError("script_asset_pack_mismatch")
        if int(version["asset_pack_id"]) != int(vn_asset_pack_id):
            raise ValueError("script_asset_pack_mismatch")
        pack = VNAssetPackService(
            self.repo.db,
            owner_user_id=self.owner_user_id,
        ).get_pack(int(version["asset_pack_id"]))
        script_primary_character_id = int(pack.primary_character_id)
        script_content_rating = str(script.get("content_rating") or pack.content_rating or "general")
        if script_primary_character_id != int(primary_character_id):
            raise ValueError("script_primary_character_mismatch")
        if script_content_rating != str(content_rating):
            raise ValueError("script_content_rating_mismatch")

        policy_snapshot = VNProfileSnapshotRepository.initialized(
            self.repo.db
        ).get_profile_snapshot(
            int(version["policy_snapshot_id"]),
            owner_user_id=self.owner_user_id,
        )
        if policy_snapshot is None:
            raise ValueError("policy_snapshot_not_found")

        policy_decision = evaluate_character_safety_definition(
            profile_definition=policy_snapshot["definition"],
            policy_profile_id=str(policy_snapshot["profile_id"]),
            content_rating=script_content_rating,
            metadata_status=self._character_safety_status(script_primary_character_id),
        )
        if policy_decision.get("blocked"):
            raise ValueError("script_session_policy_blocked")
        required_acknowledgements = {
            str(reason["code"])
            for reason in policy_decision.get("reasons", [])
            if isinstance(reason, Mapping)
            and reason.get("requires_acknowledgement")
            and reason.get("code")
        }
        if not required_acknowledgements.issubset(set(acknowledgements or [])):
            raise ValueError("script_session_acknowledgement_required")

        program = version.get("program")
        return {
            "script_id": int(version["script_id"]),
            "script_version_id": int(version["id"]),
            "script_manifest_snapshot_id": int(version["manifest_snapshot_id"]),
            "script_policy_snapshot_id": int(version["policy_snapshot_id"]),
            "script_generation_profile_snapshot_id": int(
                version["generation_profile_snapshot_id"]
            ),
            "script_position": _initial_script_position(program),
            "primary_character_id": script_primary_character_id,
            "content_rating": script_content_rating,
        }

    def _character_safety_status(self, character_id: int) -> str:
        character = self.repo.db.get_character_card_by_id(character_id)
        if not isinstance(character, Mapping):
            return "missing"
        return _character_safety_status(character)

    def list_sessions(self, *, include_deleted: bool = False) -> list[VNPlaySession]:
        return [
            VNPlaySession.from_row(row)
            for row in self.repo.list_sessions(
                owner_user_id=self.owner_user_id,
                include_deleted=include_deleted,
            )
        ]

    def get_session(self, session_id: int) -> VNPlaySession:
        row = self.repo.get_session(session_id, owner_user_id=self.owner_user_id)
        if row is None:
            raise VNPlayNotFoundError("session_not_found")
        return VNPlaySession.from_row(row)

    def public_session_payload(self, session: VNPlaySession) -> dict[str, Any]:
        """Return a session payload without raw scripted-story interpreter internals."""
        return _public_session_payload(session)

    def public_checkpoint_payload(self, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
        """Return a checkpoint payload without raw scripted-story interpreter internals."""
        payload = dict(checkpoint)
        snapshot = dict(payload.get("scene_state_snapshot") or {})
        script_position = _script_position_from_snapshot(snapshot)
        if script_position is not None:
            snapshot["script_position"] = {
                "progress_token": _script_progress_token(script_position)
            }
        payload["scene_state_snapshot"] = snapshot
        return payload

    async def submit_turn(
        self,
        session_id: int,
        *,
        input_text: str | None = None,
        choice_id: str | None = None,
        custom_action: Mapping[str, Any] | None = None,
        client_scene_version: int,
        idempotency_key: str,
    ) -> VNPlayTurnResponse:
        if not idempotency_key:
            raise VNPlayTurnError("idempotency_key_required")

        input_payload = _normalize_input_payload(
            input_text=input_text,
            choice_id=choice_id,
            custom_action=custom_action,
        )
        request_payload_hash = _payload_hash(
            {
                "session_id": session_id,
                "input": input_payload,
            }
        )

        existing = self.repo.get_turn_request_by_key(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            return self._response_for_existing_turn(existing, request_payload_hash)

        session = self.get_session(session_id)
        persisted_scene_state = self.repo.get_scene_state(
            session_id,
            owner_user_id=self.owner_user_id,
        )
        if session.scene_version != client_scene_version:
            turn_request = self.repo.create_turn_request(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                idempotency_key=idempotency_key,
                request_payload_hash=request_payload_hash,
                base_scene_version=client_scene_version,
                status=TURN_STATUS_ABANDONED,
            )
            self.repo.update_turn_request(
                int(turn_request["id"]),
                {
                    "error": {"code": ERROR_STALE_SCENE_VERSION},
                },
                owner_user_id=self.owner_user_id,
            )
            raise VNPlayConflictError(ERROR_STALE_SCENE_VERSION)
        if session.active_turn_request_id is not None:
            raise VNPlayConflictError(ERROR_TURN_IN_PROGRESS)
        _validate_turn_input_for_mode(session, input_payload)

        selected_choice: dict[str, Any] | None = None
        parent_choice_event_id: int | None = None
        expected_scene_last_event_id: int | None = None
        branch_path: list[dict[str, Any]] | None = None
        if session.mode == MODE_STORY and choice_id is not None:
            events_before_input = self.repo.list_events(session_id)
            selected_choice = _selected_visible_choice(persisted_scene_state, choice_id)
            expected_scene_last_event_id = _optional_int(
                persisted_scene_state.get("last_event_id")
                if persisted_scene_state is not None
                else None
            )
            parent_choice_event_id = _parent_choice_event_id(
                events_before_input,
                expected_scene_last_event_id,
                choice_id,
            )
            branch_path = _branch_path_for_choice(
                selected_choice,
                scene_version=client_scene_version,
                choice_presented_event_id=parent_choice_event_id,
                parent_branch_path=_active_branch_path(
                    self.repo.list_branches(
                        session_id,
                        owner_user_id=self.owner_user_id,
                    ),
                    persisted_scene_state,
                ),
            )
            input_payload = {
                "choice_id": choice_id,
                "choice": selected_choice,
            }

        turn_request = self.repo.create_turn_request(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            idempotency_key=idempotency_key,
            request_payload_hash=request_payload_hash,
            base_scene_version=client_scene_version,
            status=TURN_STATUS_PENDING,
        )
        lock_acquired = self.repo.try_acquire_turn_lock(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            turn_request_id=int(turn_request["id"]),
            expected_scene_version=client_scene_version,
        )
        if not lock_acquired:
            return self._abandon_conflicting_turn(turn_request, session_id)

        if selected_choice is not None:
            try:
                persisted_choice = self.repo.record_story_choice_selection(
                    session_id=session_id,
                    owner_user_id=self.owner_user_id,
                    turn_request_id=int(turn_request["id"]),
                    client_scene_version=client_scene_version,
                    selected_choice=selected_choice,
                    parent_event_id=parent_choice_event_id,
                    expected_scene_last_event_id=expected_scene_last_event_id,
                    branch_label=_choice_text(selected_choice),
                    branch_path=branch_path,
                )
            except RuntimeError as exc:
                if str(exc) not in {"choice_not_visible", "scene_state_moved"}:
                    raise
                self.repo.update_turn_request(
                    int(turn_request["id"]),
                    {
                        "status": TURN_STATUS_ABANDONED,
                        "error": {
                            "code": ERROR_INVALID_CHOICE_ID,
                            "detail": str(exc),
                        },
                    },
                    owner_user_id=self.owner_user_id,
                )
                self.repo.update_session(
                    session_id,
                    {"active_turn_request_id": None},
                    owner_user_id=self.owner_user_id,
                )
                raise VNPlayTurnError(ERROR_INVALID_CHOICE_ID) from exc
            turn_events = [
                persisted_choice["turn_started"],
                persisted_choice["choice_selected"],
            ]
        else:
            turn_events = self._append_accepted_turn_events(
                session_id=session_id,
                turn_request_id=int(turn_request["id"]),
                input_payload=input_payload,
                client_scene_version=client_scene_version,
            )
        scene_state = derive_scene_state(self.repo.list_events(session_id))
        context = VNPlayTurnContext(
            session=session,
            input_payload=input_payload,
            scene_state=scene_state,
            recent_events=self.repo.list_events(session_id, limit=50),
            turn_request_id=int(turn_request["id"]),
        )

        try:
            result = coerce_turn_result(await self.adapter.generate_turn(context))
        except VNPlayParseError as exc:
            self._mark_turn_failed(
                session_id=session_id,
                turn_request_id=int(turn_request["id"]),
                error_code=TURN_STATUS_PARSE_FAILED,
                error_message=str(exc),
                client_scene_version=client_scene_version,
                event_type=EVENT_MODEL_TURN_PARSE_FAILED,
            )
            raise VNPlayTurnError(TURN_STATUS_PARSE_FAILED) from exc
        except Exception as exc:
            self._mark_turn_failed(
                session_id=session_id,
                turn_request_id=int(turn_request["id"]),
                error_code=TURN_STATUS_MODEL_FAILED,
                error_message=str(exc),
                client_scene_version=client_scene_version,
            )
            raise VNPlayTurnError(TURN_STATUS_MODEL_FAILED) from exc

        return self._complete_turn(
            session=session,
            session_id=session_id,
            turn_request_id=int(turn_request["id"]),
            prior_events=turn_events,
            result=result,
            next_scene_version=client_scene_version + 1,
        )

    async def retry_last_turn(
        self,
        session_id: int,
        *,
        client_scene_version: int,
        idempotency_key: str,
    ) -> VNPlayTurnResponse:
        if not idempotency_key:
            raise VNPlayTurnError("idempotency_key_required")

        existing = self.repo.get_turn_request_by_key(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            return self._response_for_existing_turn(
                existing,
                str(existing["request_payload_hash"]),
            )

        session = self.get_session(session_id)
        if session.scene_version != client_scene_version:
            raise VNPlayConflictError(ERROR_STALE_SCENE_VERSION)
        if session.active_turn_request_id is not None:
            raise VNPlayConflictError(ERROR_TURN_IN_PROGRESS)

        retry_source = self.repo.latest_retryable_turn_request(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
        )
        if retry_source is None:
            raise VNPlayTurnError(ERROR_RETRY_LAST_TURN_NOT_FAILED)
        source_event = self.repo.get_event(int(retry_source["input_event_id"]))
        if source_event is None:
            raise VNPlayTurnError(ERROR_RETRY_LAST_TURN_NOT_FAILED)
        input_payload = _retry_input_payload_from_event(source_event)
        request_payload_hash = _payload_hash(
            {
                "session_id": session_id,
                "retry_of_turn_request_id": int(retry_source["id"]),
                "retry_source_event_id": int(source_event["id"]),
                "input": input_payload,
            }
        )

        turn_request = self.repo.create_turn_request(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            idempotency_key=idempotency_key,
            request_payload_hash=request_payload_hash,
            base_scene_version=client_scene_version,
            status=TURN_STATUS_PENDING,
        )
        lock_acquired = self.repo.try_acquire_turn_lock(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            turn_request_id=int(turn_request["id"]),
            expected_scene_version=client_scene_version,
        )
        if not lock_acquired:
            return self._abandon_conflicting_turn(turn_request, session_id)

        turn_started = self.repo.append_event(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            event_type=EVENT_TURN_STARTED,
            event_payload={
                "turn_request_id": int(turn_request["id"]),
                "retry_of_turn_request_id": int(retry_source["id"]),
                "retry_source_event_id": int(source_event["id"]),
                "scene_version": client_scene_version,
            },
            source="runtime",
        )
        self.repo.update_turn_request(
            int(turn_request["id"]),
            {
                "status": TURN_STATUS_MODEL_CALLING,
                "turn_started_event_id": turn_started["id"],
                "input_event_id": source_event["id"],
            },
            owner_user_id=self.owner_user_id,
        )
        scene_state = derive_scene_state(self.repo.list_events(session_id))
        context = VNPlayTurnContext(
            session=session,
            input_payload=input_payload,
            scene_state=scene_state,
            recent_events=self.repo.list_events(session_id, limit=50),
            turn_request_id=int(turn_request["id"]),
        )

        try:
            result = coerce_turn_result(await self.adapter.generate_turn(context))
        except VNPlayParseError as exc:
            self._mark_turn_failed(
                session_id=session_id,
                turn_request_id=int(turn_request["id"]),
                error_code=TURN_STATUS_PARSE_FAILED,
                error_message=str(exc),
                client_scene_version=client_scene_version,
                event_type=EVENT_MODEL_TURN_PARSE_FAILED,
            )
            raise VNPlayTurnError(TURN_STATUS_PARSE_FAILED) from exc
        except Exception as exc:
            self._mark_turn_failed(
                session_id=session_id,
                turn_request_id=int(turn_request["id"]),
                error_code=TURN_STATUS_MODEL_FAILED,
                error_message=str(exc),
                client_scene_version=client_scene_version,
            )
            raise VNPlayTurnError(TURN_STATUS_MODEL_FAILED) from exc

        return self._complete_turn(
            session=session,
            session_id=session_id,
            turn_request_id=int(turn_request["id"]),
            prior_events=[turn_started],
            result=result,
            next_scene_version=client_scene_version + 1,
        )

    def list_events(self, session_id: int) -> list[dict[str, Any]]:
        return self.list_events_with_metadata(session_id)["events"]

    def list_events_with_metadata(
        self,
        session_id: int,
        *,
        branch_id: int | None = None,
        after_sequence: int | None = None,
        limit: int | None = None,
        include_descendants: bool = False,
    ) -> dict[str, Any]:
        self.get_session(session_id)
        if branch_id is None:
            return {
                "events": self.repo.list_events(
                    session_id,
                    after_sequence=after_sequence,
                    limit=limit,
                ),
                "warnings": [],
            }

        branches = self.repo.list_branches(
            session_id,
            owner_user_id=self.owner_user_id,
        )
        if not any(int(branch["id"]) == branch_id for branch in branches):
            raise VNPlayNotFoundError(ERROR_BRANCH_NOT_FOUND)

        branch_ids = branch_filter_ids(
            branch_id=branch_id,
            branches=branches,
            include_descendants=include_descendants,
        )
        if self.repo.can_filter_branch_events_by_tags(session_id):
            return {
                "events": self.repo.list_events_for_branch_nodes(
                    session_id,
                    sorted(branch_ids),
                    after_sequence=after_sequence,
                    limit=limit,
                ),
                "warnings": [],
            }

        events = self.repo.list_events(session_id)
        filtered_events, warnings = filter_branch_events(
            branch_id=branch_id,
            branches=branches,
            events=events,
            include_descendants=include_descendants,
            after_sequence=after_sequence,
            limit=limit if limit is not None else len(events),
        )
        return {"events": filtered_events, "warnings": warnings}

    def get_branch_navigation(self, session_id: int) -> dict[str, Any]:
        session = self.get_session(session_id)
        branches = self.repo.list_branches(
            session_id,
            owner_user_id=self.owner_user_id,
        )
        events = self.repo.list_events(session_id)
        scene_state = self.repo.get_scene_state(
            session_id,
            owner_user_id=self.owner_user_id,
        )
        return build_branch_navigation(
            session=_session_payload(session),
            branches=branches,
            events=events,
            scene_state=scene_state,
        )

    def get_enriched_scene_state(self, session_id: int) -> dict[str, Any] | None:
        session = self.get_session(session_id)
        state = self.repo.get_scene_state(
            session_id,
            owner_user_id=self.owner_user_id,
        )
        if state is None:
            return None

        enriched = dict(state)
        try:
            manifest = self._build_pack_manifest(session.vn_asset_pack_id)
        except Exception as exc:
            logger.exception(
                "Failed to enrich VN Play scene state from VN asset manifest: "
                "session_id={}, pack_id={}",
                session_id,
                session.vn_asset_pack_id,
            )
            _append_enrichment_warning(enriched, exc)
            return enriched

        items_by_id = _manifest_items_by_id(manifest)
        background_id = enriched.get("current_background_item_id")
        if isinstance(background_id, int):
            enriched["background"] = items_by_id.get(background_id)

        depth_id = enriched.get("current_depth_item_id")
        if isinstance(depth_id, int):
            enriched["depth"] = items_by_id.get(depth_id)

        active_sprites: list[dict[str, Any]] = []
        for sprite in _list_of_dicts(enriched.get("active_sprite_items")):
            item_id = sprite.get("item_id")
            if isinstance(item_id, int) and item_id in items_by_id:
                active_sprites.append(items_by_id[item_id])
        enriched["active_sprites"] = active_sprites
        return enriched

    def create_checkpoint(self, session_id: int, *, label: str) -> dict[str, Any]:
        session = self.get_session(session_id)
        events = self.repo.list_events(session_id)
        state = self.repo.get_scene_state(session_id, owner_user_id=self.owner_user_id)
        if state is None:
            raise VNPlayNotFoundError("scene_state_not_found")
        scene_state_snapshot = dict(state)
        if session.mode == MODE_SCRIPTED_STORY:
            scene_state_snapshot["script_position"] = dict(session.script_position)
        last_event_id = int(events[-1]["id"]) if events else None
        checkpoint = self.repo.create_checkpoint(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            label=label,
            event_id=last_event_id,
            scene_version=int(state["scene_version"]),
            scene_state_snapshot=scene_state_snapshot,
        )
        self.repo.append_event(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            event_type=EVENT_SESSION_CHECKPOINT_CREATED,
            event_payload={
                "checkpoint_id": checkpoint["id"],
                "label": label,
                "scene_version": state["scene_version"],
            },
            source="runtime",
        )
        return checkpoint

    def restore_branch(
        self,
        session_id: int,
        *,
        branch_id: int,
        client_scene_version: int,
        idempotency_key: str,
        target: str = BRANCH_RESTORE_TARGET_LATEST,
    ) -> dict[str, Any]:
        """Restore a story branch with idempotent session-action serialization.

        Completed actions replay their stored response. Failed or abandoned
        terminal actions return their persisted error code without mutating the
        original action state.
        """
        if not idempotency_key:
            raise VNPlayTurnError("idempotency_key_required")
        if target not in {BRANCH_RESTORE_TARGET_LATEST, BRANCH_RESTORE_TARGET_CHOICE_POINT}:
            raise VNPlayConflictError(ERROR_BRANCH_RESTORE_TARGET_UNAVAILABLE)

        session = self.get_session(session_id)
        if session.mode != MODE_STORY:
            raise VNPlayConflictError(ERROR_BRANCH_RESTORE_NOT_ALLOWED)

        branches = self.repo.list_branches(
            session_id,
            owner_user_id=self.owner_user_id,
        )
        branch = next(
            (item for item in branches if int(item["id"]) == branch_id),
            None,
        )
        if branch is None:
            raise VNPlayNotFoundError(ERROR_BRANCH_NOT_FOUND)

        request_payload_hash = _payload_hash(
            {
                "action_type": "branch_restore",
                "branch_id": branch_id,
                "target": target,
                "client_scene_version": client_scene_version,
            }
        )
        action = self._create_or_replay_session_action(
            session_id=session_id,
            action_type="branch_restore",
            idempotency_key=idempotency_key,
            request_payload_hash=request_payload_hash,
        )
        if action["status"] == SESSION_ACTION_STATUS_COMPLETED:
            return self._completed_session_action_response(action)
        self._raise_for_terminal_session_action(action)

        self._validate_restore_can_start(
            session_id=session_id,
            action_id=int(action["id"]),
            expected_scene_version=client_scene_version,
        )

        try:
            target_event_id = self._branch_restore_target_event_id(
                session_id=session_id,
                branch_id=branch_id,
                target=target,
            )
            target_state = self._scene_state_through_event(
                session_id=session_id,
                target_event_id=target_event_id,
            )
            previous_scene_version = session.scene_version
            next_scene_version = previous_scene_version + 1
            restore_snapshot = _scene_state_payload(target_state)
            restored_state = dict(restore_snapshot)
            restored_state["scene_version"] = next_scene_version
            event_payload = {
                "restore_kind": "branch",
                "branch_id": branch_id,
                "target": target,
                "target_event_id": target_event_id,
                "scene_state_snapshot": restore_snapshot,
                "previous_scene_version": previous_scene_version,
                "scene_version": next_scene_version,
                "idempotency_key": idempotency_key,
            }
            response = self.repo.commit_session_restore_action(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                action_id=int(action["id"]),
                event_payload=event_payload,
                scene_state=restored_state,
                scene_version=next_scene_version,
                branch_node_id=_optional_int(restored_state.get("active_branch_node_id")),
                response_payload_factory=lambda payload: self._restore_response_payload(
                    payload,
                    branch_id=branch_id,
                    target=target,
                    target_event_id=target_event_id,
                    replayed=False,
                ),
            )
        except VNPlayConflictError as exc:
            self._abandon_session_action(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=str(exc),
            )
            raise
        except Exception as exc:
            self._mark_session_action_failed(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=ERROR_INTERNAL_ERROR,
                error_type=exc.__class__.__name__,
            )
            raise
        return response

    def restore_checkpoint(
        self,
        session_id: int,
        checkpoint_id: int,
        *,
        client_scene_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Restore a checkpoint through the shared idempotent restore pipeline.

        The stored session action owns replay, active mutation locking, and
        terminal failure semantics so duplicate client retries cannot fork
        session state or overwrite prior failure diagnostics.
        """
        if not idempotency_key:
            raise VNPlayTurnError("idempotency_key_required")

        session = self.get_session(session_id)
        checkpoints = self.repo.list_checkpoints(
            session_id,
            owner_user_id=self.owner_user_id,
        )
        checkpoint = next(
            (item for item in checkpoints if int(item["id"]) == checkpoint_id),
            None,
        )
        if checkpoint is None:
            raise VNPlayNotFoundError("checkpoint_not_found")

        request_payload_hash = _payload_hash(
            {
                "action_type": "checkpoint_restore",
                "checkpoint_id": checkpoint_id,
                "client_scene_version": client_scene_version,
            }
        )
        action = self._create_or_replay_session_action(
            session_id=session_id,
            action_type="checkpoint_restore",
            idempotency_key=idempotency_key,
            request_payload_hash=request_payload_hash,
        )
        if action["status"] == SESSION_ACTION_STATUS_COMPLETED:
            return self._completed_session_action_response(action)
        self._raise_for_terminal_session_action(action)

        self._validate_restore_can_start(
            session_id=session_id,
            action_id=int(action["id"]),
            expected_scene_version=client_scene_version,
        )

        snapshot = dict(checkpoint["scene_state_snapshot"])
        script_position = _script_position_from_snapshot(snapshot)
        previous_scene_version = session.scene_version
        next_scene_version = previous_scene_version + 1
        restored_state = _scene_snapshot_without_script_position(snapshot)
        restored_state["scene_version"] = next_scene_version
        event_payload = {
            "restore_kind": "checkpoint",
            "checkpoint_id": checkpoint_id,
            "scene_state_snapshot": snapshot,
            "previous_scene_version": previous_scene_version,
            "scene_version": next_scene_version,
            "idempotency_key": idempotency_key,
        }
        try:
            response = self.repo.commit_session_restore_action(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                action_id=int(action["id"]),
                event_payload=event_payload,
                scene_state=restored_state,
                scene_version=next_scene_version,
                script_position=script_position,
                branch_node_id=_optional_int(restored_state.get("active_branch_node_id")),
                response_payload_factory=lambda payload: self._restore_response_payload(
                    payload,
                    checkpoint_id=checkpoint_id,
                    target_event_id=_optional_int(checkpoint.get("event_id")),
                    replayed=False,
                ),
            )
        except Exception as exc:
            self._mark_session_action_failed(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=ERROR_INTERNAL_ERROR,
                error_type=exc.__class__.__name__,
            )
            raise
        return response

    async def start_story(
        self,
        session_id: int,
        *,
        client_scene_version: int,
        idempotency_key: str,
    ) -> VNPlayTurnResponse | dict[str, Any]:
        """Start a model Story session or advance a scripted story from its entry."""
        session = self.get_session(session_id)
        if session.mode == MODE_SCRIPTED_STORY:
            return self.advance_script(
                session_id,
                client_scene_version=client_scene_version,
                idempotency_key=idempotency_key,
            )
        if session.mode != MODE_STORY:
            raise VNPlayConflictError("story_start_requires_story_mode")
        return await self.submit_turn(
            session_id,
            custom_action={"verb": "start_story"},
            client_scene_version=client_scene_version,
            idempotency_key=idempotency_key,
        )

    def create_save_slot(
        self,
        session_id: int,
        *,
        slot_key: str,
        title: str,
        metadata: Mapping[str, Any] | None = None,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not idempotency_key:
            raise VNPlayTurnError("idempotency_key_required")
        session = self.get_session(session_id)
        normalized_metadata = dict(metadata or {})
        request_payload_hash = _payload_hash(
            {
                "action_type": "save_slot_create",
                "slot_key": slot_key,
                "title": title,
                "metadata": normalized_metadata,
            }
        )
        action = self._create_or_replay_session_action(
            session_id=session_id,
            action_type="save_slot_create",
            idempotency_key=idempotency_key,
            request_payload_hash=request_payload_hash,
        )
        if action["status"] == SESSION_ACTION_STATUS_COMPLETED:
            return self._completed_session_action_response(action)
        self._raise_for_terminal_session_action(action)
        response_payload = action.get("response_payload")
        if response_payload is not None:
            raise VNPlayConflictError(ERROR_RESTORE_ACTION_IN_PROGRESS)

        self._validate_restore_can_start(
            session_id=session_id,
            action_id=int(action["id"]),
            expected_scene_version=session.scene_version,
        )

        try:
            events = self.repo.list_events(session_id)
            state = self.repo.get_scene_state(session_id, owner_user_id=self.owner_user_id)
            if state is None:
                raise VNPlayNotFoundError("scene_state_not_found")
            scene_state_snapshot = dict(state)
            if session.mode == MODE_SCRIPTED_STORY:
                scene_state_snapshot["script_position"] = dict(session.script_position)
            last_event_id = int(events[-1]["id"]) if events else None
            response = self.repo.commit_save_slot_create_action(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                action_id=int(action["id"]),
                slot_key=slot_key,
                title=title,
                metadata=normalized_metadata,
                event_id=last_event_id,
                scene_version=int(state["scene_version"]),
                scene_state_snapshot=scene_state_snapshot,
                response_payload_factory=lambda save_slot: _save_slot_response(
                    save_slot,
                    replayed=False,
                ),
            )
            return response
        except VNPlayConflictError as exc:
            self._abandon_session_action(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=str(exc),
            )
            raise
        except Exception as exc:
            self._mark_session_action_failed(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=ERROR_INTERNAL_ERROR,
                error_type=exc.__class__.__name__,
            )
            raise

    def list_save_slots(
        self,
        session_id: int,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        self.get_session(session_id)
        return self.repo.list_save_slots(
            session_id,
            owner_user_id=self.owner_user_id,
            include_deleted=include_deleted,
        )

    def get_save_slot(
        self,
        session_id: int,
        save_slot_id: int,
        *,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        self.get_session(session_id)
        slot = self.repo.get_save_slot(
            save_slot_id,
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            include_deleted=include_deleted,
        )
        if slot is None:
            raise VNPlayNotFoundError("save_slot_not_found")
        return slot

    def update_save_slot(
        self,
        session_id: int,
        save_slot_id: int,
        *,
        title: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.get_session(session_id)
        slot = self.repo.update_save_slot(
            save_slot_id,
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            title=title,
            metadata=metadata,
        )
        if slot is None or slot.get("deleted"):
            raise VNPlayNotFoundError("save_slot_not_found")
        return slot

    def delete_save_slot(self, session_id: int, save_slot_id: int) -> None:
        self.get_session(session_id)
        slot = self.repo.update_save_slot(
            save_slot_id,
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            deleted=True,
        )
        if slot is None:
            raise VNPlayNotFoundError("save_slot_not_found")

    def restore_save_slot(
        self,
        session_id: int,
        save_slot_id: int,
        *,
        client_scene_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not idempotency_key:
            raise VNPlayTurnError("idempotency_key_required")

        session = self.get_session(session_id)
        slot = self.get_save_slot(session_id, save_slot_id)
        checkpoint_id = int(slot["checkpoint_id"])
        request_payload_hash = _payload_hash(
            {
                "action_type": "save_slot_restore",
                "save_slot_id": save_slot_id,
                "checkpoint_id": checkpoint_id,
                "client_scene_version": client_scene_version,
            }
        )
        action = self._create_or_replay_session_action(
            session_id=session_id,
            action_type="save_slot_restore",
            idempotency_key=idempotency_key,
            request_payload_hash=request_payload_hash,
        )
        if action["status"] == SESSION_ACTION_STATUS_COMPLETED:
            return self._completed_session_action_response(action)
        self._raise_for_terminal_session_action(action)

        try:
            if session.scene_version != client_scene_version:
                raise VNPlayConflictError(ERROR_STALE_SCENE_VERSION)
            self._validate_restore_can_start(
                session_id=session_id,
                action_id=int(action["id"]),
                expected_scene_version=client_scene_version,
            )
            checkpoint = self.repo.get_checkpoint(checkpoint_id)
            if checkpoint is None or int(checkpoint["session_id"]) != session_id:
                raise VNPlayNotFoundError("checkpoint_not_found")
            snapshot = dict(checkpoint["scene_state_snapshot"])
            script_position = _script_position_from_snapshot(snapshot)
            previous_scene_version = session.scene_version
            next_scene_version = previous_scene_version + 1
            restored_state = _scene_snapshot_without_script_position(snapshot)
            restored_state["scene_version"] = next_scene_version
            event_payload = {
                "restore_kind": "save_slot",
                "save_slot_id": save_slot_id,
                "checkpoint_id": checkpoint_id,
                "scene_state_snapshot": snapshot,
                "previous_scene_version": previous_scene_version,
                "scene_version": next_scene_version,
                "idempotency_key": idempotency_key,
            }
            response = self.repo.commit_session_restore_action(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                action_id=int(action["id"]),
                event_payload=event_payload,
                scene_state=restored_state,
                scene_version=next_scene_version,
                script_position=script_position,
                branch_node_id=_optional_int(restored_state.get("active_branch_node_id")),
                response_payload_factory=lambda payload: self._restore_response_payload(
                    payload,
                    checkpoint_id=checkpoint_id,
                    save_slot_id=save_slot_id,
                    target_event_id=_optional_int(checkpoint.get("event_id")),
                    replayed=False,
                ),
            )
        except VNPlayConflictError as exc:
            self._abandon_session_action(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=str(exc),
            )
            raise
        except Exception as exc:
            self._mark_session_action_failed(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=ERROR_INTERNAL_ERROR,
                error_type=exc.__class__.__name__,
            )
            raise
        return response

    def advance_script(
        self,
        session_id: int,
        *,
        client_scene_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Advance a scripted story until it reaches a visible choice or end."""
        return self._run_script_action(
            session_id=session_id,
            action_type="script_advance",
            client_scene_version=client_scene_version,
            idempotency_key=idempotency_key,
            choice_id=None,
        )

    def choose_script_option(
        self,
        session_id: int,
        *,
        choice_id: str,
        client_scene_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Select a visible scripted-story choice and advance the target label."""
        return self._run_script_action(
            session_id=session_id,
            action_type="script_choice",
            client_scene_version=client_scene_version,
            idempotency_key=idempotency_key,
            choice_id=choice_id,
        )

    def regenerate_script_expansion(
        self,
        session_id: int,
        *,
        client_scene_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        """Append a regenerated model-expansion lineage event without rewriting history."""
        return self._run_script_regenerate_action(
            session_id=session_id,
            client_scene_version=client_scene_version,
            idempotency_key=idempotency_key,
        )

    def get_script_state(self, session_id: int) -> dict[str, Any]:
        """Return spoiler-safe interpreter state for a scripted-story session."""
        session = self.get_session(session_id)
        if session.mode != MODE_SCRIPTED_STORY:
            raise VNPlayConflictError("scripted_story_required")
        version = self._script_version_for_session(session)
        return _script_public_state_payload(
            session_id=session.id,
            scene_version=session.scene_version,
            position=session.script_position,
            program=version.get("program"),
        )

    def get_script_debug_state(self, session_id: int) -> dict[str, Any]:
        """Return owner-visible pinned script metadata and program state."""
        session = self.get_session(session_id)
        if session.mode != MODE_SCRIPTED_STORY:
            raise VNPlayConflictError("scripted_story_required")
        version = self._script_version_for_session(session)
        state = _script_state_payload(
            session_id=session.id,
            scene_version=session.scene_version,
            position=session.script_position,
        )
        state.update(
            {
                "script_id": session.script_id,
                "script_version_id": session.script_version_id,
                "script_manifest_snapshot_id": session.script_manifest_snapshot_id,
                "script_policy_snapshot_id": session.script_policy_snapshot_id,
                "script_generation_profile_snapshot_id": (
                    session.script_generation_profile_snapshot_id
                ),
                "version_number": version.get("version_number"),
                "version_label": version.get("label"),
                "program": version.get("program") if isinstance(version.get("program"), Mapping) else {},
                "script_defaults": (
                    version.get("script_defaults")
                    if isinstance(version.get("script_defaults"), Mapping)
                    else {}
                ),
                "validation": version.get("validation"),
            }
        )
        return state

    def _run_script_action(
        self,
        *,
        session_id: int,
        action_type: str,
        client_scene_version: int,
        idempotency_key: str,
        choice_id: str | None,
    ) -> dict[str, Any]:
        if not idempotency_key:
            raise VNPlayTurnError("idempotency_key_required")
        session = self.get_session(session_id)
        if session.mode != MODE_SCRIPTED_STORY:
            raise VNPlayConflictError("scripted_story_required")
        request_payload_hash = _payload_hash(
            {
                "action_type": action_type,
                "choice_id": choice_id,
                "client_scene_version": client_scene_version,
            }
        )
        action = self._create_or_replay_session_action(
            session_id=session_id,
            action_type=action_type,
            idempotency_key=idempotency_key,
            request_payload_hash=request_payload_hash,
        )
        if action["status"] == SESSION_ACTION_STATUS_COMPLETED:
            return self._completed_session_action_response(action)
        self._raise_for_terminal_session_action(action)

        lock_acquired = False
        try:
            if choice_id is None:
                if _list_of_dicts(session.script_position.get("waiting_choices")):
                    self._abandon_session_action(
                        session_id=session_id,
                        action_id=int(action["id"]),
                        error_code=ERROR_SCRIPT_ADVANCE_BLOCKED,
                    )
                    raise VNPlayConflictError(ERROR_SCRIPT_ADVANCE_BLOCKED)
                if session.script_position.get("ended"):
                    self._abandon_session_action(
                        session_id=session_id,
                        action_id=int(action["id"]),
                        error_code=ERROR_SCRIPT_ENDED,
                    )
                    raise VNPlayConflictError(ERROR_SCRIPT_ENDED)
            self._validate_restore_can_start(
                session_id=session_id,
                action_id=int(action["id"]),
                expected_scene_version=client_scene_version,
            )
            lock_acquired = True
            version = self._script_version_for_session(session)
            execution = _execute_script_program(
                version.get("program"),
                session.script_position,
                choice_id=choice_id,
                seed=session.seed,
            )
            next_scene_version = client_scene_version + 1
            events = self._append_script_events(
                session_id=session_id,
                action_id=int(action["id"]),
                execution=execution,
                scene_version=next_scene_version,
                selected_choice_id=choice_id,
            )
            state = derive_scene_state(self.repo.list_events(session_id))
            last_event_id = int(events[-1]["id"]) if events else None
            self._persist_scene_state(
                session_id=session_id,
                state=state,
                last_event_id=last_event_id,
            )
            self.repo.update_session(
                session_id,
                {
                    "scene_version": next_scene_version,
                    "script_position": execution["position"],
                },
                owner_user_id=self.owner_user_id,
            )
            response = self._script_action_response(
                session_id=session_id,
                scene_version=next_scene_version,
                position=execution["position"],
                events=events,
                replayed=False,
            )
            self.repo.mark_session_action_terminal(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                action_id=int(action["id"]),
                status=SESSION_ACTION_STATUS_COMPLETED,
                response_payload=response,
            )
        except VNPlayTurnError as exc:
            self._abandon_session_action(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=str(exc) or ERROR_INTERNAL_ERROR,
            )
            raise
        except VNPlayConflictError:
            raise
        except VNPlayNotFoundError as exc:
            if lock_acquired:
                self._mark_session_action_failed(
                    session_id=session_id,
                    action_id=int(action["id"]),
                    error_code=str(exc) or ERROR_INTERNAL_ERROR,
                    error_type=exc.__class__.__name__,
                )
            raise
        except Exception as exc:
            self._mark_session_action_failed(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=ERROR_INTERNAL_ERROR,
                error_type=exc.__class__.__name__,
            )
            raise
        return response

    def _run_script_regenerate_action(
        self,
        *,
        session_id: int,
        client_scene_version: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if not idempotency_key:
            raise VNPlayTurnError("idempotency_key_required")
        session = self.get_session(session_id)
        if session.mode != MODE_SCRIPTED_STORY:
            raise VNPlayConflictError("scripted_story_required")
        generation = session.script_position.get("last_generation")
        if not isinstance(generation, Mapping):
            raise VNPlayConflictError("script_regenerate_unavailable")
        request_payload_hash = _payload_hash(
            {
                "action_type": "script_regenerate",
                "client_scene_version": client_scene_version,
                "generation_id": generation.get("id"),
            }
        )
        action = self._create_or_replay_session_action(
            session_id=session_id,
            action_type="script_regenerate",
            idempotency_key=idempotency_key,
            request_payload_hash=request_payload_hash,
        )
        if action["status"] == SESSION_ACTION_STATUS_COMPLETED:
            return self._completed_session_action_response(action)
        self._raise_for_terminal_session_action(action)

        lock_acquired = False
        try:
            self._validate_restore_can_start(
                session_id=session_id,
                action_id=int(action["id"]),
                expected_scene_version=client_scene_version,
            )
            lock_acquired = True
            result = _script_regeneration_result(generation, idempotency_key=idempotency_key)
            next_position = dict(session.script_position)
            next_position["last_generation"] = result
            execution = _script_execution_payload(
                position=next_position,
                variables=dict(next_position.get("variables") or {}),
                narrative_lines=[str(result.get("narrative_text") or "")],
                dialogue=_list_of_dicts(result.get("dialogue")),
                visible_choices=[],
                selected_choice=None,
                random_results=[],
                generation_results=[result],
            )
            next_scene_version = client_scene_version + 1
            events = self._append_script_events(
                session_id=session_id,
                action_id=int(action["id"]),
                execution=execution,
                scene_version=next_scene_version,
                selected_choice_id=None,
            )
            state = derive_scene_state(self.repo.list_events(session_id))
            last_event_id = int(events[-1]["id"]) if events else None
            self._persist_scene_state(
                session_id=session_id,
                state=state,
                last_event_id=last_event_id,
            )
            self.repo.update_session(
                session_id,
                {
                    "scene_version": next_scene_version,
                    "script_position": next_position,
                },
                owner_user_id=self.owner_user_id,
            )
            response = self._script_action_response(
                session_id=session_id,
                scene_version=next_scene_version,
                position=next_position,
                events=events,
                replayed=False,
            )
            self.repo.mark_session_action_terminal(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                action_id=int(action["id"]),
                status=SESSION_ACTION_STATUS_COMPLETED,
                response_payload=response,
            )
        except VNPlayConflictError:
            raise
        except Exception as exc:
            if lock_acquired:
                self._mark_session_action_failed(
                    session_id=session_id,
                    action_id=int(action["id"]),
                    error_code=ERROR_INTERNAL_ERROR,
                    error_type=exc.__class__.__name__,
                )
            raise
        return response

    def _script_version_for_session(self, session: VNPlaySession) -> dict[str, Any]:
        if session.script_id is None or session.script_version_id is None:
            raise VNPlayNotFoundError("script_version_not_found")
        version = VNScriptsRepository.initialized(self.repo.db).get_version(
            session.script_id,
            session.script_version_id,
            owner_user_id=self.owner_user_id,
        )
        if version is None:
            raise VNPlayNotFoundError("script_version_not_found")
        return version

    def _append_script_events(
        self,
        *,
        session_id: int,
        action_id: int,
        execution: Mapping[str, Any],
        scene_version: int,
        selected_choice_id: str | None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if selected_choice_id is not None:
            selected_choice = execution.get("selected_choice")
            events.append(
                self.repo.append_event(
                    session_id=session_id,
                    owner_user_id=self.owner_user_id,
                    event_type=EVENT_CHOICE_SELECTED,
                    event_payload={
                        "session_action_id": action_id,
                        "choice_id": selected_choice_id,
                        "choice": (
                            _script_public_choice(selected_choice)
                            if isinstance(selected_choice, Mapping)
                            else {}
                        ),
                        "scene_version": scene_version,
                    },
                    source="user",
                )
            )
        narrative_text = str(execution.get("narrative_text") or "")
        dialogue = _list_of_dicts(execution.get("dialogue"))
        generation_results = _list_of_dicts(execution.get("generation_results"))
        if narrative_text or dialogue or generation_results:
            events.append(
                self.repo.append_event(
                    session_id=session_id,
                    owner_user_id=self.owner_user_id,
                    event_type=EVENT_MODEL_TURN,
                    event_payload={
                        "session_action_id": action_id,
                        "narrative_text": narrative_text,
                        "dialogue": dialogue,
                        "scripted": True,
                        "generation_results": generation_results,
                        "scene_version": scene_version,
                    },
                    source="runtime",
                )
            )
        choices = _script_public_choices(execution.get("visible_choices"))
        random_results = _list_of_dicts(execution.get("random_results"))
        if choices:
            events.append(
                self.repo.append_event(
                    session_id=session_id,
                    owner_user_id=self.owner_user_id,
                    event_type=EVENT_CHOICE_PRESENTED,
                    event_payload={
                        "session_action_id": action_id,
                        "choices": choices,
                        "scene_version": scene_version,
                    },
                    source="runtime",
                )
            )
        events.append(
            self.repo.append_event(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                event_type=EVENT_SCENE_STATE_CHANGED,
                event_payload={
                    "session_action_id": action_id,
                    "visible_choices": choices,
                    "random_results": random_results,
                    "scene_version": scene_version,
                },
                source="runtime",
            )
        )
        return events

    def _script_action_response(
        self,
        *,
        session_id: int,
        scene_version: int,
        position: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
        replayed: bool,
    ) -> dict[str, Any]:
        session = self.get_session(session_id)
        version = self._script_version_for_session(session)
        current_scene = self.get_enriched_scene_state(session_id)
        return {
            "status": "completed",
            "replayed": replayed,
            "scene_version": scene_version,
            "session": _public_session_payload(session),
            "current_scene": current_scene,
            "script_state": _script_public_state_payload(
                session_id=session_id,
                scene_version=scene_version,
                position=position,
                program=version.get("program"),
            ),
            "events": [dict(event) for event in events],
            "warnings": [],
        }

    def list_branches(self, session_id: int) -> list[dict[str, Any]]:
        self.get_session(session_id)
        return self.repo.list_branches(
            session_id,
            owner_user_id=self.owner_user_id,
        )

    def _create_or_replay_session_action(
        self,
        *,
        session_id: int,
        action_type: str,
        idempotency_key: str,
        request_payload_hash: str,
    ) -> dict[str, Any]:
        try:
            return self.repo.create_session_action(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                action_type=action_type,
                idempotency_key=idempotency_key,
                request_payload_hash=request_payload_hash,
                status=SESSION_ACTION_STATUS_PENDING,
            )
        except ValueError as exc:
            if str(exc) == ERROR_IDEMPOTENCY_KEY_CONFLICT:
                raise VNPlayConflictError(ERROR_IDEMPOTENCY_KEY_CONFLICT) from exc
            raise

    def _completed_session_action_response(
        self,
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        response_payload = action.get("response_payload")
        if not isinstance(response_payload, Mapping):
            raise VNPlayConflictError(ERROR_RESTORE_ACTION_IN_PROGRESS)
        replayed = dict(response_payload)
        replayed["replayed"] = True
        return replayed

    def _raise_for_terminal_session_action(self, action: Mapping[str, Any]) -> None:
        status = str(action.get("status"))
        if status not in {
            SESSION_ACTION_STATUS_FAILED,
            SESSION_ACTION_STATUS_ABANDONED,
        }:
            return
        error = action.get("error")
        error_code = (
            str(error["code"])
            if isinstance(error, Mapping) and error.get("code")
            else ERROR_RESTORE_ACTION_IN_PROGRESS
        )
        raise VNPlayConflictError(error_code)

    def _validate_restore_can_start(
        self,
        *,
        session_id: int,
        action_id: int,
        expected_scene_version: int,
    ) -> None:
        session = self.get_session(session_id)
        if session.scene_version != expected_scene_version:
            self._abandon_session_action(
                session_id=session_id,
                action_id=action_id,
                error_code=ERROR_STALE_SCENE_VERSION,
            )
            raise VNPlayConflictError(ERROR_STALE_SCENE_VERSION)
        if session.active_turn_request_id is not None:
            self._abandon_session_action(
                session_id=session_id,
                action_id=action_id,
                error_code=ERROR_TURN_IN_PROGRESS,
            )
            raise VNPlayConflictError(ERROR_TURN_IN_PROGRESS)
        if session.active_session_action_id == action_id:
            raise VNPlayConflictError(ERROR_RESTORE_ACTION_IN_PROGRESS)
        if (
            session.active_session_action_id is not None
            and session.active_session_action_id != action_id
        ):
            self._abandon_session_action(
                session_id=session_id,
                action_id=action_id,
                error_code=ERROR_RESTORE_ACTION_IN_PROGRESS,
            )
            raise VNPlayConflictError(ERROR_RESTORE_ACTION_IN_PROGRESS)

        if not self.repo.try_acquire_session_action_lock(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            action_id=action_id,
            expected_scene_version=expected_scene_version,
        ):
            current = self.get_session(session_id)
            if current.active_turn_request_id is not None:
                error_code = ERROR_TURN_IN_PROGRESS
            elif current.active_session_action_id is not None:
                error_code = ERROR_RESTORE_ACTION_IN_PROGRESS
            else:
                error_code = ERROR_STALE_SCENE_VERSION
            self._abandon_session_action(
                session_id=session_id,
                action_id=action_id,
                error_code=error_code,
            )
            raise VNPlayConflictError(error_code)

    def _branch_restore_target_event_id(
        self,
        *,
        session_id: int,
        branch_id: int,
        target: str,
    ) -> int:
        navigation = self.get_branch_navigation(session_id)
        branch_node = next(
            (
                branch
                for branch in navigation["branches"]
                if int(branch["branch_id"]) == branch_id
            ),
            None,
        )
        if branch_node is None:
            raise VNPlayNotFoundError(ERROR_BRANCH_NOT_FOUND)
        if target == BRANCH_RESTORE_TARGET_LATEST:
            target_event_id = _optional_int(
                branch_node.get("event_range", {}).get("latest_event_id")
            )
        else:
            target_event_id = _optional_int(branch_node.get("parent_event_id"))
            parent_event = (
                self.repo.get_event(target_event_id)
                if target_event_id is not None
                else None
            )
            if parent_event is None or parent_event.get("event_type") != EVENT_CHOICE_PRESENTED:
                target_event_id = None
        if target_event_id is None:
            raise VNPlayConflictError(ERROR_BRANCH_RESTORE_TARGET_UNAVAILABLE)
        return target_event_id

    def _scene_state_through_event(
        self,
        *,
        session_id: int,
        target_event_id: int,
    ) -> SceneState:
        bounded_events: list[dict[str, Any]] = []
        for event in self.repo.list_events(session_id):
            event_id = _event_int(event, "id")
            if event_id is not None and event_id <= target_event_id:
                bounded_events.append(event)
        if not any(_event_int(event, "id") == target_event_id for event in bounded_events):
            raise VNPlayConflictError(ERROR_BRANCH_RESTORE_TARGET_UNAVAILABLE)
        return derive_scene_state(bounded_events)

    def _restore_response_payload(
        self,
        payload: Mapping[str, Any],
        *,
        branch_id: int | None = None,
        checkpoint_id: int | None = None,
        save_slot_id: int | None = None,
        target: str | None = None,
        target_event_id: int | None,
        replayed: bool,
    ) -> dict[str, Any]:
        session_payload = _public_session_payload(VNPlaySession.from_row(payload["session"]))
        scene_state = dict(payload["scene_state"])
        navigation = build_branch_navigation(
            session=session_payload,
            branches=_list_of_dicts(payload.get("branches")),
            events=_list_of_dicts(payload.get("events")),
            scene_state=scene_state,
        )
        restore_event = dict(payload["restore_event"])
        response: dict[str, Any] = {
            "status": SESSION_ACTION_STATUS_COMPLETED,
            "replayed": replayed,
            "restore_event_id": int(restore_event["id"]),
            "target_event_id": target_event_id,
            "scene_version": int(scene_state["scene_version"]),
            "session": session_payload,
            "current_scene": scene_state,
            "branch_navigation": navigation,
        }
        if branch_id is not None:
            response["branch_id"] = branch_id
        if checkpoint_id is not None:
            response["checkpoint_id"] = checkpoint_id
        if save_slot_id is not None:
            response["save_slot_id"] = save_slot_id
        if target is not None:
            response["target"] = target
        return response

    def _abandon_session_action(
        self,
        *,
        session_id: int,
        action_id: int,
        error_code: str,
    ) -> None:
        self.repo.mark_session_action_terminal(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            action_id=action_id,
            status=SESSION_ACTION_STATUS_ABANDONED,
            error={"code": error_code},
        )

    def _mark_session_action_failed(
        self,
        *,
        session_id: int,
        action_id: int,
        error_code: str,
        error_type: str | None = None,
    ) -> None:
        error_payload = {"code": error_code}
        if error_type:
            error_payload["error_type"] = error_type
        self.repo.mark_session_action_terminal(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            action_id=action_id,
            status=SESSION_ACTION_STATUS_FAILED,
            error=error_payload,
        )

    def _response_for_existing_turn(
        self,
        turn_request: Mapping[str, Any],
        request_payload_hash: str,
    ) -> VNPlayTurnResponse:
        if turn_request["request_payload_hash"] != request_payload_hash:
            raise VNPlayConflictError(ERROR_IDEMPOTENCY_KEY_CONFLICT)

        response_payload = turn_request.get("response_payload")
        if (
            turn_request.get("status") == TURN_STATUS_COMPLETED
            and isinstance(response_payload, Mapping)
        ):
            return VNPlayTurnResponse.from_payload(response_payload)

        status = str(turn_request.get("status") or "")
        error = turn_request.get("error")
        error_code = None
        if isinstance(error, Mapping):
            if error.get("code") is not None:
                error_code = str(error["code"])
        if status in {TURN_STATUS_MODEL_FAILED, TURN_STATUS_PARSE_FAILED}:
            raise VNPlayTurnError(error_code or status)
        if status == TURN_STATUS_ABANDONED and error_code:
            if error_code in _CONFLICT_REPLAY_ERROR_CODES:
                raise VNPlayConflictError(error_code)
            raise VNPlayTurnError(error_code)
        if status in {TURN_STATUS_PENDING, TURN_STATUS_MODEL_CALLING}:
            raise VNPlayConflictError(ERROR_TURN_IN_PROGRESS)
        if error_code is not None:
            if error_code in _CONFLICT_REPLAY_ERROR_CODES:
                raise VNPlayConflictError(error_code)
            raise VNPlayTurnError(error_code)
        raise VNPlayConflictError(ERROR_TURN_IN_PROGRESS)

    def _abandon_conflicting_turn(
        self,
        turn_request: Mapping[str, Any],
        session_id: int,
    ) -> VNPlayTurnResponse:
        current = self.get_session(session_id)
        if current.active_turn_request_id is not None:
            status = ERROR_TURN_IN_PROGRESS
        elif current.active_session_action_id is not None:
            status = ERROR_RESTORE_ACTION_IN_PROGRESS
        else:
            status = ERROR_STALE_SCENE_VERSION
        self.repo.update_turn_request(
            int(turn_request["id"]),
            {
                "status": TURN_STATUS_ABANDONED,
                "error": {"code": status},
            },
            owner_user_id=self.owner_user_id,
        )
        raise VNPlayConflictError(status)

    def _append_accepted_turn_events(
        self,
        *,
        session_id: int,
        turn_request_id: int,
        input_payload: Mapping[str, Any],
        client_scene_version: int,
    ) -> list[dict[str, Any]]:
        turn_started = self.repo.append_event(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            event_type=EVENT_TURN_STARTED,
            event_payload={
                "turn_request_id": turn_request_id,
                "scene_version": client_scene_version,
            },
            source="runtime",
        )
        user_turn = self.repo.append_event(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            event_type=EVENT_USER_TURN,
            event_payload={
                "turn_request_id": turn_request_id,
                "input": dict(input_payload),
                "scene_version": client_scene_version,
            },
            source="user",
        )
        self.repo.update_turn_request(
            turn_request_id,
            {
                "status": TURN_STATUS_MODEL_CALLING,
                "turn_started_event_id": turn_started["id"],
                "input_event_id": user_turn["id"],
            },
            owner_user_id=self.owner_user_id,
        )
        return [turn_started, user_turn]

    def _complete_turn(
        self,
        *,
        session: VNPlaySession,
        session_id: int,
        turn_request_id: int,
        prior_events: Sequence[Mapping[str, Any]],
        result: TurnResult,
        next_scene_version: int,
    ) -> VNPlayTurnResponse:
        active_branch_id = _story_active_branch_id(
            session,
            self.repo.get_scene_state(
                session_id,
                owner_user_id=self.owner_user_id,
            ),
        )
        model_turn = self.repo.append_event(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            event_type=EVENT_MODEL_TURN,
            event_payload={
                "turn_request_id": turn_request_id,
                "narrative_text": result.narrative_text,
                "dialogue": result.dialogue,
                "visual_directives": result.visual_directives,
                "warnings": result.warnings,
                "scene_version": next_scene_version,
            },
            source="model",
            branch_node_id=active_branch_id,
        )
        new_events: list[dict[str, Any]] = [dict(event) for event in prior_events]
        new_events.append(model_turn)

        visual_events, visual_scene_updates, visual_warnings = self._apply_visual_directives(
            session=session,
            session_id=session_id,
            turn_request_id=turn_request_id,
            directives=result.visual_directives,
            scene_version=next_scene_version,
            branch_node_id=active_branch_id,
        )
        new_events.extend(visual_events)

        if result.choices:
            choice_presented = self.repo.append_event(
                session_id=session_id,
                owner_user_id=self.owner_user_id,
                event_type=EVENT_CHOICE_PRESENTED,
                event_payload={
                    "turn_request_id": turn_request_id,
                    "choices": result.choices,
                    "scene_version": next_scene_version,
                },
                source="runtime",
                branch_node_id=active_branch_id,
            )
            new_events.append(choice_presented)

        scene_payload = dict(result.scene_updates)
        scene_payload.update(visual_scene_updates)
        scene_payload["scene_version"] = next_scene_version
        all_warnings = [*result.warnings, *visual_warnings]
        if result.warnings:
            scene_payload["warnings"] = result.warnings
        scene_state_changed = self.repo.append_event(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            event_type=EVENT_SCENE_STATE_CHANGED,
            event_payload=scene_payload,
            source="runtime",
            branch_node_id=active_branch_id,
        )
        turn_completed = self.repo.append_event(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            event_type=EVENT_TURN_COMPLETED,
            event_payload={
                "turn_request_id": turn_request_id,
                "scene_version": next_scene_version,
            },
            source="runtime",
            branch_node_id=active_branch_id,
        )
        new_events.extend([scene_state_changed, turn_completed])

        state = derive_scene_state(self.repo.list_events(session_id))
        self._persist_scene_state(
            session_id=session_id,
            state=state,
            last_event_id=int(turn_completed["id"]),
        )
        self.repo.update_session(
            session_id,
            {"active_turn_request_id": None, "scene_version": next_scene_version},
            owner_user_id=self.owner_user_id,
        )
        response = VNPlayTurnResponse(
            turn_request_id=turn_request_id,
            status=TURN_STATUS_COMPLETED,
            scene_version=next_scene_version,
            events=new_events,
            warnings=all_warnings,
        )
        self.repo.update_turn_request(
            turn_request_id,
            {
                "status": TURN_STATUS_COMPLETED,
                "turn_completed_event_id": turn_completed["id"],
                "response_payload": response.to_payload(),
            },
            owner_user_id=self.owner_user_id,
        )
        return response

    def _apply_visual_directives(
        self,
        *,
        session: VNPlaySession,
        session_id: int,
        turn_request_id: int,
        directives: Sequence[Mapping[str, Any]],
        scene_version: int,
        branch_node_id: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
        if not directives:
            return [], {}, []

        events: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        scene_updates: dict[str, Any] = {}
        sprite_items: list[dict[str, Any]] = []
        manifest: Mapping[str, Any] | None = None
        manifest_error: str | None = None
        default_rejection_reason = "manifest_unavailable"

        try:
            manifest = self._build_pack_manifest(session.vn_asset_pack_id)
        except Exception as exc:
            logger.exception(
                "Failed to build VN asset manifest for VN Play visual directives: "
                "session_id={}, pack_id={}",
                session_id,
                session.vn_asset_pack_id,
            )
            manifest_error = exc.__class__.__name__

        if manifest is not None:
            try:
                resolutions = resolve_scene_directives(
                    manifest,
                    directives,
                    seed=session.seed or f"session-{session_id}",
                )
            except Exception as exc:
                logger.exception(
                    "Failed to resolve VN Play visual directives: session_id={}, pack_id={}",
                    session_id,
                    session.vn_asset_pack_id,
                )
                manifest_error = exc.__class__.__name__
                default_rejection_reason = "resolver_error"
                resolutions = []
        else:
            resolutions = []

        for index, directive in enumerate(directives):
            directive_payload = dict(directive)
            events.append(
                self.repo.append_event(
                    session_id=session_id,
                    owner_user_id=self.owner_user_id,
                    event_type=EVENT_VISUAL_DIRECTIVE_REQUESTED,
                    event_payload={
                        "turn_request_id": turn_request_id,
                        "directive": directive_payload,
                        "scene_version": scene_version,
                    },
                    source="runtime",
                    branch_node_id=branch_node_id,
                )
            )

            resolution = resolutions[index] if index < len(resolutions) else None
            if resolution is not None and resolution.applied and resolution.item is not None:
                item = dict(resolution.item)
                asset_type = _visual_asset_type(directive_payload, item)
                events.append(
                    self.repo.append_event(
                        session_id=session_id,
                        owner_user_id=self.owner_user_id,
                        event_type=EVENT_VISUAL_DIRECTIVE_APPLIED,
                        event_payload={
                            "turn_request_id": turn_request_id,
                            "asset_type": asset_type,
                            "directive": directive_payload,
                            "item": item,
                            "scene_version": scene_version,
                        },
                        source="runtime",
                        branch_node_id=branch_node_id,
                    )
                )
                _merge_visual_item_scene_update(
                    scene_updates,
                    sprite_items,
                    asset_type=asset_type,
                    item=item,
                )
                continue

            reason = (
                resolution.reason
                if resolution is not None and resolution.reason
                else default_rejection_reason
            )
            warning = _visual_directive_warning(
                directive_payload,
                reason=reason,
                scene_version=scene_version,
                error_type=manifest_error,
            )
            warnings.append(warning)
            events.append(
                self.repo.append_event(
                    session_id=session_id,
                    owner_user_id=self.owner_user_id,
                    event_type=EVENT_VISUAL_DIRECTIVE_REJECTED,
                    event_payload={
                        "turn_request_id": turn_request_id,
                        "directive": directive_payload,
                        **warning,
                    },
                    source="runtime",
                    branch_node_id=branch_node_id,
                )
            )

        if sprite_items:
            scene_updates["active_sprite_items"] = sprite_items
        return events, scene_updates, warnings

    def _build_pack_manifest(self, pack_id: int) -> dict[str, Any]:
        manifest = self._manifest_cache.get(pack_id)
        if manifest is None:
            manifest_response = VNAssetPackService(
                self.repo.db,
                owner_user_id=self.owner_user_id,
            ).build_manifest(pack_id)
            manifest = manifest_response.model_dump()
            self._manifest_cache[pack_id] = manifest
        return manifest

    def _mark_turn_failed(
        self,
        *,
        session_id: int,
        turn_request_id: int,
        error_code: str,
        error_message: str,
        client_scene_version: int,
        event_type: str = EVENT_TURN_FAILED,
    ) -> None:
        failed_event = self.repo.append_event(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            event_type=event_type,
            event_payload={
                "turn_request_id": turn_request_id,
                "code": error_code,
                "message": error_message,
                "scene_version": client_scene_version,
            },
            source="runtime",
        )
        state = derive_scene_state(self.repo.list_events(session_id))
        self._persist_scene_state(
            session_id=session_id,
            state=state,
            last_event_id=int(failed_event["id"]),
        )
        self.repo.update_turn_request(
            turn_request_id,
            {
                "status": error_code,
                "turn_completed_event_id": failed_event["id"],
                "error": {
                    "code": error_code,
                    "message": error_message,
                },
            },
            owner_user_id=self.owner_user_id,
        )
        self.repo.update_session(
            session_id,
            {"active_turn_request_id": None},
            owner_user_id=self.owner_user_id,
        )

    def _persist_scene_state(
        self,
        *,
        session_id: int,
        state: SceneState,
        last_event_id: int | None,
    ) -> None:
        self.repo.set_scene_state(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            last_event_id=last_event_id,
            current_background_item_id=state.current_background_item_id,
            current_depth_item_id=state.current_depth_item_id,
            active_sprite_items=state.active_sprite_items,
            location_key=state.location_key,
            mood=state.mood,
            time_of_day=state.time_of_day,
            weather=state.weather,
            active_branch_node_id=state.active_branch_node_id,
            visible_choices=state.visible_choices,
            transcript_cursor=state.transcript_cursor,
            scene_version=state.scene_version,
            warnings=state.warnings,
        )


def _validate_turn_input_for_mode(
    session: VNPlaySession,
    input_payload: Mapping[str, Any],
) -> None:
    if session.mode == MODE_FREEFORM and "choice_id" in input_payload:
        raise VNPlayTurnError(ERROR_CHOICE_NOT_ALLOWED)
    if session.mode == MODE_STORY and "input_text" in input_payload:
        raise VNPlayTurnError(ERROR_CHOICE_NOT_ALLOWED)
    if session.mode == MODE_SCRIPTED_STORY:
        raise VNPlayTurnError(ERROR_SCRIPTED_STORY_TURN_NOT_ALLOWED)


def _initial_script_position(program: Any) -> dict[str, Any]:
    """Build the deterministic initial interpreter position for a script."""
    if not isinstance(program, Mapping):
        return {"label": "start", "index": 0, "ended": False, "variables": {}}
    entry_label = str(program.get("entry_label") or "start")
    return {
        "label": entry_label,
        "index": 0,
        "ended": False,
        "variables": _initial_script_variables(program.get("variables")),
    }


def _initial_script_variables(raw_variables: Any) -> dict[str, Any]:
    if not isinstance(raw_variables, Mapping):
        return {}
    variables: dict[str, Any] = {}
    for name, definition in raw_variables.items():
        if isinstance(name, str) and isinstance(definition, Mapping):
            variables[name] = definition.get("default")
    return variables


def _execute_script_program(
    program: Any,
    position: Mapping[str, Any],
    *,
    choice_id: str | None,
    seed: str | None,
) -> dict[str, Any]:
    """Run a deterministic script segment until the next visible boundary."""
    if not isinstance(program, Mapping):
        raise VNPlayTurnError("script_program_missing")
    labels = program.get("labels")
    if not isinstance(labels, Mapping):
        raise VNPlayTurnError("script_labels_missing")

    current_position = dict(position or _initial_script_position(program))
    variables = dict(current_position.get("variables") or {})
    selected_choice: dict[str, Any] | None = None
    if choice_id is not None:
        selected_choice = _script_selected_choice(current_position, choice_id)
        current_position = {
            "label": selected_choice["target"],
            "index": 0,
            "ended": False,
            "variables": variables,
        }

    label = str(current_position.get("label") or program.get("entry_label") or "start")
    index = int(current_position.get("index") or 0)
    narrative_lines: list[str] = []
    dialogue: list[dict[str, Any]] = []
    visible_choices: list[dict[str, Any]] = []
    random_results: list[dict[str, Any]] = []
    generation_results: list[dict[str, Any]] = []
    ended = False

    for _ in range(MAX_SCRIPT_EXECUTION_STEPS):
        ops = labels.get(label)
        if not isinstance(ops, list):
            raise VNPlayTurnError("script_label_missing")
        if index >= len(ops):
            ended = True
            break
        opcode = ops[index]
        index += 1
        if not isinstance(opcode, Mapping):
            continue
        if not _script_condition_matches(opcode.get("if"), variables):
            continue

        op = str(opcode.get("op") or "")
        if op == "narrate":
            text = str(opcode.get("text") or "")
            if text:
                narrative_lines.append(text)
                dialogue.append({"speaker": "Narrator", "text": text})
        elif op == "say":
            text = str(opcode.get("text") or "")
            speaker = str(opcode.get("speaker") or opcode.get("character") or "")
            if text:
                dialogue.append({"speaker": speaker or "Narrator", "text": text})
        elif op == "set":
            var_name = str(opcode.get("var") or "")
            if var_name:
                variables[var_name] = opcode.get("value")
        elif op == "increment":
            var_name = str(opcode.get("var") or "")
            amount = opcode.get("amount", 1)
            current = variables.get(var_name, 0)
            if isinstance(current, (int, float)) and isinstance(amount, (int, float)):
                variables[var_name] = current + amount
        elif op == "random":
            result = _script_random_result(
                opcode,
                seed=seed,
                label=label,
                index=index - 1,
            )
            var_name = result.get("var")
            if isinstance(var_name, str) and var_name:
                variables[var_name] = result.get("value")
            random_results.append(result)
        elif op == "generate":
            result = _script_generation_result(
                opcode,
                seed=seed,
                label=label,
                index=index - 1,
            )
            generation_results.append(result)
            text = str(result.get("narrative_text") or "")
            if text:
                narrative_lines.append(text)
            dialogue.extend(_list_of_dicts(result.get("dialogue")))
            current_position = {
                "label": label,
                "index": index,
                "ended": False,
                "variables": variables,
                "last_generation": result,
            }
            return _script_execution_payload(
                position=current_position,
                variables=variables,
                narrative_lines=narrative_lines,
                dialogue=dialogue,
                visible_choices=[],
                selected_choice=selected_choice,
                random_results=random_results,
                generation_results=generation_results,
            )
        elif op == "jump":
            label = str(opcode.get("target") or "")
            index = 0
        elif op == "choice":
            visible_choices = _script_visible_choices(opcode)
            current_position = {
                "label": label,
                "index": index - 1,
                "ended": False,
                "variables": variables,
                "waiting_choice_id": str(opcode.get("id") or ""),
                "waiting_choices": visible_choices,
            }
            return _script_execution_payload(
                position=current_position,
                variables=variables,
                narrative_lines=narrative_lines,
                dialogue=dialogue,
                visible_choices=visible_choices,
                selected_choice=selected_choice,
                random_results=random_results,
                generation_results=generation_results,
            )
        elif op == "end":
            ended = True
            break
        elif op == "return":
            ended = True
            break

    current_position = {
        "label": label,
        "index": index,
        "ended": ended,
        "variables": variables,
    }
    return _script_execution_payload(
        position=current_position,
        variables=variables,
        narrative_lines=narrative_lines,
        dialogue=dialogue,
        visible_choices=[],
        selected_choice=selected_choice,
        random_results=random_results,
        generation_results=generation_results,
    )


def _script_selected_choice(position: Mapping[str, Any], choice_id: str) -> dict[str, Any]:
    choices = _list_of_dicts(position.get("waiting_choices"))
    for choice in choices:
        if str(choice.get("id")) == choice_id and choice.get("target"):
            return choice
    raise VNPlayTurnError(ERROR_INVALID_CHOICE_ID)


def _script_visible_choices(opcode: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices = _list_of_dicts(opcode.get("choices"))
    return [
        {
            "id": str(choice.get("id") or ""),
            "text": str(choice.get("text") or choice.get("id") or ""),
            "target": str(choice.get("target") or ""),
        }
        for choice in choices
        if choice.get("id") and choice.get("target")
    ]


def _script_execution_payload(
    *,
    position: Mapping[str, Any],
    variables: Mapping[str, Any],
    narrative_lines: Sequence[str],
    dialogue: Sequence[Mapping[str, Any]],
    visible_choices: Sequence[Mapping[str, Any]],
    selected_choice: Mapping[str, Any] | None,
    random_results: Sequence[Mapping[str, Any]],
    generation_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "position": dict(position),
        "variables": dict(variables),
        "narrative_text": "\n".join(narrative_lines),
        "dialogue": [dict(item) for item in dialogue],
        "visible_choices": [dict(choice) for choice in visible_choices],
        "random_results": [dict(result) for result in random_results],
        "generation_results": [dict(result) for result in generation_results],
    }
    if selected_choice is not None:
        payload["selected_choice"] = dict(selected_choice)
    return payload


def _script_random_result(
    opcode: Mapping[str, Any],
    *,
    seed: str | None,
    label: str,
    index: int,
) -> dict[str, Any]:
    random_id = str(opcode.get("id") or f"{label}:{index}")
    var_name = str(opcode.get("var") or "")
    digest = int(
        _payload_hash(
            {
                "seed": seed or "",
                "id": random_id,
                "label": label,
                "index": index,
            }
        ),
        16,
    )
    choices = opcode.get("choices")
    if (
        isinstance(choices, Sequence)
        and not isinstance(choices, (str, bytes))
        and len(choices) > 0
    ):
        value = choices[digest % len(choices)]
        result_type = "choice"
    else:
        minimum = opcode.get("min", 0)
        maximum = opcode.get("max", 1)
        if not isinstance(minimum, int) or isinstance(minimum, bool):
            minimum = 0
        if not isinstance(maximum, int) or isinstance(maximum, bool):
            maximum = 1
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        value = minimum + (digest % (maximum - minimum + 1))
        result_type = "integer"

    return {
        "id": random_id,
        "var": var_name,
        "type": result_type,
        "value": value,
    }


def _script_generation_result(
    opcode: Mapping[str, Any],
    *,
    seed: str | None,
    label: str,
    index: int,
) -> dict[str, Any]:
    generation_id = str(opcode.get("id") or f"{label}:{index}")
    prompt = str(opcode.get("prompt") or opcode.get("text") or generation_id)
    narrative_text = str(opcode.get("narrative_text") or opcode.get("text") or "")
    if not narrative_text:
        raise VNPlayTurnError("script_generation_unavailable")
    speaker = str(opcode.get("speaker") or "Narrator")
    regeneration_text = str(
        opcode.get("regeneration_text")
        or opcode.get("regenerate_text")
        or ""
    )
    return {
        "id": generation_id,
        "label": label,
        "index": index,
        "prompt_hash": _payload_hash({"seed": seed or "", "prompt": prompt})[:16],
        "source": SCRIPT_GENERATION_SOURCE_LITERAL,
        "model_invoked": False,
        "narrative_text": narrative_text,
        "dialogue": [{"speaker": speaker, "text": narrative_text}],
        "regeneration_text": regeneration_text or None,
        "regeneration_supported": bool(regeneration_text),
        "regenerated": False,
    }


def _script_regeneration_result(
    generation: Mapping[str, Any],
    *,
    idempotency_key: str,
) -> dict[str, Any]:
    regenerated_text = str(generation.get("regeneration_text") or "")
    if not regenerated_text:
        raise VNPlayConflictError("script_regenerate_unavailable")
    return {
        **dict(generation),
        "regenerated": True,
        "regeneration_key_hash": _payload_hash({"idempotency_key": idempotency_key})[:16],
        "narrative_text": regenerated_text,
        "dialogue": [{"speaker": "Narrator", "text": regenerated_text}],
    }


def _script_state_payload(
    *,
    session_id: int,
    scene_version: int,
    position: Mapping[str, Any],
) -> dict[str, Any]:
    variables = dict(position.get("variables") or {})
    waiting_choices = _list_of_dicts(position.get("waiting_choices"))
    waiting_choice = None
    if waiting_choices:
        waiting_choice = {
            "id": position.get("waiting_choice_id"),
            "choices": waiting_choices,
        }
    return {
        "session_id": session_id,
        "scene_version": scene_version,
        "position": dict(position),
        "variables": variables,
        "waiting_choice": waiting_choice,
        "ended": bool(position.get("ended")),
    }


def _script_public_state_payload(
    *,
    session_id: int,
    scene_version: int,
    position: Mapping[str, Any],
    program: Any,
) -> dict[str, Any]:
    waiting_choices = _script_public_choices(position.get("waiting_choices"))
    waiting_choice = None
    if waiting_choices:
        waiting_choice = {
            "id": position.get("waiting_choice_id"),
            "choices": waiting_choices,
        }
    return {
        "session_id": session_id,
        "scene_version": scene_version,
        "position": {"progress_token": _script_progress_token(position)},
        "variables": _script_public_variables(program, position.get("variables")),
        "waiting_choice": waiting_choice,
        "ended": bool(position.get("ended")),
    }


def _script_progress_token(position: Mapping[str, Any]) -> str:
    return _payload_hash(
        {
            "label": position.get("label"),
            "index": position.get("index"),
            "waiting_choice_id": position.get("waiting_choice_id"),
            "ended": bool(position.get("ended")),
        }
    )[:16]


def _script_public_variables(program: Any, raw_variables: Any) -> dict[str, Any]:
    if not isinstance(program, Mapping) or not isinstance(raw_variables, Mapping):
        return {}
    definitions = program.get("variables")
    if not isinstance(definitions, Mapping):
        return {}
    public_variables: dict[str, Any] = {}
    for name, value in raw_variables.items():
        if not isinstance(name, str):
            continue
        definition = definitions.get(name)
        if isinstance(definition, Mapping) and definition.get("public") is True:
            public_variables[name] = value
    return public_variables


def _script_public_choices(raw_choices: Any) -> list[dict[str, Any]]:
    return [
        _script_public_choice(choice)
        for choice in _list_of_dicts(raw_choices)
        if choice.get("id")
    ]


def _script_public_choice(choice: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": str(choice.get("id") or ""),
        "text": str(choice.get("text") or choice.get("id") or ""),
    }


def _script_condition_matches(condition: Any, variables: Mapping[str, Any]) -> bool:
    if condition is None:
        return True
    if not isinstance(condition, Mapping):
        return False
    if (
        "all" in condition
        and isinstance(condition["all"], Sequence)
        and not isinstance(condition["all"], (str, bytes))
    ):
        return all(_script_condition_matches(item, variables) for item in condition["all"])
    if (
        "any" in condition
        and isinstance(condition["any"], Sequence)
        and not isinstance(condition["any"], (str, bytes))
    ):
        return any(_script_condition_matches(item, variables) for item in condition["any"])
    if "not" in condition:
        return not _script_condition_matches(condition["not"], variables)
    value = variables.get(str(condition.get("var") or ""))
    expected = condition.get("value")
    operator = str(condition.get("op") or "eq")
    if operator == "eq":
        return value == expected
    if operator in {"ne", "neq"}:
        return value != expected
    if operator == "in" and isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return value in expected
    if operator == "not_in" and isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        return value not in expected
    if operator in {"lt", "lte", "gt", "gte"}:
        return _compare_script_values(value, expected, operator)
    return False


def _compare_script_values(value: Any, expected: Any, operator: str) -> bool:
    if not isinstance(value, (int, float)) or not isinstance(expected, (int, float)):
        return False
    if operator == "lt":
        return value < expected
    if operator == "lte":
        return value <= expected
    if operator == "gt":
        return value > expected
    if operator == "gte":
        return value >= expected
    return False


def _selected_visible_choice(
    state: Mapping[str, Any] | None,
    choice_id: str,
) -> dict[str, Any]:
    raw_choices = state.get("visible_choices") if state is not None else []
    choices = _list_of_dicts(raw_choices)
    for choice in choices:
        if str(choice.get("id")) == choice_id:
            return choice
    raise VNPlayTurnError(ERROR_INVALID_CHOICE_ID)


def _latest_restore_sequence(events: Sequence[Mapping[str, Any]]) -> int:
    latest = 0
    for event in events:
        if event.get("event_type") != EVENT_SESSION_RESTORED:
            continue
        sequence_number = _event_int(event, "sequence_number")
        if sequence_number is not None:
            latest = max(latest, sequence_number)
    return latest


def _parent_choice_event_id(
    events: Sequence[Mapping[str, Any]],
    scene_last_event_id: int | None,
    choice_id: str,
) -> int | None:
    bounded_events: list[Mapping[str, Any]] = []
    for event in events:
        event_id = _event_int(event, "id")
        if (
            scene_last_event_id is not None
            and event_id is not None
            and event_id > scene_last_event_id
        ):
            continue
        bounded_events.append(event)

    restore_sequence = _latest_restore_sequence(bounded_events)
    for event in reversed(bounded_events):
        sequence_number = _event_int(event, "sequence_number")
        if (
            restore_sequence
            and sequence_number is not None
            and sequence_number <= restore_sequence
        ):
            break
        if event.get("event_type") != EVENT_CHOICE_PRESENTED:
            continue
        payload = event.get("event_payload")
        if not isinstance(payload, Mapping):
            continue
        raw_choices = payload.get("choices", payload.get("visible_choices", []))
        choices = _list_of_dicts(raw_choices)
        if any(str(choice.get("id")) == choice_id for choice in choices):
            return _event_int(event, "id")
    return None


def _choice_text(choice: Mapping[str, Any]) -> str:
    for key in ("text", "label"):
        value = choice.get(key)
        if isinstance(value, str) and value:
            return value[:STORY_BRANCH_LABEL_MAX_LENGTH]
    return str(choice.get("id") or "")[:STORY_BRANCH_LABEL_MAX_LENGTH]


def _branch_path_for_choice(
    choice: Mapping[str, Any],
    *,
    scene_version: int,
    choice_presented_event_id: int | None,
    parent_branch_path: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    path = [dict(step) for step in parent_branch_path or []]
    path.append(
        {
            "schema_version": 1,
            "type": "choice",
            "choice_id": str(choice.get("id")),
            "choice_text": _choice_text(choice),
            "choice_presented_event_id": choice_presented_event_id,
            "scene_version": scene_version,
        }
    )
    return path


def _active_branch_path(
    branches: Sequence[Mapping[str, Any]],
    scene_state: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    active_branch_id = _optional_int(
        scene_state.get("active_branch_node_id") if scene_state is not None else None
    )
    if active_branch_id is None:
        return []
    for branch in branches:
        if _optional_int(branch.get("id")) == active_branch_id:
            return _list_of_dicts(branch.get("branch_path"))
    return []


def _story_active_branch_id(
    session: VNPlaySession,
    scene_state: Mapping[str, Any] | None,
) -> int | None:
    if session.mode != MODE_STORY or scene_state is None:
        return None
    return _optional_int(scene_state.get("active_branch_node_id"))


def _session_payload(session: VNPlaySession) -> dict[str, Any]:
    return {
        "id": session.id,
        "owner_user_id": session.owner_user_id,
        "mode": session.mode,
        "title": session.title,
        "status": session.status,
        "primary_character_id": session.primary_character_id,
        "scene_version": session.scene_version,
    }


def _public_session_payload(session: VNPlaySession) -> dict[str, Any]:
    payload = asdict(session)
    payload["deleted"] = False
    if session.mode == MODE_SCRIPTED_STORY:
        payload["script_position"] = {
            "progress_token": _script_progress_token(session.script_position)
        }
    return payload


def _event_int(event: Mapping[str, Any], key: str) -> int | None:
    value = event.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_input_payload(
    *,
    input_text: str | None,
    choice_id: str | None,
    custom_action: Mapping[str, Any] | None,
) -> dict[str, Any]:
    populated = [
        value is not None
        for value in (input_text, choice_id, custom_action)
    ]
    if sum(populated) != 1:
        raise VNPlayTurnError("exactly_one_turn_input_required")
    if input_text is not None:
        return {"input_text": input_text}
    if choice_id is not None:
        return {"choice_id": choice_id}
    return {"custom_action": dict(custom_action or {})}


def _retry_input_payload_from_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild adapter input from the persisted event accepted by the failed turn."""
    payload = event.get("event_payload")
    if not isinstance(payload, Mapping):
        raise VNPlayTurnError(ERROR_RETRY_LAST_TURN_NOT_FAILED)

    event_type = event.get("event_type")
    if event_type == EVENT_CHOICE_SELECTED:
        choice_id = payload.get("choice_id")
        if choice_id is None:
            raise VNPlayTurnError(ERROR_RETRY_LAST_TURN_NOT_FAILED)
        retry_payload: dict[str, Any] = {"choice_id": str(choice_id)}
        choice = payload.get("choice")
        if isinstance(choice, Mapping):
            retry_payload["choice"] = dict(choice)
        branch_node_id = payload.get("branch_node_id", event.get("branch_node_id"))
        if branch_node_id is not None:
            retry_payload["branch_node_id"] = branch_node_id
        return retry_payload

    if event_type == EVENT_USER_TURN:
        input_payload = payload.get("input")
        if isinstance(input_payload, Mapping):
            return dict(input_payload)

    raise VNPlayTurnError(ERROR_RETRY_LAST_TURN_NOT_FAILED)


def _visual_asset_type(
    directive: Mapping[str, Any],
    item: Mapping[str, Any],
) -> str:
    raw_asset_type = directive.get("asset_type") or item.get("asset_type")
    if not isinstance(raw_asset_type, str):
        return ""
    normalized = raw_asset_type.strip().lower()
    if normalized in {"background", "backgrounds"}:
        return "background"
    if normalized in {"depth", "depth_companion", "depth_companions"}:
        return "depth_companion"
    if normalized in {"sprite", "sprites"}:
        return "sprite"
    if normalized in {"cg", "cgs"}:
        return "cg"
    return normalized


def _merge_visual_item_scene_update(
    scene_updates: dict[str, Any],
    sprite_items: list[dict[str, Any]],
    *,
    asset_type: str,
    item: Mapping[str, Any],
) -> None:
    item_id = item.get("item_id")
    if asset_type == "background" and isinstance(item_id, int):
        scene_updates["current_background_item_id"] = item_id
        depth_item_id = item.get("depth_companion_item_id")
        if isinstance(depth_item_id, int):
            scene_updates["current_depth_item_id"] = depth_item_id
    elif asset_type == "depth_companion" and isinstance(item_id, int):
        scene_updates["current_depth_item_id"] = item_id
    elif asset_type == "sprite":
        sprite_items.append(dict(item))


def _visual_directive_warning(
    directive: Mapping[str, Any],
    *,
    reason: str,
    scene_version: int,
    error_type: str | None = None,
) -> dict[str, Any]:
    warning: dict[str, Any] = {
        "code": "visual_directive_rejected",
        "reason": reason,
        "scene_version": scene_version,
    }
    asset_type = directive.get("asset_type")
    if isinstance(asset_type, str):
        warning["asset_type"] = asset_type
    slot_key = directive.get("slot_key")
    if isinstance(slot_key, str):
        warning["slot_key"] = slot_key
    if error_type:
        warning["error_type"] = error_type
    return warning


def _manifest_items_by_id(manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    assets = manifest.get("assets")
    if not isinstance(assets, Mapping):
        return {}

    items_by_id: dict[int, dict[str, Any]] = {}
    for collection in assets.values():
        for item in _list_of_dicts(collection):
            item_id = item.get("item_id")
            if isinstance(item_id, int):
                items_by_id[item_id] = item
    return items_by_id


def _append_enrichment_warning(state: dict[str, Any], exc: Exception) -> None:
    warnings = list(state.get("warnings") or [])
    warnings.append(
        {
            "code": "scene_asset_enrichment_failed",
            "reason": "manifest_unavailable",
            "error_type": exc.__class__.__name__,
        }
    )
    state["warnings"] = warnings


def _payload_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded, usedforsecurity=False).hexdigest()


def _scene_state_payload(state: SceneState) -> dict[str, Any]:
    return asdict(state)


def _scene_snapshot_without_script_position(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(snapshot)
    payload.pop("script_position", None)
    return payload


def _script_position_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    script_position = snapshot.get("script_position")
    if isinstance(script_position, Mapping):
        return dict(script_position)
    return None


def _save_slot_response(slot: Mapping[str, Any], *, replayed: bool) -> dict[str, Any]:
    payload = dict(slot)
    payload["replayed"] = replayed
    return payload


def _input_text(payload: Mapping[str, Any]) -> str:
    if "input_text" in payload:
        return str(payload["input_text"])
    if "choice_id" in payload:
        return f"choice:{payload['choice_id']}"
    return json.dumps(payload.get("custom_action", {}), sort_keys=True)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
