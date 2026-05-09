"""VN Play runtime service orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.DB_Management.VNPlay_DB import VNPlayRepository
from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
from tldw_Server_API.app.core.VN_Play.assets import resolve_scene_directives
from tldw_Server_API.app.core.VN_Play.branch_navigation import (
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
    ERROR_INVALID_CHOICE_ID,
    ERROR_RESTORE_ACTION_IN_PROGRESS,
    ERROR_RETRY_LAST_TURN_NOT_FAILED,
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
from tldw_Server_API.app.core.VN_Play.models import SceneState, TurnResult
from tldw_Server_API.app.core.VN_Play.parser import VNPlayParseError, coerce_turn_result
from tldw_Server_API.app.core.VN_Play.state import derive_scene_state


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

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> VNPlayTurnResponse:
        return cls(
            turn_request_id=int(payload["turn_request_id"]),
            status=str(payload["status"]),
            scene_version=int(payload["scene_version"]),
            events=_list_of_dicts(payload.get("events")),
            warnings=_list_of_dicts(payload.get("warnings")),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "turn_request_id": self.turn_request_id,
            "status": self.status,
            "scene_version": self.scene_version,
            "events": self.events,
            "warnings": self.warnings,
        }


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
    ) -> VNPlaySession:
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
        self.get_session(session_id)
        events = self.repo.list_events(session_id)
        state = self.repo.get_scene_state(session_id, owner_user_id=self.owner_user_id)
        if state is None:
            raise VNPlayNotFoundError("scene_state_not_found")
        last_event_id = int(events[-1]["id"]) if events else None
        checkpoint = self.repo.create_checkpoint(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            label=label,
            event_id=last_event_id,
            scene_version=int(state["scene_version"]),
            scene_state_snapshot=state,
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
        except Exception:
            self._mark_session_action_failed(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=SESSION_ACTION_STATUS_FAILED,
            )
            raise
        return response

    def restore_checkpoint(
        self,
        session_id: int,
        checkpoint_id: int,
        *,
        idempotency_key: str,
    ) -> dict[str, Any]:
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

        self._validate_restore_can_start(
            session_id=session_id,
            action_id=int(action["id"]),
            expected_scene_version=session.scene_version,
        )

        snapshot = dict(checkpoint["scene_state_snapshot"])
        previous_scene_version = session.scene_version
        next_scene_version = previous_scene_version + 1
        restored_state = dict(snapshot)
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
                branch_node_id=_optional_int(restored_state.get("active_branch_node_id")),
                response_payload_factory=lambda payload: self._restore_response_payload(
                    payload,
                    checkpoint_id=checkpoint_id,
                    target_event_id=_optional_int(checkpoint.get("event_id")),
                    replayed=False,
                ),
            )
        except Exception:
            self._mark_session_action_failed(
                session_id=session_id,
                action_id=int(action["id"]),
                error_code=SESSION_ACTION_STATUS_FAILED,
            )
            raise
        return response

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
        target: str | None = None,
        target_event_id: int | None,
        replayed: bool,
    ) -> dict[str, Any]:
        session_payload = dict(payload["session"])
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
    ) -> None:
        self.repo.mark_session_action_terminal(
            session_id=session_id,
            owner_user_id=self.owner_user_id,
            action_id=action_id,
            status=SESSION_ACTION_STATUS_FAILED,
            error={"code": error_code},
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

        return VNPlayTurnResponse(
            turn_request_id=int(turn_request["id"]),
            status=str(turn_request["status"]),
            scene_version=int(turn_request["base_scene_version"]),
        )

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
