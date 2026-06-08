from __future__ import annotations

import base64
import json

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    WorkspaceContextMembershipSummary,
    WorkspaceMembershipCreateRequest,
    WorkspaceMembershipListResponse,
    WorkspaceMembershipListSummary,
    WorkspaceMembershipResponse,
    WorkspaceMembershipSummaryResponse,
)
from tldw_Server_API.app.core.Workspaces import membership_adapters
from tldw_Server_API.app.core.Workspaces.membership_adapters import (
    ChatMembershipAdapter,
    MediaMembershipAdapter,
    WorkspaceArtifactMembershipAdapter,
    WorkspaceMembershipAdapterError,
    WorkspaceMembershipContext,
    WorkspaceNoteMembershipAdapter,
    WorkspaceSourceMembershipAdapter,
    get_workspace_membership_adapter,
)
from tldw_Server_API.app.core.Workspaces import membership_models
from tldw_Server_API.app.core.Workspaces.membership_models import (
    WORKSPACE_MEMBERSHIP_CURSOR_MAX_BYTES,
    WorkspaceMembershipCursor,
    WorkspaceResourceRef,
    WorkspaceResourceMembershipCursor,
    decode_membership_cursor,
    decode_resource_membership_cursor,
    encode_membership_cursor,
    encode_resource_membership_cursor,
)
from tldw_Server_API.app.core.Workspaces.membership_service import (
    WorkspaceMembershipService,
    WorkspaceMembershipServiceError,
)


def _encoded_json(payload: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii").rstrip("=")


def test_workspace_membership_cursor_round_trips() -> None:
    cursor = WorkspaceMembershipCursor(
        updated_at="2026-06-07T12:30:00Z",
        resource_type="media",
        resource_id="42",
    )

    encoded = encode_membership_cursor(cursor)

    assert encoded != cursor.updated_at
    assert decode_membership_cursor(encoded) == cursor


def test_workspace_resource_membership_cursor_round_trips() -> None:
    cursor = WorkspaceResourceMembershipCursor(
        updated_at="2026-06-07T12:30:00Z",
        workspace_id="workspace-1",
    )

    encoded = encode_resource_membership_cursor(cursor)

    assert encoded != cursor.updated_at
    assert decode_resource_membership_cursor(encoded) == cursor


@pytest.mark.parametrize(
    "payload",
    [
        "not-base64",
        _encoded_json(["not", "an", "object"]),
        _encoded_json({"updated_at": "2026-06-07T12:30:00Z", "resource_type": "media"}),
        _encoded_json({"updated_at": "2026-06-07T12:30:00Z", "resource_type": "media", "resource_id": 42}),
        _encoded_json(
            {
                "updated_at": "2026-06-07T12:30:00Z",
                "resource_type": "future_resource",
                "resource_id": "42",
            }
        ),
    ],
)
def test_workspace_membership_cursor_rejects_invalid_values(payload: str) -> None:
    with pytest.raises(ValueError):
        decode_membership_cursor(payload)


@pytest.mark.parametrize("version", [True, 1.0])
def test_workspace_membership_cursor_rejects_non_integer_versions(version: object) -> None:
    payload = _encoded_json(
        {
            "v": version,
            "updated_at": "2026-06-07T12:30:00Z",
            "resource_type": "media",
            "resource_id": "42",
        }
    )

    with pytest.raises(ValueError):
        decode_membership_cursor(payload)


def test_workspace_membership_cursor_rejects_oversized_values() -> None:
    payload = _encoded_json(
        {
            "v": 1,
            "updated_at": "2026-06-07T12:30:00Z" + ("x" * 5000),
            "resource_type": "media",
            "resource_id": "42",
        }
    )

    with pytest.raises(ValueError):
        decode_membership_cursor(payload)


@pytest.mark.parametrize(
    "payload",
    [
        "not-base64",
        _encoded_json({"updated_at": "2026-06-07T12:30:00Z", "resource_type": "media", "resource_id": "42"}),
        _encoded_json({"updated_at": "2026-06-07T12:30:00Z"}),
        _encoded_json({"updated_at": "2026-06-07T12:30:00Z", "workspace_id": 7}),
    ],
)
def test_workspace_resource_membership_cursor_rejects_invalid_values(payload: str) -> None:
    with pytest.raises(ValueError):
        decode_resource_membership_cursor(payload)


@pytest.mark.parametrize("version", [True, 1.0])
def test_workspace_resource_membership_cursor_rejects_non_integer_versions(version: object) -> None:
    payload = _encoded_json(
        {
            "v": version,
            "updated_at": "2026-06-07T12:30:00Z",
            "workspace_id": "workspace-1",
        }
    )

    with pytest.raises(ValueError):
        decode_resource_membership_cursor(payload)


def test_workspace_resource_membership_cursor_rejects_oversized_values() -> None:
    payload = _encoded_json(
        {
            "v": 1,
            "updated_at": "2026-06-07T12:30:00Z" + ("x" * 5000),
            "workspace_id": "workspace-1",
        }
    )

    with pytest.raises(ValueError):
        decode_resource_membership_cursor(payload)


def test_workspace_membership_cursors_reject_oversized_values_before_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_decode(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("base64 decode should not run for oversized cursors")

    monkeypatch.setattr(membership_models.base64, "b64decode", fail_decode)
    payload = "A" * (WORKSPACE_MEMBERSHIP_CURSOR_MAX_BYTES + 1)

    with pytest.raises(ValueError):
        decode_membership_cursor(payload)
    with pytest.raises(ValueError):
        decode_resource_membership_cursor(payload)


def test_workspace_membership_create_request_accepts_valid_payload() -> None:
    request = WorkspaceMembershipCreateRequest(
        resource_type="media",
        resource_id="42",
        role="source",
        label="Research paper",
        transfer_policy="link",
        provenance={"source_surface": "library", "operation_id": "op-1"},
        metadata={"rank": 1},
    )

    assert request.resource_type == "media"
    assert request.resource_id == "42"
    assert request.role == "source"
    assert request.transfer_policy == "link"
    assert request.provenance["source_surface"] == "library"
    assert request.metadata["rank"] == 1


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("role", "owner"),
        ("transfer_policy", "move"),
    ],
)
def test_workspace_membership_create_request_rejects_unsupported_literals(field_name: str, value: str) -> None:
    payload = {
        "resource_type": "media",
        "resource_id": "42",
        "role": "source",
        "transfer_policy": "link",
    }
    payload[field_name] = value

    with pytest.raises(ValidationError):
        WorkspaceMembershipCreateRequest(**payload)


@pytest.mark.parametrize("field_name", ["provenance", "metadata"])
def test_workspace_membership_create_request_rejects_oversized_json(field_name: str) -> None:
    payload = {
        "resource_type": "media",
        "resource_id": "42",
        "role": "source",
        "transfer_policy": "link",
        field_name: {"value": "x" * (16 * 1024 + 1)},
    }

    with pytest.raises(ValidationError, match=f"{field_name} exceeds"):
        WorkspaceMembershipCreateRequest(**payload)


@pytest.mark.parametrize("field_name", ["provenance", "metadata"])
def test_workspace_membership_create_request_rejects_non_json_serializable_values(field_name: str) -> None:
    payload = {
        "resource_type": "media",
        "resource_id": "42",
        "role": "source",
        "transfer_policy": "link",
        field_name: {"bad": object()},
    }

    with pytest.raises(ValidationError, match=f"{field_name} must be JSON serializable"):
        WorkspaceMembershipCreateRequest(**payload)


@pytest.mark.parametrize("field_name", ["provenance", "metadata"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_workspace_membership_create_request_rejects_non_finite_json_values(field_name: str, value: float) -> None:
    payload = {
        "resource_type": "media",
        "resource_id": "42",
        "role": "source",
        "transfer_policy": "link",
        field_name: {"value": value},
    }

    with pytest.raises(ValidationError, match=f"{field_name} must be JSON serializable"):
        WorkspaceMembershipCreateRequest(**payload)


@pytest.mark.parametrize("field_name", ["provenance", "metadata"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_membership_service_rejects_non_finite_json_values(field_name: str, value: float) -> None:
    db = FakeChaChaDB()
    adapter = RecordingAdapter()
    service = WorkspaceMembershipService(db, adapters={"media": adapter})

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.link_membership(
            "workspace-1",
            {
                "resource_type": "media",
                "resource_id": "42",
                "role": "source",
                field_name: {"value": value},
            },
            media_db=object(),
        )

    assert exc_info.value.code == "invalid_membership_request"
    assert exc_info.value.status_code == 400
    assert adapter.validated == []
    assert db.last_add_data is None


@pytest.mark.parametrize("field_name", ["provenance", "metadata"])
def test_membership_service_rejects_oversized_json_values(field_name: str) -> None:
    db = FakeChaChaDB()
    service = WorkspaceMembershipService(db, adapters={"media": RecordingAdapter()})

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.link_membership(
            "workspace-1",
            {
                "resource_type": "media",
                "resource_id": "42",
                "role": "source",
                field_name: {"value": "x" * (16 * 1024 + 1)},
            },
            media_db=object(),
        )

    assert exc_info.value.code == "invalid_membership_request"
    assert db.last_add_data is None


def test_workspace_membership_list_response_constructs_grouped_totals() -> None:
    item = WorkspaceMembershipResponse(
        workspace_id="workspace-1",
        resource_type="media",
        resource_id="42",
        role="source",
        label="Research paper",
        transfer_policy="link",
        provenance={"source_surface": "library"},
        metadata={"rank": 1},
        summary=WorkspaceMembershipSummaryResponse(
            title="Research paper",
            subtitle="PDF",
            href="/media/42",
            updated_at="2026-06-07T12:30:00Z",
            state="available",
        ),
        created_at="2026-06-07T12:00:00Z",
        updated_at="2026-06-07T12:30:00Z",
        version=2,
        deleted=False,
    )

    response = WorkspaceMembershipListResponse(
        workspace_id="workspace-1",
        items=[item],
        total=1,
        next_cursor=None,
        summary=WorkspaceMembershipListSummary(
            total=1,
            by_resource_type={"media": 1},
            by_role={"source": 1},
        ),
    )

    assert response.items[0].summary is not None
    assert response.summary.by_resource_type == {"media": 1}
    assert response.summary.by_role == {"source": 1}


def test_workspace_context_membership_summary_constructs_grouped_totals() -> None:
    summary = WorkspaceContextMembershipSummary(
        total=3,
        by_resource_type={"media": 2, "chat": 1},
        by_role={"source": 2, "conversation": 1},
    )

    assert summary.total == 3
    assert summary.by_resource_type["media"] == 2
    assert summary.by_role["conversation"] == 1


class FakeChaChaDB:
    def __init__(self) -> None:
        self.workspaces = {
            "workspace-1": {"id": "workspace-1", "archived": False},
            "workspace-archived": {"id": "workspace-archived", "archived": True},
        }
        self.notes: dict[tuple[str, int], dict[str, object]] = {}
        self.sources: dict[tuple[str, str], dict[str, object]] = {}
        self.artifacts: dict[tuple[str, str], dict[str, object]] = {}
        self.conversations: dict[str, dict[str, object]] = {}
        self.memberships: dict[tuple[str, str, str], dict[str, object]] = {}
        self.last_add_data: dict[str, object] | None = None
        self.last_list_resource_key: tuple[str, str] | None = None
        self.list_workspace_calls = 0
        self.summary_calls = 0
        self._clock = 0

    def _timestamp(self) -> str:
        self._clock += 1
        return f"2026-06-07T12:00:{self._clock:02d}Z"

    def get_workspace(self, workspace_id: str) -> dict[str, object] | None:
        return self.workspaces.get(workspace_id)

    def get_workspace_note(
        self,
        workspace_id: str,
        note_id: int,
        *,
        include_deleted: bool = False,
    ) -> dict[str, object] | None:
        return self.notes.get((workspace_id, note_id))

    def get_workspace_source(
        self,
        workspace_id: str,
        source_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, object] | None:
        return self.sources.get((workspace_id, source_id))

    def get_workspace_artifact(
        self,
        workspace_id: str,
        artifact_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, object] | None:
        return self.artifacts.get((workspace_id, artifact_id))

    def get_conversation_for_workspace_membership(
        self,
        conversation_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, object] | None:
        return self.conversations.get(conversation_id)

    def add_workspace_resource_membership(
        self,
        workspace_id: str,
        data: dict[str, object],
        *,
        user_id: str | None = None,
    ) -> dict[str, object]:
        self.last_add_data = dict(data)
        key = (workspace_id, str(data["resource_type"]), str(data["resource_id"]))
        existing = self.memberships.get(key)
        if existing is not None:
            if existing.get("deleted") and data.get("restore_deleted") is True:
                existing.update(
                    {
                        "role": data.get("role", "member"),
                        "label": data.get("label"),
                        "transfer_policy": data.get("transfer_policy", "link"),
                        "provenance": data.get("provenance", {}),
                        "metadata": data.get("metadata", {}),
                        "deleted": False,
                        "updated_at": self._timestamp(),
                        "version": int(existing.get("version", 1)) + 1,
                    }
                )
            return dict(existing)

        now = self._timestamp()
        row = {
            "workspace_id": workspace_id,
            "resource_type": str(data["resource_type"]),
            "resource_id": str(data["resource_id"]),
            "role": data.get("role", "member"),
            "label": data.get("label"),
            "transfer_policy": data.get("transfer_policy", "link"),
            "provenance": data.get("provenance", {}),
            "metadata": data.get("metadata", {}),
            "summary": None,
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "deleted": False,
            "created_by_user_id": user_id,
            "updated_by_user_id": user_id,
        }
        self.memberships[key] = row
        return dict(row)

    def get_workspace_resource_membership(
        self,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, object] | None:
        row = self.memberships.get((workspace_id, resource_type, resource_id))
        if row is None or (row.get("deleted") and not include_deleted):
            return None
        return dict(row)

    def list_workspace_resource_memberships(
        self,
        workspace_id: str,
        *,
        resource_type: str | None = None,
        role: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        cursor: tuple[str, str, str] | None = None,
    ) -> list[dict[str, object]]:
        self.list_workspace_calls += 1
        rows = [
            row
            for (row_workspace_id, _, _), row in self.memberships.items()
            if row_workspace_id == workspace_id
            and (resource_type is None or row["resource_type"] == resource_type)
            and (role is None or row["role"] == role)
            and (include_deleted or not row.get("deleted"))
        ]
        rows.sort(
            key=lambda row: (str(row["updated_at"]), str(row["resource_type"]), str(row["resource_id"])),
            reverse=True,
        )
        return [dict(row) for row in rows[:limit]]

    def workspace_resource_membership_summary(self, workspace_id: str) -> dict[str, object]:
        self.summary_calls += 1
        rows = [
            row
            for (row_workspace_id, _, _), row in self.memberships.items()
            if row_workspace_id == workspace_id and not row.get("deleted")
        ]
        by_resource_type: dict[str, int] = {}
        by_role: dict[str, int] = {}
        for row in rows:
            by_resource_type[str(row["resource_type"])] = by_resource_type.get(str(row["resource_type"]), 0) + 1
            by_role[str(row["role"])] = by_role.get(str(row["role"]), 0) + 1
        return {
            "total": len(rows),
            "by_resource_type": dict(sorted(by_resource_type.items())),
            "by_role": dict(sorted(by_role.items())),
        }

    def list_resource_workspace_memberships(
        self,
        resource_type: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        cursor: tuple[str, str] | None = None,
    ) -> list[dict[str, object]]:
        self.last_list_resource_key = (resource_type, resource_id)
        rows = [
            row
            for (_, row_resource_type, row_resource_id), row in self.memberships.items()
            if row_resource_type == resource_type
            and row_resource_id == resource_id
            and (include_deleted or not row.get("deleted"))
        ]
        rows.sort(key=lambda row: (str(row["updated_at"]), str(row["workspace_id"])), reverse=True)
        return [dict(row) for row in rows[:limit]]

    def delete_workspace_resource_membership(
        self,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        row = self.memberships.get((workspace_id, resource_type, resource_id))
        if row is None or row.get("deleted"):
            return None
        row["deleted"] = True
        row["updated_by_user_id"] = user_id
        row["updated_at"] = self._timestamp()
        row["version"] = int(row.get("version", 1)) + 1
        return dict(row)


def _membership_row(
    workspace_id: str = "workspace-1",
    resource_type: str = "media",
    resource_id: str = "42",
    role: str = "source",
    *,
    deleted: bool = False,
) -> dict[str, object]:
    return {
        "workspace_id": workspace_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "role": role,
        "label": "Stored label",
        "transfer_policy": "link",
        "provenance": {"source_surface": "test"},
        "metadata": {"rank": 1},
        "created_at": "2026-06-07T12:00:00Z",
        "updated_at": "2026-06-07T12:30:00Z",
        "version": 1,
        "deleted": deleted,
    }


def _context(db: FakeChaChaDB, *, media_db: object | None = None) -> WorkspaceMembershipContext:
    return WorkspaceMembershipContext(
        workspace_id="workspace-1",
        user_id="user-1",
        chacha_db=db,
        media_db=media_db,
    )


def test_unsupported_resource_type_fails_closed() -> None:
    service = WorkspaceMembershipService(FakeChaChaDB())

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.link_membership(
            "workspace-1",
            {"resource_type": "note", "resource_id": "1"},
            user_id="user-1",
        )

    assert exc_info.value.code == "unsupported_resource_type"
    assert exc_info.value.status_code == 400
    with pytest.raises(WorkspaceMembershipAdapterError):
        get_workspace_membership_adapter("note")


def test_workspace_note_adapter_validates_same_workspace_note() -> None:
    db = FakeChaChaDB()
    db.notes[("workspace-1", 7)] = {
        "id": 7,
        "workspace_id": "workspace-1",
        "title": "Research note",
        "last_modified": "2026-06-07T12:05:00Z",
        "deleted": 0,
    }

    ref = WorkspaceNoteMembershipAdapter().validate_access("7", _context(db))

    assert ref.resource_type == "workspace_note"
    assert ref.resource_id == "7"
    assert ref.title == "Research note"
    assert ref.updated_at == "2026-06-07T12:05:00Z"


def test_workspace_source_adapter_validates_same_workspace_source() -> None:
    db = FakeChaChaDB()
    db.sources[("workspace-1", "source-1")] = {
        "id": "source-1",
        "workspace_id": "workspace-1",
        "title": "Source title",
        "source_type": "pdf",
        "media_id": 42,
        "added_at": "2026-06-07T12:05:00Z",
    }

    ref = WorkspaceSourceMembershipAdapter().validate_access("source-1", _context(db))

    assert ref.resource_type == "workspace_source"
    assert ref.resource_id == "source-1"
    assert ref.title == "Source title"
    assert ref.subtitle == "pdf"
    assert ref.metadata["media_id"] == 42


def test_workspace_artifact_adapter_validates_same_workspace_artifact() -> None:
    db = FakeChaChaDB()
    db.artifacts[("workspace-1", "artifact-1")] = {
        "id": "artifact-1",
        "workspace_id": "workspace-1",
        "title": "Draft report",
        "artifact_type": "report",
        "review_state": "approved",
        "created_at": "2026-06-07T12:05:00Z",
    }

    ref = WorkspaceArtifactMembershipAdapter().validate_access("artifact-1", _context(db))

    assert ref.resource_type == "workspace_artifact"
    assert ref.resource_id == "artifact-1"
    assert ref.title == "Draft report"
    assert ref.subtitle == "report"
    assert ref.metadata["review_state"] == "approved"


def test_media_adapter_validates_via_media_db_api_and_canonicalizes_id(monkeypatch: pytest.MonkeyPatch) -> None:
    db = FakeChaChaDB()
    media_db = object()
    calls: list[tuple[object, int, bool, bool]] = []

    def fake_get_media_by_id(
        passed_db: object,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, object]:
        calls.append((passed_db, media_id, include_deleted, include_trash))
        return {
            "id": media_id,
            "title": "Library item",
            "type": "video",
            "url": "https://example.test/item",
            "last_modified": "2026-06-07T12:05:00Z",
        }

    monkeypatch.setattr(membership_adapters.media_db_api, "get_media_by_id", fake_get_media_by_id)

    ref = MediaMembershipAdapter().validate_access("0042", _context(db, media_db=media_db))

    assert calls == [(media_db, 42, False, False)]
    assert ref.resource_type == "media"
    assert ref.resource_id == "42"
    assert ref.title == "Library item"


def test_media_service_requires_media_db_for_validation() -> None:
    service = WorkspaceMembershipService(FakeChaChaDB())

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.link_membership(
            "workspace-1",
            {"resource_type": "media", "resource_id": "42", "role": "source"},
            user_id="user-1",
        )

    assert exc_info.value.code == "media_db_unavailable"
    assert exc_info.value.status_code == 503


def test_chat_adapter_allows_global_and_same_workspace_conversations() -> None:
    db = FakeChaChaDB()
    db.conversations["chat-global"] = {
        "id": "chat-global",
        "title": "Global chat",
        "scope_type": "global",
        "workspace_id": None,
        "last_modified": "2026-06-07T12:05:00Z",
    }
    db.conversations["chat-workspace"] = {
        "id": "chat-workspace",
        "title": "Workspace chat",
        "scope_type": "workspace",
        "workspace_id": "workspace-1",
        "last_modified": "2026-06-07T12:06:00Z",
    }

    global_ref = ChatMembershipAdapter().validate_access("chat-global", _context(db))
    workspace_ref = ChatMembershipAdapter().validate_access("chat-workspace", _context(db))

    assert global_ref.resource_id == "chat-global"
    assert global_ref.title == "Global chat"
    assert workspace_ref.resource_id == "chat-workspace"
    assert workspace_ref.title == "Workspace chat"


def test_chat_adapter_rejects_other_workspace_conversation() -> None:
    db = FakeChaChaDB()
    db.conversations["chat-other"] = {
        "id": "chat-other",
        "title": "Other workspace chat",
        "scope_type": "workspace",
        "workspace_id": "workspace-2",
        "last_modified": "2026-06-07T12:06:00Z",
    }

    with pytest.raises(WorkspaceMembershipAdapterError) as exc_info:
        ChatMembershipAdapter().validate_access("chat-other", _context(db))

    assert exc_info.value.code == "resource_not_found"


def test_archived_workspace_rejects_link_membership() -> None:
    service = WorkspaceMembershipService(FakeChaChaDB())

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.link_membership(
            "workspace-archived",
            {"resource_type": "chat", "resource_id": "chat-1"},
            user_id="user-1",
        )

    assert exc_info.value.code == "workspace_archived"
    assert exc_info.value.status_code == 409


def test_missing_workspace_rejects_membership_read_path() -> None:
    service = WorkspaceMembershipService(FakeChaChaDB())

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.list_workspace_memberships("workspace-missing")

    assert exc_info.value.code == "workspace_not_found"
    assert exc_info.value.status_code == 404


def test_list_workspace_memberships_resolve_false_omits_adapter_summary() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row()
    service = WorkspaceMembershipService(db)

    payload = service.list_workspace_memberships("workspace-1", resolve=False)

    assert payload["workspace_id"] == "workspace-1"
    assert payload["items"][0]["summary"] is None
    assert payload["summary"]["by_resource_type"] == {"media": 1}


@pytest.mark.parametrize(
    "cursor",
    [123, object(), ("only-updated-at", "media"), ("", "media", "42"), ("2026-06-07T12:00:00Z", "media", 42)],
)
def test_list_workspace_memberships_rejects_invalid_cursor_types(cursor: object) -> None:
    service = WorkspaceMembershipService(FakeChaChaDB())

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.list_workspace_memberships("workspace-1", cursor=cursor)

    assert exc_info.value.code == "invalid_cursor"
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    "cursor",
    [123, object(), ("only-updated-at",), ("", "workspace-1"), ("2026-06-07T12:00:00Z", 7)],
)
def test_list_resource_memberships_rejects_invalid_cursor_types(cursor: object) -> None:
    service = WorkspaceMembershipService(FakeChaChaDB(), adapters={"media": RecordingAdapter()})

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.list_resource_memberships("media", "42", cursor=cursor, media_db=object())

    assert exc_info.value.code == "invalid_cursor"
    assert exc_info.value.status_code == 400


class FailingSummaryAdapter:
    resource_type = "media"

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return WorkspaceResourceRef(resource_type=self.resource_type, resource_id=str(int(resource_id)))

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        raise RuntimeError("summary backend unavailable at /tmp/secret-token api_key=sk-test")

    def on_link(self, membership: dict[str, object], context: WorkspaceMembershipContext) -> None:
        return None

    def on_unlink(self, membership: dict[str, object], context: WorkspaceMembershipContext) -> None:
        return None


def test_adapter_summarize_failure_during_list_marks_item_unresolved() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row()
    service = WorkspaceMembershipService(db, adapters={"media": FailingSummaryAdapter()})

    payload = service.list_workspace_memberships("workspace-1", media_db=object())

    summary = payload["items"][0]["summary"]
    assert summary["state"] == "unresolved"
    assert summary["metadata"]["code"] == "summary_unavailable"


def test_generic_summarize_failure_uses_safe_unresolved_message() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row()
    service = WorkspaceMembershipService(db, adapters={"media": FailingSummaryAdapter()})

    payload = service.list_workspace_memberships("workspace-1", media_db=object())

    message = payload["items"][0]["summary"]["metadata"]["message"]
    assert message == "Workspace resource summary is unavailable."
    assert "/tmp/secret-token" not in message
    assert "sk-test" not in message


@pytest.mark.parametrize(
    "resource_type,resource_id,store_key,row",
    [
        (
            "workspace_note",
            "7",
            ("workspace-1", 7),
            {"id": 7, "workspace_id": "workspace-1", "title": "Deleted note", "deleted": 1},
        ),
        (
            "workspace_source",
            "source-1",
            ("workspace-1", "source-1"),
            {
                "id": "source-1",
                "workspace_id": "workspace-1",
                "title": "Deleted source",
                "source_type": "pdf",
                "deleted": 1,
            },
        ),
        (
            "workspace_artifact",
            "artifact-1",
            ("workspace-1", "artifact-1"),
            {
                "id": "artifact-1",
                "workspace_id": "workspace-1",
                "title": "Deleted artifact",
                "artifact_type": "report",
                "deleted": 1,
            },
        ),
    ],
)
def test_deleted_workspace_subresource_summary_reports_deleted_state(
    resource_type: str,
    resource_id: str,
    store_key: tuple[object, ...],
    row: dict[str, object],
) -> None:
    db = FakeChaChaDB()
    if resource_type == "workspace_note":
        db.notes[store_key] = row
    elif resource_type == "workspace_source":
        db.sources[store_key] = row
    else:
        db.artifacts[store_key] = row
    db.memberships[("workspace-1", resource_type, resource_id)] = _membership_row(
        resource_type=resource_type,
        resource_id=resource_id,
    )
    service = WorkspaceMembershipService(db)

    payload = service.list_workspace_memberships("workspace-1")

    summary = payload["items"][0]["summary"]
    assert summary["state"] == "deleted"
    assert summary["title"] == row["title"]


def test_deleted_chat_summary_reports_deleted_state() -> None:
    db = FakeChaChaDB()
    db.conversations["chat-1"] = {
        "id": "chat-1",
        "title": "Deleted chat",
        "scope_type": "workspace",
        "workspace_id": "workspace-1",
        "deleted": 1,
    }
    db.memberships[("workspace-1", "chat", "chat-1")] = _membership_row(
        resource_type="chat",
        resource_id="chat-1",
        role="conversation",
    )
    service = WorkspaceMembershipService(db)

    payload = service.list_workspace_memberships("workspace-1")

    summary = payload["items"][0]["summary"]
    assert summary["state"] == "deleted"
    assert summary["title"] == "Deleted chat"


def test_deleted_media_summary_reports_deleted_state(monkeypatch: pytest.MonkeyPatch) -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row()
    media_db = object()
    calls: list[tuple[int, bool, bool]] = []

    def fake_get_media_by_id(
        passed_db: object,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, object] | None:
        assert passed_db is media_db
        calls.append((media_id, include_deleted, include_trash))
        if include_deleted and include_trash:
            return {"id": media_id, "title": "Deleted media", "deleted": 1}
        return None

    monkeypatch.setattr(membership_adapters.media_db_api, "get_media_by_id", fake_get_media_by_id)
    service = WorkspaceMembershipService(db)

    payload = service.list_workspace_memberships("workspace-1", media_db=media_db)

    summary = payload["items"][0]["summary"]
    assert summary["state"] == "deleted"
    assert summary["title"] == "Deleted media"
    assert calls == [(42, True, True)]


class RecordingAdapter:
    resource_type = "media"

    def __init__(self) -> None:
        self.validated: list[str] = []
        self.linked: list[dict[str, object]] = []
        self.unlinked: list[dict[str, object]] = []

    def validate_access(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        self.validated.append(resource_id)
        return WorkspaceResourceRef(resource_type=self.resource_type, resource_id=str(int(resource_id)), title="Restored")

    def summarize(self, resource_id: str, context: WorkspaceMembershipContext) -> WorkspaceResourceRef:
        return WorkspaceResourceRef(resource_type=self.resource_type, resource_id=str(int(resource_id)), title="Restored")

    def on_link(self, membership: dict[str, object], context: WorkspaceMembershipContext) -> None:
        self.linked.append(dict(membership))

    def on_unlink(self, membership: dict[str, object], context: WorkspaceMembershipContext) -> None:
        self.unlinked.append(dict(membership))


def test_link_membership_restores_deleted_row_after_adapter_validation() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row(deleted=True)
    adapter = RecordingAdapter()
    service = WorkspaceMembershipService(db, adapters={"media": adapter})

    payload = service.link_membership(
        "workspace-1",
        {"resource_type": "media", "resource_id": "0042", "role": "source"},
        user_id="user-1",
        media_db=object(),
    )

    assert adapter.validated == ["0042"]
    assert db.last_add_data is not None
    assert db.last_add_data["resource_id"] == "42"
    assert db.last_add_data["restore_deleted"] is True
    assert payload["resource_id"] == "42"
    assert payload["deleted"] is False
    assert adapter.linked == []


def test_link_membership_idempotent_retry_does_not_invoke_reserved_on_link_hook() -> None:
    db = FakeChaChaDB()
    adapter = RecordingAdapter()
    service = WorkspaceMembershipService(db, adapters={"media": adapter})

    first_payload = service.link_membership(
        "workspace-1",
        {"resource_type": "media", "resource_id": "0042", "role": "source"},
        user_id="user-1",
        media_db=object(),
    )
    retry_payload = service.link_membership(
        "workspace-1",
        {"resource_type": "media", "resource_id": "0042", "role": "source"},
        user_id="user-1",
        media_db=object(),
    )

    assert first_payload["resource_id"] == "42"
    assert retry_payload["resource_id"] == "42"
    assert adapter.linked == []


def test_get_membership_returns_unresolved_summary_when_media_db_is_unavailable() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row()
    service = WorkspaceMembershipService(db)

    payload = service.get_membership("workspace-1", "media", "0042")

    assert payload is not None
    assert payload["resource_id"] == "42"
    assert payload["summary"]["state"] == "unresolved"
    assert payload["summary"]["metadata"]["code"] == "media_db_unavailable"


def test_list_resource_memberships_canonicalizes_resource_id_before_listing() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row()
    service = WorkspaceMembershipService(db, adapters={"media": RecordingAdapter()})

    payload = service.list_resource_memberships("media", "0042", media_db=object())

    assert db.last_list_resource_key == ("media", "42")
    assert payload["resource_type"] == "media"
    assert payload["resource_id"] == "42"
    assert payload["items"][0]["resource_id"] == "42"


def test_list_resource_memberships_unsupported_type_fails_closed() -> None:
    service = WorkspaceMembershipService(FakeChaChaDB())

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.list_resource_memberships("note", "1")

    assert exc_info.value.code == "unsupported_resource_type"
    assert exc_info.value.status_code == 400


def test_unlink_membership_soft_deletes_active_membership_and_calls_adapter_hook() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row()
    adapter = RecordingAdapter()
    service = WorkspaceMembershipService(db, adapters={"media": adapter})

    payload = service.unlink_membership("workspace-1", "media", "0042", user_id="user-1", media_db=object())

    assert payload is not None
    assert payload["deleted"] is True
    assert db.memberships[("workspace-1", "media", "42")]["deleted"] is True
    assert adapter.unlinked == [db.memberships[("workspace-1", "media", "42")]]


def test_unlink_membership_noops_without_adapter_hook_for_missing_or_deleted_membership() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row(deleted=True)
    adapter = RecordingAdapter()
    service = WorkspaceMembershipService(db, adapters={"media": adapter})

    missing_payload = service.unlink_membership("workspace-1", "media", "43", user_id="user-1", media_db=object())
    deleted_payload = service.unlink_membership("workspace-1", "media", "42", user_id="user-1", media_db=object())

    assert missing_payload is None
    assert deleted_payload is None
    assert adapter.unlinked == []


class FailingUnlinkAdapter(RecordingAdapter):
    def on_unlink(self, membership: dict[str, object], context: WorkspaceMembershipContext) -> None:
        raise RuntimeError("external cleanup failed")


def test_unlink_membership_returns_deleted_row_when_adapter_hook_fails() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row()
    service = WorkspaceMembershipService(db, adapters={"media": FailingUnlinkAdapter()})

    payload = service.unlink_membership("workspace-1", "media", "0042", user_id="user-1", media_db=object())

    assert payload is not None
    assert payload["deleted"] is True
    assert db.memberships[("workspace-1", "media", "42")]["deleted"] is True


def test_workspace_membership_summary_uses_db_aggregate_for_active_compact_totals() -> None:
    db = FakeChaChaDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row(role="source")
    db.memberships[("workspace-1", "chat", "chat-1")] = _membership_row(
        resource_type="chat",
        resource_id="chat-1",
        role="conversation",
    )
    db.memberships[("workspace-1", "media", "43")] = _membership_row(resource_id="43", deleted=True)
    service = WorkspaceMembershipService(db)

    summary = service.workspace_membership_summary("workspace-1")

    assert summary == {
        "total": 2,
        "by_resource_type": {"chat": 1, "media": 1},
        "by_role": {"conversation": 1, "source": 1},
    }
    assert db.summary_calls == 1
    assert db.list_workspace_calls == 0
