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
from tldw_Server_API.app.core.Workspaces.membership_models import (
    WorkspaceMembershipCursor,
    WorkspaceResourceMembershipCursor,
    decode_membership_cursor,
    decode_resource_membership_cursor,
    encode_membership_cursor,
    encode_resource_membership_cursor,
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
        ("resource_type", "note"),
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
