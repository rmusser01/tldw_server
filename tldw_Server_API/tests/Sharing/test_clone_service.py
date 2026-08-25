"""Unit tests for deterministic, operation-owned Workspace cloning."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from types import MappingProxyType
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.repositories.clone_snapshot_repository import (
    OperationOwnedMediaResult,
)
from tldw_Server_API.app.core.Sharing import clone_models as clone_models_module
from tldw_Server_API.app.core.Sharing.clone_models import (
    CloneCancelled,
    CloneCopyCounts,
    CloneRetrievalReadiness,
    CloneSnapshotUnavailable,
    CloneWarning,
    MediaCloneSnapshot,
    WorkspaceCloneRequest,
    WorkspaceCloneResult,
    WorkspaceCloneSnapshot,
)
from tldw_Server_API.app.core.Sharing.clone_service import CloneService

pytestmark = pytest.mark.unit


def test_clone_contracts_are_frozen_and_slotted():
    contracts = (
        WorkspaceCloneRequest,
        WorkspaceCloneSnapshot,
        MediaCloneSnapshot,
        CloneCopyCounts,
        CloneRetrievalReadiness,
        CloneWarning,
        WorkspaceCloneResult,
    )

    for contract in contracts:
        assert is_dataclass(contract)
        assert contract.__dataclass_params__.frozen is True
        assert "__dict__" not in contract.__slots__

    readiness = CloneRetrievalReadiness("ready", "ready", "needs_indexing")
    with pytest.raises(FrozenInstanceError):
        readiness.text_search = "unavailable"


def test_clone_request_normalizes_name():
    request = WorkspaceCloneRequest(
        source_workspace_id="source",
        target_workspace_id="target",
        operation_id="operation",
        request_fingerprint="fingerprint",
        name="  Research\t Workspace  ",
    )

    assert request.name == "Research Workspace"


def test_clone_request_rejects_empty_normalized_name():
    with pytest.raises(ValueError, match="name"):
        WorkspaceCloneRequest(
            source_workspace_id="source",
            target_workspace_id="target",
            operation_id="operation",
            request_fingerprint="fingerprint",
            name=" \t\n ",
        )


def test_clone_contract_rejects_non_ascii_identifier():
    with pytest.raises(ValueError, match="ASCII"):
        WorkspaceCloneRequest(
            source_workspace_id="sourcé",
            target_workspace_id="target",
            operation_id="operation",
            request_fingerprint="fingerprint",
            name="Copy",
        )


def test_clone_contract_rejects_operation_owned_media_above_media_copied():
    with pytest.raises(ValueError, match="operation_owned_media_count.*media_copied"):
        CloneCopyCounts(
            media_attempted=2,
            media_copied=1,
            operation_owned_media_count=2,
        )


def test_snapshot_defensively_copies_mutable_rows():
    row = {"id": "source-1", "title": "Original"}
    snapshot = WorkspaceCloneSnapshot.from_rows(
        workspace={"id": "ws"}, sources=[row], notes=[], artifacts=[]
    )
    row["title"] = "Changed"
    assert snapshot.sources[0]["title"] == "Original"


def test_snapshot_rows_are_recursive_immutable_views():
    row = {"id": "source-1", "metadata": {"tags": ["one"]}}
    snapshot = WorkspaceCloneSnapshot.from_rows(
        workspace={"id": "ws"}, sources=[row], notes=[], artifacts=[]
    )

    assert isinstance(snapshot.workspace, MappingProxyType)
    assert isinstance(snapshot.sources, tuple)
    assert isinstance(snapshot.sources[0], MappingProxyType)
    assert isinstance(snapshot.sources[0]["metadata"], MappingProxyType)
    assert snapshot.sources[0]["metadata"]["tags"] == ("one",)
    with pytest.raises(TypeError):
        snapshot.sources[0]["title"] = "Changed"


def test_media_snapshot_defensively_copies_rows():
    media = {"id": 1, "metadata": {"labels": ["source"]}}
    chunks = [{"text": "chunk"}]
    transcripts = [{"transcription": "text"}]
    snapshot = MediaCloneSnapshot.from_rows(media, chunks, transcripts)

    media["metadata"]["labels"].append("changed")
    chunks[0]["text"] = "changed"
    transcripts[0]["transcription"] = "changed"

    assert snapshot.media["metadata"]["labels"] == ("source",)
    assert snapshot.chunks[0]["text"] == "chunk"
    assert snapshot.transcripts[0]["transcription"] == "text"


def test_direct_snapshot_construction_is_also_immutable():
    row = {"id": "source-1", "nested": {"value": "Original"}}
    snapshot = WorkspaceCloneSnapshot(
        workspace={"id": "ws"},
        memberships=[],
        sources=[row],
        notes=[],
        artifacts=[],
    )

    row["nested"]["value"] = "Changed"
    assert snapshot.sources[0]["nested"]["value"] == "Original"


def test_snapshot_converts_bytearray_to_bytes():
    buffer = bytearray(b"original")
    snapshot = WorkspaceCloneSnapshot.from_rows(
        workspace={"id": "ws", "buffer": buffer},
        sources=[],
        notes=[],
        artifacts=[],
    )

    buffer[:] = b"changed!"
    assert snapshot.workspace["buffer"] == b"original"
    assert isinstance(snapshot.workspace["buffer"], bytes)


def test_snapshot_converts_memoryview_to_bytes():
    buffer = bytearray(b"original")
    snapshot = WorkspaceCloneSnapshot.from_rows(
        workspace={"id": "ws", "buffer": memoryview(buffer)},
        sources=[],
        notes=[],
        artifacts=[],
    )

    buffer[:] = b"changed!"
    assert snapshot.workspace["buffer"] == b"original"
    assert isinstance(snapshot.workspace["buffer"], bytes)


def test_snapshot_rejects_unsupported_custom_values():
    class MutablePayload:
        def __init__(self) -> None:
            self.values = []

    with pytest.raises(TypeError, match="unsupported clone snapshot value type"):
        WorkspaceCloneSnapshot.from_rows(
            workspace={"id": "ws", "payload": MutablePayload()},
            sources=[],
            notes=[],
            artifacts=[],
        )


@pytest.mark.parametrize(
    "value",
    (
        None,
        True,
        7,
        3.5,
        "text",
        b"bytes",
        date(2026, 8, 25),
        datetime(2026, 8, 25, 12, 30, 45),
        time(12, 30, 45),
        Decimal("12.34"),
        UUID("12345678-1234-5678-1234-567812345678"),
    ),
)
def test_snapshot_preserves_expected_immutable_db_scalars(value):
    snapshot = WorkspaceCloneSnapshot.from_rows(
        workspace={"id": "ws", "value": value},
        sources=[],
        notes=[],
        artifacts=[],
    )

    assert snapshot.workspace["value"] == value
    assert type(snapshot.workspace["value"]) is type(value)


def test_clone_warning_rejects_unbounded_or_invalid_count():
    with pytest.raises(ValueError, match="count"):
        CloneWarning(code="warning", count=-1)
    with pytest.raises(ValueError, match="ASCII"):
        CloneWarning(code="warning-é", count=1)


@pytest.mark.parametrize(
    "code",
    (
        "Warning_Code",
        "warning-code",
        "https://example.com",
        "/tmp/private",
        "ValueError:failed",
        "_warning",
        "a" * 65,
    ),
)
def test_clone_contract_rejects_non_structural_warning_codes(code):
    with pytest.raises(ValueError, match="warning code"):
        CloneWarning(code=code, count=1)


def test_clone_result_rejects_unbounded_warnings():
    with pytest.raises(ValueError, match="at most 8"):
        WorkspaceCloneResult(
            workspace_id="target",
            name="Copy",
            outcome="partial",
            publication_confirmed=False,
            counts=CloneCopyCounts.empty(),
            readiness=CloneRetrievalReadiness("ready", "ready", "needs_indexing"),
            warnings=tuple(CloneWarning(code=f"w{i}", count=1) for i in range(9)),
        )


def test_clone_contract_allows_complete_before_publication_confirmation():
    result = WorkspaceCloneResult(
        workspace_id="target",
        name="Copy",
        outcome="complete",
        publication_confirmed=False,
        counts=CloneCopyCounts.empty(),
        readiness=CloneRetrievalReadiness("ready", "ready", "needs_indexing"),
    )

    assert result.publication_confirmed is False
    with pytest.raises(FrozenInstanceError):
        result.publication_confirmed = True


@pytest.mark.parametrize("publication_confirmed", (False, True))
def test_clone_contract_allows_partial_with_either_publication_state(
    publication_confirmed,
):
    result = WorkspaceCloneResult(
        workspace_id="target",
        name="Copy",
        outcome="partial",
        publication_confirmed=publication_confirmed,
        counts=CloneCopyCounts.empty(),
        readiness=CloneRetrievalReadiness("ready", "ready", "needs_indexing"),
    )

    assert result.publication_confirmed is publication_confirmed


def _request() -> WorkspaceCloneRequest:
    return WorkspaceCloneRequest(
        source_workspace_id="source-ws",
        target_workspace_id="target-ws",
        operation_id="operation-1",
        request_fingerprint="fingerprint-1",
        name="Workspace Copy",
    )


def _workspace_snapshot(
    *,
    workspace: dict | None = None,
    sources: list[dict] | None = None,
    notes: list[dict] | None = None,
    artifacts: list[dict] | None = None,
    memberships: list[dict] | None = None,
) -> WorkspaceCloneSnapshot:
    return WorkspaceCloneSnapshot.from_rows(
        workspace=workspace
        or {
            "id": "source-ws",
            "name": "Source Workspace",
            "description": "Source description",
            "workspace_profile": "research",
        },
        sources=sources or [],
        notes=notes or [],
        artifacts=artifacts or [],
        memberships=memberships or [],
    )


def _media_snapshot(media_id: int, *, chunks: bool = True) -> MediaCloneSnapshot:
    return MediaCloneSnapshot.from_rows(
        media={
            "id": media_id,
            "uuid": f"source-media-{media_id}",
            "url": f"https://source.invalid/{media_id}",
            "title": f"Source media {media_id}",
            "type": "document",
            "content": f"content-{media_id}",
            "keywords": ("alpha", "beta"),
        },
        chunks=[{"text": f"chunk-{media_id}", "chunk_index": 0}] if chunks else [],
        transcripts=[],
    )


def _make_service(
    snapshot: WorkspaceCloneSnapshot | None = None,
    *,
    media_snapshots: dict[int, MediaCloneSnapshot] | None = None,
    vector_retrieval_configured: bool = False,
) -> tuple[CloneService, MagicMock, MagicMock, MagicMock, MagicMock]:
    source_chacha = MagicMock()
    source_media = MagicMock()
    target_chacha = MagicMock()
    target_media = MagicMock()
    snapshot = snapshot or _workspace_snapshot()
    media_snapshots = media_snapshots or {}

    source_chacha.read_workspace_clone_snapshot.return_value = snapshot
    source_media.read_media_clone_snapshots.side_effect = lambda media_ids: {
        media_id: media_snapshots[media_id] for media_id in media_ids
    }
    target_chacha.reserve_clone_target.return_value = {
        "id": "target-ws",
        "system_operation_state": "staged",
    }

    def add_source(workspace_id: str, data: dict) -> dict:
        return {
            **data,
            "workspace_id": workspace_id,
            "reviewed_at": None,
            "reviewed_by_user_id": None,
        }

    target_chacha.add_workspace_source.side_effect = add_source
    target_chacha.get_workspace_source.return_value = None
    next_note_id = iter(range(1001, 2000))
    target_chacha.add_workspace_note.side_effect = lambda workspace_id, data: {
        **data,
        "workspace_id": workspace_id,
        "id": next(next_note_id),
    }
    target_chacha.add_workspace_artifact.side_effect = lambda workspace_id, data: {
        **data,
        "workspace_id": workspace_id,
    }
    target_chacha.add_workspace_resource_membership.side_effect = (
        lambda workspace_id, data: {**data, "workspace_id": workspace_id}
    )
    target_chacha.publish_clone_target.return_value = {
        "id": "target-ws",
        "system_operation_state": "publication_pending",
    }

    def insert_media(
        *,
        snapshot: MediaCloneSnapshot,
        operation_id: str,
        source_identity: str,
        expected_content_hash: str,
    ) -> OperationOwnedMediaResult:
        del operation_id, source_identity, expected_content_hash
        media_id = int(snapshot.media["id"])
        return OperationOwnedMediaResult(
            media_id=1000 + media_id,
            media_uuid=f"target-media-{media_id}",
            created=True,
            replayed=False,
        )

    target_media.insert_operation_owned_clone_media.side_effect = insert_media
    target_media.delete_operation_owned_clone_media.return_value = 1
    target_chacha.discard_clone_target.return_value = True

    service = CloneService(
        source_chacha_db=source_chacha,
        source_media_db=source_media,
        target_chacha_db=target_chacha,
        target_media_db=target_media,
        vector_retrieval_configured=vector_retrieval_configured,
    )
    return service, source_chacha, source_media, target_chacha, target_media


def _warning_counts(result: WorkspaceCloneResult) -> dict[str, int]:
    return {warning.code: warning.count for warning in result.warnings}


def _assert_persistence_error(
    exc: BaseException,
    *,
    code: str,
    cleanup_state: str,
) -> None:
    assert type(exc).__name__ == "ClonePersistenceError"
    assert exc.code == code
    assert exc.cleanup_state == cleanup_state
    assert str(exc) == code


def test_clone_persistence_error_contract_is_bounded():
    error_type = getattr(clone_models_module, "ClonePersistenceError", None)

    assert error_type is not None
    error = error_type(code="clone_publication_failed", cleanup_state="pending")
    assert error.code == "clone_publication_failed"
    assert error.cleanup_state == "pending"
    with pytest.raises(ValueError, match="code"):
        error_type(code="/tmp/private: raw failure", cleanup_state="complete")


def test_clone_reads_stable_unique_snapshots_before_deterministic_reservation():
    events: list[object] = []
    snapshot = _workspace_snapshot(
        workspace={
            "id": "source-ws",
            "description": "Keep this description",
            "workspace_profile": "analysis",
        },
        sources=[
            {
                "id": "source-2",
                "media_id": "7",
                "position": 20,
                "selected": 0,
                "title": "Second",
                "source_type": "media",
                "url": "https://source.invalid/7",
            },
            {
                "id": "source-1",
                "media_id": 9,
                "position": 30,
                "selected": 1,
                "title": "Third",
                "source_type": "media",
                "url": "https://source.invalid/9",
            },
        ],
        memberships=[
            {"resource_type": "media", "resource_id": "7", "role": "context"},
            {"resource_type": "media", "resource_id": "11", "role": "context"},
        ],
    )
    service, source_chacha, source_media, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={
            7: _media_snapshot(7),
            9: _media_snapshot(9),
            11: _media_snapshot(11),
        },
    )
    source_chacha.read_workspace_clone_snapshot.side_effect = lambda workspace_id: (
        events.append(("workspace_snapshot", workspace_id)) or snapshot
    )

    def read_media(media_ids):
        events.append(("media_snapshots", tuple(media_ids)))
        return {
            media_id: {7: _media_snapshot(7), 9: _media_snapshot(9), 11: _media_snapshot(11)}[
                media_id
            ]
            for media_id in media_ids
        }

    source_media.read_media_clone_snapshots.side_effect = read_media

    def reserve(**kwargs):
        events.append(("reserve", kwargs["workspace_id"]))
        return {"id": kwargs["workspace_id"], "system_operation_state": "staged"}

    target_chacha.reserve_clone_target.side_effect = reserve

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert events[:3] == [
        ("workspace_snapshot", "source-ws"),
        ("media_snapshots", (7, 9, 11)),
        ("reserve", "target-ws"),
    ]
    source_chacha.read_workspace_clone_snapshot.assert_called_once_with("source-ws")
    source_media.read_media_clone_snapshots.assert_called_once_with((7, 9, 11))
    target_chacha.reserve_clone_target.assert_called_once_with(
        workspace_id="target-ws",
        operation_id="operation-1",
        request_fingerprint="fingerprint-1",
        name="Workspace Copy",
        description="Keep this description",
        workspace_profile="analysis",
    )
    assert [call.args[1]["id"] for call in target_chacha.add_workspace_source.call_args_list] == [
        "source-2",
        "source-1",
    ]
    assert [
        call.args[1]["media_id"] for call in target_chacha.add_workspace_source.call_args_list
    ] == [1007, 1009]
    assert [
        call.kwargs["source_identity"]
        for call in target_media.insert_operation_owned_clone_media.call_args_list
    ] == ["media:7", "media:9", "media:11"]
    target_chacha.create_workspace.assert_not_called()
    target_media.add_media_with_keywords.assert_not_called()
    target_chacha.confirm_clone_target_publication.assert_not_called()
    target_media.confirm_operation_owned_clone_media.assert_not_called()
    target_chacha.publish_clone_target.assert_called_once_with(
        workspace_id="target-ws",
        operation_id="operation-1",
    )
    assert result.workspace_id == "target-ws"
    assert result.publication_confirmed is False
    assert result.counts == CloneCopyCounts(
        sources_attempted=2,
        sources_copied=2,
        sources_failed=0,
        media_attempted=3,
        media_copied=3,
        media_failed=0,
        operation_owned_media_count=3,
    )


@pytest.mark.parametrize("bad_reference", (-1, "-2", "+3", "secret://media", 1.5, True))
def test_clone_rejects_malformed_nonzero_media_references_before_writes(bad_reference):
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": bad_reference, "source_type": "media"}]
    )
    service, _, source_media, target_chacha, target_media = _make_service(snapshot)

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_validation_failed",
        cleanup_state="complete",
    )
    source_media.read_media_clone_snapshots.assert_not_called()
    target_chacha.reserve_clone_target.assert_not_called()
    target_media.insert_operation_owned_clone_media.assert_not_called()


@pytest.mark.parametrize(
    "bad_reference",
    (None, False, True, 0, "0", "", " ", -1, "-2", "media:7", 1.5),
)
def test_clone_rejects_nonpositive_active_media_memberships_before_writes(bad_reference):
    snapshot = _workspace_snapshot(
        memberships=[
            {
                "resource_type": "media",
                "resource_id": bad_reference,
                "role": "context",
                "deleted": 0,
            }
        ]
    )
    service, _, source_media, target_chacha, target_media = _make_service(snapshot)

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_validation_failed",
        cleanup_state="complete",
    )
    source_media.read_media_clone_snapshots.assert_not_called()
    target_chacha.reserve_clone_target.assert_not_called()
    target_media.insert_operation_owned_clone_media.assert_not_called()
    target_chacha.add_workspace_resource_membership.assert_not_called()


def test_clone_snapshot_failure_is_controlled_and_precedes_reservation():
    service, source_chacha, source_media, target_chacha, _ = _make_service()
    source_chacha.read_workspace_clone_snapshot.side_effect = CloneSnapshotUnavailable(
        cleanup_state="complete"
    )

    with pytest.raises(CloneSnapshotUnavailable) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    assert exc_info.value.cleanup_state == "complete"
    source_media.read_media_clone_snapshots.assert_not_called()
    target_chacha.reserve_clone_target.assert_not_called()


@pytest.mark.parametrize(
    ("cancel_on_call", "expected_cleanup_media", "reservation_expected"),
    (
        (1, (), False),
        (2, (), False),
        (3, (), True),
        (4, (1,), True),
        (5, (1, 2), True),
        (6, (1, 2), True),
        (7, (1, 2), True),
        (8, (1, 2), True),
        (9, (1, 2), True),
    ),
)
def test_clone_cancels_at_every_required_boundary_with_exact_cleanup(
    cancel_on_call,
    expected_cleanup_media,
    reservation_expected,
):
    snapshot = _workspace_snapshot(
        sources=[
            {"id": "source-1", "media_id": 1, "source_type": "media"},
            {"id": "source-2", "media_id": 2, "source_type": "media"},
        ],
        notes=[
            {"id": 1, "title": "One", "content": "one", "keywords_json": "[]"},
            {"id": 2, "title": "Two", "content": "two", "keywords_json": "[]"},
        ],
        artifacts=[
            {"id": "artifact-1", "artifact_type": "text", "title": "One"},
            {"id": "artifact-2", "artifact_type": "text", "title": "Two"},
        ],
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={1: _media_snapshot(1), 2: _media_snapshot(2)},
    )
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks == cancel_on_call

    with pytest.raises(CloneCancelled) as exc_info:
        service.clone_workspace(_request(), should_cancel=should_cancel)

    assert checks == cancel_on_call
    assert exc_info.value.cleanup_state == "complete"
    assert [
        call.kwargs["source_identity"]
        for call in target_media.delete_operation_owned_clone_media.call_args_list
    ] == [f"media:{media_id}" for media_id in expected_cleanup_media]
    if reservation_expected:
        target_chacha.discard_clone_target.assert_called_once_with(
            workspace_id="target-ws",
            operation_id="operation-1",
        )
    else:
        target_chacha.discard_clone_target.assert_not_called()
    target_chacha.publish_clone_target.assert_not_called()


def test_duplicate_media_references_converge_to_one_target_media():
    snapshot = _workspace_snapshot(
        sources=[
            {"id": "source-1", "media_id": 7, "source_type": "media"},
            {"id": "source-2", "media_id": "7", "source_type": "media"},
        ],
        memberships=[
            {
                "resource_type": "media",
                "resource_id": "7",
                "role": "context",
                "label": "Primary",
            }
        ],
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    target_media.insert_operation_owned_clone_media.assert_called_once()
    assert [
        call.args[1]["media_id"] for call in target_chacha.add_workspace_source.call_args_list
    ] == [1007, 1007]
    membership = target_chacha.add_workspace_resource_membership.call_args.args[1]
    assert membership["resource_type"] == "media"
    assert membership["resource_id"] == "1007"
    assert result.counts.media_attempted == 1
    assert result.counts.media_copied == 1
    assert result.counts.operation_owned_media_count == 1


def test_source_link_failure_deletes_unreferenced_media_and_later_source_recreates_it():
    snapshot = _workspace_snapshot(
        sources=[
            {"id": "source-1", "media_id": 7, "source_type": "media"},
            {"id": "source-2", "media_id": 7, "source_type": "media"},
        ]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_media.insert_operation_owned_clone_media.side_effect = (
        OperationOwnedMediaResult(101, "target-101", created=True, replayed=False),
        OperationOwnedMediaResult(202, "target-202", created=True, replayed=False),
    )
    add_attempt = 0

    def add_source(workspace_id: str, data: dict) -> dict:
        nonlocal add_attempt
        add_attempt += 1
        if add_attempt == 1:
            raise RuntimeError("first source insert failed")
        return {
            **data,
            "workspace_id": workspace_id,
            "reviewed_at": None,
            "reviewed_by_user_id": None,
        }

    target_chacha.add_workspace_source.side_effect = add_source
    target_chacha.get_workspace_source.return_value = None

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert target_media.insert_operation_owned_clone_media.call_count == 2
    target_media.delete_operation_owned_clone_media.assert_called_once_with(
        operation_id="operation-1",
        source_identity="media:7",
        expected_content_hash=target_media.insert_operation_owned_clone_media.call_args_list[0].kwargs[
            "expected_content_hash"
        ],
    )
    assert target_chacha.add_workspace_source.call_args_list[1].args[1]["media_id"] == 202
    assert result.counts.sources_copied == 1
    assert result.counts.sources_failed == 1
    assert result.counts.media_copied == 1
    assert result.counts.media_failed == 0
    assert result.counts.operation_owned_media_count == 1
    assert _warning_counts(result)["source_copy_failed"] == 1


def test_source_response_loss_accepts_exact_persisted_row_without_deleting_media():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_chacha.add_workspace_source.side_effect = RuntimeError("response lost")
    target_chacha.get_workspace_source.return_value = {
        "id": "source-1",
        "workspace_id": "target-ws",
        "media_id": 1007,
        "source_type": "media",
        "title": "",
        "url": None,
        "position": 0,
        "selected": 1,
        "review_state": "unset",
        "reviewed_at": None,
        "reviewed_by_user_id": None,
    }

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    target_chacha.get_workspace_source.assert_called_once_with("target-ws", "source-1")
    target_media.delete_operation_owned_clone_media.assert_not_called()
    target_chacha.publish_clone_target.assert_called_once()
    assert result.outcome == "complete"
    assert result.counts.sources_copied == 1
    assert result.counts.sources_failed == 0
    assert result.counts.media_copied == 1
    assert result.warnings == ()


def test_source_response_loss_with_failed_lookup_fails_closed():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_chacha.add_workspace_source.side_effect = RuntimeError("response lost")
    target_chacha.get_workspace_source.side_effect = RuntimeError("lookup failed")

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_validation_failed",
        cleanup_state="complete",
    )
    target_media.delete_operation_owned_clone_media.assert_called_once()
    target_chacha.discard_clone_target.assert_called_once()
    target_chacha.publish_clone_target.assert_not_called()


def test_source_response_loss_with_mismatched_row_fails_closed():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_chacha.add_workspace_source.side_effect = RuntimeError("response lost")
    target_chacha.get_workspace_source.return_value = {
        "id": "source-1",
        "workspace_id": "target-ws",
        "media_id": 9999,
        "source_type": "media",
        "title": "",
        "url": None,
        "position": 0,
        "selected": 1,
        "review_state": "unset",
        "reviewed_at": None,
        "reviewed_by_user_id": None,
    }

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_validation_failed",
        cleanup_state="complete",
    )
    target_media.delete_operation_owned_clone_media.assert_called_once()
    target_chacha.discard_clone_target.assert_called_once()
    target_chacha.publish_clone_target.assert_not_called()


def test_replayed_media_counts_as_copied_but_not_newly_operation_owned():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, _, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_media.insert_operation_owned_clone_media.return_value = OperationOwnedMediaResult(
        707,
        "target-707",
        created=False,
        replayed=True,
    )
    target_media.insert_operation_owned_clone_media.side_effect = None

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert result.counts.media_attempted == 1
    assert result.counts.media_copied == 1
    assert result.counts.media_failed == 0
    assert result.counts.operation_owned_media_count == 0


def test_media_response_loss_retries_exact_insert_and_accepts_replay():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, _, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_media.insert_operation_owned_clone_media.side_effect = (
        RuntimeError("response lost"),
        OperationOwnedMediaResult(707, "target-707", created=False, replayed=True),
    )

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert target_media.insert_operation_owned_clone_media.call_count == 2
    assert (
        target_media.insert_operation_owned_clone_media.call_args_list[0]
        == target_media.insert_operation_owned_clone_media.call_args_list[1]
    )
    target_media.delete_operation_owned_clone_media.assert_not_called()
    assert result.outcome == "complete"
    assert result.counts.media_copied == 1
    assert result.counts.media_failed == 0
    assert result.counts.operation_owned_media_count == 0


@pytest.mark.parametrize("deleted", (0, 1))
def test_unresolved_media_write_exact_cleanup_becomes_partial(deleted):
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_media.insert_operation_owned_clone_media.side_effect = RuntimeError("write ambiguous")
    target_media.delete_operation_owned_clone_media.return_value = deleted

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert target_media.insert_operation_owned_clone_media.call_count == 2
    target_media.delete_operation_owned_clone_media.assert_called_once()
    target_chacha.add_workspace_source.assert_not_called()
    target_chacha.publish_clone_target.assert_called_once()
    assert result.outcome == "partial"
    assert result.counts.media_attempted == 1
    assert result.counts.media_copied == 0
    assert result.counts.media_failed == 1
    assert result.counts.operation_owned_media_count == 0
    assert _warning_counts(result) == {
        "media_copy_failed": 1,
        "source_copy_failed": 1,
    }


@pytest.mark.parametrize("cleanup_outcome", (RuntimeError("cleanup failed"), 2))
def test_unresolved_media_write_with_ambiguous_cleanup_fails_pending(cleanup_outcome):
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_media.insert_operation_owned_clone_media.side_effect = RuntimeError("write ambiguous")
    if isinstance(cleanup_outcome, BaseException):
        target_media.delete_operation_owned_clone_media.side_effect = cleanup_outcome
    else:
        target_media.delete_operation_owned_clone_media.return_value = cleanup_outcome

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_cleanup_incomplete",
        cleanup_state="pending",
    )
    assert target_media.insert_operation_owned_clone_media.call_count == 2
    target_media.delete_operation_owned_clone_media.assert_called_once()
    target_chacha.discard_clone_target.assert_called_once()
    target_chacha.publish_clone_target.assert_not_called()


def test_memberships_map_supported_resources_reset_review_and_aggregate_skips():
    snapshot = _workspace_snapshot(
        sources=[
            {
                "id": "source-1",
                "media_id": 7,
                "source_type": "media",
                "title": "Source title",
                "url": "https://source.invalid/7",
                "position": 4,
                "selected": 0,
                "review_state": "reviewed",
                "reviewed_at": "2026-08-25T00:00:00Z",
                "reviewed_by_user_id": "owner-secret",
            }
        ],
        notes=[
            {
                "id": 5,
                "title": "Note",
                "content": "Body",
                "keywords_json": '["one", "two"]',
            }
        ],
        artifacts=[
            {
                "id": "artifact-1",
                "artifact_type": "report",
                "title": "Artifact",
                "status": "completed",
                "content": "Artifact body",
                "content_type": "text/markdown",
                "preview_text": "Preview",
                "summary": "Summary",
                "review_state": "accepted",
                "producer_metadata": {"provider": "local"},
                "source_lineage": {"source_ids": ["source-1"]},
                "review_metadata": {"reviewer": "owner-secret"},
                "version_metadata": {"kind": "final"},
                "export_refs": ["export-1"],
                "redaction": {"support_safe": True, "redacted": False},
                "schema_version": 2,
                "owner_id": "owner-secret",
            }
        ],
        memberships=[
            {"resource_type": "media", "resource_id": "7", "role": "context", "label": "M"},
            {
                "resource_type": "workspace_source",
                "resource_id": "source-1",
                "role": "evidence",
                "label": "S",
            },
            {"resource_type": "workspace_note", "resource_id": "5", "role": "annotation"},
            {
                "resource_type": "workspace_artifact",
                "resource_id": "artifact-1",
                "role": "output",
            },
            {"resource_type": "conversation", "resource_id": "conversation-1", "role": "chat"},
            {"resource_type": "workspace_note", "resource_id": "999", "role": "annotation"},
        ],
    )
    service, _, _, target_chacha, _ = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_chacha.add_workspace_note.side_effect = lambda workspace_id, data: {
        **data,
        "workspace_id": workspace_id,
        "id": 105,
    }

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    source_payload = target_chacha.add_workspace_source.call_args.args[1]
    assert source_payload == {
        "id": "source-1",
        "media_id": 1007,
        "source_type": "media",
        "title": "Source title",
        "url": "https://source.invalid/7",
        "position": 4,
        "selected": False,
        "review_state": "unset",
    }
    note_payload = target_chacha.add_workspace_note.call_args.args[1]
    assert note_payload == {"title": "Note", "content": "Body", "keywords": ["one", "two"]}
    artifact_payload = target_chacha.add_workspace_artifact.call_args.args[1]
    assert artifact_payload == {
        "id": "artifact-1",
        "artifact_type": "report",
        "title": "Artifact",
        "status": "completed",
        "content": "Artifact body",
        "content_type": "text/markdown",
        "preview_text": "Preview",
        "summary": "Summary",
        "review_state": "accepted",
        "producer_metadata": {"provider": "local"},
        "source_lineage": {"source_ids": ["source-1"]},
        "version_metadata": {"kind": "final"},
        "export_refs": ["export-1"],
        "redaction": {"support_safe": True, "redacted": False},
        "schema_version": 2,
    }
    membership_payloads = [
        call.args[1] for call in target_chacha.add_workspace_resource_membership.call_args_list
    ]
    assert [(item["resource_type"], item["resource_id"]) for item in membership_payloads] == [
        ("media", "1007"),
        ("workspace_source", "source-1"),
        ("workspace_note", "105"),
        ("workspace_artifact", "artifact-1"),
    ]
    for item in membership_payloads:
        assert item["transfer_policy"] == "copy"
        assert item["provenance"] == {
            "kind": "shared_workspace_clone",
            "operation_id": "operation-1",
            "source_workspace_id": "source-ws",
        }
        assert item["metadata"] == {}
    assert result.outcome == "partial"
    assert _warning_counts(result) == {"membership_skipped": 2}


def test_item_failures_aggregate_stable_warnings_and_truthful_final_counts():
    snapshot = _workspace_snapshot(
        sources=[
            {"id": "media-source", "media_id": 7, "source_type": "media"},
            {"id": "web-source", "media_id": 0, "source_type": "web"},
        ],
        notes=[
            {"id": 1, "title": "One", "content": "one", "keywords_json": "[]"},
            {"id": 2, "title": "Two", "content": "two", "keywords_json": "[]"},
            {"id": 3, "title": "Three", "content": "three", "keywords_json": "[]"},
        ],
        artifacts=[{"id": "artifact-1", "artifact_type": "text", "title": "Artifact"}],
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_media.insert_operation_owned_clone_media.side_effect = RuntimeError("media failed")
    target_chacha.add_workspace_note.side_effect = RuntimeError("note failed")
    target_chacha.add_workspace_artifact.side_effect = RuntimeError("artifact failed")

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert result.outcome == "partial"
    assert result.counts == CloneCopyCounts(
        sources_attempted=2,
        sources_copied=1,
        sources_failed=1,
        notes_attempted=3,
        notes_copied=0,
        notes_failed=3,
        artifacts_attempted=1,
        artifacts_copied=0,
        artifacts_failed=1,
        media_attempted=1,
        media_copied=0,
        media_failed=1,
        operation_owned_media_count=0,
    )
    assert _warning_counts(result) == {
        "media_copy_failed": 1,
        "source_copy_failed": 1,
        "note_copy_failed": 3,
        "artifact_copy_failed": 1,
    }
    assert len(result.warnings) <= 8


@pytest.mark.parametrize(
    ("configured", "expected_vector", "expected_outcome", "expected_warning"),
    (
        (False, "not_configured", "complete", None),
        (True, "needs_indexing", "partial", "vector_index_not_generated"),
    ),
)
def test_vector_readiness_uses_explicit_deployment_configuration(
    configured,
    expected_vector,
    expected_outcome,
    expected_warning,
):
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, _, _ = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
        vector_retrieval_configured=configured,
    )

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert result.readiness == CloneRetrievalReadiness("ready", "ready", expected_vector)
    assert result.outcome == expected_outcome
    warning_codes = [warning.code for warning in result.warnings]
    if expected_warning is None:
        assert warning_codes == []
    else:
        assert warning_codes.count(expected_warning) == 1


def test_configured_vectors_are_ready_without_retained_media_and_need_no_warning():
    service, _, source_media, _, _ = _make_service(
        vector_retrieval_configured=True,
    )

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    source_media.read_media_clone_snapshots.assert_called_once_with(())
    assert result.readiness == CloneRetrievalReadiness("unavailable", "unavailable", "ready")
    assert result.outcome == "complete"
    assert result.warnings == ()


def test_text_and_citations_require_a_copied_source_with_copied_chunks():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}],
        memberships=[{"resource_type": "media", "resource_id": "8", "role": "context"}],
    )
    service, _, _, _, _ = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7, chunks=False), 8: _media_snapshot(8)},
    )

    result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert result.counts.media_copied == 2
    assert result.readiness.text_search == "unavailable"
    assert result.readiness.citations == "unavailable"


def test_same_operation_different_fingerprint_conflict_preserves_existing_target():
    service, _, _, target_chacha, target_media = _make_service()
    target_chacha.reserve_clone_target.side_effect = RuntimeError(
        "same operation different fingerprint"
    )

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_reservation_failed",
        cleanup_state="pending",
    )
    target_chacha.discard_clone_target.assert_not_called()
    target_media.delete_operation_owned_clone_media.assert_not_called()


def test_exact_publication_pending_reservation_exits_without_cleanup_or_copy():
    progress: list[tuple[str, float]] = []
    service, _, _, target_chacha, target_media = _make_service()
    target_chacha.reserve_clone_target.return_value = {
        "id": "target-ws",
        "system_operation_state": "publication_pending",
        "system_operation_id": "operation-1",
        "system_request_fingerprint": "fingerprint-1",
        "name": "Workspace Copy",
        "description": "Source description",
        "workspace_profile": "research",
    }

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(
            _request(),
            should_cancel=lambda: False,
            on_progress=lambda phase, fraction: progress.append((phase, fraction)),
        )

    _assert_persistence_error(
        exc_info.value,
        code="clone_publication_pending",
        cleanup_state="pending",
    )
    assert progress == [
        ("queued", 0.0),
        ("authorizing", 0.05),
        ("preparing", 0.1),
        ("preparing", 0.2),
    ]
    target_media.insert_operation_owned_clone_media.assert_not_called()
    target_media.delete_operation_owned_clone_media.assert_not_called()
    target_chacha.add_workspace_source.assert_not_called()
    target_chacha.publish_clone_target.assert_not_called()
    target_chacha.discard_clone_target.assert_not_called()


@pytest.mark.parametrize(
    ("field_name", "wrong_value"),
    (
        ("system_operation_id", "operation-other"),
        ("system_request_fingerprint", "fingerprint-other"),
        ("name", "Different Copy"),
        ("description", "Different description"),
        ("workspace_profile", "analysis"),
    ),
)
def test_mismatched_returned_reservation_fields_preserve_ambiguous_target(
    field_name,
    wrong_value,
):
    service, _, _, target_chacha, target_media = _make_service()
    reservation = {
        "id": "target-ws",
        "system_operation_state": "staged",
        "system_operation_id": "operation-1",
        "system_request_fingerprint": "fingerprint-1",
        "name": "Workspace Copy",
        "description": "Source description",
        "workspace_profile": "research",
    }
    reservation[field_name] = wrong_value
    target_chacha.reserve_clone_target.return_value = reservation

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_validation_failed",
        cleanup_state="pending",
    )
    target_media.insert_operation_owned_clone_media.assert_not_called()
    target_media.delete_operation_owned_clone_media.assert_not_called()
    target_chacha.add_workspace_source.assert_not_called()
    target_chacha.publish_clone_target.assert_not_called()
    target_chacha.discard_clone_target.assert_not_called()


def test_source_validation_failure_is_fatal_and_cleans_exact_media():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_chacha.add_workspace_source.side_effect = lambda workspace_id, data: {
        **data,
        "workspace_id": workspace_id,
        "media_id": 999,
        "reviewed_by_user_id": None,
    }

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_validation_failed",
        cleanup_state="complete",
    )
    target_media.delete_operation_owned_clone_media.assert_called_once()
    target_chacha.discard_clone_target.assert_called_once()
    target_chacha.publish_clone_target.assert_not_called()


def test_invalid_media_persistence_result_is_exact_deleted_before_validation_failure():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_media.insert_operation_owned_clone_media.side_effect = None
    target_media.insert_operation_owned_clone_media.return_value = {
        "media_id": 1007,
        "created": True,
    }

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_validation_failed",
        cleanup_state="complete",
    )
    target_media.delete_operation_owned_clone_media.assert_called_once_with(
        operation_id="operation-1",
        source_identity="media:7",
        expected_content_hash=target_media.insert_operation_owned_clone_media.call_args.kwargs[
            "expected_content_hash"
        ],
    )
    target_chacha.discard_clone_target.assert_called_once()


def test_publication_failure_cleans_all_exact_media_and_discards_target():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_chacha.publish_clone_target.side_effect = RuntimeError("publication failed")

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_publication_failed",
        cleanup_state="complete",
    )
    target_media.delete_operation_owned_clone_media.assert_called_once_with(
        operation_id="operation-1",
        source_identity="media:7",
        expected_content_hash=target_media.insert_operation_owned_clone_media.call_args.kwargs[
            "expected_content_hash"
        ],
    )
    target_chacha.discard_clone_target.assert_called_once_with(
        workspace_id="target-ws",
        operation_id="operation-1",
    )
    target_chacha.confirm_clone_target_publication.assert_not_called()


def test_cleanup_failure_is_pending_and_every_tracked_identity_is_attempted():
    snapshot = _workspace_snapshot(
        sources=[
            {"id": "source-1", "media_id": 7, "source_type": "media"},
            {"id": "source-2", "media_id": 8, "source_type": "media"},
        ]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7), 8: _media_snapshot(8)},
    )
    target_chacha.publish_clone_target.side_effect = RuntimeError("publication failed")
    target_media.delete_operation_owned_clone_media.side_effect = (
        RuntimeError("first cleanup failed"),
        1,
    )

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_publication_failed",
        cleanup_state="pending",
    )
    assert [
        call.kwargs["source_identity"]
        for call in target_media.delete_operation_owned_clone_media.call_args_list
    ] == ["media:7", "media:8"]
    target_chacha.discard_clone_target.assert_called_once()


def test_immediate_source_cleanup_failure_becomes_fatal_and_never_publishes():
    snapshot = _workspace_snapshot(
        sources=[{"id": "source-1", "media_id": 7, "source_type": "media"}]
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_chacha.add_workspace_source.side_effect = RuntimeError("source insert failed")
    target_media.delete_operation_owned_clone_media.side_effect = RuntimeError("cleanup failed")

    with pytest.raises(Exception) as exc_info:
        service.clone_workspace(_request(), should_cancel=lambda: False)

    _assert_persistence_error(
        exc_info.value,
        code="clone_cleanup_incomplete",
        cleanup_state="pending",
    )
    assert target_media.delete_operation_owned_clone_media.call_count == 2
    target_chacha.discard_clone_target.assert_called_once()
    target_chacha.publish_clone_target.assert_not_called()


def test_progress_uses_only_stable_monotonic_content_free_phases():
    snapshot = _workspace_snapshot(
        workspace={
            "id": "source-ws",
            "description": "SECRET_TOKEN workspace description",
            "workspace_profile": "research",
        },
        sources=[
            {
                "id": "source-1",
                "media_id": 7,
                "source_type": "media",
                "title": "SECRET_TOKEN title",
                "url": "https://source.invalid/supersecret",
            }
        ],
        notes=[{"id": 1, "title": "password=bad", "content": "private", "keywords_json": "[]"}],
        artifacts=[{"id": "artifact-1", "title": "/tmp/private", "content": "private"}],
    )
    service, _, _, _, _ = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    progress: list[tuple[str, float]] = []

    service.clone_workspace(
        _request(),
        should_cancel=lambda: False,
        on_progress=lambda phase, fraction: progress.append((phase, fraction)),
    )

    allowed = {
        "queued",
        "authorizing",
        "preparing",
        "sources",
        "notes",
        "artifacts",
        "finalizing",
    }
    assert progress[0] == ("queued", 0.0)
    assert progress[-1] == ("finalizing", 1.0)
    assert {phase for phase, _ in progress} <= allowed
    fractions = [fraction for _, fraction in progress]
    assert fractions == sorted(fractions)
    assert all(0.0 <= fraction <= 1.0 for fraction in fractions)
    serialized = repr(progress)
    for secret in ("SECRET_TOKEN", "supersecret", "password=", "/tmp/private", "private"):
        assert secret not in serialized


def test_partial_failure_logs_use_only_bounded_ids_and_exception_classes():
    snapshot = _workspace_snapshot(
        sources=[
            {
                "id": "source-media",
                "media_id": 7,
                "source_type": "media",
                "title": "SECRET_TOKEN source title",
                "url": "sqlite:///tmp/private",
            },
            {
                "id": "source-web",
                "media_id": 0,
                "source_type": "web",
                "title": "password=bad",
            },
        ],
        notes=[{"id": 1, "title": "token=abc123", "content": "supersecret", "keywords_json": "[]"}],
        artifacts=[{"id": "artifact-1", "title": "/tmp/private", "content": "supersecret"}],
    )
    service, _, _, target_chacha, target_media = _make_service(
        snapshot,
        media_snapshots={7: _media_snapshot(7)},
    )
    target_media.insert_operation_owned_clone_media.side_effect = RuntimeError(
        "SECRET_TOKEN sqlite:///tmp/private"
    )
    target_chacha.add_workspace_source.side_effect = RuntimeError("password=bad")
    target_chacha.add_workspace_note.side_effect = RuntimeError("token=abc123")
    target_chacha.add_workspace_artifact.side_effect = RuntimeError("supersecret")

    with patch("tldw_Server_API.app.core.Sharing.clone_service.logger") as fake_logger:
        result = service.clone_workspace(_request(), should_cancel=lambda: False)

    assert result.outcome == "partial"
    logged = repr(fake_logger.mock_calls)
    assert "RuntimeError" in logged
    for secret in (
        "SECRET_TOKEN",
        "abc123",
        "supersecret",
        "password=",
        "token=",
        "sqlite://",
        "/tmp/private",
    ):
        assert secret not in logged


def test_constructor_rejects_non_boolean_vector_configuration():
    with pytest.raises(TypeError, match="vector_retrieval_configured"):
        CloneService(MagicMock(), MagicMock(), MagicMock(), MagicMock(), vector_retrieval_configured=1)
