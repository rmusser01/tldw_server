"""A device's delete of a chat or a note the server does not have is already satisfied (§5.4, Q7).

``test_sync_v2_satisfied_tombstone.py`` covers a ``chat.message`` tombstone. The
owner extended the same rule to ``chat.conversation`` and ``notes.note``
(decided 2026-10-19). Before it:

* a tombstone with no base was refused by the store contract ("tombstones
  require base metadata"), which raised out of ``push`` and failed the whole
  batch;
* the conversation and notes materializers rejected a tombstone of an object
  with no Sync state, as a ``missing_server_object`` conflict (with a base) or a
  *failed* envelope (without one, when no row carried the id).

The rule under test, for both domains:

* the tombstone is accepted and applied; no conflict is left and the dataset is
  not blocked;
* object state records the object as tombstoned, so a later create of the same
  id is refused at push, which blocks nothing;
* a copy the owner holds outside Sync is soft-deleted; another owner's row is
  never reached, and neither is a row another dataset of the owner holds;
* the tombstone is delivered to the other devices, which converge on "gone";
* a tombstone that disagrees with an object the dataset holds is still a
  conflict;
* a dataset already holding such a tombstone in conflict recovers on the next
  write, and a failed one on ``repair``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import (
    ChatConversationMaterializer,
    ChatMessageMaterializer,
    NotesMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    M1_SYNC_OPERATIONS,
    SyncConflictCreate,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import SERVER_ORIGIN_DEVICE_ID
from tldw_Server_API.app.core.Sync.v2.service import SyncPushResult, SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import _client, _ready_encryption

pytestmark = pytest.mark.integration

USER = "user-1"
GONE = "object-the-server-never-had"
PUSHER = "offline-device"
THIRD = "third-device"
OBJECT_DOMAINS = ("chat.conversation", "notes.note")
_STATE_DOMAINS = ["chat.conversation", "chat.message", "notes.note"]


class _NoopRateLimiter:
    async def check_user_rate_limit(self, user_id: int, endpoint: str, role: str = "user"):
        return True, {}


def _service(sync_database: SyncDatabase, projection_db: CharactersRAGDB) -> SyncV2Service:
    """An active profile whose chat and notes envelopes project into ``projection_db``."""
    registry = SyncAdapterRegistry(
        [
            NotesDomainAdapter() if domain == "notes.note" else StaticSyncAdapter(domain=domain, supported_adapter_versions={1})
            for domain in M1_SYNC_DOMAINS
        ]
    )
    ids = itertools.count(1)
    service = SyncV2Service(
        store=SyncV2Store(sync_database),
        adapters=registry,
        materializers={
            "chat.conversation": ChatConversationMaterializer(projection_db),
            "chat.message": ChatMessageMaterializer(projection_db),
            "notes.note": NotesMaterializer(projection_db),
        },
        clock=lambda: "2026-10-19T12:00:00+00:00",
        id_factory=lambda prefix: f"{prefix}-{next(ids)}",
        settings=SyncV2Settings(
            supported_domains=list(M1_SYNC_DOMAINS),
            operations={domain: list(operations) for domain, operations in M1_SYNC_OPERATIONS.items()},
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.bootstrap_profile(user_id=USER, mode="server_frontend", device_id="frontend-device", device_name="Server")
    service.register_device(user_id=USER, display_name="Offline laptop", client_type="chatbook", device_id=PUSHER)
    return service


@pytest.fixture()
def chacha_db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(db_path=str(tmp_path / "ChaChaNotes.db"), client_id=USER)
    try:
        yield database
    finally:
        database.close_all_connections()


@pytest.fixture()
def sync_service(tmp_path: Path, chacha_db: CharactersRAGDB) -> Iterator[SyncV2Service]:
    projection = CharactersRAGDB(db_path=chacha_db.db_path_str, client_id=USER)
    try:
        yield _service(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db"), projection)
    finally:
        projection.close_all_connections()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, chacha_db: CharactersRAGDB, sync_service: SyncV2Service) -> TestClient:
    return _client(monkeypatch, chacha_db, sync_service)


@pytest.fixture()
def notes_client(monkeypatch: pytest.MonkeyPatch, chacha_db: CharactersRAGDB, sync_service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(notes_endpoint.router, prefix="/api/v1/notes")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return User(id=USER, username=USER, is_admin=True)

    app.dependency_overrides[notes_endpoint.get_chacha_db_for_user] = _db_override
    app.dependency_overrides[notes_endpoint.get_request_user] = _user_override
    app.dependency_overrides[notes_endpoint.get_rate_limiter_dep] = lambda: _NoopRateLimiter()
    monkeypatch.setattr(
        notes_endpoint, "get_active_server_origin_sync_service_for_user", lambda user_id: sync_service, raising=False
    )
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(params=OBJECT_DOMAINS)
def domain(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture()
def third_device(sync_service: SyncV2Service) -> str:
    sync_service.register_device(user_id=USER, display_name="Phone", client_type="chatbook", device_id=THIRD)
    return THIRD


@pytest.fixture()
def replica(tmp_path: Path) -> Iterator[tuple[SyncV2Service, CharactersRAGDB]]:
    """Another replica that runs the same projection: its own Sync store and its own database."""
    database = CharactersRAGDB(db_path=str(tmp_path / "replica" / "ChaChaNotes.db"), client_id=USER)
    projection = CharactersRAGDB(db_path=database.db_path_str, client_id=USER)
    try:
        yield _service(SyncDatabase(sqlite_path=tmp_path / "replica" / "Sync_v2.db"), projection), database
    finally:
        projection.close_all_connections()
        database.close_all_connections()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dataset_id(service: SyncV2Service) -> str:
    return service.profile(user_id=USER).active_dataset_id or ""


_SEQUENCES: dict[tuple[str, str], int] = {}


def _client_sequence(device: str, envelope_id: str) -> int:
    """One device sequence per envelope id: a replay repeats it, and two envelopes never share one."""
    return _SEQUENCES.setdefault((device, envelope_id), len(_SEQUENCES) + 1)


def _payload(domain: str, object_id: str) -> dict[str, Any]:
    if domain == "chat.conversation":
        return {"title": f"{object_id} title"}
    return {"title": f"{object_id} title", "content": f"{object_id} body", "conversation_id": None, "message_id": None}


def _envelope(
    service: SyncV2Service,
    domain: str,
    operation: str,
    object_id: str = GONE,
    *,
    device: str = PUSHER,
    dataset_id: str | None = None,
    envelope_id: str | None = None,
    **overrides: Any,
) -> SyncEnvelopeCreate:
    deleted = operation == "tombstone"
    envelope_id = envelope_id or f"env-{device}-{domain}-{operation}-{object_id}"
    values: dict[str, Any] = {
        "dataset_id": dataset_id or _dataset_id(service),
        "client_envelope_id": envelope_id,
        "domain": domain,
        "operation": operation,
        "object_id": object_id,
        "device_id": device,
        "client_sequence": _client_sequence(device, envelope_id),
        "object_revision": 1,
        "payload": {"id": object_id, "deleted": True} if deleted else _payload(domain, object_id),
        "payload_hash": f"sha256:{device}-{domain}-{operation}-{object_id}",
        "deleted": deleted,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    values.update(overrides)
    return SyncEnvelopeCreate(**values)


def _push(service: SyncV2Service, envelope: SyncEnvelopeCreate) -> SyncPushResult:
    return service.push(
        user_id=USER, dataset_id=envelope.dataset_id, device_id=envelope.device_id or PUSHER, envelopes=[envelope]
    )


def _push_tombstone(service: SyncV2Service, domain: str, object_id: str = GONE, **overrides: Any) -> SyncPushResult:
    return _push(service, _envelope(service, domain, "tombstone", object_id, **overrides))


def _state(service: SyncV2Service, domain: str, object_id: str = GONE, dataset_id: str | None = None):
    return service.store.get_object_state(dataset_id or _dataset_id(service), domain, object_id)


def _blocker(service: SyncV2Service):
    return service.store.get_unresolved_materialization_conflict(_dataset_id(service))


def _log(service: SyncV2Service) -> list[tuple[str, str, str, str]]:
    envelopes = service.store.list_envelopes_after(_dataset_id(service), 0, domains=_STATE_DOMAINS, limit=200)
    return [(item.domain, item.operation, item.object_id, item.apply_status) for item in envelopes]


def _pull(service: SyncV2Service, device_id: str = THIRD) -> list[SyncEnvelope]:
    return service.pull(
        user_id=USER, dataset_id=_dataset_id(service), device_id=device_id, cursor="0", domains=_STATE_DOMAINS
    ).envelopes


def _row(db: CharactersRAGDB, domain: str, object_id: str = GONE) -> dict[str, Any] | None:
    if domain == "chat.conversation":
        return db.get_conversation_by_id(object_id, include_deleted=True)
    return db.get_note_by_id(object_id, include_deleted=True)


def _is_deleted(db: CharactersRAGDB, domain: str, object_id: str = GONE) -> bool:
    row = _row(db, domain, object_id)
    assert row is not None, f"{domain} {object_id} has no row"
    return bool(row["deleted"])


def _copy_outside_sync(db: CharactersRAGDB, domain: str, object_id: str = GONE) -> None:
    """Write the row the way a path that publishes nothing does: owner row, no envelope, no state."""
    if domain == "chat.conversation":
        db.add_conversation({"id": object_id, "character_id": None, "title": "Before the profile", "client_id": USER})
    else:
        db.add_note(title="Before the profile", content="kept outside Sync", note_id=object_id)


def _another_owners_row(db: CharactersRAGDB, domain: str, object_id: str) -> None:
    with db.transaction() as conn:
        if domain == "chat.conversation":
            conn.execute(
                "INSERT INTO conversations (id, root_id, title, client_id) VALUES (?, ?, ?, ?)",
                (object_id, object_id, "Theirs", "someone-else"),
            )
        else:
            conn.execute(
                "INSERT INTO notes (id, title, content, client_id) VALUES (?, ?, ?, ?)",
                (object_id, "Theirs", "not yours to delete", "someone-else"),
            )


def _held(service: SyncV2Service, domain: str, object_id: str) -> SyncEnvelope:
    """An object the dataset holds: created by a device push and projected."""
    pushed = _push(service, _envelope(service, domain, "upsert", object_id, envelope_id=f"env-create-{object_id}"))
    assert [item.apply_status for item in pushed.accepted] == ["applied"], (pushed.rejected, pushed.conflicts)
    return service.store.get_current_head(_dataset_id(service), domain, object_id)


def _deliver(replica_service: SyncV2Service, envelopes: list[SyncEnvelope]) -> SyncPushResult:
    """Hand pulled envelopes to the replica, which projects them under its own rules."""
    dataset_id = _dataset_id(replica_service)
    return replica_service.push(
        user_id=USER,
        dataset_id=dataset_id,
        device_id=PUSHER,
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id=f"replica-{item.client_envelope_id}",
                domain=item.domain,
                operation=item.operation,
                object_id=item.object_id,
                device_id=PUSHER,
                client_sequence=index,
                object_revision=item.object_revision,
                parent_id=item.parent_id,
                payload=dict(item.payload),
                payload_hash=item.payload_hash,
                deleted=item.deleted,
                encryption_metadata={"policy": "server_trusted_v1"},
            )
            for index, item in enumerate(envelopes, start=1)
        ],
    )


def _strand(
    service: SyncV2Service,
    domain: str,
    object_id: str = GONE,
    *,
    device: str = PUSHER,
    with_base: bool = False,
) -> SyncEnvelope:
    """Leave the tombstone accepted with its projection in conflict, as the old materializers did.

    With ``with_base`` it is the shape they produced: a tombstone based on a head
    whose projection never happened (here an upsert a reviewer skipped), which
    they rejected as ``missing_server_object`` because no object state exists.
    """
    dataset_id = _dataset_id(service)
    overrides: dict[str, Any] = {}
    if with_base:
        head = service.store.insert_envelope(
            replace(_envelope(service, domain, "upsert", object_id, envelope_id="env-never-projected"), status="accepted")
        )
        service.store.mark_envelope_apply_status(head.server_cursor, apply_status="superseded")
        overrides = {
            "object_revision": 2,
            "base_server_cursor": head.server_cursor,
            "base_object_revision": head.object_revision,
            "base_object_hash": head.payload_hash,
        }
    stranded = service.store.insert_envelope(
        replace(_envelope(service, domain, "tombstone", object_id, device=device, **overrides), status="accepted")
    )
    stranded = service.store.mark_envelope_apply_status(
        stranded.server_cursor,
        apply_status="conflict",
        apply_error_code="whole_object_conflict",
        apply_error_message=f"{domain} base state does not match the current server projection",
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id=f"conflict-stranded-{domain}",
            dataset_id=dataset_id,
            domain=domain,
            entity_id=object_id,
            conflict_type="whole_object_conflict",
            local_envelope_id=stranded.client_envelope_id,
            server_sequence=stranded.server_cursor,
            metadata={"reason": "missing_server_object"},
        )
    )
    assert _blocker(service).conflict_id == f"conflict-stranded-{domain}"
    return stranded


def _block_dataset(service: SyncV2Service) -> None:
    """An unrelated projection conflict that only a reviewer may resolve."""
    dataset_id = _dataset_id(service)
    blocked = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-unresolved",
            domain="chat.message",
            operation="append",
            object_id="device-message",
            device_id=PUSHER,
            client_sequence=_client_sequence(PUSHER, "env-unresolved"),
            object_revision=1,
            payload={"conversation_id": "elsewhere", "sender": "user", "content": "Never projected"},
            payload_hash="sha256:unresolved",
            encryption_metadata={"policy": "server_trusted_v1"},
            status="accepted",
        )
    )
    blocked = service.store.mark_envelope_apply_status(
        blocked.server_cursor, apply_status="conflict", apply_error_code="message_stable_id_conflict"
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-unresolved",
            dataset_id=dataset_id,
            domain=blocked.domain,
            entity_id=blocked.object_id,
            conflict_type="message_stable_id_conflict",
            local_envelope_id=blocked.client_envelope_id,
            server_sequence=blocked.server_cursor,
        )
    )


# ---------------------------------------------------------------------------
# The store contract
# ---------------------------------------------------------------------------


def test_the_store_accepts_a_tombstone_without_a_base_for_a_chat_and_a_note(
    sync_service: SyncV2Service, domain: str
) -> None:
    stored = sync_service.store.insert_envelope(
        replace(_envelope(sync_service, domain, "tombstone"), status="accepted")
    )

    assert (stored.operation, stored.base_server_cursor, stored.base_object_revision) == ("tombstone", None, None)


def test_the_store_still_refuses_a_partial_base_and_an_update_without_one(
    sync_service: SyncV2Service, domain: str
) -> None:
    with pytest.raises(SyncStoreError, match="complete set"):
        sync_service.store.insert_envelope(
            replace(_envelope(sync_service, domain, "tombstone", base_server_cursor=1), status="accepted")
        )
    with pytest.raises(SyncStoreError, match="updates require base metadata"):
        sync_service.store.insert_envelope(
            replace(_envelope(sync_service, domain, "upsert", object_revision=2), status="accepted")
        )


def test_the_store_still_requires_a_base_for_a_provenance_tombstone(sync_service: SyncV2Service) -> None:
    """Provenance is deleted with its note, from the note's own head; it is not part of the rule."""
    with pytest.raises(SyncStoreError, match="tombstones require base metadata"):
        sync_service.store.insert_envelope(
            replace(_envelope(sync_service, "notes.provenance", "tombstone"), status="accepted")
        )


# ---------------------------------------------------------------------------
# The tombstone is applied as a no-op
# ---------------------------------------------------------------------------


def test_device_tombstone_of_an_object_the_server_never_had_is_applied(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    pushed = _push_tombstone(sync_service, domain)

    assert pushed.conflicts == [] and pushed.rejected == []
    assert [(item.entity_id, item.apply_status) for item in pushed.accepted] == [(GONE, "applied")]
    assert _log(sync_service) == [(domain, "tombstone", GONE, "applied")]
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
    assert _blocker(sync_service) is None
    state = _state(sync_service, domain)
    assert (state.deleted, state.object_revision, state.latest_server_cursor) == (
        True, 1, pushed.accepted[0].server_sequence,
    )
    # Nothing was invented to delete.
    assert _row(chacha_db, domain) is None


def test_the_tombstone_no_longer_fails_the_rest_of_the_push(sync_service: SyncV2Service, domain: str) -> None:
    """The contract error used to raise out of push, and every envelope after it went with it."""
    after = _envelope(sync_service, "chat.conversation", "upsert", "after-the-tombstone", envelope_id="env-after")

    pushed = sync_service.push(
        user_id=USER,
        dataset_id=_dataset_id(sync_service),
        device_id=PUSHER,
        envelopes=[_envelope(sync_service, domain, "tombstone"), after],
    )

    assert [(item.entity_id, item.apply_status) for item in pushed.accepted] == [
        (GONE, "applied"),
        ("after-the-tombstone", "applied"),
    ]


def test_the_dataset_still_takes_writes_after_a_satisfied_tombstone(
    client: TestClient, sync_service: SyncV2Service, domain: str
) -> None:
    assert _push_tombstone(sync_service, domain).conflicts == []

    created = client.post("/api/v1/chats/", json={"title": "After the tombstone"})
    pushed = _push(sync_service, _envelope(sync_service, domain, "upsert", "a-new-object", envelope_id="env-new"))

    assert created.status_code == 201, created.text
    assert [item.apply_status for item in pushed.accepted] == ["applied"], (pushed.rejected, pushed.conflicts)
    assert _blocker(sync_service) is None
    assert {item[3] for item in _log(sync_service)} == {"applied"}


def test_pushing_the_same_tombstone_again_is_an_idempotent_replay(sync_service: SyncV2Service, domain: str) -> None:
    first = _push_tombstone(sync_service, domain)
    again = _push_tombstone(sync_service, domain)

    assert [item.apply_status for item in again.accepted] == ["applied"]
    assert again.conflicts == []
    assert again.accepted[0].server_sequence == first.accepted[0].server_sequence
    assert len(_log(sync_service)) == 1


def test_a_copy_the_owner_holds_outside_sync_is_deleted_with_the_tombstone(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    """The device deleted an object both sides had before the profile: it ends up deleted on both."""
    _copy_outside_sync(chacha_db, domain)
    _copy_outside_sync(chacha_db, domain, "a-neighbour")

    pushed = _push_tombstone(sync_service, domain)

    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _is_deleted(chacha_db, domain)
    assert not _is_deleted(chacha_db, domain, "a-neighbour")
    assert _state(sync_service, domain).deleted is True
    assert _blocker(sync_service) is None


def test_a_copy_that_is_already_deleted_stays_deleted(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    _copy_outside_sync(chacha_db, domain)
    if domain == "chat.conversation":
        chacha_db.soft_delete_conversation(GONE, 1)
    else:
        chacha_db.soft_delete_note(GONE, 1)

    pushed = _push_tombstone(sync_service, domain)

    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _is_deleted(chacha_db, domain)
    assert _state(sync_service, domain).deleted is True


# ---------------------------------------------------------------------------
# The other devices converge
# ---------------------------------------------------------------------------


def test_a_third_device_receives_the_tombstone(sync_service: SyncV2Service, third_device: str, domain: str) -> None:
    _push_tombstone(sync_service, domain)

    pulled = _pull(sync_service, third_device)

    assert [(item.domain, item.operation, item.object_id, item.device_id) for item in pulled] == [
        (domain, "tombstone", GONE, PUSHER)
    ]


def test_a_replica_that_never_had_the_object_applies_the_tombstone_as_a_no_op(
    sync_service: SyncV2Service,
    third_device: str,
    replica: tuple[SyncV2Service, CharactersRAGDB],
    domain: str,
) -> None:
    replica_service, replica_db = replica
    _push_tombstone(sync_service, domain)

    delivered = _deliver(replica_service, _pull(sync_service, third_device))

    assert delivered.conflicts == [] and delivered.rejected == []
    assert [item.apply_status for item in delivered.accepted] == ["applied"]
    assert _row(replica_db, domain) is None
    assert _state(replica_service, domain).deleted is True
    assert _blocker(replica_service) is None


def test_a_replica_that_still_has_the_object_deletes_it(
    sync_service: SyncV2Service,
    third_device: str,
    replica: tuple[SyncV2Service, CharactersRAGDB],
    domain: str,
) -> None:
    """The copy a third device kept outside Sync goes when the tombstone arrives: every replica ends at "gone"."""
    replica_service, replica_db = replica
    _copy_outside_sync(replica_db, domain)
    _push_tombstone(sync_service, domain)

    delivered = _deliver(replica_service, _pull(sync_service, third_device))

    assert delivered.conflicts == []
    assert [item.apply_status for item in delivered.accepted] == ["applied"]
    assert _is_deleted(replica_db, domain)
    assert _blocker(replica_service) is None


def test_a_later_create_of_the_same_id_is_a_push_conflict_that_blocks_nothing(
    client: TestClient, sync_service: SyncV2Service, chacha_db: CharactersRAGDB, third_device: str, domain: str
) -> None:
    """A third device that created the object offline learns it was deleted; nothing is resurrected."""
    _push_tombstone(sync_service, domain)

    late = _push(sync_service, _envelope(sync_service, domain, "upsert", device=third_device))

    assert late.accepted == []
    assert [item.client_envelope_id for item in late.conflicts] == [f"env-{third_device}-{domain}-upsert-{GONE}"]
    assert _row(chacha_db, domain) is None
    assert _state(sync_service, domain).deleted is True
    # A push-time conflict is not a projection conflict: the dataset is healthy and the device can pull the delete.
    assert _blocker(sync_service) is None
    assert [(item.operation, item.object_id) for item in _pull(sync_service, third_device)] == [("tombstone", GONE)]
    assert client.post("/api/v1/chats/", json={"title": "Still healthy"}).status_code == 201


# ---------------------------------------------------------------------------
# What is still a conflict
# ---------------------------------------------------------------------------


def test_a_tombstone_whose_base_disagrees_with_a_held_object_is_still_a_conflict(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    head = _held(sync_service, domain, "held")

    stale = _push_tombstone(
        sync_service,
        domain,
        "held",
        object_revision=2,
        base_server_cursor=head.server_cursor,
        base_object_revision=head.object_revision,
        base_object_hash="sha256:not-what-the-server-holds",
    )

    assert stale.accepted == []
    assert [item.entity_id for item in stale.conflicts] == ["held"]
    assert not _is_deleted(chacha_db, domain, "held")
    assert _state(sync_service, domain, "held").deleted is False


def test_a_baseless_tombstone_of_a_held_object_is_still_a_conflict(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    """Only an object the dataset has no history for is "already gone"; this one is published and live."""
    _held(sync_service, domain, "held")

    baseless = _push_tombstone(sync_service, domain, "held", device=PUSHER, envelope_id="env-baseless-held")

    assert baseless.accepted == []
    assert [item.entity_id for item in baseless.conflicts] == ["held"]
    assert not _is_deleted(chacha_db, domain, "held")
    assert _state(sync_service, domain, "held").deleted is False


def test_a_tombstone_that_names_a_base_the_dataset_never_issued_is_a_push_conflict_that_blocks_nothing(
    client: TestClient, sync_service: SyncV2Service, third_device: str, domain: str
) -> None:
    """The device claims server history that does not exist. It is told so, and nothing is held up."""
    claimed = _push_tombstone(
        sync_service,
        domain,
        object_revision=2,
        base_server_cursor=7,
        base_object_revision=1,
        base_object_hash="sha256:a-base-this-dataset-never-issued",
    )

    assert claimed.accepted == []
    assert [item.entity_id for item in claimed.conflicts] == [GONE]
    assert _state(sync_service, domain) is None
    assert _blocker(sync_service) is None
    assert _pull(sync_service, third_device) == []
    assert client.post("/api/v1/chats/", json={"title": "Still healthy"}).status_code == 201


def test_a_tombstone_never_deletes_another_owners_row(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    """The object is found by id alone, and one PostgreSQL database holds every owner's rows.

    With no Sync state tying the id to this owner, only a row this owner holds
    can be reached. For this owner the object does not exist, so the tombstone
    is satisfied and the other owner's row is untouched.
    """
    _another_owners_row(chacha_db, domain, "theirs")

    pushed = _push_tombstone(sync_service, domain, "theirs")

    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert not _is_deleted(chacha_db, domain, "theirs")
    assert _state(sync_service, domain, "theirs").deleted is True
    assert _blocker(sync_service) is None


def test_a_row_another_dataset_of_the_owner_holds_is_not_deleted(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    """A row projected from other Sync history is not this tombstone's to delete.

    Messages carry that history on the row; chats and notes do not, so the
    owner's other datasets are asked instead.
    """
    sync_service.enroll_dataset(user_id=USER, dataset_id="second-dataset", domains=list(M1_SYNC_DOMAINS))
    other = _push(
        sync_service,
        _envelope(sync_service, domain, "upsert", "from-elsewhere", dataset_id="second-dataset", envelope_id="env-other"),
    )
    assert [item.apply_status for item in other.accepted] == ["applied"], (other.rejected, other.conflicts)

    pushed = _push_tombstone(sync_service, domain, "from-elsewhere")

    assert pushed.accepted == []
    assert [item.entity_id for item in pushed.conflicts] == ["from-elsewhere"]
    assert not _is_deleted(chacha_db, domain, "from-elsewhere")
    blocker = _blocker(sync_service)
    assert (blocker.conflict_type, blocker.metadata["reason"]) == ("whole_object_conflict", "missing_server_object")
    assert _state(sync_service, domain, "from-elsewhere", dataset_id="second-dataset").deleted is False


def test_a_native_chat_sync_cannot_carry_is_not_deleted(sync_service: SyncV2Service, chacha_db: CharactersRAGDB) -> None:
    """A chat whose native bundle Sync cannot represent is the server's own; a device cannot delete it."""
    _copy_outside_sync(chacha_db, "chat.conversation", "native-chat")
    with chacha_db.transaction() as conn:
        conn.execute("UPDATE conversations SET native_bundle_json = ? WHERE id = ?", ("{}", "native-chat"))

    pushed = _push_tombstone(sync_service, "chat.conversation", "native-chat")

    assert pushed.accepted == []
    assert [item.entity_id for item in pushed.conflicts] == ["native-chat"]
    assert not _is_deleted(chacha_db, "chat.conversation", "native-chat")


# ---------------------------------------------------------------------------
# The server's own delete of a note Sync never saw
# ---------------------------------------------------------------------------


def test_deleting_a_note_sync_never_saw_publishes_an_id_only_tombstone(
    notes_client: TestClient, sync_service: SyncV2Service, chacha_db: CharactersRAGDB, third_device: str
) -> None:
    """The note route tombstones through the log. A note with no Sync state used to fail it with 503."""
    _copy_outside_sync(chacha_db, "notes.note", "legacy-note")
    version = chacha_db.get_note_by_id("legacy-note")["version"]

    deleted = notes_client.delete("/api/v1/notes/legacy-note", headers={"expected-version": str(version)})

    assert deleted.status_code == 204, deleted.text
    assert _is_deleted(chacha_db, "notes.note", "legacy-note")
    assert _log(sync_service) == [("notes.note", "tombstone", "legacy-note", "applied")]
    tombstone = _pull(sync_service, third_device)[0]
    assert (tombstone.device_id, tombstone.base_server_cursor, tombstone.object_revision) == (
        SERVER_ORIGIN_DEVICE_ID, None, 1,
    )
    assert set(tombstone.payload) == {"deleted_at", "reason"}
    assert _state(sync_service, "notes.note", "legacy-note").deleted is True
    assert _blocker(sync_service) is None


# ---------------------------------------------------------------------------
# A dataset already holding such a tombstone
# ---------------------------------------------------------------------------


def test_a_stranded_device_tombstone_is_applied_on_the_next_server_write(
    client: TestClient, sync_service: SyncV2Service, third_device: str, domain: str
) -> None:
    stranded = _strand(sync_service, domain, with_base=True)

    created = client.post("/api/v1/chats/", json={"title": "After the block"})

    assert created.status_code == 201, created.text
    assert _blocker(sync_service) is None
    settled = sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor)
    assert (settled.apply_status, settled.apply_error_code) == ("applied", None)
    assert _state(sync_service, domain).deleted is True
    conflict = sync_service.store.get_conflict(f"conflict-stranded-{domain}")
    assert (conflict.status, conflict.resolution_action) == ("resolved", "apply_on_retry")
    assert [(item.operation, item.object_id) for item in _pull(sync_service, third_device)] == [
        ("tombstone", GONE),
        ("upsert", created.json()["id"]),
    ]


def test_a_stranded_device_tombstone_is_applied_on_the_next_push(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, third_device: str, domain: str
) -> None:
    """A user who only writes from devices is not left waiting for a server write."""
    _copy_outside_sync(chacha_db, domain)
    stranded = _strand(sync_service, domain, with_base=True)

    pushed = _push(sync_service, _envelope(sync_service, domain, "upsert", "from-the-phone", device=third_device))

    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _blocker(sync_service) is None
    assert sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor).apply_status == "applied"
    assert _is_deleted(chacha_db, domain)


def test_a_device_that_pushes_its_stranded_tombstone_again_gets_it_applied(
    sync_service: SyncV2Service, domain: str
) -> None:
    stranded = _strand(sync_service, domain)

    again = _push_tombstone(sync_service, domain)

    assert again.conflicts == []
    assert [(item.server_sequence, item.apply_status) for item in again.accepted] == [
        (stranded.server_cursor, "applied"),
    ]
    assert sync_service.store.get_conflict(f"conflict-stranded-{domain}").status == "resolved"


def test_a_stranded_server_tombstone_is_skipped_on_the_next_write(
    client: TestClient, sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    """The server's own stranded delete answered 503: it is dismissed, and the row it named is left alone."""
    _copy_outside_sync(chacha_db, domain)
    stranded = _strand(sync_service, domain, device=SERVER_ORIGIN_DEVICE_ID, with_base=True)

    created = client.post("/api/v1/chats/", json={"title": "After the block"})

    assert created.status_code == 201, created.text
    assert _blocker(sync_service) is None
    skipped = sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor)
    assert skipped.apply_status == "superseded"
    assert not _is_deleted(chacha_db, domain)


def test_recovery_leaves_every_other_conflict_for_review(
    client: TestClient, sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    _block_dataset(sync_service)

    refused = client.post("/api/v1/chats/", json={"title": "Still blocked"})
    pushed = _push(sync_service, _envelope(sync_service, domain, "upsert", "from-the-phone"))

    assert refused.status_code == 503, refused.text
    assert pushed.accepted == []
    assert [item.conflict_id for item in pushed.conflicts] == ["conflict-unresolved"]
    assert _blocker(sync_service).conflict_id == "conflict-unresolved"


def test_a_stranded_tombstone_of_a_row_another_dataset_holds_is_left_for_review(
    client: TestClient, sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    sync_service.enroll_dataset(user_id=USER, dataset_id="second-dataset", domains=list(M1_SYNC_DOMAINS))
    _push(
        sync_service,
        _envelope(sync_service, domain, "upsert", GONE, dataset_id="second-dataset", envelope_id="env-other"),
    )
    _strand(sync_service, domain)

    refused = client.post("/api/v1/chats/", json={"title": "Still blocked"})

    assert refused.status_code == 503, refused.text
    assert _blocker(sync_service).conflict_id == f"conflict-stranded-{domain}"
    assert not _is_deleted(chacha_db, domain)


def test_a_failed_tombstone_of_an_unknown_object_is_applied_by_repair(
    sync_service: SyncV2Service, chacha_db: CharactersRAGDB, domain: str
) -> None:
    """The old materializer failed a tombstone that found no row; replaying it now applies it."""
    failed = sync_service.store.insert_envelope(
        replace(_envelope(sync_service, domain, "tombstone"), status="accepted")
    )
    sync_service.store.mark_envelope_apply_status(
        failed.server_cursor,
        apply_status="failed",
        apply_error_code="chat_projection_failed",
        apply_error_message="not found for Sync v2 tombstone",
    )

    sync_service.repair(user_id=USER, dataset_id=_dataset_id(sync_service), failed_only=True)

    assert sync_service.store.get_envelope_by_server_cursor(failed.server_cursor).apply_status == "applied"
    assert _state(sync_service, domain).deleted is True
    assert _row(chacha_db, domain) is None
