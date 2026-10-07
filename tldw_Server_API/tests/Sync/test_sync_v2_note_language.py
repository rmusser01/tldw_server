"""Complete Notes snapshot contract and version negotiation."""

from dataclasses import replace

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.sync_v2_models import SyncV2Envelope
from tldw_Server_API.app.core.Ingestion_Sources.sinks.notes_sink import _capture_ingestion_note
from tldw_Server_API.app.core.Notes import Notes_Library
from tldw_Server_API.app.core.Sync.v2 import models
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import NotesOrganizationCoordinator
from tldw_Server_API.app.core.Sync.v2.server_origin import canonical_payload_hash
from tldw_Server_API.tests.Sync import test_sync_v2_notes_materializer as materializer_tests
from tldw_Server_API.tests.Sync import test_sync_v2_server_origin_capture as origin_tests
from tldw_Server_API.tests.Sync.notes_organization_test_support import build_ready_notes_sync_stack
from tldw_Server_API.tests.Sync.test_sync_v2_endpoints import _client_for_service


@pytest.mark.parametrize("language", [None, "json", " Future Mode ", "x" * 100])
def test_v2_keeps_exact_source_and_explicit_language(language):
    payload = {
        "title": "  Source  ",
        "content": '\ufeff{\r\n  "n": 1e2\r\n}\r\n',
        "conversation_id": None,
        "message_id": None,
        "content_language": language,
    }
    assert models.validate_notes_note_upsert_payload_v2(payload) == payload
    with pytest.raises(ValueError, match="unsupported fields"):
        models.validate_notes_note_upsert_payload(payload)


@pytest.mark.parametrize("language", ["", "x" * 101, 1, False, [], {}])
def test_v2_rejects_invalid_language(language):
    with pytest.raises(ValueError):
        models.validate_notes_note_upsert_payload_v2(
            {
                "title": "T",
                "content": "{}",
                "conversation_id": None,
                "message_id": None,
                "content_language": language,
            }
        )


def test_v2_requires_explicit_complete_snapshot():
    with pytest.raises(ValueError):
        models.validate_notes_note_upsert_payload_v2({"title": "T", "content": "{}"})


chacha_db = materializer_tests.chacha_db
sync_store = materializer_tests.sync_store
sync_service = materializer_tests.sync_service


@pytest.fixture
def service(sync_service):
    sync_service.adapters = SyncAdapterRegistry([NotesDomainAdapter()])
    sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
        capabilities={"supported_adapter_versions": {"notes.note": [1, 2]}},
    )
    return sync_service


def snapshot(service, language, *, version=2, operation="upsert", restore=False):
    state = service.store.get_object_state("dataset-1", "notes.note", "note-1")
    revision = state.object_revision + 1 if state else 1
    payload = {
        "title": "  Config  ",
        "content": '\ufeff{\r\n "x":1e2\r\n}\r\n',
        "conversation_id": None,
        "message_id": None,
    }
    if version == 2:
        payload["content_language"] = language
    if operation == "tombstone":
        payload = {}
    return materializer_tests._note_envelope(
        client_envelope_id=f"env-{revision}-{version}",
        client_sequence=revision,
        schema_version=version,
        adapter_version=version,
        object_revision=revision,
        operation=operation,
        deleted=operation == "tombstone",
        payload=payload,
        payload_hash=canonical_payload_hash(payload)[0],
        base_server_cursor=state.latest_server_cursor if state else None,
        base_object_revision=state.object_revision if state else None,
        base_object_hash=state.object_hash if state else None,
        routing_metadata={"restore_intent": True} if restore else {},
    )


def push(service, envelope):
    return materializer_tests._push_one(service, envelope)


def test_v1_upgrade_complete_source_language_reset_retry_and_restore(service, chacha_db):
    assert push(service, snapshot(service, None, version=1)).accepted
    envelope = snapshot(service, "json")
    assert push(service, envelope).accepted
    assert push(service, envelope).accepted
    note = chacha_db.get_note_by_id("note-1")
    assert (note["version"], note["content_language"], note["content"], note["title"]) == (
        2,
        "json",
        envelope.payload["content"],
        envelope.payload["title"],
    )
    assert push(service, snapshot(service, None, operation="tombstone")).accepted
    assert push(service, snapshot(service, "json", restore=True)).accepted
    assert chacha_db.get_note_by_id("note-1")["content_language"] == "json"
    assert push(service, snapshot(service, None)).accepted
    assert chacha_db.get_note_by_id("note-1")["content_language"] is None


def test_old_writer_cannot_downgrade_even_with_current_base(service, chacha_db):
    assert push(service, snapshot(service, "yaml")).accepted
    before = service.store.list_envelopes_after("dataset-1", 0, limit=100)
    result = push(service, snapshot(service, None, version=1))
    assert result.rejected[0].error_code == "notes_snapshot_upgrade_required"
    assert service.store.list_envelopes_after("dataset-1", 0, limit=100) == before
    assert chacha_db.get_note_by_id("note-1")["content_language"] == "yaml"


def test_language_only_stale_snapshot_conflicts(service, chacha_db):
    assert push(service, snapshot(service, None)).accepted
    stale = snapshot(service, "yaml")
    assert push(service, replace(snapshot(service, "json"), client_envelope_id="winner")).accepted
    result = push(service, replace(stale, client_sequence=3))
    assert result.conflicts or result.rejected
    assert chacha_db.get_note_by_id("note-1")["content_language"] == "json"


@pytest.mark.parametrize("adapter,schema", [(1, 2), (2, 1), (3, 3)])
def test_unsupported_pair_never_appended(service, adapter, schema):
    envelope = replace(snapshot(service, None), adapter_version=adapter, schema_version=schema)
    result = push(service, envelope)
    assert result.rejected
    assert not service.store.list_envelopes_after("dataset-1", 0, limit=100)


@pytest.fixture
def origin(tmp_path):
    db = origin_tests.chacha_db.__wrapped__(tmp_path)
    service = origin_tests.sync_service.__wrapped__(tmp_path, db)
    service.adapters = default_sync_v2_registry()
    yield service, db
    db.close_connection()


def test_rest_complete_language_omission_reset_and_projection_equality(origin, monkeypatch):
    service, db = origin
    client = origin_tests._notes_app(monkeypatch, chacha_db=db, sync_service=service)
    source = '\ufeff{\r\n  "x":1e2\r\n}\r\n'
    response = client.post(
        "/api/v1/notes/", json={"title": "  Config  ", "content": source, "content_language": "json"}
    )
    assert response.status_code == 201, response.text
    note_id = response.json()["id"]
    assert client.get("/api/v1/notes/capabilities").json()["content_language"] is True
    for version, patch, language in [(1, {"title": "  Next  "}, "json"), (2, {"content_language": None}, None)]:
        response = client.patch(f"/api/v1/notes/{note_id}", json=patch, headers={"expected-version": str(version)})
        assert response.status_code == 200, response.text
        note = response.json()
        dataset = service.store.list_datasets_for_user("user-1")[0]
        head = service.store.get_current_head(dataset.dataset_id, "notes.note", note_id)
        assert head.payload == models.complete_note_snapshot(note)
        assert head.payload["content"] == source
        assert head.payload["content_language"] == language
        assert head.payload_hash == canonical_payload_hash(head.payload)[0]
        assert (head.adapter_version, head.schema_version) == (2, 2)
        assert "content_language" not in head.routing_metadata


def test_private_transport_is_not_advertised_or_accepted(service):
    assert "notes" not in service.capabilities().supported_domains
    assert service.capabilities().encryption_policies == ["server_trusted_v1"]
    with pytest.raises(ValidationError, match="not accepted"):
        SyncV2Envelope.model_validate(
            {
                "dataset_id": "dataset-1",
                "client_envelope_id": "private",
                "object_id": "note-1",
                "domain": "notes.note",
                "operation": "upsert",
                "schema_version": 2,
                "adapter_version": 2,
                "payload": {},
                "payload_ciphertext": "opaque",
                "payload_hash": "sha256:opaque",
                "encryption_metadata": {"policy": "client_private_v1"},
            }
        )
    assert not service.store.list_envelopes_after("dataset-1", 0, limit=100)


@pytest.mark.parametrize("interleave", [False, True])
def test_old_http_pull_reports_upgrade_without_advancing_cursor(service, monkeypatch, interleave):
    service.register_device(user_id="user-1", device_id="old", display_name="Old", client_type="chatbook")
    assert push(service, snapshot(service, None, version=1)).accepted
    client = _client_for_service(service)
    params = {"dataset_id": "dataset-1", "device_id": "old"}
    control = client.get("/api/v1/sync/pull", params=params)
    assert control.status_code == 200, control.text
    before = service.store.get_device_cursor("dataset-1", "old", "notes.note")
    if interleave:
        original = service._scan_pull_page

        def scan_then_upgrade(**kwargs):
            page = original(**kwargs)
            assert push(service, snapshot(service, "json")).accepted
            return page

        monkeypatch.setattr(service, "_scan_pull_page", scan_then_upgrade)
    else:
        assert push(service, snapshot(service, "json")).accepted
    response = client.get("/api/v1/sync/pull", params=params)
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["error_code"] == "notes_snapshot_upgrade_required"
    assert "Upgrade" in response.json()["detail"]["message"]
    assert service.store.get_device_cursor("dataset-1", "old", "notes.note") == before
    new_pull = service.pull(user_id="user-1", dataset_id="dataset-1", device_id="device-1", include_own_changes=True)
    latest = new_pull.envelopes[-1]
    assert latest.adapter_version == 2
    assert latest.payload["content_language"] == "json"
    assert latest.payload_hash == canonical_payload_hash(latest.payload)[0]


def test_interop_compound_and_ingestion_publish_complete_snapshots(tmp_path, monkeypatch):
    db, store, sync = build_ready_notes_sync_stack(tmp_path)
    sync.settings = replace(sync.settings, pull_token_signing_secret="synthetic-f3-test-signing-key")
    coordinator = NotesOrganizationCoordinator(service=sync, note_db=db, user_id="user-1")
    monkeypatch.setattr(Notes_Library, "active_coordinator", lambda *args, **kwargs: coordinator)
    service = Notes_Library.NotesInteropService(tmp_path, "user-1")
    monkeypatch.setattr(service, "_get_db", lambda user: db)
    note_id = service.add_note("user-1", "  Config  ", "a: 1\r\n", content_language=" Future ")
    service.update_note("user-1", note_id, {"title": "  Changed  "}, 1)
    _capture_ingestion_note(
        coordinator,
        note_id=note_id,
        title="  Ingested  ",
        content="a: 2\r\n",
        expected_version=2,
        request_key="ingest-existing",
    )
    note = db.get_note_by_id(note_id)
    head = store.get_current_head(coordinator.require_ready().dataset_id, "notes.note", note_id)
    assert (note["content_language"], note["content"], note["title"]) == (" Future ", "a: 2\r\n", "  Ingested  ")
    assert head.payload == models.complete_note_snapshot(note)
    assert (head.adapter_version, head.schema_version) == (2, 2)
    service.update_note(user_id="user-1", note_id=note_id, content_language=None, expected_version=3)
    assert db.get_note_by_id(note_id)["content_language"] is None
    db.close_connection()


def test_missing_signer_keeps_safe_v1_but_refuses_language_before_append(origin, monkeypatch):
    service, db = origin
    monkeypatch.delenv("SYNC_V2_PULL_TOKEN_SIGNING_SECRET", raising=False)
    service.settings = replace(service.settings, pull_token_signing_secret=None)
    client = origin_tests._notes_app(monkeypatch, chacha_db=db, sync_service=service)
    assert client.get("/api/v1/notes/capabilities").json()["content_language"] is False
    created = client.post("/api/v1/notes/", json={"title": "Legacy", "content": "source"})
    assert created.status_code == 201, created.text
    dataset = service.store.list_datasets_for_user("user-1")[0]
    before = service.store.list_envelopes_after(dataset.dataset_id, 0, limit=100)
    assert before[-1].adapter_version == 1
    rejected = client.patch(
        f"/api/v1/notes/{created.json()['id']}", json={"content_language": "yaml"}, headers={"expected-version": "1"}
    )
    assert rejected.status_code == 503, rejected.text
    assert rejected.json()["detail"]["error_code"] == "sync_pull_token_signing_unavailable"
    assert service.store.list_envelopes_after(dataset.dataset_id, 0, limit=100) == before
    assert models.sync_v2_dataset_writable_adapter_versions(dataset)["notes.note"] == [1]


def test_lost_signer_never_downgrades_existing_v2_or_advances_pull(origin, monkeypatch):
    service, db = origin
    client = origin_tests._notes_app(monkeypatch, chacha_db=db, sync_service=service)
    created = client.post("/api/v1/notes/", json={"title": "Config", "content": "source"})
    assert created.status_code == 201, created.text
    dataset = service.store.list_datasets_for_user("user-1")[0]
    before = service.store.list_envelopes_after(dataset.dataset_id, 0, limit=100)
    monkeypatch.delenv("SYNC_V2_PULL_TOKEN_SIGNING_SECRET", raising=False)
    service.settings = replace(service.settings, pull_token_signing_secret=None)
    rejected = client.patch(
        f"/api/v1/notes/{created.json()['id']}", json={"title": "Later"}, headers={"expected-version": "1"}
    )
    assert rejected.status_code == 503, rejected.text
    assert service.store.list_envelopes_after(dataset.dataset_id, 0, limit=100) == before
    response = _client_for_service(service).get(
        "/api/v1/sync/pull", params={"dataset_id": dataset.dataset_id, "device_id": "offline-device"}
    )
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error_code"] == "sync_pull_token_signing_unavailable"
    assert (
        service.store.get_device_cursor(dataset.dataset_id, "offline-device", "notes.note", adapter_version=2) is None
    )


def test_derived_ingestion_rechecks_canonical_language_version_before_append(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.errors import SyncNotesAdmissionError
    from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import NotesOrganizationPreflightError

    db, store, sync = build_ready_notes_sync_stack(tmp_path)
    sync.settings = replace(sync.settings, pull_token_signing_secret="synthetic-f3-test-signing-key")
    coordinator = NotesOrganizationCoordinator(service=sync, note_db=db, user_id="user-1")
    _capture_ingestion_note(coordinator, note_id="derived-race", title="Config", content="source", request_key="create")
    before = store.list_envelopes_after(coordinator.require_ready().dataset_id, 0, limit=100)
    capture = coordinator.capture

    def interleave(self, **kwargs):
        db.update_note("derived-race", {"content_language": "yaml"}, expected_version=1)
        return capture(**kwargs)

    monkeypatch.setattr(NotesOrganizationCoordinator, "capture", interleave)
    with pytest.raises(NotesOrganizationPreflightError) as exc:
        _capture_ingestion_note(
            coordinator,
            note_id="derived-race",
            title="Config",
            content="later",
            expected_version=1,
            request_key="update",
        )
    assert isinstance(exc.value.__cause__, SyncNotesAdmissionError)
    assert store.list_envelopes_after(coordinator.require_ready().dataset_id, 0, limit=100) == before
    assert db.get_note_by_id("derived-race")["content_language"] == "yaml"
    db.close_connection()


def test_durable_v2_compound_retry_survives_signer_loss(tmp_path, monkeypatch):
    db, store, sync = build_ready_notes_sync_stack(tmp_path)
    sync.settings = replace(sync.settings, pull_token_signing_secret="synthetic-f3-test-signing-key")
    coordinator = NotesOrganizationCoordinator(service=sync, note_db=db, user_id="user-1")
    monkeypatch.setattr(Notes_Library, "active_coordinator", lambda *args, **kwargs: coordinator)
    service = Notes_Library.NotesInteropService(tmp_path, "user-1")
    monkeypatch.setattr(service, "_get_db", lambda user: db)
    note_id = service.add_note("user-1", "Config", "source", content_language="yaml")
    before = store.list_envelopes_after(coordinator.require_ready().dataset_id, 0, limit=100)
    sync.settings = replace(sync.settings, pull_token_signing_secret=None)
    monkeypatch.delenv("SYNC_V2_PULL_TOKEN_SIGNING_SECRET", raising=False)
    assert service.add_note("user-1", "Config", "source", content_language="yaml") == note_id
    assert store.list_envelopes_after(coordinator.require_ready().dataset_id, 0, limit=100) == before
    assert db.get_note_by_id(note_id)["content_language"] == "yaml"
    db.close_connection()


def test_retry_retains_legacy_optional_field_shape_and_hash(origin):
    from tldw_Server_API.app.core.Sync.v2.server_origin import capture_server_origin_mutation

    service, db = origin
    dataset = service.store.list_datasets_for_user("user-1")[0]
    payload = {"title": "Legacy", "content": "source"}
    legacy = models.SyncEnvelopeCreate(
        dataset_id=dataset.dataset_id,
        client_envelope_id="old-optional-fields",
        domain="notes.note",
        operation="upsert",
        object_id="legacy-retry",
        device_id="server-origin",
        object_revision=1,
        payload=payload,
        payload_hash=canonical_payload_hash(payload)[0],
        stable_key="legacy-key",
    )
    accepted = service.store.insert_envelope(legacy)
    service._materialize_envelope(accepted)
    result = capture_server_origin_mutation(
        service,
        user_id="user-1",
        domain="notes.note",
        operation="upsert",
        object_id="legacy-retry",
        payload=payload,
        source="server_api",
        stable_key="legacy-key",
    )
    assert result.envelope.payload == payload
    assert result.envelope.payload_hash == legacy.payload_hash
    assert result.envelope.adapter_version == 1
    assert db.get_note_by_id("legacy-retry")["version"] == 1


@pytest.mark.parametrize("signer", [False, True])
def test_modern_reader_receives_safe_v1_with_runtime_negotiation(service, monkeypatch, signer):
    monkeypatch.delenv("SYNC_V2_PULL_TOKEN_SIGNING_SECRET", raising=False)
    service.settings = replace(service.settings, pull_token_signing_secret="synthetic-fix-key" if signer else None)
    assert push(service, snapshot(service, None, version=1)).accepted
    assert service.capabilities().supported_adapter_versions["notes.note"] == ([1, 2] if signer else [1])
    result = service.pull(user_id="user-1", dataset_id="dataset-1", device_id="device-1", include_own_changes=True)
    assert len(result.envelopes) == 1
    assert result.envelopes[0].adapter_version == 1
    assert result.envelopes[0].payload["content"] == '\ufeff{\r\n "x":1e2\r\n}\r\n'


@pytest.mark.parametrize("interleave, versions", [(False, [1, 2]), (True, [1, 2]), (False, [2])])
def test_missing_signer_never_filters_v2_for_modern_reader(service, monkeypatch, interleave, versions):
    assert push(service, snapshot(service, None, version=1)).accepted
    candidate = snapshot(service, "json")
    if not interleave:
        assert push(service, candidate).accepted
    service.register_device(
        user_id="user-1",
        device_id="modern-reader",
        display_name="Modern",
        client_type="chatbook",
        capabilities={"supported_adapter_versions": {"notes.note": versions}},
    )
    monkeypatch.delenv("SYNC_V2_PULL_TOKEN_SIGNING_SECRET", raising=False)
    service.settings = replace(service.settings, pull_token_signing_secret=None)
    if interleave:
        scan = service._scan_pull_page

        def scan_then_append(**kwargs):
            page = scan(**kwargs)
            # Another already-admitted writer becomes visible after the scan.
            service.store.insert_envelope(candidate)
            return page

        monkeypatch.setattr(service, "_scan_pull_page", scan_then_append)
    before = service.store.get_device_cursor("dataset-1", "modern-reader", "notes.note")
    response = _client_for_service(service).get(
        "/api/v1/sync/pull", params={"dataset_id": "dataset-1", "device_id": "modern-reader"}
    )
    assert service.store.get_current_head("dataset-1", "notes.note", "note-1").adapter_version == 2
    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error_code"] == "sync_pull_token_signing_unavailable"
    assert service.store.get_device_cursor("dataset-1", "modern-reader", "notes.note") == before


@pytest.mark.parametrize("caller_version", [None, 1])
def test_preserved_language_binds_cached_row_version(tmp_path, monkeypatch, caller_version):
    from tldw_Server_API.app.core.Notes.organization_capture import capture_note_upsert
    from tldw_Server_API.app.core.Sync.v2.errors import SyncNotesAdmissionError
    from tldw_Server_API.app.core.Sync.v2.notes_organization_coordinator import NotesOrganizationPreflightError

    db, store, sync = build_ready_notes_sync_stack(tmp_path)
    sync.settings = replace(sync.settings, pull_token_signing_secret="synthetic-fix-key")
    coordinator = NotesOrganizationCoordinator(service=sync, note_db=db, user_id="user-1")
    capture_note_upsert(
        coordinator,
        note_id="cached",
        title="Config",
        content="source",
        content_language="json",
        source="test",
        key="create",
    )
    before = store.list_envelopes_after(coordinator.require_ready().dataset_id, 0, limit=100)
    capture = coordinator.capture

    def interleave(self, **kwargs):
        db.update_note("cached", {"content_language": "yaml"}, expected_version=1)
        return capture(**kwargs)

    monkeypatch.setattr(NotesOrganizationCoordinator, "capture", interleave)
    with pytest.raises(NotesOrganizationPreflightError) as exc:
        capture_note_upsert(
            coordinator,
            note_id="cached",
            title="Config",
            content="later",
            expected_version=caller_version,
            source="test",
            key="update",
        )
    assert isinstance(exc.value.__cause__, SyncNotesAdmissionError)
    assert store.list_envelopes_after(coordinator.require_ready().dataset_id, 0, limit=100) == before
    assert db.get_note_by_id("cached")["content_language"] == "yaml"
    assert db.get_note_by_id("cached")["content"] == "source"
    db.close_connection()
