"""Real-store authority, atomic projection, and upgrade for Notes provenance."""

from dataclasses import replace

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.models import SyncDatasetCreate, SyncDeviceUpsert, SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.notes_provenance_contract import (
    notes_provenance_object_hash,
    retain_notes_provenance,
)
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.server_origin import canonical_payload_hash, capture_server_origin_mutation
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    ServerOriginMutationStep,
    capture_server_origin_mutation_batch,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

PAYLOAD = {"origin": "knowledge_qa", "trust_state": "cited_answer"}
NOTE = {"title": "Title", "content": "Body", "conversation_id": None, "message_id": None}


@pytest.fixture
def env(tmp_path, request, monkeypatch):
    if getattr(request, "param", "sqlite") == "postgres":
        from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

        config = request.getfixturevalue("pg_database_config")
        db = CharactersRAGDB(":memory:", client_id="alice", backend=DatabaseBackendFactory.create_backend(config))
        store = SyncV2Store(SyncDatabase(backend=DatabaseBackendFactory.create_backend(config)))
        pool = store.db.backend.get_pool()
        get_connection = pool.get_connection

        def bounded_connection():
            connection = get_connection()
            store.db.execute("SET lock_timeout = '500ms'", connection=connection)
            connection.commit()
            return connection

        monkeypatch.setattr(pool, "get_connection", bounded_connection)
    else:
        db = CharactersRAGDB(tmp_path / "notes.db", client_id="alice")
        store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="personal",
            owner_user_id="alice",
            domains=["notes.note"],
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    service = SyncV2Service(
        store=store,
        adapters=default_sync_v2_registry(),
        materializers={"notes.note": NotesMaterializer(db)},
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="encrypted_volume", server_trusted_enabled=True, auth_mode="multi_user"
            )
        ),
    )
    yield service, db
    db.close_all_connections()


def save(env, *, expected=0, core_expected=0, payload=PAYLOAD, content="Body", key="save"):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import capture_note_with_provenance

    service, db = env
    return capture_note_with_provenance(
        service=service,
        note_db=db,
        user_id="alice",
        note_id="note",
        note_payload={**NOTE, "content": content},
        expected_note_version=core_expected,
        provenance=payload,
        expected_provenance_version=expected,
        idempotency_key=key,
        source="test",
    )


def head(service, domain="notes.provenance"):
    return service.store.get_current_head("personal", domain, "note")


def client(
    service,
    *,
    domain="notes.note",
    operation="upsert",
    payload=None,
    key="client",
    device="old",
    base=None,
    restore=False,
):
    base = base or head(service, domain)
    value = (NOTE if domain == "notes.note" else PAYLOAD) if payload is None else payload
    digest = (
        notes_provenance_object_hash(value, deleted=operation == "tombstone")
        if domain == "notes.provenance"
        else canonical_payload_hash(value)[0]
    )
    return SyncEnvelopeCreate(
        dataset_id="personal",
        device_id=device,
        client_envelope_id=key,
        domain=domain,
        operation=operation,
        object_id="note",
        parent_id="note" if domain == "notes.provenance" else None,
        payload=value,
        payload_hash=digest,
        base_server_cursor=base.server_cursor if base else None,
        base_object_revision=base.object_revision if base else None,
        base_object_hash=base.payload_hash if base else None,
        object_revision=base.object_revision + 1 if base else 1,
        routing_metadata={"restore_intent": True} if restore else {},
    )


def register(service, domains=("notes.note",), device="old"):
    service.store.upsert_device(
        SyncDeviceUpsert(
            device_id=device,
            user_id="alice",
            display_name=device,
            client_type="test",
            capabilities={"supported_domains": list(domains), "supported_adapter_versions": {d: [1] for d in domains}},
        )
    )


def test_capability_registry_is_strict(env):
    service, _ = env
    assert "notes.provenance" in service.capabilities().supported_domains
    assert service.adapters.has_domain("notes.provenance")


def test_new_pair_and_old_device_markdown_preserves_history(env):
    service, db = env
    result = save(env)
    assert result.fully_applied
    assert [e.domain for e in result.envelopes] == ["notes.note", "notes.provenance"]
    retained = db.note_provenance_store.get("note")
    register(service)
    outcome = service.push(
        user_id="alice",
        dataset_id="personal",
        device_id="old",
        envelopes=[client(service, payload={**NOTE, "content": "Old editor stripped comments"})],
    )
    assert outcome.accepted and not outcome.conflicts
    assert db.note_provenance_store.get("note") == retained
    assert db.get_note_by_id("note")["content"] == "Old editor stripped comments"
    assert save(env).fully_applied  # exact lost acknowledgment, despite a newer core head


def test_stale_independent_base_changes_neither_head(env):
    service, db = env
    save(env)
    before = (head(service, "notes.note"), head(service))
    with pytest.raises(SyncStoreError):
        save(env, expected=0, core_expected=1, key="stale", content="Rejected")
    assert (head(service, "notes.note"), head(service)) == before
    assert db.get_note_by_id("note")["content"] == "Body"


def test_second_product_write_rolls_back_pair_and_filtered_repair_recovers(env, monkeypatch):
    service, db = env
    original = db.note_provenance_store.apply_sync

    def fail(*args, **kwargs):
        raise RuntimeError("injected second product write")

    monkeypatch.setattr(db.note_provenance_store, "apply_sync", fail)
    with pytest.raises(SyncStoreError):
        save(env)
    assert db.get_note_by_id("note") is None
    assert service.store.get_object_state("personal", "notes.note", "note") is None
    second = head(service)
    monkeypatch.setattr(db.note_provenance_store, "apply_sync", original)
    from tldw_Server_API.app.core.Sync.v2.replay import SyncReplayRepairer

    result = SyncReplayRepairer(
        store=service.store,
        materializers=service.materializers,
        materialize=lambda e, s: service._materialize_envelope(e, store=s),
        snapshot=lambda e, s: service._envelope_snapshot(e, store=s),
        scan_limit=10,
    ).run(dataset_id="personal", domains=["notes.provenance"], since_cursor=second.server_cursor - 1)
    assert result.failed_count == 0
    assert db.get_note_by_id("note")["version"] == 1
    assert db.note_provenance_store.get("note")["version"] == 1
    assert save(env).fully_applied


def test_post_product_pre_sync_failure_retries_without_half_checkpoint(env, monkeypatch):
    service, db = env
    original = service.store.db.upsert_object_state
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected checkpoint failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(service.store.db, "upsert_object_state", fail_second)
    with pytest.raises(SyncStoreError):
        save(env)
    assert db.get_note_by_id("note")["version"] == 1
    assert db.note_provenance_store.get("note")["version"] == 1
    assert service.store.get_object_state("personal", "notes.note", "note") is None
    monkeypatch.setattr(service.store.db, "upsert_object_state", original)
    assert save(env).fully_applied
    assert db.note_provenance_store.get("note")["version"] == 1


@pytest.mark.parametrize("route", ["client", "singleton", "batch"])
def test_parent_delete_expands_atomically_and_restore_is_explicit(env, route):
    service, db = env
    save(env)
    if route == "client":
        register(service)
        request = client(service, operation="tombstone", payload={}, key="delete")
        result = service.push(user_id="alice", dataset_id="personal", device_id="old", envelopes=[request])
        assert result.accepted and not result.conflicts
        assert service.push(user_id="alice", dataset_id="personal", device_id="old", envelopes=[request]).accepted
    elif route == "singleton":
        capture_server_origin_mutation(
            service,
            user_id="alice",
            domain="notes.note",
            operation="tombstone",
            object_id="note",
            payload={},
            source="delete",
            stable_key="delete",
        )
    else:
        for _ in range(2):
            capture_server_origin_mutation_batch(
                service=service,
                user_id="alice",
                steps=[
                    ServerOriginMutationStep(domain="notes.note", operation="tombstone", object_id="note", payload={})
                ],
                source="delete",
                idempotency_key="delete",
            )
    assert head(service).operation == "tombstone"
    assert db.note_provenance_store.get("note", include_deleted=True)["version"] == 2
    register(service)
    result = service.push(
        user_id="alice",
        dataset_id="personal",
        device_id="old",
        envelopes=[client(service, key="restore", restore=True)],
    )
    assert result.accepted
    assert db.note_provenance_store.get("note") is None
    register(service, ("notes.note", "notes.provenance"), device="new")
    ordinary = client(service, domain="notes.provenance", device="new", key="late")
    assert service.push(user_id="alice", dataset_id="personal", device_id="new", envelopes=[ordinary]).conflicts
    restored = client(service, domain="notes.provenance", device="new", key="restore-history", restore=True)
    assert service.push(user_id="alice", dataset_id="personal", device_id="new", envelopes=[restored]).accepted
    assert db.note_provenance_store.get("note")["version"] == 3


def test_preflight_delete_race_rejects_child_append(env, monkeypatch):
    service, db = env
    save(env)
    original = service.store.insert_envelopes_atomic

    def delete_before_append(envelopes, **kwargs):
        monkeypatch.setattr(service.store, "insert_envelopes_atomic", original)
        capture_server_origin_mutation(
            service,
            user_id="alice",
            domain="notes.note",
            operation="tombstone",
            object_id="note",
            payload={},
            source="racer",
        )
        return original(envelopes, **kwargs)

    monkeypatch.setattr(service.store, "insert_envelopes_atomic", delete_before_append)
    with pytest.raises(SyncStoreError):
        capture_server_origin_mutation_batch(
            service=service,
            user_id="alice",
            steps=[
                ServerOriginMutationStep(
                    domain="notes.provenance",
                    operation="upsert",
                    object_id="note",
                    parent_id="note",
                    payload={"origin": "reviewed_sources"},
                )
            ],
            source="late",
            idempotency_key="late",
        )
    assert head(service).operation == "tombstone"


def test_upgrade_preserves_product_revisions_and_tombstones(env):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    db.upsert_note_from_sync(
        note_id="note",
        title="Existing",
        content=retain_notes_provenance("text", PAYLOAD),
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=7,
        object_hash="ignored",
    )
    dataset = ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    assert dataset.metadata["notes_provenance_v1"]["state"] == "ready"
    assert db.get_note_by_id("note")["version"] == head(service, "notes.note").object_revision == 7
    assert head(service).object_revision == 1
    capture_server_origin_mutation(
        service,
        user_id="alice",
        domain="notes.provenance",
        operation="tombstone",
        object_id="note",
        parent_id="note",
        payload=PAYLOAD,
        source="remove-history",
    )
    ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    assert head(service).operation == "tombstone"
    assert db.note_provenance_store.get("note") is None


def test_client_private_and_bad_hash_rejected(env):
    service, _ = env
    save(env)
    dataset = service.store.get_dataset("personal")
    request = client(service, domain="notes.provenance", key="bad-hash")
    outcome = service._evaluate_envelope(dataset, replace(request, payload_hash="sha256:" + "0" * 64))
    assert outcome.error_code == "notes_provenance_payload_invalid"
    outcome = service.adapters.get("notes.provenance").evaluate_envelope(
        request, dataset=replace(dataset, encryption_policy="client_private_v1")
    )
    assert outcome.error_code == "notes_provenance_encryption_unsupported"


def test_old_plain_parent_is_source_verified_before_first_provenance_save(env):
    service, db = env
    db.upsert_note_from_sync(
        note_id="note",
        title="Plain",
        content="old",
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=9,
        object_hash="ignored",
    )
    result = save(env, core_expected=9)
    assert result.fully_applied
    assert db.get_note_by_id("note")["version"] == 10
    assert head(service).object_revision == 1


def test_independent_adapter_requires_all_three_exact_base_tokens(env):
    service, _ = env
    save(env)
    envelope = replace(client(service, domain="notes.provenance"), base_object_hash="wrong")
    from tldw_Server_API.app.core.Sync.v2.adapters import AdapterConflict

    assert isinstance(service._evaluate_envelope(service.store.get_dataset("personal"), envelope), AdapterConflict)


def test_new_child_after_preflight_cannot_follow_deleted_plain_parent(env, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    capture_server_origin_mutation(
        service,
        user_id="alice",
        domain="notes.note",
        operation="upsert",
        object_id="note",
        payload=NOTE,
        source="plain",
    )
    original = service.store.insert_envelopes_atomic

    def delete_before_append(envelopes, **kwargs):
        monkeypatch.setattr(service.store, "insert_envelopes_atomic", original)
        capture_server_origin_mutation(
            service,
            user_id="alice",
            domain="notes.note",
            operation="tombstone",
            object_id="note",
            payload={},
            source="delete",
        )
        return original(envelopes, **kwargs)

    monkeypatch.setattr(service.store, "insert_envelopes_atomic", delete_before_append)
    with pytest.raises(SyncStoreError):
        capture_server_origin_mutation_batch(
            service=service,
            user_id="alice",
            steps=[
                ServerOriginMutationStep(
                    domain="notes.provenance", operation="upsert", object_id="note", parent_id="note", payload=PAYLOAD
                )
            ],
            source="late",
            idempotency_key="late",
        )
    assert head(service) is None
    assert db.note_provenance_store.get("note", include_deleted=True) is None


def test_profile_explicit_enrollment_publishes_ready_and_blocks_metadata_forgery(env):
    service, db = env
    profile = service.bootstrap_profile(
        user_id="alice",
        mode="server_frontend",
        device_id="explicit",
        requested_domains=["notes.note", "notes.provenance"],
    )
    assert profile.dataset.notes_provenance["state"] == "ready"
    assert "notes.provenance" in profile.dataset.domains
    with pytest.raises(SyncStoreError):
        service.enroll_dataset(
            user_id="alice",
            domains=["notes.note", "notes.provenance"],
            metadata={"notes_provenance_v1": {"state": "ready"}},
        )


def test_interrupted_bootstrap_resumes_same_heads_and_versions(env, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    db.upsert_note_from_sync(
        note_id="note",
        title="Legacy",
        content=retain_notes_provenance("body", PAYLOAD),
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=6,
        object_hash="ignored",
    )
    original = service.store.db.mark_bootstrap_envelope_verified

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted after durable append")

    monkeypatch.setattr(service.store.db, "mark_bootstrap_envelope_verified", fail)
    with pytest.raises(SyncStoreError):
        ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    before = (head(service, "notes.note"), head(service))
    assert service.store.get_dataset("personal").metadata["notes_provenance_v1"]["state"] == "initializing"
    monkeypatch.setattr(service.store.db, "mark_bootstrap_envelope_verified", original)
    assert (
        ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice").metadata["notes_provenance_v1"][
            "state"
        ]
        == "ready"
    )
    assert (head(service, "notes.note").server_cursor, head(service).server_cursor) == tuple(
        item.server_cursor for item in before
    )
    assert db.get_note_by_id("note")["version"] == 6
    assert db.note_provenance_store.get("note")["version"] == 1


def test_bootstrap_never_revives_independently_deleted_marker_history(env):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    db.add_note("Legacy", retain_notes_provenance("body", PAYLOAD), note_id="note")
    db.note_provenance_store.put("note", PAYLOAD, expected_version=0)
    db.note_provenance_store.tombstone("note", expected_version=1)
    ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    assert head(service).operation == "tombstone"
    assert head(service).object_revision == 2
    assert db.note_provenance_store.get("note") is None


def test_existing_canonical_tombstone_wins_on_reenrollment(env):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    save(env, content=retain_notes_provenance("body", PAYLOAD))
    capture_server_origin_mutation(
        service,
        user_id="alice",
        domain="notes.provenance",
        operation="tombstone",
        object_id="note",
        parent_id="note",
        payload=PAYLOAD,
        source="remove-history",
    )
    current = service.store.get_dataset("personal").metadata["notes_provenance_v1"]
    service.store.db.transition_notes_provenance_bootstrap(
        "personal",
        bootstrap_id=current["bootstrap_id"],
        expected_state="ready",
        state="failed",
        captured_count=0,
        expected_count=0,
        source_hash=current["source_hash"],
    )
    ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    assert head(service).operation == "tombstone"
    assert head(service).object_revision == 2


def test_deleted_parent_blocks_explicit_child_restore(env):
    service, db = env
    save(env)
    capture_server_origin_mutation(
        service,
        user_id="alice",
        domain="notes.note",
        operation="tombstone",
        object_id="note",
        payload={},
        source="delete",
    )
    register(service, ("notes.note", "notes.provenance"), device="new")
    result = service.push(
        user_id="alice",
        dataset_id="personal",
        device_id="new",
        envelopes=[client(service, domain="notes.provenance", device="new", restore=True)],
    )
    assert result.rejected
    assert result.rejected[0].error_code == "notes_provenance_parent_missing"
    assert db.note_provenance_store.get("note") is None


def test_bootstrap_pages_only_authenticated_owner_parents(env):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    foreign = CharactersRAGDB(db.db_path, client_id="bob")
    try:
        foreign.add_note("Foreign", retain_notes_provenance("body", PAYLOAD), note_id="foreign")
        db.add_note("Own", "body", note_id="note")
        pages = db.note_provenance_store.list_parent_notes(limit=1)
        assert [row["id"] for row in pages] == ["note"]
        assert db.note_provenance_store.list_parent_notes(limit=1, after_note_id="note") == []
        ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
        assert service.store.get_current_head("personal", "notes.note", "foreign") is None
    finally:
        foreign.close_all_connections()


@pytest.mark.postgres
def test_postgres_pair_rollback_replay_and_parent_lifecycle(pg_database_config, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

    db = CharactersRAGDB(
        ":memory:", client_id="alice", backend=DatabaseBackendFactory.create_backend(pg_database_config)
    )
    store = SyncV2Store(SyncDatabase(backend=DatabaseBackendFactory.create_backend(pg_database_config)))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="personal",
            owner_user_id="alice",
            domains=["notes.note"],
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    service = SyncV2Service(
        store=store,
        adapters=default_sync_v2_registry(),
        materializers={"notes.note": NotesMaterializer(db)},
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="encrypted_volume", server_trusted_enabled=True, auth_mode="multi_user"
            )
        ),
    )
    original = db.note_provenance_store.apply_sync
    try:

        def fail(*args, **kwargs):
            raise RuntimeError("second product write")

        monkeypatch.setattr(db.note_provenance_store, "apply_sync", fail)
        with pytest.raises(SyncStoreError):
            save((service, db))
        assert db.get_note_by_id("note") is None
        monkeypatch.setattr(db.note_provenance_store, "apply_sync", original)
        assert save((service, db)).fully_applied
        capture_server_origin_mutation(
            service,
            user_id="alice",
            domain="notes.note",
            operation="tombstone",
            object_id="note",
            payload={},
            source="delete",
        )
        assert head(service).operation == "tombstone"
        assert db.note_provenance_store.get("note", include_deleted=True)["version"] == 2
    finally:
        db.close_all_connections()


def test_nonadjacent_core_and_provenance_plan_is_rejected_before_append(env):
    service, db = env
    save(env)
    previous = (head(service, "notes.note"), head(service))
    steps = [
        ServerOriginMutationStep(
            domain="notes.provenance", operation="upsert", object_id="note", parent_id="note", payload=PAYLOAD
        ),
        ServerOriginMutationStep(
            domain="notes.note", operation="upsert", object_id="note", payload={**NOTE, "content": "reordered"}
        ),
    ]
    with pytest.raises(SyncStoreError):
        capture_server_origin_mutation_batch(
            service=service, user_id="alice", steps=steps, source="invalid", idempotency_key="invalid"
        )
    assert (head(service, "notes.note"), head(service)) == previous
    assert db.get_note_by_id("note")["content"] == "Body"


def test_parent_delete_race_with_first_child_fails_without_orphan(env, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    capture_server_origin_mutation(
        service,
        user_id="alice",
        domain="notes.note",
        operation="upsert",
        object_id="note",
        payload=NOTE,
        source="plain",
    )
    original = service.store.db.insert_envelope

    def create_child_before_delete(envelope, **kwargs):
        monkeypatch.setattr(service.store.db, "insert_envelope", original)
        capture_server_origin_mutation(
            service,
            user_id="alice",
            domain="notes.provenance",
            operation="upsert",
            object_id="note",
            parent_id="note",
            payload=PAYLOAD,
            source="child",
        )
        return original(envelope, **kwargs)

    monkeypatch.setattr(service.store.db, "insert_envelope", create_child_before_delete)
    with pytest.raises(SyncStoreError):
        capture_server_origin_mutation(
            service,
            user_id="alice",
            domain="notes.note",
            operation="tombstone",
            object_id="note",
            payload={},
            source="delete",
        )
    assert head(service, "notes.note").operation == "upsert"
    assert db.note_provenance_store.get("note") is not None


def test_second_product_delete_failure_preserves_active_pair(env, monkeypatch):
    service, db = env
    save(env)
    original = db.note_provenance_store.apply_sync

    def fail(*args, **kwargs):
        raise RuntimeError("child delete projection failed")

    monkeypatch.setattr(db.note_provenance_store, "apply_sync", fail)
    with pytest.raises(SyncStoreError):
        capture_server_origin_mutation(
            service,
            user_id="alice",
            domain="notes.note",
            operation="tombstone",
            object_id="note",
            payload={},
            source="delete",
            stable_key="delete",
        )
    assert db.get_note_by_id("note") is not None
    assert db.note_provenance_store.get("note")["version"] == 1
    monkeypatch.setattr(db.note_provenance_store, "apply_sync", original)
    capture_server_origin_mutation(
        service,
        user_id="alice",
        domain="notes.note",
        operation="tombstone",
        object_id="note",
        payload={},
        source="delete",
        stable_key="delete",
    )
    assert db.note_provenance_store.get("note", include_deleted=True)["version"] == 2


def test_repair_of_interrupted_bootstrap_never_rewinds_changed_product(env, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready
    from tldw_Server_API.app.core.Sync.v2.replay import SyncReplayRepairer

    service, db = env
    db.upsert_note_from_sync(
        note_id="note",
        title="Legacy",
        content=retain_notes_provenance("body", PAYLOAD),
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=6,
        object_hash="ignored",
    )
    original = service.store.db.mark_bootstrap_envelope_verified

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted before verified checkpoint")

    monkeypatch.setattr(service.store.db, "mark_bootstrap_envelope_verified", fail)
    with pytest.raises(SyncStoreError):
        ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    monkeypatch.setattr(service.store.db, "mark_bootstrap_envelope_verified", original)
    db.upsert_note_from_sync(
        note_id="note",
        title="Changed",
        content="Current version",
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=7,
        object_hash="ignored",
    )
    result = SyncReplayRepairer(
        store=service.store,
        materializers=service.materializers,
        materialize=lambda e, s: service._materialize_envelope(e, store=s),
        snapshot=lambda e, s: service._envelope_snapshot(e, store=s),
        scan_limit=10,
    ).run(dataset_id="personal", domains=["notes.provenance"])
    assert result.failed_count > 0
    assert db.get_note_by_id("note")["version"] == 7
    assert db.get_note_by_id("note")["content"] == "Current version"


def test_single_parent_bootstrap_repair_is_source_verified(env, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    db.upsert_note_from_sync(
        note_id="note",
        title="Plain",
        content="Before",
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=6,
        object_hash="ignored",
    )
    original = service.store.db.mark_bootstrap_envelope_verified

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted before verified checkpoint")

    monkeypatch.setattr(service.store.db, "mark_bootstrap_envelope_verified", fail)
    with pytest.raises(SyncStoreError):
        ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice", required_note_id="note")
    monkeypatch.setattr(service.store.db, "mark_bootstrap_envelope_verified", original)
    db.upsert_note_from_sync(
        note_id="note",
        title="Plain",
        content="After",
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=7,
        object_hash="ignored",
    )
    assert service._materialize_envelope(head(service, "notes.note")).status == "failed"
    assert db.get_note_by_id("note")["version"] == 7


@pytest.mark.parametrize("revision", [True, 2.0, 9_007_199_254_740_992])
def test_provenance_adapter_rejects_noncanonical_revision(env, revision):
    from tldw_Server_API.app.core.Sync.v2.adapters import AdapterRejected

    service, _ = env
    save(env)
    request = replace(client(service, domain="notes.provenance"), object_revision=revision)
    assert isinstance(service._evaluate_envelope(service.store.get_dataset("personal"), request), AdapterRejected)


def conflict_delete(env):
    service, _ = env
    save(env)
    register(service)
    stale_base = head(service, "notes.note")
    assert service.push(
        user_id="alice",
        dataset_id="personal",
        device_id="old",
        envelopes=[client(service, payload={**NOTE, "content": "Edited"}, key="edit")],
    ).accepted
    conflicted = service.push(
        user_id="alice",
        dataset_id="personal",
        device_id="old",
        envelopes=[client(service, base=stale_base, key="stale")],
    )
    return conflicted.conflicts[0].conflict_id, client(service, operation="tombstone", payload={}, key="resolution")


@pytest.mark.parametrize("env", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)], indirect=True)
@pytest.mark.parametrize("route", ["single", "batch"])
def test_conflict_parent_delete_retries_complete_pair_and_requires_explicit_restore(env, route):
    service, db = env
    conflict_id, request = conflict_delete(env)

    def resolve(envelope=request):
        if route == "batch":
            resolved, rejected, _ = service.resolve_conflicts_batch(
                user_id="alice",
                dataset_id="personal",
                device_id="old",
                resolutions=[(conflict_id, "overwrite", envelope, None, None, None)],
            )
            assert not rejected
            return resolved[0][1]
        return service.resolve_conflict(
            user_id="alice",
            dataset_id="personal",
            conflict_id=conflict_id,
            action="overwrite",
            resolved_by_device_id="old",
            resolution_envelope=envelope,
        )

    resolved = resolve()
    assert resolved.status == "resolved"
    parent, child = head(service, "notes.note"), head(service)
    assert child.operation == "tombstone"
    group = service.store.list_mutation_group("personal", parent.mutation_group_id)
    assert [(item.domain, item.operation, item.apply_status) for item in group] == [
        ("notes.note", "tombstone", "applied"),
        ("notes.provenance", "tombstone", "applied"),
    ]
    assert child.server_cursor == parent.server_cursor + 1
    retained = db.note_provenance_store.get("note", include_deleted=True)
    assert retained["version"] == child.object_revision == 2
    assert retained["object_hash"] == child.payload_hash
    assert resolve() == resolved  # Lost response must replay the complete original decision.
    assert (head(service, "notes.note"), head(service)) == (parent, child)

    assert service.push(
        user_id="alice",
        dataset_id="personal",
        device_id="old",
        envelopes=[client(service, key="parent-restore", restore=True)],
    ).accepted
    assert db.note_provenance_store.get("note") is None
    register(service, ("notes.note", "notes.provenance"), device="new")
    assert service.push(
        user_id="alice",
        dataset_id="personal",
        device_id="new",
        envelopes=[client(service, domain="notes.provenance", device="new", key="child-restore", restore=True)],
    ).accepted
    assert db.note_provenance_store.get("note")["version"] == 3
    assert resolve() == resolved  # Later restores cannot change retry identity or re-delete either head.
    assert head(service).operation == head(service, "notes.note").operation == "upsert"
    changed = replace(request, payload={"changed": True}, payload_hash=canonical_payload_hash({"changed": True})[0])
    if route == "single":
        with pytest.raises(SyncStoreError, match="already resolved"):
            resolve(changed)
    else:
        _, rejected, _ = service.resolve_conflicts_batch(
            user_id="alice",
            dataset_id="personal",
            device_id="old",
            resolutions=[(conflict_id, "overwrite", changed, None, None, None)],
        )
        assert rejected == [0]


@pytest.mark.parametrize("failure", ["product", "checkpoint"])
def test_conflict_parent_delete_failure_preserves_claim_retry_and_pair(env, monkeypatch, failure):
    service, db = env
    conflict_id, request = conflict_delete(env)
    before = (head(service, "notes.note"), head(service))
    target = db.note_provenance_store if failure == "product" else service.store.db
    method = "apply_sync" if failure == "product" else "upsert_object_state"
    original = getattr(target, method)

    def fail(*args, **kwargs):
        if failure == "product" or args[0].domain == "notes.provenance":
            raise RuntimeError("injected pair failure")
        return original(*args, **kwargs)

    def resolve():
        return service.resolve_conflict(
            user_id="alice",
            dataset_id="personal",
            conflict_id=conflict_id,
            action="overwrite",
            resolved_by_device_id="old",
            resolution_envelope=request,
        )

    monkeypatch.setattr(target, method, fail)
    with pytest.raises(SyncStoreError):
        resolve()
    assert (head(service, "notes.note"), head(service)) == before
    conflict = service.store.get_conflict(conflict_id)
    assert conflict.status == "unresolved" and conflict.resolution_action is None
    assert service.store.get_object_state("personal", "notes.provenance", "note").object_revision == 1
    retained = db.note_provenance_store.get("note", include_deleted=True)
    assert retained["deleted"] == (failure == "checkpoint")
    assert (db.get_note_by_id("note") is None) == (failure == "checkpoint")
    monkeypatch.setattr(target, method, original)
    assert resolve().status == "resolved"
    assert head(service).operation == "tombstone"
    assert db.note_provenance_store.get("note", include_deleted=True)["version"] == 2


def test_claimed_database_append_fences_unexpanded_parent_delete(env):
    service, _ = env
    conflict_id, request = conflict_delete(env)
    conflict = service.store.get_conflict(conflict_id)
    source = service.store.get_envelope_by_server_cursor(conflict.server_sequence)
    before = (head(service, "notes.note"), head(service))
    with pytest.raises(SyncStoreError):
        with service.store.materialization_guard([source], require_predecessors=False) as guarded:
            guarded.claim_conflict_resolution(
                conflict_id,
                dataset_id="personal",
                resolved_by_device_id="old",
                resolution_action="overwrite",
                resolution_notes=None,
            )
            guarded.db.insert_claimed_conflict_resolution_envelope(
                request,
                conflict_id=conflict_id,
                dataset_id="personal",
                resolved_by_device_id="old",
                resolution_action="overwrite",
                resolution_notes=None,
                connection=guarded._connection,
            )
    assert (head(service, "notes.note"), head(service)) == before
    assert service.store.get_conflict(conflict_id).resolution_action is None


def test_unrelated_large_plain_note_does_not_block_small_sourced_create(env):
    service, db = env
    db.add_note("Large plain", "x" * 300_000, note_id="unrelated-large")
    assert save(env).fully_applied
    assert service.store.get_current_head("personal", "notes.note", "unrelated-large") is None
    assert len(db.get_note_by_id("unrelated-large")["content"]) == 300_000
    assert service.store.get_dataset("personal").metadata["notes_provenance_v1"]["expected_count"] == 0


def test_ready_profile_source_verifies_only_requested_existing_plain_parent(env):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    db.add_note("Unrelated", "x" * 300_000, note_id="unrelated-large")
    db.upsert_note_from_sync(
        note_id="note",
        title="Plain",
        content="Original",
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=9,
        object_hash="ignored",
    )
    assert save(env, core_expected=9).fully_applied
    assert db.get_note_by_id("note")["version"] == 10
    assert head(service).object_revision == 1
    assert service.store.get_current_head("personal", "notes.note", "unrelated-large") is None


def test_oversized_provenance_parent_still_fails_closed_without_losing_evidence(env):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    db.add_note("Required source", "x" * 300_000, note_id="required")
    retained = db.note_provenance_store.put("required", PAYLOAD, expected_version=0)
    with pytest.raises(SyncStoreError, match="notes_provenance_sync_not_ready"):
        ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    assert db.note_provenance_store.get("required") == retained
    assert service.store.get_dataset("personal").metadata["notes_provenance_v1"]["state"] == "failed"


def test_required_large_plain_parent_does_not_poison_ready_profile(env):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    db.add_note("Large requested parent", "x" * 300_000, note_id="large-target")
    with pytest.raises(SyncStoreError, match="payload exceeds"):
        ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice", required_note_id="large-target")
    assert service.store.get_current_head("personal", "notes.note", "large-target") is None
    assert save(env).fully_applied


def test_ready_profile_required_parent_bootstrap_resumes_exact_source(env, monkeypatch):
    from tldw_Server_API.app.core.Sync.v2.notes_provenance import ensure_notes_provenance_ready

    service, db = env
    ensure_notes_provenance_ready(service=service, note_db=db, user_id="alice")
    db.upsert_note_from_sync(
        note_id="note",
        title="Plain",
        content="Original",
        conversation_id=None,
        message_id=None,
        sync_client_id="alice",
        object_revision=9,
        object_hash="ignored",
    )
    original = service.store.db.mark_bootstrap_envelope_verified

    def fail(*args, **kwargs):
        raise RuntimeError("interrupted required parent checkpoint")

    monkeypatch.setattr(service.store.db, "mark_bootstrap_envelope_verified", fail)
    with pytest.raises(SyncStoreError):
        save(env, core_expected=9)
    source_head = head(service, "notes.note")
    assert source_head.object_revision == db.get_note_by_id("note")["version"] == 9
    assert service.store.get_dataset("personal").metadata["notes_provenance_v1"]["state"] == "ready"
    monkeypatch.setattr(service.store.db, "mark_bootstrap_envelope_verified", original)
    result = save(env, core_expected=9)
    assert result.fully_applied
    assert result.envelopes[0].base_server_cursor == source_head.server_cursor
    assert db.get_note_by_id("note")["version"] == 10
    assert save(env, core_expected=9).fully_applied
