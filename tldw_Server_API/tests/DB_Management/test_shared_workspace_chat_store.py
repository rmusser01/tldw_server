"""SQLite behavior contracts for the recipient shared-workspace chat store."""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store import (
    SharedWorkspaceChatStore,
    SharedWorkspaceCursorInputError,
    StaleSharedWorkspaceChatClaim,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)

NOW = datetime(2026, 8, 21, 20, 0, tzinfo=timezone.utc)


@pytest.fixture()
def db(tmp_path) -> CharactersRAGDB:
    database = CharactersRAGDB(tmp_path / "recipient-chat.db", client_id=" recipient-a ")
    try:
        yield database
    finally:
        database.close_all_connections()


def _thread(db: CharactersRAGDB, *, share_id: int = 41):
    return db.shared_workspace_chat_store.get_or_create_thread(
        share_id=share_id,
        owner_user_id="owner-7",
        workspace_id="workspace-alpha",
        workspace_name="Evidence review",
    )


def _claim(
    db: CharactersRAGDB,
    *,
    share_id: int = 41,
    request_id: UUID | None = None,
    fingerprint: str = "fingerprint-a",
    now: datetime = NOW,
    lease_seconds: int = 600,
):
    thread = db.shared_workspace_chat_store.get_thread(share_id=share_id) or _thread(
        db, share_id=share_id
    )
    return db.shared_workspace_chat_store.claim_request(
        share_id=share_id,
        request_id=request_id or uuid4(),
        request_fingerprint=fingerprint,
        conversation_id=thread.conversation_id,
        lease_seconds=lease_seconds,
        now=now,
    )


def _citations() -> list[dict[str, object]]:
    return [
        {
            "citation_id": "citation-1",
            "source_id": "source-a",
            "source_title": "Primary report",
            "locator": {"chunk": 4},
            "quote": "Bounded supporting passage",
            "score": 0.87,
        }
    ]


def _complete(db: CharactersRAGDB, claim, *, query: str = "Question", answer: str = "Answer"):
    assert db.shared_workspace_chat_store.freeze_sources(
        claim=claim,
        source_mode="include",
        source_ids=("source-a",),
        snapshot_hash="snapshot-a",
        provider="llama",
        model="configured-model",
    )
    return db.shared_workspace_chat_store.complete_turn(
        claim=claim,
        query=query,
        answer=answer,
        citations=_citations(),
        provider="llama",
        model="configured-model",
        source_mode="include",
        effective_source_count=1,
    )


def test_store_normalizes_private_recipient_key_and_rejects_blank() -> None:
    db = SimpleNamespace(client_id=123)
    store = SharedWorkspaceChatStore(db)

    assert store._recipient_user_id == "123"
    with pytest.raises(InputError, match="recipient"):
        SharedWorkspaceChatStore(SimpleNamespace(client_id="   "))
    with pytest.raises(InputError, match="recipient"):
        SharedWorkspaceChatStore(SimpleNamespace(client_id=None))


def test_get_or_create_thread_uses_canonical_conversation_fields(db: CharactersRAGDB) -> None:
    thread = _thread(db)

    conversation = db.get_conversation_by_id(thread.conversation_id)
    assert conversation is not None
    assert thread.share_id == 41
    assert thread.owner_user_id == "owner-7"
    assert thread.workspace_id == "workspace-alpha"
    assert conversation["source"] == "shared_workspace"
    assert conversation["external_ref"] == "share:41"
    assert conversation["scope_type"] == "global"
    assert conversation["workspace_id"] is None
    assert conversation["client_id"] == "recipient-a"

    row = db.execute_query(
        "SELECT recipient_user_id FROM shared_workspace_chat_threads WHERE share_id = ?",
        (41,),
    ).fetchone()
    assert row["recipient_user_id"] == "recipient-a"
    assert _thread(db).conversation_id == thread.conversation_id


def test_get_or_create_thread_initializes_empty_chat_settings(db: CharactersRAGDB) -> None:
    thread = _thread(db)

    settings = db.get_conversation_settings(thread.conversation_id)

    assert settings is not None
    assert settings["settings"] == {}


def test_get_or_create_thread_repairs_missing_chat_settings(db: CharactersRAGDB) -> None:
    thread = _thread(db)
    db.execute_query(
        "DELETE FROM conversation_settings WHERE conversation_id = ?",
        (thread.conversation_id,),
        commit=True,
    )
    assert db.get_conversation_settings(thread.conversation_id) is None

    reopened = _thread(db)

    assert reopened.conversation_id == thread.conversation_id
    assert db.get_conversation_settings(thread.conversation_id)["settings"] == {}


def test_get_or_create_thread_preserves_existing_chat_settings(db: CharactersRAGDB) -> None:
    thread = _thread(db)
    expected = {"authorNote": "Keep this recipient setting."}
    assert db.upsert_conversation_settings(thread.conversation_id, expected)

    reopened = _thread(db)

    assert reopened.conversation_id == thread.conversation_id
    assert db.get_conversation_settings(thread.conversation_id)["settings"] == expected


def test_concurrent_first_thread_creation_returns_one_mapping(tmp_path) -> None:
    path = tmp_path / "concurrent-recipient-chat.db"
    databases = [CharactersRAGDB(path, client_id="recipient-a") for _ in range(2)]
    try:
        def create(database: CharactersRAGDB):
            return database.shared_workspace_chat_store.get_or_create_thread(
                share_id=91,
                owner_user_id="owner-a",
                workspace_id="workspace-a",
                workspace_name="Concurrent workspace",
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            threads = list(executor.map(create, databases))

        assert threads[0].conversation_id == threads[1].conversation_id
        rows = databases[0].execute_query(
            "SELECT conversation_id FROM shared_workspace_chat_threads WHERE share_id = ?",
            (91,),
        ).fetchall()
        conversations = databases[0].execute_query(
            "SELECT id FROM conversations WHERE external_ref = ? AND deleted = 0",
            ("share:91",),
        ).fetchall()
        settings_rows = databases[0].execute_query(
            "SELECT conversation_id FROM conversation_settings WHERE conversation_id = ?",
            (threads[0].conversation_id,),
        ).fetchall()
        assert len(rows) == 1
        assert len(conversations) == 1
        assert len(settings_rows) == 1
    finally:
        for database in databases:
            database.close_all_connections()


def test_claim_is_insert_first_clamps_lease_and_preserves_fingerprint_owner(
    db: CharactersRAGDB,
) -> None:
    thread = _thread(db)
    request_id = uuid4()
    claimed = _claim(db, request_id=request_id, lease_seconds=1)

    assert claimed.disposition == "claimed"
    assert claimed.lease_epoch == 1
    assert claimed.lease_token
    assert claimed.lease_expires_at == NOW + timedelta(minutes=5)

    active = _claim(db, request_id=request_id, lease_seconds=60, now=NOW)
    assert active.disposition == "in_progress"
    assert active.retry_after_ms == 300_000
    assert active.lease_token is None

    before = dict(
        db.execute_query(
            "SELECT * FROM shared_workspace_chat_requests WHERE request_id = ?",
            (str(request_id),),
        ).fetchone()
    )
    conflict = _claim(db, request_id=request_id, fingerprint="different")
    after = dict(
        db.execute_query(
            "SELECT * FROM shared_workspace_chat_requests WHERE request_id = ?",
            (str(request_id),),
        ).fetchone()
    )
    assert conflict.disposition == "request_id_conflict"
    assert before == after
    assert before["conversation_id"] == thread.conversation_id
    assert before["recipient_user_id"] == "recipient-a"

    maximum = _claim(db, request_id=uuid4(), lease_seconds=99_999)
    assert maximum.lease_expires_at == NOW + timedelta(minutes=30)


def test_retryable_and_expired_claims_reclaim_with_one_new_fence(db: CharactersRAGDB) -> None:
    _thread(db)
    retry_id = uuid4()
    first = _claim(db, request_id=retry_id)
    assert db.shared_workspace_chat_store.mark_retryable(
        claim=first, error_code="generation_failed"
    )

    reclaimed = _claim(db, request_id=retry_id, now=NOW + timedelta(seconds=1))
    assert reclaimed.disposition == "claimed"
    assert reclaimed.lease_epoch == first.lease_epoch + 1
    assert reclaimed.lease_token != first.lease_token

    expired_id = uuid4()
    expired = _claim(db, request_id=expired_id, lease_seconds=300)
    reclaimed_expired = _claim(
        db,
        request_id=expired_id,
        now=NOW + timedelta(minutes=5, seconds=1),
    )
    assert reclaimed_expired.disposition == "claimed"
    assert reclaimed_expired.lease_epoch == expired.lease_epoch + 1
    assert reclaimed_expired.lease_token != expired.lease_token


@pytest.mark.parametrize(
    ("winner_status", "expected_disposition"),
    [
        ("completed", "replay"),
        ("retryable", "claimed"),
        ("conflicted", "request_id_conflict"),
        ("in_progress", "in_progress"),
    ],
)
def test_reclaim_cas_loser_reclassifies_fresh_winner_state(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    winner_status: str,
    expected_disposition: str,
) -> None:
    thread = _thread(db)
    request_id = uuid4()
    first = _claim(db, request_id=request_id)
    assert db.shared_workspace_chat_store.mark_retryable(
        claim=first, error_code="generation_failed"
    )

    store = db.shared_workspace_chat_store
    original_fetch = store._fetch_receipt_with_conn
    raced = False
    replay_marker = object()

    def fetch_then_publish_winner(conn, share_id, loaded_request_id):
        nonlocal raced
        row = original_fetch(conn, share_id, loaded_request_id)
        if not raced:
            raced = True
            lease_token = "winner-token" if winner_status == "in_progress" else None
            lease_expires_at = (
                store._db_datetime(NOW + timedelta(minutes=10))
                if winner_status == "in_progress"
                else None
            )
            conn.execute(
                "UPDATE shared_workspace_chat_requests "
                "SET status = ?, lease_epoch = lease_epoch + 1, lease_token = ?, "
                "lease_expires_at = ?, completed_at = ? "
                "WHERE share_id = ? AND request_id = ?",
                (
                    winner_status,
                    lease_token,
                    lease_expires_at,
                    store._db_datetime(NOW) if winner_status == "completed" else None,
                    share_id,
                    str(loaded_request_id),
                ),
            )
        return row

    monkeypatch.setattr(store, "_fetch_receipt_with_conn", fetch_then_publish_winner)
    if winner_status == "completed":
        monkeypatch.setattr(store, "load_completed_turn", lambda **_kwargs: replay_marker)

    result = store.claim_request(
        share_id=41,
        request_id=request_id,
        request_fingerprint="fingerprint-a",
        conversation_id=thread.conversation_id,
        lease_seconds=600,
        now=NOW + timedelta(seconds=1),
    )

    assert result.disposition == expected_disposition
    assert result.lease_epoch == first.lease_epoch + (
        2 if winner_status == "retryable" else 1
    )
    if winner_status == "completed":
        assert result.completed_turn is replay_marker
    elif winner_status == "retryable":
        assert result.lease_token not in {None, first.lease_token, "winner-token"}
    elif winner_status == "in_progress":
        assert result.lease_token is None
        assert result.retry_after_ms == 599_000


def test_claim_requires_aware_utc_now_and_bounds_retry_timing(db: CharactersRAGDB) -> None:
    _thread(db)
    with pytest.raises(InputError, match="aware"):
        _claim(db, now=datetime(2026, 8, 21, 20, 0))
    for invalid_lease in (True, "600"):
        with pytest.raises(InputError, match="lease_seconds"):
            _claim(db, lease_seconds=invalid_lease)  # type: ignore[arg-type]

    request_id = uuid4()
    _claim(db, request_id=request_id, lease_seconds=1)
    late = _claim(
        db,
        request_id=request_id,
        now=NOW + timedelta(minutes=4, seconds=59, milliseconds=900),
    )
    assert late.disposition == "in_progress"
    assert 0 <= late.retry_after_ms <= 1_800_000


def test_reload_claim_state_reads_active_and_completed_without_reclaim(
    db: CharactersRAGDB,
) -> None:
    claim = _claim(db)

    active = db.shared_workspace_chat_store.reload_claim_state(claim=claim, now=NOW)

    assert active is not None
    assert active.disposition == "in_progress"
    assert active.lease_epoch == claim.lease_epoch
    assert active.retry_after_ms == 600_000

    completed_turn = _complete(db, claim)
    completed = db.shared_workspace_chat_store.reload_claim_state(claim=claim, now=NOW)

    assert completed is not None
    assert completed.disposition == "replay"
    assert completed.completed_turn == completed_turn


@pytest.mark.parametrize(
    ("source_mode", "source_ids"),
    [
        ("exclude", ("source-a",)),
        ("include", ()),
        ("include", ("source-b", "source-a")),
        ("include", ("source-a", "source-a")),
        ("include", ("",)),
        ("include", tuple(f"source-{index:03d}" for index in range(501))),
    ],
)
def test_freeze_sources_rejects_noncanonical_or_unbounded_source_json(
    db: CharactersRAGDB,
    source_mode: str,
    source_ids: tuple[str, ...],
) -> None:
    claim = _claim(db)
    with pytest.raises(InputError):
        db.shared_workspace_chat_store.freeze_sources(
            claim=claim,
            source_mode=source_mode,
            source_ids=source_ids,
            snapshot_hash="snapshot",
            provider="llama",
            model="model",
        )


def test_frozen_sources_are_immutable_and_every_transition_is_fenced(db: CharactersRAGDB) -> None:
    claim = _claim(db)
    stale = replace(claim, lease_token="stale-token")

    assert not db.shared_workspace_chat_store.freeze_sources(
        claim=stale,
        source_mode="include",
        source_ids=("source-a",),
        snapshot_hash="snapshot-a",
        provider="llama",
        model="model-a",
    )
    assert db.shared_workspace_chat_store.freeze_sources(
        claim=claim,
        source_mode="include",
        source_ids=("source-a",),
        snapshot_hash="snapshot-a",
        provider="llama",
        model="model-a",
    )
    assert db.shared_workspace_chat_store.freeze_sources(
        claim=claim,
        source_mode="include",
        source_ids=("source-a",),
        snapshot_hash="snapshot-a",
        provider="llama",
        model="model-a",
    )
    assert not db.shared_workspace_chat_store.freeze_sources(
        claim=claim,
        source_mode="include",
        source_ids=("source-b",),
        snapshot_hash="snapshot-b",
        provider="llama",
        model="model-b",
    )
    assert not db.shared_workspace_chat_store.mark_retryable(
        claim=stale, error_code="stale"
    )
    assert not db.shared_workspace_chat_store.mark_conflicted(
        claim=stale, error_code="stale"
    )
    with pytest.raises(StaleSharedWorkspaceChatClaim):
        db.shared_workspace_chat_store.complete_turn(
            claim=stale,
            query="Question",
            answer="Answer",
            citations=_citations(),
            provider="llama",
            model="model-a",
            source_mode="include",
            effective_source_count=1,
        )


def test_conflicted_transition_requires_a_frozen_source_set(db: CharactersRAGDB) -> None:
    claim = _claim(db)

    assert not db.shared_workspace_chat_store.mark_conflicted(
        claim=claim,
        error_code="shared_source_changed",
    )
    assert db.shared_workspace_chat_store.mark_retryable(
        claim=claim,
        error_code="generation_failed",
    )


def test_idempotent_freeze_does_not_mutate_an_existing_frozen_receipt(
    db: CharactersRAGDB,
) -> None:
    claim = _claim(db)
    assert db.shared_workspace_chat_store.freeze_sources(
        claim=claim,
        source_mode="include",
        source_ids=("source-a",),
        snapshot_hash="snapshot-a",
        provider="llama",
        model="model-a",
    )
    first_updated_at = db.execute_query(
        "SELECT updated_at FROM shared_workspace_chat_requests WHERE request_id = ?",
        (str(claim.request_id),),
    ).fetchone()["updated_at"]
    assert "T" in first_updated_at
    assert first_updated_at.endswith("+00:00")
    sentinel = "2026-08-21T19:00:00+00:00"
    with db.transaction() as conn:
        conn.execute(
            "UPDATE shared_workspace_chat_requests SET updated_at = ? WHERE request_id = ?",
            (sentinel, str(claim.request_id)),
        )

    assert db.shared_workspace_chat_store.freeze_sources(
        claim=claim,
        source_mode="include",
        source_ids=("source-a",),
        snapshot_hash="snapshot-a",
        provider="llama",
        model="model-a",
    )
    row = db.execute_query(
        "SELECT updated_at FROM shared_workspace_chat_requests WHERE request_id = ?",
        (str(claim.request_id),),
    ).fetchone()
    assert row["updated_at"] == sentinel


def test_completion_is_atomic_and_loads_exact_stored_turn(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = _claim(db)
    assert db.shared_workspace_chat_store.freeze_sources(
        claim=claim,
        source_mode="include",
        source_ids=("source-a",),
        snapshot_hash="snapshot-a",
        provider="llama",
        model="configured-model",
    )

    def fail_metadata(*_args, **_kwargs) -> None:
        raise CharactersRAGDBError("strict metadata write failed")

    monkeypatch.setattr(
        db.shared_workspace_chat_store,
        "_write_message_metadata_strict",
        fail_metadata,
    )
    with pytest.raises(CharactersRAGDBError, match="metadata"):
        db.shared_workspace_chat_store.complete_turn(
            claim=claim,
            query="Question",
            answer="Answer",
            citations=_citations(),
            provider="llama",
            model="configured-model",
            source_mode="include",
            effective_source_count=1,
        )

    assert db.get_messages_for_conversation(claim.conversation_id) == []
    receipt = db.execute_query(
        "SELECT status, user_message_id, assistant_message_id FROM shared_workspace_chat_requests "
        "WHERE request_id = ?",
        (str(claim.request_id),),
    ).fetchone()
    assert tuple(receipt) == ("in_progress", None, None)

    monkeypatch.undo()
    stored = db.shared_workspace_chat_store.complete_turn(
        claim=claim,
        query="Question",
        answer="Answer",
        citations=_citations(),
        provider="llama",
        model="configured-model",
        source_mode="include",
        effective_source_count=1,
    )
    loaded = db.shared_workspace_chat_store.load_completed_turn(
        share_id=41, request_id=claim.request_id
    )

    assert loaded == stored
    assert stored.user_message.role == "user"
    assert stored.user_message.content == "Question"
    assert stored.assistant_message.role == "assistant"
    assert stored.assistant_message.content == "Answer"
    assert stored.citations == tuple(_citations())
    assert stored.provider == "llama"
    assert stored.model == "configured-model"
    assert stored.source_mode == "include"
    assert stored.effective_source_count == 1

    metadata = db.get_message_metadata(stored.assistant_message.message_id)
    rag_context = metadata["extra"]["rag_context"]
    assert rag_context["retrieved_documents"] == [
        {"source_id": "source-a", "quote": "Bounded supporting passage"}
    ]
    assert "media_id" not in json.dumps(rag_context)
    assert "Question" not in json.dumps(rag_context)

    replay = _claim(db, request_id=claim.request_id)
    assert replay.disposition == "replay"
    assert replay.completed_turn == stored


@pytest.mark.parametrize(
    "citations",
    [
        [{**_citations()[0], "media_id": 123}],
        [{**_citations()[0], "quote": "x" * 1001}],
        [_citations()[0] for _ in range(21)],
        [{**_citations()[0], "source_id": ""}],
        [{**_citations()[0], "source_id": "source-outside-frozen-scope"}],
        [{**_citations()[0], "score": float("nan")}],
    ],
)
def test_strict_citation_validation_precedes_message_persistence(
    db: CharactersRAGDB,
    citations: list[dict[str, object]],
) -> None:
    claim = _claim(db)
    assert db.shared_workspace_chat_store.freeze_sources(
        claim=claim,
        source_mode="include",
        source_ids=("source-a",),
        snapshot_hash="snapshot-a",
        provider="llama",
        model="model-a",
    )
    with pytest.raises(InputError):
        db.shared_workspace_chat_store.complete_turn(
            claim=claim,
            query="Question",
            answer="Answer",
            citations=citations,
            provider="llama",
            model="model-a",
            source_mode="include",
            effective_source_count=1,
        )
    assert db.get_messages_for_conversation(claim.conversation_id) == []


def test_history_uses_opaque_stable_cursor_and_returns_chronological_pages(
    db: CharactersRAGDB,
) -> None:
    _thread(db)
    turns = [
        _complete(db, _claim(db, request_id=uuid4()), query=f"Q{index}", answer=f"A{index}")
        for index in range(3)
    ]

    first = db.shared_workspace_chat_store.list_messages(share_id=41, before=None, limit=3)
    second = db.shared_workspace_chat_store.list_messages(
        share_id=41, before=first.next_before, limit=3
    )
    first_ids = [message.message_id for message in first.messages]
    second_ids = [message.message_id for message in second.messages]

    assert [message.content for message in first.messages] == ["A1", "Q2", "A2"]
    assert [message.content for message in second.messages] == ["Q0", "A0", "Q1"]
    assert set(first_ids).isdisjoint(second_ids)
    assert first.next_before is not None
    assert second.next_before is None
    assert first.messages[-1].citations == turns[-1].citations

    invalid_timestamp_cursor = base64.urlsafe_b64encode(
        json.dumps(
            ["not-a-timestamp", NOW.isoformat(), "message-1"],
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    invalid_message_id_cursor = base64.urlsafe_b64encode(
        json.dumps(
            [NOW.isoformat(), NOW.isoformat(), "\ud800"],
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii").rstrip("=")
    for invalid in (
        "not-base64",
        "e30",
        "",
        "\ud800",
        "A" * 2049,
        invalid_timestamp_cursor,
        invalid_message_id_cursor,
    ):
        with pytest.raises(SharedWorkspaceCursorInputError, match="cursor"):
            db.shared_workspace_chat_store.list_messages(
                share_id=41, before=invalid, limit=3
            )
    with pytest.raises(InputError, match="limit"):
        db.shared_workspace_chat_store.list_messages(share_id=41, before=None, limit=101)


def test_history_cursor_preserves_equal_timestamp_text_for_tuple_pagination(
    db: CharactersRAGDB,
) -> None:
    _thread(db)
    for index in range(3):
        _complete(db, _claim(db), query=f"Q{index}", answer=f"A{index}")
    timestamp = "2026-08-21T20:00:00.123Z"
    with db.transaction() as conn:
        conn.execute(
            "UPDATE messages SET timestamp = ?, last_modified = ? WHERE conversation_id = ?",
            (timestamp, timestamp, _thread(db).conversation_id),
        )

    seen: list[str] = []
    before = None
    while True:
        page = db.shared_workspace_chat_store.list_messages(
            share_id=41,
            before=before,
            limit=2,
        )
        seen.extend(message.message_id for message in page.messages)
        if page.next_before is None:
            break
        before = page.next_before

    expected = db.execute_query(
        "SELECT id FROM messages WHERE conversation_id = ? ORDER BY id ASC",
        (_thread(db).conversation_id,),
    ).fetchall()
    assert sorted(seen) == [row["id"] for row in expected]
    assert len(seen) == len(set(seen)) == 6


def test_hard_delete_cascades_thread_and_receipts(db: CharactersRAGDB) -> None:
    thread = _thread(db)
    _claim(db)

    with db.transaction() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (thread.conversation_id,))

    assert db.shared_workspace_chat_store.get_thread(share_id=41) is None
    assert db.execute_query(
        "SELECT count(*) FROM shared_workspace_chat_requests WHERE share_id = ?",
        (41,),
    ).fetchone()[0] == 0


def test_cleanup_is_bounded_and_never_deletes_completed_or_retryable(
    db: CharactersRAGDB,
) -> None:
    thread = _thread(db)
    old = NOW - timedelta(hours=25)
    conflict_ids = [str(uuid4()) for _ in range(103)]
    with db.transaction() as conn:
        for request_id in conflict_ids:
            conn.execute(
                "INSERT INTO shared_workspace_chat_requests("
                "recipient_user_id, share_id, request_id, request_fingerprint, conversation_id, "
                "status, updated_at) VALUES (?, ?, ?, ?, ?, 'conflicted', ?)",
                (
                    "recipient-a",
                    41,
                    request_id,
                    f"fingerprint-{request_id}",
                    thread.conversation_id,
                    old.isoformat(),
                ),
            )
        for status in ("completed", "retryable"):
            request_id = str(uuid4())
            conn.execute(
                "INSERT INTO shared_workspace_chat_requests("
                "recipient_user_id, share_id, request_id, request_fingerprint, conversation_id, "
                "status, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "recipient-a",
                    41,
                    request_id,
                    f"fingerprint-{request_id}",
                    thread.conversation_id,
                    status,
                    old.isoformat(),
                ),
            )

    assert db.shared_workspace_chat_store.purge_expired_conflicts(now=NOW) == 100
    assert db.shared_workspace_chat_store.purge_expired_conflicts(now=NOW) == 3
    statuses = db.execute_query(
        "SELECT status FROM shared_workspace_chat_requests ORDER BY status"
    ).fetchall()
    assert [row["status"] for row in statuses] == ["completed", "retryable"]


def test_failure_timestamps_are_canonical_and_cleanup_honors_24_hour_boundary(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.DB_Management.chacha.shared_workspace_chat_store as store_module

    class FrozenDateTime(datetime):
        current = NOW

        @classmethod
        def now(cls, tz=None):
            return cls.fromtimestamp(cls.current.timestamp(), tz=tz)

    claims = [_claim(db, request_id=uuid4()) for _ in range(2)]
    for claim in claims:
        assert db.shared_workspace_chat_store.freeze_sources(
            claim=claim,
            source_mode="include",
            source_ids=("source-a",),
            snapshot_hash="snapshot-a",
            provider="llama",
            model="model-a",
        )

    monkeypatch.setattr(store_module, "datetime", FrozenDateTime)
    for claim, age_hours in zip(claims, (23, 25), strict=True):
        FrozenDateTime.current = FrozenDateTime.fromtimestamp(
            (NOW - timedelta(hours=age_hours)).timestamp(),
            tz=timezone.utc,
        )
        assert db.shared_workspace_chat_store.mark_conflicted(
            claim=claim, error_code="shared_source_changed"
        )

    rows = db.execute_query(
        "SELECT request_id, updated_at FROM shared_workspace_chat_requests "
        "WHERE status = 'conflicted' ORDER BY request_id"
    ).fetchall()
    assert all("T" in row["updated_at"] and row["updated_at"].endswith("+00:00") for row in rows)

    cleanup_now = FrozenDateTime.fromtimestamp(NOW.timestamp(), tz=timezone.utc)
    assert db.shared_workspace_chat_store.purge_expired_conflicts(now=cleanup_now) == 1
    remaining = db.execute_query(
        "SELECT request_id FROM shared_workspace_chat_requests WHERE status = 'conflicted'"
    ).fetchall()
    assert [row["request_id"] for row in remaining] == [str(claims[0].request_id)]


def test_cleanup_compares_and_orders_legacy_sqlite_timestamps_chronologically(
    db: CharactersRAGDB,
) -> None:
    thread = _thread(db)
    canonical_old_id = str(uuid4())
    legacy_old_id = str(uuid4())
    legacy_recent_id = str(uuid4())
    timestamps = {
        canonical_old_id: (NOW - timedelta(hours=26)).isoformat(),
        legacy_old_id: (NOW - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S"),
        legacy_recent_id: (NOW - timedelta(hours=23)).strftime("%Y-%m-%d %H:%M:%S"),
    }
    with db.transaction() as conn:
        for request_id, updated_at in timestamps.items():
            conn.execute(
                "INSERT INTO shared_workspace_chat_requests("
                "recipient_user_id, share_id, request_id, request_fingerprint, conversation_id, "
                "status, updated_at) VALUES (?, ?, ?, ?, ?, 'conflicted', ?)",
                (
                    "recipient-a",
                    41,
                    request_id,
                    f"fingerprint-{request_id}",
                    thread.conversation_id,
                    updated_at,
                ),
            )

    assert db.shared_workspace_chat_store.purge_expired_conflicts(now=NOW, limit=1) == 1
    after_first_delete = {
        row["request_id"]
        for row in db.execute_query(
            "SELECT request_id FROM shared_workspace_chat_requests WHERE status = 'conflicted'"
        ).fetchall()
    }
    assert after_first_delete == {legacy_old_id, legacy_recent_id}

    assert db.shared_workspace_chat_store.purge_expired_conflicts(now=NOW) == 1
    remaining = db.execute_query(
        "SELECT request_id FROM shared_workspace_chat_requests WHERE status = 'conflicted'"
    ).fetchall()
    assert [row["request_id"] for row in remaining] == [legacy_recent_id]


def test_cleanup_failure_cannot_weaken_claim_correctness(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread = _thread(db)

    def fail_cleanup(*_args, **_kwargs) -> int:
        raise CharactersRAGDBError("cleanup unavailable")

    monkeypatch.setattr(db.shared_workspace_chat_store, "purge_expired_conflicts", fail_cleanup)
    claimed = db.shared_workspace_chat_store.claim_request(
        share_id=41,
        request_id=uuid4(),
        request_fingerprint="fingerprint-cleanup",
        conversation_id=thread.conversation_id,
        lease_seconds=600,
        now=NOW,
    )
    assert claimed.disposition == "claimed"
