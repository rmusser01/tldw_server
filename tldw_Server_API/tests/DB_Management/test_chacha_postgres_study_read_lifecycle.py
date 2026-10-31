"""Standalone Study/Buddy reads release locks without settling caller work."""

from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Buddy.service import BuddyService
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.Buddy_DB import BuddyRepository
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.app.core.Flashcards.study_assistant import build_flashcard_assistant_context

pytestmark = pytest.mark.integration


def _seed(db):
    deck = db.add_deck("Rowan")
    card = db.add_flashcard({"deck_id": deck, "front": "Marker?", "back": "Amber", "tags": ["rowan"]})
    second = db.add_flashcard({"deck_id": deck, "front": "Second?", "back": "East"})
    template = db.add_flashcard_template(
        name="Basic", model_type="basic", front_template="Marker?", back_template="Amber"
    )
    db.upsert_deck_share(deck, user_id=2, shared_by=1)
    asset = db.add_flashcard_asset(image_bytes=b"fixture", mime_type="image/png")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck, review_mode="cram", tag_filter=None, scope_key=f"cram:deck:{deck}"
    )
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO flashcard_reviews(card_id, rating, review_session_id, client_id) SELECT id, ?, ?, ? FROM flashcards WHERE uuid = ?",
            (3, session["id"], "1", card),
        )
    thread = db.get_or_create_study_assistant_thread(context_type="flashcard", flashcard_uuid=card)
    db.append_study_assistant_message(
        thread_id=thread["id"], role="user", action_type="freeform", input_modality="text", content="Explain"
    )
    persona = db.create_persona_profile({"user_id": "1", "name": "Fixture persona"})
    buddy = "a" * 32
    repo = BuddyRepository(db, "1")
    repo.create(
        {
            "id": buddy,
            "name": "Fixture Buddy",
            "optional_persona_id": persona,
            "display_mode": "static",
            "manifest": {},
            "attribution": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        [],
    )
    workspace = db.upsert_workspace("fixture-workspace", "Fixture workspace")["id"]
    conversation = db.add_conversation(
        {"title": "Fixture conversation", "scope_type": "workspace", "workspace_id": workspace}
    )
    db.add_message({"conversation_id": conversation, "sender": "assistant", "content": "Saved result"})
    for scope, scope_id in (("conversation", conversation), ("workspace", workspace)):
        repo.set_attachment(
            scope, expected_version=0, attachment={"buddy_id": buddy, "scope_type": scope, "scope_id": scope_id}
        )
    db.close_connection()
    return SimpleNamespace(
        db=db,
        deck=deck,
        card=card,
        second=second,
        template=template,
        asset=asset,
        session=session["id"],
        thread=thread["id"],
        persona=persona,
        buddy=buddy,
        repo=repo,
        workspace=workspace,
        conversation=conversation,
    )


@pytest.fixture
def pg_study(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "study.db", client_id="1", backend=backend)
    try:
        yield _seed(db), pg_database_config
    finally:
        db.close_connection()
        backend.get_pool().close_all()


READS = [
    "decks",
    "shared_decks",
    "deck",
    "deck_name",
    "deck_shares",
    "deck_share",
    "template_count",
    "templates",
    "template",
    "cards",
    "card_count",
    "tags",
    "uuids",
    "next_review",
    "sessions",
    "session",
    "complete_session",
    "reviewed_cards",
    "latest_review",
    "card",
    "asset",
    "missing_asset_content",
    "populated_asset_content",
    "keywords",
    "citations",
    "study_pack",
    "existing_thread",
    "new_thread",
    "thread",
    "messages",
    "buddy",
    "buddies",
    "buddy_assets",
    "attachment",
    "latest_results",
    "persona",
    "workspace",
    "workspace_including_deleted",
    "conversation",
    "conversations",
]


def _read(f, operation):
    db = f.db
    calls = {
        "decks": lambda: db.list_decks(),
        "shared_decks": lambda: db.list_decks(shared_with_user_id=2),
        "deck": lambda: db.get_deck(f.deck),
        "deck_name": lambda: db.get_deck_by_name("Rowan"),
        "deck_shares": lambda: db.list_deck_shares(f.deck),
        "deck_share": lambda: db.get_deck_share(f.deck, user_id=2),
        "template_count": lambda: db.count_flashcard_templates(),
        "templates": lambda: db.list_flashcard_templates(),
        "template": lambda: db.get_flashcard_template(f.template),
        "cards": lambda: db.list_flashcards(deck_id=f.deck),
        "card_count": lambda: db.count_flashcards(deck_id=f.deck),
        "tags": lambda: db.list_flashcard_tag_suggestions(),
        "uuids": lambda: db.get_flashcards_by_uuids([f.card]),
        "next_review": lambda: db.get_next_review_card(deck_id=f.deck),
        "sessions": lambda: db.list_flashcard_review_sessions(status="active"),
        "session": lambda: db.get_flashcard_review_session(f.session),
        "complete_session": lambda: db.mark_flashcard_review_session_completed(f.session),
        "reviewed_cards": lambda: db.get_flashcard_reviewed_cards(f.session),
        "latest_review": lambda: db.get_latest_flashcard_review(f.card),
        "card": lambda: db.get_flashcard(f.card),
        "asset": lambda: db.get_flashcard_asset(f.asset),
        "missing_asset_content": lambda: db.get_flashcard_asset_content("missing"),
        "populated_asset_content": lambda: db.get_flashcard_asset_content(f.asset),
        "keywords": lambda: db.get_keywords_for_flashcard(f.card),
        "citations": lambda: db.list_flashcard_citations(f.card),
        "study_pack": lambda: db.get_study_pack_for_flashcard(f.card),
        "existing_thread": lambda: db.get_or_create_study_assistant_thread(
            context_type="flashcard", flashcard_uuid=f.card
        ),
        "new_thread": lambda: db.get_or_create_study_assistant_thread(
            context_type="flashcard", flashcard_uuid=f.second
        ),
        "thread": lambda: db.get_study_assistant_thread(f.thread),
        "messages": lambda: db.list_study_assistant_messages(f.thread),
        "buddy": lambda: f.repo.get(f.buddy),
        "buddies": lambda: f.repo.list_profiles(limit=10, offset=0),
        "buddy_assets": lambda: f.repo.assets(f.buddy),
        "attachment": lambda: f.repo.attachment("default"),
        "latest_results": lambda: f.repo.latest_results("default", ["missing-conversation"]),
        "persona": lambda: db.get_persona_profile(f.persona, user_id="1"),
        "workspace": lambda: db.get_workspace(f.workspace),
        "workspace_including_deleted": lambda: db.get_workspace(f.workspace, include_deleted=True),
        "conversation": lambda: db.get_conversation_by_id(f.conversation),
        "conversations": lambda: db.get_conversations_for_user("1", scope_type="workspace", workspace_id=f.workspace),
    }
    result = calls[operation]()
    if operation == "populated_asset_content":
        assert result == b"fixture"
    return result


def _read_chain(f):
    service = BuddyService(f.db, "1")
    assert service.list_profiles(limit=10, offset=0)["buddies"][0]["optional_persona_available"]
    assert service.attachment("default")["attachment"] is None
    for scope in ("conversation", "workspace"):
        assert service.attachment(scope)["attachment"]["buddy_id"] == f.buddy
        assert service.activity(scope, limit=10, offset=0)["items"][0]["result"]["content"] == "Saved result"
    assert f.db.get_flashcard(f.card)["back"] == "Amber"
    assert len(f.db.list_decks()) == 1
    assert f.db.count_flashcards(deck_id=f.deck) == 2
    assert len(f.db.list_flashcards(deck_id=f.deck)) == 2
    assert f.db.get_next_review_card(deck_id=f.deck)[1] == "new"
    assert build_flashcard_assistant_context(f.db, f.card)["history"][0]["content"] == "Explain"


def _caller_reads(f):
    # These three methods contain existing deliberate writes/maintenance. Their
    # new SELECT flags do not redefine ownership of the surrounding mutation.
    for operation in READS:
        if operation not in {"sessions", "complete_session", "new_thread"}:
            _read(f, operation)
    _read_chain(f)


def _assert_bootstrap_unblocked(db):
    with db.backend.transaction() as observer:
        observer.execute("SET LOCAL lock_timeout = '100ms'")
        observer.execute("ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS front_search TEXT")
    assert db._get_thread_connection().info.transaction_status.name == "IDLE"


@pytest.mark.parametrize("operation", READS)
def test_each_standalone_study_read_releases_its_transaction(pg_study, operation):
    f, _ = pg_study
    _read(f, operation)
    assert f.db._get_thread_connection().info.transaction_status.name == "IDLE"
    _assert_bootstrap_unblocked(f.db)


@pytest.mark.parametrize("scope", ["conversation", "workspace"])
@pytest.mark.parametrize("operation", ["attachment", "activity"])
def test_populated_buddy_target_read_then_cards_releases_transaction(pg_study, scope, operation):
    f, _ = pg_study
    service = BuddyService(f.db, "1")
    if operation == "attachment":
        assert service.attachment(scope)["attachment"]["buddy_id"] == f.buddy
    else:
        assert service.activity(scope, limit=10, offset=0)["items"][0]["result"]["content"] == "Saved result"
    assert len(f.db.list_flashcards(deck_id=f.deck)) == 2
    _assert_bootstrap_unblocked(f.db)


@pytest.mark.parametrize("owner", ["chacha", "nested", "backend"])
def test_first_study_read_in_explicit_scope_keeps_its_lock(pg_study, owner):
    f, _ = pg_study
    db = f.db
    raw = db._get_thread_connection()
    with ExitStack() as stack:
        if owner == "backend":
            stack.enter_context(db.backend.transaction(connection=raw))
        else:
            stack.enter_context(db.transaction())
            if owner == "nested":
                stack.enter_context(db.transaction())
        assert raw.info.transaction_status.name == "IDLE"
        assert db.get_flashcard(f.card)["back"] == "Amber"
        assert raw.info.transaction_status.name == "INTRANS"
        assert (
            db.backend.execute(
                "SELECT COUNT(*) FROM pg_locks WHERE pid=%s AND relation='flashcards'::regclass",
                (raw.info.backend_pid,),
            ).scalar
            > 0
        )
    _assert_bootstrap_unblocked(db)


def test_real_buddy_study_chain_allows_replacement_initialization(pg_study, tmp_path):
    from psycopg.conninfo import make_conninfo

    f, config = pg_study
    raw = f.db._get_thread_connection()
    _read_chain(f)
    assert f.db._get_thread_connection() is raw
    replacement_config = replace(
        config,
        connection_string=make_conninfo(
            host=config.pg_host,
            port=str(config.pg_port),
            dbname=config.pg_database,
            user=config.pg_user,
            password=config.pg_password,
            options="-c lock_timeout=100ms",
        ),
    )
    backend = DatabaseBackendFactory.create_backend(replacement_config)
    replacement = None
    try:
        # Actual constructor/bootstrap, not a simulated readiness response.
        replacement = CharactersRAGDB(tmp_path / "replacement.db", client_id="1", backend=backend)
        assert replacement.get_flashcard(f.card)["back"] == "Amber"
    finally:
        if replacement is not None:
            replacement.close_connection()
        backend.get_pool().close_all()
    _assert_bootstrap_unblocked(f.db)


@pytest.mark.parametrize("owner", ["implicit", "chacha", "nested", "backend"])
@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_read_chain_preserves_caller_writes(pg_study, owner, commit):
    f, _ = pg_study
    db = f.db
    raw = db._get_thread_connection()
    with ExitStack() as stack:
        if owner in {"chacha", "nested"}:
            stack.enter_context(db.transaction())
            if owner == "nested":
                stack.enter_context(db.transaction())
        elif owner == "backend":
            stack.enter_context(db.backend.transaction(connection=raw))
        db.execute_query("UPDATE decks SET description = ? WHERE id = ? RETURNING id", ("Pending", f.deck))
        _caller_reads(f)
        assert raw.info.transaction_status.name == "INTRANS"
        assert db.backend.execute("SELECT description FROM decks WHERE id = %s", (f.deck,)).scalar is None
        if commit:
            raw.commit()
        else:
            raw.rollback()
    assert db.backend.execute("SELECT description FROM decks WHERE id = %s", (f.deck,)).scalar == (
        "Pending" if commit else None
    )


@pytest.mark.parametrize(
    "query",
    [
        "SELECT id FROM flashcards FOR UPDATE",
        "SELECT set_config('app.uat181_pending', 'pending', true)",
        "WITH changed AS (UPDATE decks SET description = 'Pending' RETURNING id) SELECT id FROM changed",
    ],
)
def test_read_chain_preserves_generic_locking_function_and_write_cte(pg_study, query):
    f, _ = pg_study
    raw = f.db._get_thread_connection()
    f.db.execute_query(query)
    _caller_reads(f)
    assert raw.info.transaction_status.name == "INTRANS"
    if "FOR UPDATE" in query:
        locks = f.db.backend.execute(
            "SELECT mode FROM pg_locks WHERE pid=%s AND relation='flashcards'::regclass", (raw.info.backend_pid,)
        ).rows
        assert any(row["mode"] == "RowShareLock" for row in locks)
    elif "set_config" in query:
        assert (
            f.db.execute_query("SELECT current_setting('app.uat181_pending') AS value").fetchone()["value"] == "pending"
        )
    else:
        assert f.db.backend.execute("SELECT description FROM decks WHERE id=%s", (f.deck,)).scalar is None
    raw.rollback()
    _assert_bootstrap_unblocked(f.db)


def test_failed_standalone_card_read_releases_aborted_transaction(pg_study):
    f, _ = pg_study
    with pytest.raises(CharactersRAGDBError):
        f.db.list_flashcards(limit="invalid-integer")
    assert f.db._get_thread_connection().info.transaction_status.name == "IDLE"
    _read_chain(f)
    _assert_bootstrap_unblocked(f.db)


def test_sqlite_read_inventory_and_chain_keep_behavior(tmp_path):
    db = CharactersRAGDB(tmp_path / "study.sqlite", client_id="1")
    try:
        f = _seed(db)
        for operation in READS:
            _read(f, operation)
        _read_chain(f)
        with pytest.raises(RuntimeError, match="rollback"):
            with db.transaction():
                db.execute_query("UPDATE decks SET description=? WHERE id=?", ("Pending", f.deck))
                _read_chain(f)
                raise RuntimeError("rollback")
        assert db.get_deck(f.deck)["description"] is None
    finally:
        db.close_connection()
