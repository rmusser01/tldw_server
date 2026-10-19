"""Core chat rows must stay private across owners on both supported backends.

SQLite gives each user their own file, so a missing tenant predicate is
harmless there.  PostgreSQL puts every user in shared tables, so the same
query is a leak.  Both ship.  A test that runs only on SQLite is measuring the
deployment that does not need the guarantee, which is why these run twice.

The PostgreSQL arm runs under a role that cannot bypass RLS.  The plain
``pg_database_config`` role is a superuser, and a superuser is exempt from
row-level security even under ``FORCE``, so asserting a policy through it would
pass with the policy deleted.
"""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Chat.history_selection import (
    HistorySelectionError,
    resolve_history_selection,
    snapshot_to_wire,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError

pytestmark = pytest.mark.integration

ALICE = "1"
BOB = "2"


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def chats(request, tmp_path):
    """Two owners against separate SQLite files or one shared PostgreSQL database."""
    backend = None
    if request.param == "postgres":
        backend = request.getfixturevalue("pg_restricted_backend")

    alice = CharactersRAGDB(tmp_path / "alice.db", client_id=ALICE, backend=backend)
    bob = CharactersRAGDB(tmp_path / "bob.db", client_id=BOB, backend=backend)
    try:
        conversation = alice.add_conversation(
            {"title": "Alice private conversation", "client_id": ALICE}
        )
        message = alice.add_message(
            {
                "conversation_id": conversation,
                "sender": "user",
                "content": "Alice private message body",
                "client_id": ALICE,
            }
        )
        alice.close_connection()
        bob.close_connection()
        yield SimpleNamespace(
            alice=alice,
            bob=bob,
            conversation=conversation,
            message=message,
            backend=request.param,
        )
    finally:
        alice.close_all_connections()
        bob.close_all_connections()


def test_owner_can_read_own_conversation(chats):
    """Guard against a fixture that isolates by breaking both sides."""
    assert chats.alice.get_conversation_by_id(chats.conversation) is not None
    assert chats.alice.get_message_by_id(chats.message) is not None


def test_foreign_conversation_is_not_listed(chats):
    assert chats.bob.get_conversations_for_user(BOB) == []


def test_foreign_conversation_is_not_searchable(chats):
    assert chats.bob.search_conversations_by_title("Alice private") == []


def test_foreign_conversation_is_not_readable_by_id(chats):
    """`get_conversation_by_id` is `WHERE id = ?` with no tenant predicate.

    On SQLite the per-user file answers this. On shared PostgreSQL tables only
    the RLS policy does, which is what makes this the load-bearing assertion.
    """
    assert chats.bob.get_conversation_by_id(chats.conversation) is None


def test_foreign_message_is_not_readable_by_id(chats):
    assert chats.bob.get_message_by_id(chats.message) is None


def test_foreign_messages_are_not_listed_for_conversation(chats):
    assert chats.bob.get_messages_for_conversation(chats.conversation) == []


def test_foreign_message_content_is_not_searchable(chats):
    assert chats.bob.search_messages_by_content("private message body") == []


def test_foreign_owner_cannot_regenerate_selected_question(chats):
    """The restricted PostgreSQL role must refuse a foreign selected-input write."""
    owner_key = "server:test/account:1"
    snapshot = snapshot_to_wire(chats.alice.get_conversation_history_snapshot(
        chats.conversation, owner_client_id=ALICE, owner_key=owner_key
    ))
    resolved = resolve_history_selection(snapshot, {
        "owner_key": owner_key,
        "conversation_id": chats.conversation,
        "interpretation": {"kind": "parent_graph_v1"},
        "cursor": {"kind": "after_message", "message_id": chats.message},
        "selection_revision": 1,
    }, "send", "prepared-private-request")
    assert resolved["status"] == "ready"  # nosec B101 # pytest contract assertion
    with pytest.raises((HistorySelectionError, NotFoundError)):
        chats.bob.settle_selected_existing_input(
            chats.conversation, resolved["selection"],
            {"id": "foreign-alternative", "sender": "assistant", "content": "must not save"},
            owner_client_id=BOB, owner_key=owner_key,
        )
    assert chats.alice.get_message_by_id("foreign-alternative") is None  # nosec B101 # pytest contract assertion


@pytest.mark.postgres
def test_postgres_inline_tool_call_backfill_consumes_native_result(pg_restricted_backend):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    owner_id = 910016
    with scoped_context(user_id=owner_id, is_admin=False):
        instance = CharactersRAGDB(db_path=":memory:", client_id=str(owner_id), backend=pg_restricted_backend)
        try:
            conversation_id = instance.add_conversation({"title": "Backfill Conversation"})
            assert conversation_id is not None
            message_id = instance.add_message(
                {
                    "conversation_id": conversation_id,
                    "sender": "assistant",
                    "content": 'Answer [tool_calls]: [{"id":"call-1","type":"function"}]',
                }
            )
            assert message_id is not None
            before = instance.get_message_by_id(message_id)
            assert before is not None
            assert instance.get_message_metadata(message_id) is None

            counts = instance.backfill_tool_calls_from_inline(strip_inline=True, limit=1)

            assert counts == {"scanned": 1, "matched": 1, "backfilled": 1, "stripped": 1}
            metadata = instance.get_message_metadata(message_id)
            assert metadata is not None
            assert metadata["tool_calls"] == [{"id": "call-1", "type": "function"}]
            message = instance.get_message_by_id(message_id)
            assert message is not None
            assert message["content"] == "Answer"
            assert message["version"] == before["version"] + 1
        finally:
            instance.close_all_connections()




@pytest.mark.postgres
@pytest.mark.parametrize("include_answers", [False, True])
def test_postgres_question_search_empty_query_keeps_result_contract(pg_restricted_backend, include_answers):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    owner_id = 910018
    with scoped_context(user_id=owner_id, is_admin=False):
        instance = CharactersRAGDB(":memory:", client_id=str(owner_id), backend=pg_restricted_backend)
        try:
            quiz_id = instance.create_quiz(name="Search contract")
            question_id = instance.create_question(
                quiz_id=quiz_id,
                question_type="true_false",
                question_text="Searchable question?",
                correct_answer="true",
            )
            osce_id = instance.create_quiz(name="Other activity", activity_type="osce")
            with pytest.raises(ConflictError, match="Quiz not found"):
                instance.list_questions(osce_id, q="   ", include_answers=include_answers)
            with scoped_context(user_id=910019, is_admin=False):
                foreign = CharactersRAGDB(":memory:", client_id="910019", backend=pg_restricted_backend)
                try:
                    with pytest.raises(ConflictError, match="Quiz not found"):
                        foreign.list_questions(quiz_id, q="   ", include_answers=include_answers)
                finally:
                    foreign.close_connection()
            ordinary = instance.list_questions(quiz_id, q="Searchable", include_answers=include_answers)
            assert ordinary["count"] == 1
            assert ordinary["items"][0]["id"] == question_id
            if include_answers:
                assert ordinary["items"][0]["correct_answer"] == "true"
            else:
                assert "correct_answer" not in ordinary["items"][0]
            assert instance.list_questions(quiz_id, q="   ", include_answers=include_answers) == {
                "items": [],
                "count": 0,
            }
        finally:
            instance.close_all_connections()
