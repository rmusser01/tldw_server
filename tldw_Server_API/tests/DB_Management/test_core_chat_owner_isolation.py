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

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

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
