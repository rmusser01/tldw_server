"""Conversation search finds a chat by what was said in it, not only by its title (CS-02)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.conversation_search_snippets import (
    build_match_snippet,
    extract_search_terms,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError

pytestmark = pytest.mark.integration

TITLE_AND_CONTENT = ("title", "content")
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    """Yield an isolated SQLite chat store and close every connection afterward."""
    database = CharactersRAGDB(db_path=str(tmp_path / "content_search.sqlite"), client_id="alice")
    yield database
    database.close_all_connections()


def _add_chat(
    db: CharactersRAGDB,
    conversation_id: str,
    title: str,
    *,
    owner: str = "alice",
    workspace: str | None = None,
) -> str:
    """Create an owned persona conversation in the requested global or workspace scope."""
    db.add_conversation(
        {
            "id": conversation_id,
            "root_id": conversation_id,
            "assistant_kind": "persona",
            "assistant_id": "persona-search",
            "persona_memory_mode": "read_only",
            "title": title,
            "client_id": owner,
            "scope_type": "workspace" if workspace else "global",
            "workspace_id": workspace,
        }
    )
    return conversation_id


def _say(db: CharactersRAGDB, conversation_id: str, content: str, *, sender: str = "user", owner: str = "alice") -> str:
    """Append one live message to a conversation and return its id."""
    message_id = db.add_message(
        {"conversation_id": conversation_id, "sender": sender, "content": content, "client_id": owner}
    )
    assert message_id is not None
    return message_id


def _set_age(db: CharactersRAGDB, ages_in_hours: dict[str, float]) -> None:
    """Date each conversation ``hours`` before NOW; call after adding messages, which touch the chat."""
    for conversation_id, hours in ages_in_hours.items():
        stamp = (NOW - timedelta(hours=hours)).isoformat()
        db.execute_query(
            "UPDATE conversations SET created_at = ?, last_modified = ? WHERE id = ?",
            (stamp, stamp, conversation_id),
            commit=True,
        )


def _search(db: CharactersRAGDB, query: str, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
    """Run the paged search as alice with content search on unless a test overrides it."""
    options: dict[str, Any] = {"client_id": "alice", "search_in": TITLE_AND_CONTENT, "as_of": NOW}
    options.update(kwargs)
    rows, total, _ = db.search_conversations_page(query, **options)
    return rows, total


def test_bland_title_chat_is_found_by_a_word_from_its_messages(db: CharactersRAGDB) -> None:
    """The memorable detail lives only in the messages; the result carries a snippet of it."""
    _add_chat(db, "weekly", "Weekly sync")
    first = _say(db, "weekly", "Remember that the launch codeword is zebrafinch.")
    _say(db, "weekly", "Noted: the launch codeword is zebrafinch.", sender="assistant")
    _add_chat(db, "unrelated", "Lunch plans")
    _say(db, "unrelated", "Sandwiches again?")

    rows, total = _search(db, "zebrafinch")

    assert total == 1
    assert [row["id"] for row in rows] == ["weekly"]
    assert rows[0]["matched_in"] == ["content"]
    assert "zebrafinch" in rows[0]["match_snippet"]
    assert rows[0]["match_message_id"] == first


def test_title_only_search_stays_the_default(db: CharactersRAGDB) -> None:
    """Callers that do not opt in keep the existing title-only results and row shape."""
    _add_chat(db, "weekly", "Weekly sync")
    _say(db, "weekly", "The launch codeword is zebrafinch.")

    rows, total, _ = db.search_conversations_page("zebrafinch", client_id="alice")
    title_rows, title_total, _ = db.search_conversations_page("Weekly", client_id="alice")

    assert (rows, total) == ([], 0)
    assert title_total == 1
    assert "matched_in" not in title_rows[0]


@pytest.mark.parametrize("order_by", ["recency", "bm25", "hybrid"])
def test_title_matches_rank_first_then_content_matches_by_recency(db: CharactersRAGDB, order_by: str) -> None:
    """An old chat named after the term outranks newer chats that only mention it."""
    _add_chat(db, "title-old", "Zebrafinch migration notes")
    _say(db, "title-old", "Nothing relevant here.")
    _add_chat(db, "content-new", "Weekly sync")
    _say(db, "content-new", "We saw a zebrafinch today.")
    _add_chat(db, "content-older", "Standup")
    _say(db, "content-older", "Is the zebrafinch still around?")
    _add_chat(db, "both", "Zebrafinch care")
    _say(db, "both", "A zebrafinch needs millet.")
    _set_age(db, {"title-old": 72, "content-new": 1, "content-older": 24, "both": 48})

    rows, total = _search(db, "zebrafinch", order_by=order_by)

    assert total == 4
    ordered = [row["id"] for row in rows]
    assert set(ordered[:2]) == {"title-old", "both"}
    assert ordered[2:] == ["content-new", "content-older"]
    if order_by == "recency":
        assert ordered[:2] == ["both", "title-old"]
    matched = {row["id"]: row["matched_in"] for row in rows}
    assert matched == {
        "title-old": ["title"],
        "both": ["title", "content"],
        "content-new": ["content"],
        "content-older": ["content"],
    }
    assert rows[ordered.index("title-old")]["match_snippet"] is None
    assert "zebrafinch" in rows[ordered.index("both")]["match_snippet"]


def test_deleted_messages_do_not_match(db: CharactersRAGDB) -> None:
    """A chat whose only mention was deleted is not found; a live mention still is."""
    _add_chat(db, "gone", "Weekly sync")
    doomed = _say(db, "gone", "The codeword is zebrafinch.")
    _say(db, "gone", "Unrelated follow-up.")
    _add_chat(db, "kept", "Standup")
    doomed_too = _say(db, "kept", "zebrafinch was mentioned first here.")
    survivor = _say(db, "kept", "And zebrafinch again in a message that stays.")
    for message_id in (doomed, doomed_too):
        assert db.soft_delete_message(message_id, expected_version=1) is True

    rows, total = _search(db, "zebrafinch")

    assert total == 1
    assert [row["id"] for row in rows] == ["kept"]
    assert rows[0]["match_message_id"] == survivor
    assert "stays" in rows[0]["match_snippet"]


def test_trashed_chats_are_excluded_even_though_their_messages_stay_live(db: CharactersRAGDB) -> None:
    """Moving a chat to Trash flags only the conversation, so the conversation guard must hold."""
    _add_chat(db, "live", "Weekly sync")
    _say(db, "live", "The codeword is zebrafinch.")
    _add_chat(db, "trashed", "Standup")
    _say(db, "trashed", "The codeword is zebrafinch.")
    trashed = db.get_conversation_by_id("trashed")
    assert db.soft_delete_conversation("trashed", expected_version=trashed["version"]) is True
    live_messages = db.execute_query(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND deleted = 0", ("trashed",)
    ).fetchone()[0]
    assert live_messages == 1

    rows, total = _search(db, "zebrafinch")

    assert total == 1
    assert [row["id"] for row in rows] == ["live"]


def test_another_owners_chats_are_never_returned(db: CharactersRAGDB) -> None:
    """Content matches are scoped to the requesting owner in both directions."""
    _add_chat(db, "alice-chat", "Weekly sync")
    _say(db, "alice-chat", "The codeword is zebrafinch.")
    _add_chat(db, "bob-chat", "Weekly sync", owner="bob")
    _say(db, "bob-chat", "Bob also said zebrafinch.", owner="bob")
    _add_chat(db, "bob-title", "Zebrafinch", owner="bob")

    alice_rows, alice_total = _search(db, "zebrafinch")
    bob_rows, bob_total = _search(db, "zebrafinch", client_id="bob")

    assert (alice_total, [row["id"] for row in alice_rows]) == (1, ["alice-chat"])
    assert "Bob" not in alice_rows[0]["match_snippet"]
    assert bob_total == 2
    assert {row["id"] for row in bob_rows} == {"bob-chat", "bob-title"}


@pytest.mark.parametrize("order_by", ["recency", "bm25", "hybrid", "topic"])
def test_paging_and_totals_stay_correct(db: CharactersRAGDB, order_by: str) -> None:
    """Every page reports the same total and the pages partition the matches."""
    _add_chat(db, "title-hit", "Zebrafinch")
    for index in range(5):
        _add_chat(db, f"content-{index}", f"Chat {index}")
        _say(db, f"content-{index}", f"zebrafinch sighting {index}")
        _say(db, f"content-{index}", f"second zebrafinch mention {index}", sender="assistant")
    _add_chat(db, "miss", "Chat without it")
    _say(db, "miss", "nothing to see")
    _set_age(db, {"title-hit": 100, **{f"content-{index}": index for index in range(5)}})

    seen: list[str] = []
    for offset in range(0, 6, 2):
        rows, total = _search(db, "zebrafinch", order_by=order_by, limit=2, offset=offset)
        assert total == 6
        assert len(rows) == 2
        seen.extend(row["id"] for row in rows)
    beyond, beyond_total = _search(db, "zebrafinch", order_by=order_by, limit=2, offset=6)

    assert (beyond, beyond_total) == ([], 6)
    assert len(seen) == len(set(seen)) == 6
    assert set(seen) == {"title-hit", *(f"content-{index}" for index in range(5))}
    assert seen == ["title-hit", *(f"content-{index}" for index in range(5))]


def test_conversation_filters_still_apply_to_content_matches(db: CharactersRAGDB) -> None:
    """A workspace chat that mentions the term stays out of the global list."""
    db.upsert_workspace("ws-a", "A")
    _add_chat(db, "global", "Weekly sync")
    _say(db, "global", "zebrafinch in the global chat")
    _add_chat(db, "scoped", "Weekly sync", workspace="ws-a")
    _say(db, "scoped", "zebrafinch in the workspace chat")

    global_rows, global_total = _search(db, "zebrafinch", scope_type="global")
    workspace_rows, workspace_total = _search(db, "zebrafinch", scope_type="workspace", workspace_id="ws-a")

    assert (global_total, [row["id"] for row in global_rows]) == (1, ["global"])
    assert (workspace_total, [row["id"] for row in workspace_rows]) == (1, ["scoped"])


def test_content_only_search_ignores_titles(db: CharactersRAGDB) -> None:
    """``search_in=content`` returns only chats whose messages match."""
    _add_chat(db, "title-hit", "Zebrafinch")
    _say(db, "title-hit", "Nothing relevant.")
    _add_chat(db, "content-hit", "Weekly sync")
    _say(db, "content-hit", "A zebrafinch appeared.")

    rows, total = _search(db, "zebrafinch", search_in=("content",))

    assert (total, [row["id"] for row in rows]) == (1, ["content-hit"])
    assert rows[0]["matched_in"] == ["content"]


@pytest.mark.parametrize(
    "query,expected",
    [
        ("what's the codeword?", {"prose"}),
        ("cedar-maple", {"hyphen"}),
        ('"launch codeword"', {"prose", "phrase"}),
        ("zebra*", {"prefix"}),
        ("title:quarterly", {"column"}),
    ],
)
def test_punctuation_and_fts_syntax_behave_like_title_search(db: CharactersRAGDB, query: str, expected: set[str]) -> None:
    """Prose that FTS rejects is searched literally; valid FTS syntax keeps its meaning."""
    _add_chat(db, "prose", "Chat one")
    _say(db, "prose", "So what's the codeword? I think the launch codeword changed.")
    _add_chat(db, "hyphen", "Chat two")
    _say(db, "hyphen", "The cedar-maple table arrived.")
    _add_chat(db, "phrase", "Chat three")
    _say(db, "phrase", "Their launch codeword is secret.")
    _add_chat(db, "prefix", "Chat four")
    _say(db, "prefix", "zebrafinch")
    _add_chat(db, "column", "Quarterly review")
    _say(db, "column", "Nothing relevant.")

    rows, total = _search(db, query)

    assert {row["id"] for row in rows} == expected
    assert total == len(expected)


def test_deleted_scopes_keep_the_existing_title_text_search(db: CharactersRAGDB) -> None:
    """Trash views are searched by title, topic and state only; content search never surfaces trashed chats."""
    _add_chat(db, "trashed-title", "Zebrafinch")
    _add_chat(db, "trashed-content", "Weekly sync")
    _say(db, "trashed-content", "zebrafinch")
    for conversation_id in ("trashed-title", "trashed-content"):
        row = db.get_conversation_by_id(conversation_id)
        db.soft_delete_conversation(conversation_id, expected_version=row["version"])

    rows, total = _search(db, "zebrafinch", deleted_only=True)

    assert (total, [row["id"] for row in rows]) == (1, ["trashed-title"])
    assert "matched_in" not in rows[0]


@pytest.mark.parametrize("search_in", [("body",), ("title", "messages"), (), ("",)])
def test_unknown_search_fields_are_rejected(db: CharactersRAGDB, search_in: tuple[str, ...]) -> None:
    """A typo in ``search_in`` is an input error, not a silent title-only search."""
    with pytest.raises(InputError):
        db.search_conversations_page("zebrafinch", client_id="alice", search_in=search_in)


def test_content_search_never_writes(db: CharactersRAGDB) -> None:
    """Searching only reads: the message FTS triggers are fragile around soft deletes (#3153)."""
    _add_chat(db, "chat", "Weekly sync")
    doomed = _say(db, "chat", "zebrafinch once")
    _say(db, "chat", "zebrafinch twice")
    db.soft_delete_message(doomed, expected_version=1)
    connection = db.get_connection()
    changes_before = connection.total_changes

    for query in ("zebrafinch", "what's this?", "title:zebrafinch"):
        rows, total = _search(db, query, order_by="hybrid")
        assert total == len(rows)

    assert connection.total_changes == changes_before


@pytest.mark.parametrize(
    "query,expected",
    [
        ("zebrafinch", ["zebrafinch"]),
        ("Launch  Codeword", ["launch", "codeword"]),
        ('"launch codeword" OR zebra*', ["launch", "codeword", "zebra"]),
        ("cedar NOT maple", ["cedar", "maple"]),
        ("what's the codeword?", ["what's", "the", "codeword"]),
        ("NEAR(cedar maple, 2)", ["cedar", "maple", "2"]),
        ("!!!", []),
    ],
)
def test_extract_search_terms(query: str, expected: list[str]) -> None:
    """Operators are dropped and the remaining words are lower-cased in order."""
    assert extract_search_terms(query) == expected


def test_snippet_centres_on_the_matching_term() -> None:
    """Long messages are trimmed around the match with ellipses on the cut sides."""
    content = ("filler " * 60) + "the launch codeword is ZebraFinch today " + ("tail " * 60)

    snippet = build_match_snippet(content, ["zebrafinch"], max_chars=80)

    assert "ZebraFinch" in snippet
    assert snippet.startswith("…") and snippet.endswith("…")
    assert len(snippet) <= 82


def test_snippet_centres_on_the_most_specific_query_word() -> None:
    """A common short word early in the message must not pull the excerpt away from the real match."""
    content = "the plan is simple. " + ("filler " * 60) + "our launch codeword is zebrafinch. " + ("tail " * 60)

    snippet = build_match_snippet(content, ["the", "zebrafinch"], max_chars=80)

    assert "zebrafinch" in snippet
    assert "the plan is simple" not in snippet


def test_snippet_of_a_short_message_is_the_whole_message_on_one_line() -> None:
    """Newlines collapse so the snippet fits a single sidebar line."""
    assert build_match_snippet("The codeword\n\nis   zebrafinch.", ["zebrafinch"]) == "The codeword is zebrafinch."


def test_snippet_falls_back_to_the_start_when_no_term_is_found() -> None:
    """Stemmed or tokenised matches the plain scan cannot locate still yield a snippet."""
    snippet = build_match_snippet("alpha " * 100, ["zebrafinch"], max_chars=40)

    assert snippet.startswith("alpha alpha")
    assert snippet.endswith("…")
    assert build_match_snippet("", ["zebrafinch"]) == ""
    assert build_match_snippet(None, ["zebrafinch"]) == ""


@pytest.mark.postgres
def test_postgres_content_search_matches_messages_with_guards(pg_database_config: DatabaseConfig) -> None:
    """The PostgreSQL path finds chats by message text with the same owner, trash and delete guards."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    database = CharactersRAGDB(db_path=":memory:", client_id="alice", backend=backend)
    try:
        _add_chat(database, "title-old", "Zebrafinch migration notes")
        _say(database, "title-old", "Nothing relevant here.")
        _add_chat(database, "content-new", "Weekly sync")
        first = _say(database, "content-new", "Remember that the launch codeword is zebrafinch.")
        _say(database, "content-new", "Noted: zebrafinch.", sender="assistant")
        _add_chat(database, "content-older", "Standup")
        _say(database, "content-older", "Is the zebrafinch still around?")
        _add_chat(database, "deleted-message", "Retro")
        doomed = _say(database, "deleted-message", "zebrafinch, but deleted")
        assert database.soft_delete_message(doomed, expected_version=1) is True
        _add_chat(database, "trashed", "Planning")
        _say(database, "trashed", "zebrafinch in a trashed chat")
        trashed = database.get_conversation_by_id("trashed")
        database.soft_delete_conversation("trashed", expected_version=trashed["version"])
        _add_chat(database, "bob-chat", "Weekly sync", owner="bob")
        _say(database, "bob-chat", "Bob said zebrafinch.", owner="bob")
        _set_age(database, {"title-old": 72, "content-new": 1, "content-older": 24})

        for order_by in ("recency", "bm25", "hybrid", "topic"):
            rows, total = _search(database, "zebrafinch", order_by=order_by)
            assert total == 3
            assert [row["id"] for row in rows] == ["title-old", "content-new", "content-older"]
        assert rows[0]["matched_in"] == ["title"]
        assert rows[1]["matched_in"] == ["content"]
        assert rows[1]["match_message_id"] == first
        assert "zebrafinch" in rows[1]["match_snippet"]

        page, page_total = _search(database, "zebrafinch", limit=1, offset=1)
        assert (page_total, [row["id"] for row in page]) == (3, ["content-new"])
        default_rows, default_total, _ = database.search_conversations_page("zebrafinch", client_id="alice")
        assert (default_total, [row["id"] for row in default_rows]) == (1, ["title-old"])
        content_rows, content_total = _search(database, "zebrafinch", search_in=("content",))
        assert (content_total, [row["id"] for row in content_rows]) == (2, ["content-new", "content-older"])
    finally:
        database.close_all_connections()
