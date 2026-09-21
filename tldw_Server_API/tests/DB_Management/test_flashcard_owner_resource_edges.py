"""Owner boundaries for resources directly reachable from private Flashcards."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessService,
    SharedWorkspaceNotFound,
)
from tldw_Server_API.tests.DB_Management.test_flashcard_shared_owner_contract import (
    _client,
    _raw_card,
)
from tldw_Server_API.tests.DB_Management.test_flashcard_shared_owner_contract import (
    owners as _owners_fixture,
)

pytestmark = pytest.mark.integration
owners = _owners_fixture


@pytest.mark.parametrize("surface", ["batch", "queue", "export", "analytics", "tags", "suggestions", "sessions", "latest-review"])
def test_derived_private_reads_are_owner_scoped(owners, surface):
    alice, bob = owners.alice, owners.bob
    if surface in {"tags", "suggestions"}:
        alice.set_flashcard_tags(owners.card, ["Alice private tag"])
    if surface in {"sessions", "latest-review"}:
        alice.review_flashcard(owners.card, 3)
    if surface == "batch":
        assert [row["uuid"] for row in bob.get_flashcards_by_uuids([owners.card, owners.bob_card])] == [owners.bob_card]
    elif surface == "queue":
        assert bob.get_next_review_card()[0]["uuid"] == owners.bob_card
    elif surface == "export":
        assert b"Alice" not in bob.export_flashcards_csv(include_workspace_items=True)
        assert b"Bob Rowan" in bob.export_flashcards_csv(include_workspace_items=True)
    elif surface == "analytics":
        assert [row["deck_id"] for row in bob.get_flashcard_analytics_summary()["decks"]] == [owners.bob_deck]
    elif surface == "tags":
        assert bob.get_keywords_for_flashcard(owners.card) == []
    elif surface == "suggestions":
        assert bob.list_flashcard_tag_suggestions() == []
    elif surface == "sessions":
        assert bob.list_flashcard_review_sessions() == []
    else:
        assert bob.get_latest_flashcard_review(owners.card) is None


@pytest.mark.parametrize("operation", ["complete", "abandon", "rollup", "reviewed-cards"])
def test_foreign_session_access_preserves_owner_session(owners, operation):
    reviewed = owners.alice.review_flashcard(owners.card, 3)
    session_id = reviewed["review_session_id"]
    if operation == "abandon":
        with owners.alice.transaction() as conn:
            conn.execute("UPDATE flashcard_review_sessions SET last_activity_at = ? WHERE id = ?", ("2000-01-01T00:00:00Z", session_id))
    before = owners.alice.get_flashcard_review_session(session_id)
    if operation == "complete":
        with pytest.raises(ConflictError, match="not found"):
            owners.bob.mark_flashcard_review_session_completed(session_id)
    elif operation == "abandon":
        assert owners.bob.abandon_stale_flashcard_review_sessions() == 0
    elif operation == "rollup":
        assert owners.bob.get_flashcard_review_session_rollup(session_id) is None
    else:
        assert owners.bob.get_flashcard_reviewed_cards(session_id) == []
    assert owners.alice.get_flashcard_review_session(session_id) == before


@pytest.mark.parametrize("operation", ["list", "upsert", "delete"])
def test_owner_managed_deck_grants_cannot_be_managed_by_foreign_owner(owners, operation):
    owners.alice.upsert_deck_share(owners.deck, user_id=4, shared_by=2)
    before = owners.alice.list_deck_shares(owners.deck)
    if operation == "list":
        assert owners.bob.list_deck_shares(owners.deck) == []
        assert owners.bob.get_deck_share(owners.deck, user_id=4) is None
    elif operation == "delete":
        assert owners.bob.delete_deck_share(owners.deck, user_id=4) is False
    else:
        with pytest.raises(ConflictError, match="not found"):
            owners.bob.upsert_deck_share(owners.deck, user_id=4, shared_by=3, role="editor")
    assert owners.alice.list_deck_shares(owners.deck) == before


@pytest.mark.parametrize("operation", ["single", "bulk", "reparent"])
def test_direct_card_parent_guard_has_no_partial_write(owners, operation):
    before = _raw_card(owners.bob, owners.bob_card)
    count = owners.bob.count_flashcards(include_workspace_items=True)
    with pytest.raises((CharactersRAGDBError, InputError)):
        if operation == "single":
            owners.bob.add_flashcard({"deck_id": owners.deck, "front": "Foreign parent", "back": "Rejected"})
        elif operation == "bulk":
            owners.bob.add_flashcards_bulk([
                {"deck_id": owners.bob_deck, "front": "Valid first", "back": "Must rollback"},
                {"deck_id": owners.deck, "front": "Foreign second", "back": "Rejected"},
            ])
        else:
            owners.bob.update_flashcard(owners.bob_card, {"deck_id": owners.deck}, expected_version=1)
    assert _raw_card(owners.bob, owners.bob_card) == before
    assert owners.bob.count_flashcards(include_workspace_items=True) == count


def test_owned_operations_preserve_caller_rollback_and_deleted_parent_read(owners):
    before = _raw_card(owners.bob, owners.bob_card)
    with pytest.raises(RuntimeError, match="caller rollback"):
        with owners.bob.transaction():
            owners.bob.update_flashcard(owners.bob_card, {"front": "Temporary"}, expected_version=1)
            raise RuntimeError("caller rollback")
    assert _raw_card(owners.bob, owners.bob_card) == before
    owners.bob.soft_delete_deck_by_id(owners.bob_deck, expected_version=1)
    assert owners.bob.get_flashcard(owners.bob_card)["uuid"] == owners.bob_card


def test_asset_cleanup_does_not_delete_another_owners_upload(owners):
    asset = owners.alice.add_flashcard_asset(image_bytes=b"private", mime_type="image/png")
    with owners.alice.transaction() as conn:
        conn.execute("UPDATE flashcard_assets SET created_at = ? WHERE uuid = ?", ("2000-01-01T00:00:00Z", asset))
    before = owners.alice.get_flashcard_asset(asset)
    assert owners.bob.cleanup_stale_flashcard_assets(older_than=timedelta(hours=1)) == 0
    assert owners.alice.get_flashcard_asset(asset) == before


def test_malformed_foreign_keyword_link_does_not_expose_owner_metadata(owners):
    if not owners.postgres:
        # SQLite files cannot contain a foreign user's persisted keyword ID.
        owners.bob.set_flashcard_tags(owners.bob_card, ["Owned"])
        assert {row["client_id"] for row in owners.bob.get_keywords_for_flashcard(owners.bob_card)} == {"3"}
        return
    owners.alice.set_flashcard_tags(owners.card, ["Private"])
    keyword = owners.alice.get_keywords_for_flashcard(owners.card)[0]
    with owners.bob.transaction() as conn:
        conn.execute("INSERT INTO flashcard_keywords(card_id, keyword_id, created_at) SELECT id, ?, ? FROM flashcards WHERE uuid = ?", (keyword["id"], "2026-01-01T00:00:00Z", owners.bob_card))
    with _client(owners.bob) as client:
        response = client.get(f"/api/v1/flashcards/{owners.bob_card}/tags")
    assert response.status_code == 200, response.text
    assert response.json() == {"items": [], "count": 0}
    before = _raw_card(owners.bob, owners.bob_card)
    with pytest.raises(InputError, match="does not match"):
        owners.bob.review_flashcard(owners.bob_card, 3, review_mode="cram", review_tag_filter="Private")
    assert _raw_card(owners.bob, owners.bob_card) == before


def test_restricted_role_actual_persistence_keeps_private_rows_owner_local(owners):
    if not owners.postgres:
        assert owners.bob.get_flashcard(owners.card) is None
        return
    backend = owners.bob.backend
    role = f"flashcard_service_{uuid4().hex[:12]}"
    quoted = backend.escape_identifier(role)
    created = False
    before = _raw_card(owners.alice, owners.card)
    try:
        owners.alice.close_connection()
        owners.bob.close_connection()
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {quoted} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}", connection=conn)
            backend.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {quoted}", connection=conn)
            backend.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {quoted}", connection=conn)
            backend.execute(f"GRANT {quoted} TO CURRENT_USER", connection=conn)
        created = True
        with owners.bob.transaction() as conn:
            conn.execute(f"SET LOCAL ROLE {quoted}")
            conn.execute("SELECT set_config('app.current_user_id', ?, true)", ("3",))
            flags = conn.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=current_user").fetchone()
            assert not flags["rolsuper"] and not flags["rolbypassrls"]
            assert [row["id"] for row in owners.bob.list_decks()] == [owners.bob_deck]
            assert owners.bob.get_flashcard(owners.card) is None
            assert owners.bob.update_flashcard(owners.bob_card, {"front": "Restricted owned edit"}, expected_version=1)
            with pytest.raises(CharactersRAGDBError, match="not found"):
                owners.bob.update_flashcard(owners.card, {"front": "Foreign edit"}, expected_version=1)
        assert _raw_card(owners.alice, owners.card) == before
        assert owners.bob.get_flashcard(owners.bob_card)["front"] == "Restricted owned edit"
    finally:
        owners.bob.close_connection()
        if created:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {quoted}", connection=conn)
                backend.execute(f"DROP ROLE {quoted}", connection=conn)


@pytest.mark.parametrize("operation", ["create", "move"])
def test_deck_workspace_reference_requires_current_owner(owners, operation):
    owners.alice.upsert_workspace("alice-private", "Alice private workspace")
    owners.bob.upsert_workspace("bob-private", "Bob private workspace")
    owned = owners.bob.add_deck("Owned workspace reference", workspace_id="bob-private")
    assert owners.bob.get_deck(owned)["workspace_id"] == "bob-private"
    if not owners.postgres:
        return  # Physical files are SQLite's user boundary; client_id is sync metadata.
    before = owners.bob.get_deck(owned)
    with pytest.raises(InputError, match="Workspace not found"):
        if operation == "create":
            owners.bob.add_deck("Foreign workspace reference", workspace_id="alice-private")
        else:
            owners.bob.update_deck(owned, workspace_id="alice-private", expected_version=1)
    assert owners.bob.get_deck(owned) == before


def test_malformed_foreign_deck_join_hides_foreign_metadata(owners):
    if not owners.postgres:
        assert owners.bob.get_flashcard(owners.bob_card)["deck_name"] == "Bob private Rowan"
        return
    with owners.bob.transaction() as conn:
        conn.execute("UPDATE flashcards SET deck_id = ? WHERE uuid = ?", (owners.deck, owners.bob_card))
    direct = owners.bob.get_flashcard(owners.bob_card)
    batch = owners.bob.get_flashcards_by_uuids([owners.bob_card])[0]
    listed = owners.bob.list_flashcards(include_workspace_items=True)[0]
    for row in (direct, batch, listed):
        assert row["client_id"] == "3"
        assert row["deck_name"] is None and row["workspace_id"] is None


def test_owned_keyword_case_identity_and_deleted_restore_preserve_foreign_row(owners):
    owners.alice.set_flashcard_tags(owners.card, ["Citrine"])
    owners.bob.set_flashcard_tags(owners.bob_card, ["CITRINE"])
    foreign_before = owners.alice.get_keywords_for_flashcard(owners.card)
    owned = owners.bob.get_keywords_for_flashcard(owners.bob_card)[0]
    with owners.bob.transaction() as conn:
        table = owners.bob._map_table_for_backend("keywords")
        conn.execute(f"UPDATE {table} SET deleted = ? WHERE id = ?", (True, owned["id"]))  # nosec B608 -- Backend-owned fixed table map.
    owners.bob.set_flashcard_tags(owners.bob_card, ["citrine"])
    restored = owners.bob.get_keywords_for_flashcard(owners.bob_card)
    assert len(restored) == 1 and restored[0]["id"] == owned["id"]
    assert restored[0]["client_id"] == "3" and not restored[0]["deleted"]
    assert owners.alice.get_keywords_for_flashcard(owners.card) == foreign_before


def test_rating_cannot_use_another_owners_session(owners):
    alice_session = owners.alice.get_or_create_flashcard_review_session(deck_id=None, review_mode="due", tag_filter=None, scope_key="due:global")
    before = _raw_card(owners.bob, owners.bob_card)
    session_before = owners.alice.get_flashcard_review_session(alice_session["id"])
    with pytest.raises(ConflictError, match="not found"):
        owners.bob.review_flashcard(owners.bob_card, 3, review_session_id=alice_session["id"], review_mode="due")
    assert _raw_card(owners.bob, owners.bob_card) == before
    assert owners.alice.get_flashcard_review_session(alice_session["id"]) == session_before


def test_foreign_tombstones_and_noop_updates_remain_owner_scoped(owners):
    owners.alice.soft_delete_flashcard(owners.card, expected_version=1)
    owners.alice.soft_delete_deck_by_id(owners.deck, expected_version=1)
    assert owners.bob.get_deck_by_name("Alice private Citrine", include_deleted=True) is None
    assert owners.card not in [row["uuid"] for row in owners.bob.list_flashcards(include_deleted=True, include_workspace_items=True)]
    assert owners.deck not in [row["id"] for row in owners.bob.list_decks(include_deleted=True, include_workspace_items=True)]
    if owners.postgres:
        with pytest.raises(CharactersRAGDBError, match="not found"):
            owners.bob.update_flashcard(owners.card, {})
        with pytest.raises(ConflictError, match="not found"):
            owners.bob.update_deck(owners.deck)


def test_foreign_review_link_does_not_contribute_to_owned_analytics(owners):
    owners.bob.review_flashcard(owners.bob_card, 3)
    assert owners.bob.get_flashcard_analytics_summary()["reviewed_today"] == 1
    if not owners.postgres:
        return
    with owners.bob.transaction() as conn:
        conn.execute("UPDATE flashcard_reviews SET client_id = ? WHERE card_id = (SELECT id FROM flashcards WHERE uuid = ?)", ("2", owners.bob_card))
    summary = owners.bob.get_flashcard_analytics_summary()
    assert summary["reviewed_today"] == 0
    assert summary["study_streak_days"] == 0


@pytest.mark.parametrize("operation", ["create", "reparent"])
def test_owned_parent_with_foreign_ancestor_is_rejected(owners, operation):
    if not owners.postgres:
        created = owners.bob.add_deck("Owned nested deck", parent_deck_id=owners.bob_deck)
        assert owners.bob.get_deck(created)["parent_deck_id"] == owners.bob_deck
        return
    target = owners.bob.add_deck("Owned reparent target")
    with owners.bob.transaction() as conn:
        conn.execute("UPDATE decks SET parent_deck_id = ? WHERE id = ?", (owners.deck, owners.bob_deck))
    before = owners.bob.get_deck(target)
    with pytest.raises(InputError, match="Parent deck"):
        if operation == "create":
            owners.bob.add_deck("Must not create", parent_deck_id=owners.bob_deck)
        else:
            owners.bob.update_deck(target, parent_deck_id=owners.bob_deck, expected_version=1)
    assert owners.bob.get_deck(target) == before
    assert owners.bob.get_deck_by_name("Must not create") is None


@pytest.mark.parametrize("state,reason", [("learning", "learning_due"), ("review", "review_due"), ("new", "new")])
def test_explicit_deck_queue_filters_owner_before_limit(owners, state, reason):
    for db, card, timestamp in [(owners.alice, owners.card, "2000-01-01T00:00:00Z"), (owners.bob, owners.bob_card, "2001-01-01T00:00:00Z")]:
        with db.transaction() as conn:
            conn.execute("UPDATE flashcards SET deck_id = ?, queue_state = ?, due_at = ?, created_at = ? WHERE uuid = ?", (owners.bob_deck, state, timestamp, timestamp, card))
    next_card, selection_reason = owners.bob.get_next_review_card(deck_id=owners.bob_deck)
    assert next_card is not None
    assert next_card["uuid"] == owners.bob_card
    assert selection_reason == reason


def test_valid_owned_ancestry_and_queue_priority_remain_available(owners):
    parent = owners.bob.add_deck("Valid owned parent", parent_deck_id=owners.bob_deck)
    child = owners.bob.add_deck("Valid owned child", parent_deck_id=parent)
    assert owners.bob.get_deck(child)["parent_deck_id"] == parent
    learning = owners.bob.add_flashcard({"deck_id": child, "front": "Learning", "back": "First", "queue_state": "learning", "due_at": "2001-01-01T00:00:00Z"})
    review = owners.bob.add_flashcard({"deck_id": child, "front": "Review", "back": "Second", "queue_state": "review", "due_at": "2000-01-01T00:00:00Z"})
    new = owners.bob.add_flashcard({"deck_id": child, "front": "New", "back": "Third"})
    for card, reason in [(learning, "learning_due"), (review, "review_due"), (new, "new")]:
        actual, selected_reason = owners.bob.get_next_review_card(deck_id=child)
        assert actual["uuid"] == card and selected_reason == reason
        owners.bob.soft_delete_flashcard(card, expected_version=1)


@pytest.mark.asyncio
async def test_authorized_service_handoff_uses_real_owner_db_before_flashcard_reads(owners):
    owners.alice.upsert_workspace("shared-owner-workspace", "Shared workspace")
    deck = owners.alice.add_deck("Explicit owner share deck", workspace_id="shared-owner-workspace")
    card = owners.alice.add_flashcard({"deck_id": deck, "front": "Shared read", "back": "Authorized owner repository"})
    share = {"id": 42, "workspace_id": "shared-owner-workspace", "owner_user_id": 2, "share_scope_type": "team", "share_scope_id": 11, "access_level": "read", "allow_clone": False}
    # Stub only the already-tested AuthNZ membership result; retain the real
    # authorization service, owner handoff, and both actual content databases.
    shares = SimpleNamespace(get_active_share_for_user=AsyncMock(return_value=share))
    users = SimpleNamespace(get_user_by_id=AsyncMock(return_value={"id": 2, "username": "Alice"}))
    loader = AsyncMock(return_value=owners.alice)
    service = SharedWorkspaceAccessService(shares, users, loader)
    with scoped_context(user_id=3, team_ids=[11]):
        context = await service.resolve(share_id=42, recipient_user_id=3)
        shares.get_active_share_for_user.assert_awaited_once_with(42, 3)
        loader.assert_awaited_once_with(2)
        selected_owner_db = loader.return_value
        assert context.owner_user_id == 2 and context.recipient_user_id == 3
        assert context.policy_actions["edit_workspace"]["allowed"] is False
        assert [row["id"] for row in selected_owner_db.list_decks(workspace_id=context.workspace_id)] == [deck]
        assert selected_owner_db.get_flashcard(card)["client_id"] == "2"
        assert owners.bob.list_decks(workspace_id=context.workspace_id) == []
        shares.get_active_share_for_user.return_value = None
        with pytest.raises(SharedWorkspaceNotFound):
            await service.resolve(share_id=42, recipient_user_id=3)
        loader.assert_awaited_once_with(2)  # Denial occurs before a second owner load.
