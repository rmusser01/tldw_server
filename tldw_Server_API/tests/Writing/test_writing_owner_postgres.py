"""Private Writing records stay isolated on the supported shared PostgreSQL backend."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind", ["session", "template", "theme"])
def test_writing_records_are_private_to_their_owner(pg_restricted_backend, kind):
    """Use the real table-owning non-bypass fixture and public DB operations."""
    alice = CharactersRAGDB(":memory:", client_id="101", backend=pg_restricted_backend)
    bob = None
    try:
        bob = CharactersRAGDB(":memory:", client_id="202", backend=pg_restricted_backend)

        def create(db, name):
            if kind == "session":
                return db.add_writing_session(name, {"prompt": name})
            if kind == "template":
                db.add_writing_template(name, {"prompt": name})
            else:
                db.add_writing_theme(name, css="body { color: red; }")
            return name

        read = "get_writing_session" if kind == "session" else f"get_writing_{kind}_by_name"
        listing = f"list_writing_{kind}s"
        count = f"count_writing_{kind}s"
        update = f"update_writing_{kind}"
        delete = f"soft_delete_writing_{kind}"
        owned = create(alice, "Alice private writing")
        saved = getattr(alice, read)(owned)
        assert saved is not None  # nosec B101 - Verify the positive owner read before the denial.
        assert getattr(bob, read)(owned) is None  # nosec B101 - Actual cross-owner data disclosure regression.
        assert getattr(bob, listing)() == []  # nosec B101 - Private records must not leak through lists/exports.
        assert getattr(bob, count)() == 0  # nosec B101 - Counts must use the same owner boundary.
        change = {"css": "body { color: blue; }"} if kind == "theme" else {"payload": {"prompt": "foreign overwrite"}}
        with pytest.raises(ConflictError):
            getattr(bob, update)(owned, change, expected_version=saved["version"])
        with pytest.raises(ConflictError):
            getattr(bob, delete)(owned, expected_version=saved["version"])
        assert getattr(alice, read)(owned) == saved  # nosec B101 - Denied writes preserve the entire original record.
        peer = create(bob, "Bob private writing")
        assert getattr(bob, read)(peer) is not None  # nosec B101 - Isolation must preserve legitimate peer use.
        assert getattr(alice, read)(peer) is None  # nosec B101 - Isolation is symmetric across accounts.
        getattr(alice, update)(
            owned, {"name": "Alice revised"} if kind == "session" else change, expected_version=saved["version"]
        )
        changed = getattr(alice, read)(owned)
        assert changed["version"] == saved["version"] + 1  # nosec B101 - The owner can still make a legitimate versioned change.
        # Reopening a current schema reapplies the complete canonical policy set.
        alice.close_connection()
        reopened = CharactersRAGDB(":memory:", client_id="101", backend=pg_restricted_backend)
        try:
            assert getattr(reopened, read)(owned) == changed  # nosec B101 - Saved owner data survives normal reopen.
            assert getattr(reopened, read)(peer) is None  # nosec B101 - Reopen retains the private owner boundary.
        finally:
            reopened.close_connection()
    finally:
        if bob is not None:
            bob.close_connection()
        alice.close_connection()
