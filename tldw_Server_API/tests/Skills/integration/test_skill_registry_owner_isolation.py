"""Shared Skill registry ownership must survive another user's filesystem sync."""

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Skills.skills_service import SkillsService

pytestmark = pytest.mark.integration
RAW = "---\nname: shared-owner-check\ndescription: Owned registry test\n---\nReturn the supplied text."


@pytest.fixture
def skill_owners(pg_restricted_backend, tmp_path):
    handles = []
    try:
        services = []
        for owner in (1, 2):
            root = tmp_path / str(owner)
            with chacha_operation(independent=True):
                db = CharactersRAGDB(root / "ChaChaNotes.db", client_id=str(owner), backend=pg_restricted_backend)
                handles.append(db)
                service = SkillsService(owner, root, db=db, sync_interval=0)
            db._discard_legacy_checkout()
            services.append(service)
        yield services
    finally:
        for db in reversed(handles):
            db._discard_legacy_checkout()
            db.close_all_connections()


@chacha_operation(independent=True)
def registry(service, name):
    with service.db.transaction():
        return service.db.get_skill_registry(name, include_deleted=True)


@pytest.mark.asyncio
async def test_foreign_empty_sync_cannot_change_saved_skill_version(skill_owners):
    alice, bob = skill_owners
    with chacha_operation(independent=True):
        created = await alice.create_skill("shared-owner-check", RAW)
    before = registry(alice, "shared-owner-check")
    with chacha_operation(independent=True):
        await bob.list_skills()
    after = registry(alice, "shared-owner-check")
    assert before["uuid"] == created["id"]  # nosec B101 # Confirm the actual created registry row.
    assert after == before  # nosec B101 # Bob's empty filesystem cannot delete or version Alice's row.


@pytest.mark.asyncio
async def test_two_owners_can_save_the_same_skill_name(skill_owners):
    alice, bob = skill_owners
    with chacha_operation(independent=True):
        first = await alice.create_skill("shared-owner-check", RAW)
    with chacha_operation(independent=True):
        second = await bob.create_skill("shared-owner-check", RAW)
    assert first["id"] != second["id"]  # nosec B101 # Same public name has independent stable identities.
    assert [registry(s, "shared-owner-check")["version"] for s in (alice, bob)] == [1, 1]  # nosec B101 # Neither save mutates the peer.


@pytest.mark.asyncio
async def test_foreign_registry_read_cannot_see_owned_skill(skill_owners):
    alice, bob = skill_owners
    with chacha_operation(independent=True):
        await alice.create_skill("shared-owner-check", RAW)
    assert registry(bob, "shared-owner-check") is None  # nosec B101 # A known name does not grant tenant access.


@chacha_operation(independent=True)
def scoped_registry_call(service, method, *args, **kwargs):
    with service.db.transaction():
        return getattr(service.db, method)(*args, **kwargs)


@pytest.mark.asyncio
async def test_foreign_uuid_counts_and_mutations_remain_hidden(skill_owners):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError, InputError

    alice, bob = skill_owners
    with chacha_operation(independent=True):
        created = await alice.create_skill("shared-owner-check", RAW)
    before = registry(alice, "shared-owner-check")
    assert scoped_registry_call(bob, "get_skill_registry_by_uuid", created["id"]) is None  # nosec B101 # Known immutable ID grants no read authority.
    assert scoped_registry_call(bob, "count_skill_registry", include_deleted=True) == 0  # nosec B101 # Counts cannot expose the peer registry.
    for method, args, kwargs in (
        ("update_skill_registry", ("shared-owner-check", {"description": "foreign"}), {"expected_version": 1}),
        ("mark_skill_registry_deleted", ("shared-owner-check",), {"expected_version": 1}),
        ("restore_skill_registry", ("shared-owner-check", {}), {"expected_version": 1}),
        ("purge_skill_registry", ("shared-owner-check",), {"expected_version": 1}),
        ("bulk_mark_skill_registry_deleted", ([("shared-owner-check", 1)],), {}),
    ):
        with pytest.raises((ConflictError, InputError)):
            scoped_registry_call(bob, method, *args, **kwargs)
    assert registry(alice, "shared-owner-check") == before  # nosec B101 # Every denied mutation leaves all owner fields intact.


@pytest.mark.asyncio
async def test_same_name_mutations_trash_restore_and_purge_are_independent(skill_owners):
    alice, bob = skill_owners
    with chacha_operation(independent=True):
        await alice.create_skill("shared-owner-check", RAW)
    with chacha_operation(independent=True):
        await bob.create_skill("shared-owner-check", RAW)
    b_before = registry(bob, "shared-owner-check")
    scoped_registry_call(
        alice, "update_skill_registry", "shared-owner-check", {"description": "Alice only"}, expected_version=1
    )
    scoped_registry_call(alice, "mark_skill_registry_deleted", "shared-owner-check", expected_version=2)
    a_deleted = registry(alice, "shared-owner-check")
    assert registry(bob, "shared-owner-check") == b_before  # nosec B101 # Owner mutation/Trash does not touch equal peer names.
    scoped_registry_call(
        bob, "update_skill_registry", "shared-owner-check", {"description": "Bob only"}, expected_version=1
    )
    assert registry(alice, "shared-owner-check") == a_deleted  # nosec B101 # Peer update cannot restore or edit the deleted row.
    scoped_registry_call(alice, "restore_skill_registry", "shared-owner-check", {}, expected_version=3)
    a_restored = registry(alice, "shared-owner-check")
    scoped_registry_call(bob, "mark_skill_registry_deleted", "shared-owner-check", expected_version=2)
    scoped_registry_call(bob, "purge_skill_registry", "shared-owner-check", expected_version=3)
    assert registry(alice, "shared-owner-check") == a_restored  # nosec B101 # Peer purge leaves the complete restored owner row.
    assert registry(bob, "shared-owner-check") is None  # nosec B101 # Only Bob's equal-name row was removed.
