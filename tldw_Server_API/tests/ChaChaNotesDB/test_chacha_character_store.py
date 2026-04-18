import json

from .conftest import _create_sample_card_data


def test_character_store_add_update_restore_roundtrip(db_instance):
    store = db_instance._character_store

    created_id = store.add_character_card(_create_sample_card_data("Store"))
    assert isinstance(created_id, int)

    created = store.get_character_card_by_id(created_id)
    assert created is not None
    assert created["name"] == "Test Character Store"
    assert created["tags"] == ["test", "sample"]

    listed = store.list_character_cards(limit=200, offset=0)
    assert any(item["id"] == created_id for item in listed)

    queried, total = store.query_character_cards(
        query="character store",
        sort_by="name",
        sort_order="asc",
        limit=20,
        offset=0,
    )
    assert total >= 1
    assert any(item["id"] == created_id for item in queried)

    rename_tags = store.manage_character_tags(
        operation="rename",
        source_tag="sample",
        target_tag="renamed",
    )
    assert created_id in rename_tags["updated_character_ids"]

    renamed = store.get_character_card_by_name("Test Character Store")
    assert renamed is not None
    assert renamed["tags"] == ["test", "renamed"]

    assert store.update_character_card(
        created_id,
        {"description": "Updated by store", "tags": json.dumps(["test", "updated"])},
        expected_version=int(renamed["version"]),
    ) is True

    updated = store.get_character_card_by_id(created_id)
    assert updated is not None
    assert updated["description"] == "Updated by store"
    assert updated["tags"] == ["test", "updated"]

    assert store.soft_delete_character_card(created_id, expected_version=int(updated["version"])) is True

    deleted_row = db_instance.get_connection().execute(
        "SELECT deleted, version FROM character_cards WHERE id = ?",
        (created_id,),
    ).fetchone()
    assert deleted_row is not None
    assert int(deleted_row["deleted"]) == 1

    assert store.restore_character_card(created_id, expected_version=int(deleted_row["version"])) is True

    restored = store.get_character_card_by_id(created_id)
    assert restored is not None
    assert not restored["deleted"]
    assert restored["description"] == "Updated by store"


def test_character_store_preserves_client_id_override(db_instance):
    store = db_instance._character_store

    created_id = store.add_character_card(
        _create_sample_card_data("Override", client_id_override="external-client")
    )

    created = store.get_character_card_by_id(created_id)
    assert created is not None
    assert created["client_id"] == "external-client"
