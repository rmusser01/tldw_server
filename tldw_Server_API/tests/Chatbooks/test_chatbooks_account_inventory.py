from tldw_Server_API.app.core.Chatbooks.chatbook_account_inventory import ACCOUNT_DATA_INVENTORY
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService


EXPECTED_CATEGORIES = {
    "account_profile",
    "account_settings",
    "conversations",
    "notes",
    "characters",
    "world_books",
    "dictionaries",
    "prompts",
    "evaluations",
    "generated_documents",
    "explainer_sessions",
    "media_records",
    "media_transcripts",
    "media_chunks",
    "media_stored_artifacts",
    "media_pointers",
    "embeddings",
    "tags_categories_relationships",
    "sensitive_user_values",
}


class CountingDB:
    def execute_query(self, query, params=()):
        return []


def test_inventory_rows_have_required_restore_contract_fields():
    required = {
        "category",
        "source",
        "export_representation",
        "manifest_count_key",
        "import_handler_key",
        "dependencies",
        "sensitivity",
        "restore_status",
    }

    assert ACCOUNT_DATA_INVENTORY
    for row in ACCOUNT_DATA_INVENTORY:
        assert required <= row.to_summary().keys()
        assert row.manifest_count_key
        assert row.restore_status in {"restorable", "pointer_only", "non_restorable"}


def test_inventory_contains_expected_account_scope_categories():
    categories = {row.category for row in ACCOUNT_DATA_INVENTORY}
    assert EXPECTED_CATEGORIES <= categories


def test_limited_restore_rows_have_user_visible_warnings():
    for row in ACCOUNT_DATA_INVENTORY:
        if row.restore_status != "restorable":
            assert row.warning


def test_media_source_references_are_pointer_only():
    rows = {row.category: row for row in ACCOUNT_DATA_INVENTORY}
    assert rows["media_pointers"].restore_status == "pointer_only"


def test_scope_summary_uses_inventory_contract():
    service = ChatbookService(user_id="scope-user", db=CountingDB())

    scope = service.get_full_account_export_scope()

    categories = {row["category"]: row for row in scope["categories"]}
    assert scope["mode"] == "full_account"
    assert EXPECTED_CATEGORIES <= categories.keys()
    assert scope["total_items"] == sum(row["count"] for row in categories.values())
    assert scope["pointer_only_count"] == sum(
        1 for row in categories.values() if row["restore_status"] == "pointer_only"
    )
    assert scope["sensitive_category_count"] == sum(
        1 for row in categories.values() if row["sensitivity"] in {"sensitive", "secret"}
    )
    assert scope["warning_count"] == sum(1 for row in categories.values() if row["warning"])
    assert set(categories["account_profile"].keys()) == {
        "category",
        "label",
        "count",
        "restore_status",
        "sensitivity",
        "warning",
    }
