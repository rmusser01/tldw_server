import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chatbook_schemas import ChatbookAccountScopeCategory
from tldw_Server_API.app.core.Chatbooks.chatbook_account_inventory import ACCOUNT_DATA_INVENTORY
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError


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


class FailingCountDB:
    def execute_query(self, query, params=()):
        if str(query).lstrip().upper().startswith("SELECT"):
            raise CharactersRAGDBError("optional table is unavailable")
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
        row["count"] for row in categories.values() if row["restore_status"] == "pointer_only"
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


def test_scope_summary_treats_chacha_count_failures_as_zero():
    service = ChatbookService(user_id="scope-failing-counts", db=FailingCountDB())

    scope = service.get_full_account_export_scope()

    categories = {row["category"]: row for row in scope["categories"]}
    assert categories["world_books"]["count"] == 0
    assert categories["dictionaries"]["count"] == 0


def test_scope_summary_counts_pointer_only_items():
    service = ChatbookService(user_id="scope-pointer-count", db=CountingDB())
    service._scope_count_for_category = lambda category: 25 if category == "media_pointers" else 0

    scope = service.get_full_account_export_scope()

    categories = {row["category"]: row for row in scope["categories"]}
    assert categories["media_pointers"]["count"] == 25
    assert scope["pointer_only_count"] == 25


def test_scope_summary_skips_missing_optional_prompt_store(monkeypatch):
    service = ChatbookService(user_id="scope-missing-prompts", db=CountingDB())
    monkeypatch.setattr(
        service,
        "_get_prompts_db",
        lambda: (_ for _ in ()).throw(AssertionError("prompts DB should not initialize")),
    )

    scope = service.get_full_account_export_scope()
    categories = {row["category"]: row for row in scope["categories"]}

    assert categories["prompts"]["count"] == 0


def test_scope_category_schema_rejects_unknown_contract_values():
    valid = {
        "category": "media_pointers",
        "label": "Media source references",
        "count": 1,
        "restore_status": "pointer_only",
        "sensitivity": "personal",
    }

    with pytest.raises(ValidationError):
        ChatbookAccountScopeCategory(**{**valid, "restore_status": "partial"})
    with pytest.raises(ValidationError):
        ChatbookAccountScopeCategory(**{**valid, "sensitivity": "private"})
