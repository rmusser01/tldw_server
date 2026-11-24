"""Flashcard domain store, transitional façade (TASK-13950.22 PR J1).

Callers outside ``CharactersRAGDB`` reach the flashcard methods as ``db.flashcard_store.<name>``. Until the bodies move
here (PR J2), each name is forwarded to the DB at call time, so an instance or class patch of the DB stays live.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.context import ChaChaContext

_MOVING: frozenset[str] = frozenset(
    {
        "_build_flashcard_search_shadow_fields",
        "_flashcard_owner_filter",
        "_validate_flashcard_deck_locked",
        "_validate_flashcard_workspace_locked",
        "_normalize_deck_parent_id",
        "_validate_deck_parent_locked",
        "add_deck",
        "list_decks",
        "get_deck",
        "get_deck_by_name",
        "upsert_deck_share",
        "list_deck_shares",
        "get_deck_share",
        "delete_deck_share",
        "update_deck",
        "_flashcard_template_default_placeholders",
        "_flashcard_template_tokens",
        "_flashcard_template_deleted_value",
        "_validate_flashcard_template_contract",
        "_serialize_flashcard_template_row",
        "add_flashcard_template",
        "count_flashcard_templates",
        "list_flashcard_templates",
        "get_flashcard_template",
        "update_flashcard_template",
        "soft_delete_flashcard_template",
        "_get_flashcard_id_from_uuid",
        "add_flashcard",
        "add_flashcards_bulk",
        "_flashcard_visibility_filter",
        "_normalize_flashcard_sqlite_query",
        "list_flashcards",
        "count_flashcards",
        "list_flashcard_tag_suggestions",
        "get_flashcards_by_uuids",
        "_srs_sm2_update",
        "_normalize_flashcard_review_mode",
        "_normalize_flashcard_review_session_status",
        "_normalize_flashcard_review_session_source_bundle",
        "_deserialize_flashcard_review_session_row",
        "_get_flashcard_review_session_deck_name_snapshot_for_conn",
        "_get_flashcard_review_session_row_for_conn",
        "_get_flashcard_review_session_reconstructed_aggregates",
        "_repair_flashcard_review_session_aggregates_if_unchanged",
        "abandon_stale_flashcard_review_sessions",
        "list_flashcard_review_sessions",
        "get_or_create_flashcard_review_session",
        "get_flashcard_review_session",
        "mark_flashcard_review_session_completed",
        "get_flashcard_review_session_rollup",
        "get_flashcard_reviewed_cards",
        "get_latest_flashcard_review",
        "review_flashcard",
        "get_flashcard_analytics_summary",
        "export_flashcards_csv",
        "get_flashcard",
        "add_flashcard_asset",
        "get_flashcard_asset",
        "get_flashcard_asset_content",
        "reconcile_flashcard_asset_refs",
        "cleanup_stale_flashcard_assets",
        "update_flashcard",
        "soft_delete_flashcard",
        "reset_flashcard_scheduling",
        "get_keywords_for_flashcard",
        "set_flashcard_tags",
        "_sync_flashcard_keyword_links",
        "_flashcard_owned_keyword_id",
        "create_study_pack",
        "_is_safe_suggestion_payload_key",
        "_sanitize_suggestion_evidence_reasons",
        "_sanitize_suggestion_payload",
        "create_suggestion_snapshot",
        "get_suggestion_snapshot",
        "list_suggestion_snapshots_for_anchor",
        "_suggestion_link_owner_filter",
        "_validate_suggestion_snapshot_locked",
        "create_suggestion_generation_link",
        "find_suggestion_generation_link",
        "replace_suggestion_generation_link",
        "find_suggestion_generation_link_by_fingerprint",
        "finalize_suggestion_generation_link",
        "soft_delete_suggestion_generation_link",
        "release_suggestion_generation_link_reservation",
        "soft_delete_deck_by_id",
        "get_study_pack",
        "add_study_pack_cards",
        "list_study_pack_cards",
        "add_flashcard_citations",
        "replace_flashcard_citations",
        "replace_flashcard_citations_and_source_reference_summary",
        "list_flashcard_citations",
        "set_flashcard_source_reference_summary",
        "get_study_pack_for_flashcard",
        "soft_delete_study_pack",
        "supersede_study_pack",
    }
)  # fmt: skip


# ponytail: transitional façade, replaced by the moved bodies in TASK-13950.22 T13
class FlashcardStore:
    def __init__(self, ctx: ChaChaContext) -> None:
        self._ctx = ctx

    def __getattr__(self, name: str) -> Any:
        if name in _MOVING:
            return getattr(self._ctx.db, name)
        raise AttributeError(name)
