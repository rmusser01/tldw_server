"""Shared constants for Recurring Question scheduled tasks."""

from __future__ import annotations

DEFAULT_SEARCHABLE_SOURCES = ("media_db", "notes", "chats")
FINDING_POLICY_PRESETS = {"balanced_findings", "high_confidence_only"}
GENERATION_MODES = {"disabled", "optional", "required"}
RETENTION_POLICY_MODES = {"default", "custom"}
SUPPORTED_SCOPE_FIELDS = {
    "mode",
    "sources",
    "collection_ids",
    "tag_ids",
    "saved_search_ids",
    "source_types",
    "date_window",
    "workspace_id",
    "advanced_filters",
}

