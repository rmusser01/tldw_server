"""
Unit + integration tests for PRD Round 2 endpoints:
  - Greeting list picker (GET/PUT /{chat_id}/greetings/...)
  - Author note info (GET /{chat_id}/author-note/info)
  - Lorebook diagnostic export (GET /{chat_id}/diagnostics/lorebook)
  - Preset editor CRUD (GET/POST/PUT/DELETE /presets/...)
"""

import json
from uuid import uuid4
import pytest

from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
    _collect_character_greeting_texts,
    _check_greeting_staleness,
    _compute_greetings_checksum,
    _parse_greeting_selection_index,
    _resolve_author_note_text,
    _summary_matches_existing,
    _estimate_tokens,
    _normalize_memory_scope,
    _TOKEN_BUDGET_AUTHOR_NOTE,
)


# ========================================================================
# Unit tests for greeting helpers
# ========================================================================

class TestGreetingHelpers:
    """Unit tests for greeting-related private helpers."""

    @pytest.mark.unit
    def test_collect_greeting_texts_basic(self):
        """Collects greetings from multiple fields, deduplicates."""
        character = {
            "greeting": "Hello!",
            "alternate_greetings": ["Hi there!", "Hey!"],
        }
        result = _collect_character_greeting_texts(character)
        assert "Hello!" in result
        assert "Hi there!" in result
        assert "Hey!" in result
        assert len(result) == 3

    @pytest.mark.unit
    def test_collect_greeting_texts_dedup(self):
        """Duplicate greetings are removed."""
        character = {
            "greeting": "Hello!",
            "first_message": "Hello!",
        }
        result = _collect_character_greeting_texts(character)
        assert result.count("Hello!") == 1

    @pytest.mark.unit
    def test_collect_greeting_texts_parses_json_string_lists(self):
        """Stringified JSON list greetings are expanded."""
        character = {
            "first_message": "Hello!",
            "alternate_greetings": '["Hi there!", "Welcome!"]',
        }
        result = _collect_character_greeting_texts(character)
        assert result == ["Hello!", "Hi there!", "Welcome!"]

    @pytest.mark.unit
    def test_collect_greeting_texts_keeps_multiline_primary_message(self):
        """Multiline single greeting text is preserved as one entry."""
        character = {
            "first_message": "Line one.\nLine two.",
            "alternate_greetings": ["Alt hello"],
        }
        result = _collect_character_greeting_texts(character)
        assert result == ["Line one.\nLine two.", "Alt hello"]

    @pytest.mark.unit
    def test_compute_greetings_checksum_stable(self):
        """Checksum is stable for same greetings."""
        character = {"greeting": "Hello!", "alternate_greetings": ["Hi!"]}
        c1 = _compute_greetings_checksum(character)
        c2 = _compute_greetings_checksum(character)
        assert c1 == c2
        assert len(c1) == 16

    @pytest.mark.unit
    def test_check_greeting_staleness_no_stored(self):
        """No staleness when no checksum stored."""
        settings = {}
        character = {"greeting": "Hello!"}
        assert _check_greeting_staleness(settings, character) is None

    @pytest.mark.unit
    def test_check_greeting_staleness_match(self):
        """No staleness when checksums match."""
        character = {"greeting": "Hello!"}
        checksum = _compute_greetings_checksum(character)
        settings = {"greetingsChecksum": checksum}
        assert _check_greeting_staleness(settings, character) is None

    @pytest.mark.unit
    def test_check_greeting_staleness_mismatch(self):
        """Staleness warning when checksums differ."""
        character = {"greeting": "Hello!"}
        settings = {"greetingsChecksum": "0000000000000000"}
        warning = _check_greeting_staleness(settings, character)
        assert warning is not None
        assert "changed" in warning.lower()

    @pytest.mark.unit
    def test_parse_greeting_selection_index(self):
        """Parses 'greeting:2:selected' → 2."""
        assert _parse_greeting_selection_index("greeting:2:selected") == 2
        assert _parse_greeting_selection_index("greeting:0:selected") == 0
        assert _parse_greeting_selection_index("invalid") is None
        assert _parse_greeting_selection_index(None) is None
        assert _parse_greeting_selection_index("greeting:-1:x") is None


# ========================================================================
# Unit tests for author note helpers
# ========================================================================

class TestAuthorNoteHelpers:
    """Unit tests for author note private helpers."""

    @pytest.mark.unit
    def test_resolve_author_note_basic(self):
        """Returns note from settings."""
        settings = {"authorNote": "Be careful here."}
        character = {}
        result = _resolve_author_note_text(settings, character)
        assert result == "Be careful here."

    @pytest.mark.unit
    def test_resolve_author_note_gm_only_for_prompt(self):
        """GM-only note returns empty when for_prompt=True."""
        settings = {"authorNote": "GM secret note", "authorNoteGmOnly": True}
        character = {}
        assert _resolve_author_note_text(settings, character, for_prompt=True) == ""
        assert _resolve_author_note_text(settings, character, for_prompt=False) == "GM secret note"

    @pytest.mark.unit
    def test_resolve_author_note_disabled(self):
        """Disabled note returns empty."""
        settings = {"authorNote": "Some note", "authorNoteEnabled": False}
        character = {}
        assert _resolve_author_note_text(settings, character) == ""

    @pytest.mark.unit
    def test_estimate_tokens(self):
        """Rough token estimation: ~4 chars per token."""
        assert _estimate_tokens("") == 0
        assert _estimate_tokens("hello world") >= 1
        assert _estimate_tokens("a" * 100) == 25

    @pytest.mark.unit
    def test_normalize_memory_scope(self):
        """Returns valid scope or default 'shared'."""
        assert _normalize_memory_scope({"memoryScope": "character"}) == "character"
        assert _normalize_memory_scope({"memoryScope": "both"}) == "both"
        assert _normalize_memory_scope({"memoryScope": "shared"}) == "shared"
        assert _normalize_memory_scope({}) == "shared"
        assert _normalize_memory_scope({"memoryScope": "invalid"}) == "shared"

    @pytest.mark.unit
    def test_token_budget_constant(self):
        """Budget constant is 240."""
        assert _TOKEN_BUDGET_AUTHOR_NOTE == 240


# ========================================================================
# Integration tests for greeting endpoints
# ========================================================================

class TestGreetingEndpoints:
    """Integration tests for greeting list/select endpoints."""

    @pytest.mark.integration
    def test_list_greetings(self, test_client, auth_headers, sample_character_card):
        """GET /chats/{chat_id}/greetings returns greeting list."""
        # Create character (uses first_message as the greeting field)
        char_data = dict(sample_character_card)
        char_data["first_message"] = "Hello, adventurer!"
        char_data["alternate_greetings"] = ["Greetings, traveler!", "Welcome!"]

        char_resp = test_client.post("/api/v1/characters/", json=char_data, headers=auth_headers)
        assert char_resp.status_code == 201
        char_id = char_resp.json()["id"]

        # Create chat session
        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "test"},
            headers=auth_headers,
        )
        assert chat_resp.status_code == 201
        chat_id = chat_resp.json()["id"]

        # List greetings
        resp = test_client.get(f"/api/v1/chats/{chat_id}/greetings", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_id"] == chat_id
        # At minimum the first_message greeting should be present
        assert len(data["greetings"]) >= 1
        greeting_texts = [g["text"] for g in data["greetings"]]
        assert "Hello, adventurer!" in greeting_texts

    @pytest.mark.integration
    def test_list_greetings_staleness(self, test_client, auth_headers, sample_character_card):
        """GET /chats/{chat_id}/greetings includes staleness warning when checksum mismatches."""
        char_data = dict(sample_character_card)
        char_data["first_message"] = "Hello!"

        char_resp = test_client.post("/api/v1/characters/", json=char_data, headers=auth_headers)
        char_id = char_resp.json()["id"]

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["id"]

        # Set a stale checksum in settings (use a future updatedAt so LWW merge accepts it)
        settings_resp = test_client.put(
            f"/api/v1/chats/{chat_id}/settings",
            json={"settings": {
                "greetingsChecksum": "stale_checksum_x",
                "updatedAt": "2099-01-01T00:00:00Z",
            }},
            headers=auth_headers,
        )
        assert settings_resp.status_code == 200
        stored = settings_resp.json().get("settings", {})
        assert stored.get("greetingsChecksum") == "stale_checksum_x"

        resp = test_client.get(f"/api/v1/chats/{chat_id}/greetings", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["staleness_warning"] is not None

    @pytest.mark.integration
    def test_select_greeting(self, test_client, auth_headers, sample_character_card):
        """PUT /chats/{chat_id}/greetings/select updates selection."""
        char_data = dict(sample_character_card)
        char_data["first_message"] = "Hello!"
        char_data["alternate_greetings"] = ["Alt greeting"]

        char_resp = test_client.post("/api/v1/characters/", json=char_data, headers=auth_headers)
        char_id = char_resp.json()["id"]

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["id"]

        resp = test_client.put(
            f"/api/v1/chats/{chat_id}/greetings/select",
            json={"index": 1},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_index"] == 1
        assert data["checksum_updated"] is True

    @pytest.mark.integration
    def test_select_greeting_out_of_range(self, test_client, auth_headers, sample_character_card):
        """PUT /chats/{chat_id}/greetings/select rejects out-of-range index."""
        char_data = dict(sample_character_card)
        char_data["first_message"] = "Hello!"

        char_resp = test_client.post("/api/v1/characters/", json=char_data, headers=auth_headers)
        char_id = char_resp.json()["id"]

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["id"]

        resp = test_client.put(
            f"/api/v1/chats/{chat_id}/greetings/select",
            json={"index": 999},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ========================================================================
# Integration tests for author note info endpoint
# ========================================================================

class TestAuthorNoteInfoEndpoint:
    """Integration tests for GET /{chat_id}/author-note/info."""

    @pytest.mark.integration
    def test_author_note_info_basic(self, test_client, auth_headers, sample_character_card):
        """GET /chats/{chat_id}/author-note/info returns correct token info."""
        char_resp = test_client.post("/api/v1/characters/", json=sample_character_card, headers=auth_headers)
        char_id = char_resp.json()["id"]

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["id"]

        # Set an author note
        test_client.put(
            f"/api/v1/chats/{chat_id}/settings",
            json={"settings": {"authorNote": "Be careful in the forest."}},
            headers=auth_headers,
        )

        resp = test_client.get(f"/api/v1/chats/{chat_id}/author-note/info", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_id"] == chat_id
        assert data["text"] == "Be careful in the forest."
        assert data["tokens_estimated"] > 0
        assert data["budget"] == 240
        assert data["source"] == "settings"

    @pytest.mark.integration
    def test_author_note_info_gm_only(self, test_client, auth_headers, sample_character_card):
        """GM-only note: text present, text_for_prompt empty."""
        char_resp = test_client.post("/api/v1/characters/", json=sample_character_card, headers=auth_headers)
        char_id = char_resp.json()["id"]

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["id"]

        test_client.put(
            f"/api/v1/chats/{chat_id}/settings",
            json={"settings": {"authorNote": "Secret GM note", "authorNoteGmOnly": True}},
            headers=auth_headers,
        )

        resp = test_client.get(f"/api/v1/chats/{chat_id}/author-note/info", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Secret GM note"
        assert data["text_for_prompt"] == ""
        assert data["gm_only"] is True

    @pytest.mark.integration
    def test_author_note_info_truncated(self, test_client, auth_headers, sample_character_card):
        """Truncated flag when note exceeds budget."""
        char_resp = test_client.post("/api/v1/characters/", json=sample_character_card, headers=auth_headers)
        char_id = char_resp.json()["id"]

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["id"]

        # Create a note that exceeds 240 tokens (~960 chars at 4 chars/token)
        long_note = "word " * 300  # ~1500 chars = ~375 tokens
        test_client.put(
            f"/api/v1/chats/{chat_id}/settings",
            json={"settings": {"authorNote": long_note.strip()}},
            headers=auth_headers,
        )

        resp = test_client.get(f"/api/v1/chats/{chat_id}/author-note/info", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["truncated"] is True
        assert len(data["warnings"]) >= 1


# ========================================================================
# Integration tests for lorebook diagnostic export
# ========================================================================

class TestLorebookDiagnosticExport:
    """Integration tests for GET /{chat_id}/diagnostics/lorebook."""

    @pytest.mark.integration
    def test_empty_diagnostics(self, test_client, auth_headers, sample_character_card):
        """Returns empty list for chat with no diagnostics."""
        char_resp = test_client.post("/api/v1/characters/", json=sample_character_card, headers=auth_headers)
        char_id = char_resp.json()["id"]

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "test"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["id"]

        resp = test_client.get(f"/api/v1/chats/{chat_id}/diagnostics/lorebook", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["chat_id"] == chat_id
        assert data["total_turns_with_diagnostics"] == 0
        assert data["turns"] == []

    @pytest.mark.integration
    def test_diagnostics_ordering_and_pagination(
        self,
        test_client,
        auth_headers,
        sample_character_card,
        character_db,
    ):
        """Returns paginated diagnostics in requested sort order."""
        char_resp = test_client.post("/api/v1/characters/", json=sample_character_card, headers=auth_headers)
        char_id = char_resp.json()["id"]

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "diag-order"},
            headers=auth_headers,
        )
        chat_id = chat_resp.json()["id"]

        first_message_id = str(uuid4())
        second_message_id = str(uuid4())

        character_db.add_message(
            {
                "id": first_message_id,
                "conversation_id": chat_id,
                "sender": "assistant",
                "content": "First assistant turn",
                "parent_message_id": None,
                "deleted": 0,
                "client_id": "test_client",
                "version": 1,
            }
        )
        character_db.add_message(
            {
                "id": second_message_id,
                "conversation_id": chat_id,
                "sender": "assistant",
                "content": "Second assistant turn",
                "parent_message_id": first_message_id,
                "deleted": 0,
                "client_id": "test_client",
                "version": 1,
            }
        )

        first_diag = {
            "entry_id": 11,
            "world_book_id": 5,
            "activation_reason": "keyword_match",
            "keyword": "alpha",
            "token_cost": 12,
            "priority": 10,
            "regex_match": False,
            "content_preview": "alpha preview",
        }
        second_diag = {
            "entry_id": 22,
            "world_book_id": 6,
            "activation_reason": "regex_match",
            "keyword": "beta.*",
            "token_cost": 18,
            "priority": 20,
            "regex_match": True,
            "content_preview": "beta preview",
        }

        assert character_db.add_message_metadata(
            first_message_id,
            extra={"lorebook_diagnostics": [first_diag]},
        )
        assert character_db.add_message_metadata(
            second_message_id,
            extra={"lorebook_diagnostics": [second_diag]},
        )

        asc_resp = test_client.get(
            f"/api/v1/chats/{chat_id}/diagnostics/lorebook",
            params={"page": 1, "size": 1, "order": "asc"},
            headers=auth_headers,
        )
        assert asc_resp.status_code == 200
        asc_data = asc_resp.json()
        assert asc_data["total_turns_with_diagnostics"] == 2
        assert asc_data["pagination"] == {
            "mode": "page",
            "page": 1,
            "per_page": 1,
            "total": 2,
            "total_pages": 2,
            "has_more": True,
        }
        assert len(asc_data["turns"]) == 1
        assert asc_data["turns"][0]["message_id"] == first_message_id

        desc_resp = test_client.get(
            f"/api/v1/chats/{chat_id}/diagnostics/lorebook",
            params={"page": 1, "size": 1, "order": "desc"},
            headers=auth_headers,
        )
        assert desc_resp.status_code == 200
        desc_data = desc_resp.json()
        assert desc_data["total_turns_with_diagnostics"] == 2
        assert len(desc_data["turns"]) == 1
        assert desc_data["turns"][0]["message_id"] == second_message_id

    @pytest.mark.integration
    def test_diagnostics_include_character_name_sender_alias(
        self,
        test_client,
        auth_headers,
        sample_character_card,
        character_db,
    ):
        """Diagnostics export should include assistant turns stored under character name senders."""
        char_resp = test_client.post("/api/v1/characters/", json=sample_character_card, headers=auth_headers)
        assert char_resp.status_code == 201
        character = char_resp.json()
        char_id = character["id"]
        char_name = character.get("name") or "Assistant"

        chat_resp = test_client.post(
            "/api/v1/chats/",
            json={"character_id": char_id, "title": "diag-name-sender"},
            headers=auth_headers,
        )
        assert chat_resp.status_code == 201
        chat_id = chat_resp.json()["id"]

        message_id = str(uuid4())
        character_db.add_message(
            {
                "id": message_id,
                "conversation_id": chat_id,
                "sender": char_name,
                "content": "Assistant turn saved with character-name sender",
                "parent_message_id": None,
                "deleted": 0,
                "client_id": "test_client",
                "version": 1,
            }
        )
        assert character_db.add_message_metadata(
            message_id,
            extra={"lorebook_diagnostics": [{"entry_id": 99, "keyword": "alpha"}]},
        )

        resp = test_client.get(
            f"/api/v1/chats/{chat_id}/diagnostics/lorebook",
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_turns_with_diagnostics"] == 1
        assert len(data["turns"]) == 1
        assert data["turns"][0]["message_id"] == message_id


# ========================================================================
# Integration tests for preset editor CRUD
# ========================================================================

class TestPresetEditorEndpoints:
    """Integration tests for preset CRUD endpoints."""

    @pytest.mark.integration
    def test_list_presets_returns_builtins(self, test_client, auth_headers):
        """GET /chats/presets returns built-in presets."""
        resp = test_client.get("/api/v1/chats/presets", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        preset_ids = [p["preset_id"] for p in data["presets"]]
        assert "default" in preset_ids
        assert "st_default" in preset_ids

    @pytest.mark.integration
    def test_list_preset_tokens(self, test_client, auth_headers):
        """GET /chats/presets/tokens returns template tokens."""
        resp = test_client.get("/api/v1/chats/presets/tokens", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        tokens = [t["token"] for t in data["tokens"]]
        assert "{{char}}" in tokens
        assert "{{user}}" in tokens

    @pytest.mark.integration
    def test_crud_lifecycle(self, test_client, auth_headers):
        """Create → list → update → delete preset lifecycle."""
        # Create
        create_resp = test_client.post(
            "/api/v1/chats/presets",
            json={
                "preset_id": "my_custom",
                "name": "My Custom Preset",
                "section_order": ["identity", "description"],
                "section_templates": {
                    "identity": "You are {{char}}.",
                    "description": "{{description}}",
                },
            },
            headers=auth_headers,
        )
        assert create_resp.status_code == 201
        data = create_resp.json()
        assert data["preset_id"] == "my_custom"
        assert data["builtin"] is False

        # List should include it
        list_resp = test_client.get("/api/v1/chats/presets", headers=auth_headers)
        preset_ids = [p["preset_id"] for p in list_resp.json()["presets"]]
        assert "my_custom" in preset_ids

        # Update
        update_resp = test_client.put(
            "/api/v1/chats/presets/my_custom",
            json={"name": "Updated Preset Name"},
            headers=auth_headers,
        )
        assert update_resp.status_code == 200
        assert update_resp.json()["name"] == "Updated Preset Name"

        # Delete
        del_resp = test_client.delete("/api/v1/chats/presets/my_custom", headers=auth_headers)
        assert del_resp.status_code == 204

        # Verify gone
        list_resp2 = test_client.get("/api/v1/chats/presets", headers=auth_headers)
        preset_ids2 = [p["preset_id"] for p in list_resp2.json()["presets"]]
        assert "my_custom" not in preset_ids2

    @pytest.mark.integration
    def test_cannot_delete_builtin(self, test_client, auth_headers):
        """Cannot delete built-in presets (422)."""
        resp = test_client.delete("/api/v1/chats/presets/default", headers=auth_headers)
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_cannot_modify_builtin(self, test_client, auth_headers):
        """Cannot modify built-in presets (422)."""
        resp = test_client.put(
            "/api/v1/chats/presets/st_default",
            json={"name": "Hacked"},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_cannot_create_with_builtin_id(self, test_client, auth_headers):
        """Cannot create preset with built-in ID (422)."""
        resp = test_client.post(
            "/api/v1/chats/presets",
            json={
                "preset_id": "default",
                "name": "Fake",
                "section_order": [],
                "section_templates": {},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.integration
    def test_create_preset_returns_500_when_db_upsert_fails(self, test_client, auth_headers, monkeypatch):
        """Create should fail with 500 when DB layer reports upsert failure."""
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

        monkeypatch.setattr(CharactersRAGDB, "upsert_prompt_preset", lambda *args, **kwargs: False)

        resp = test_client.post(
            "/api/v1/chats/presets",
            json={
                "preset_id": "db_fail_create",
                "name": "DB Failure Create",
                "section_order": ["identity"],
                "section_templates": {"identity": "You are {{char}}."},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 500

    @pytest.mark.integration
    def test_delete_preset_returns_500_when_db_delete_fails(self, test_client, auth_headers, monkeypatch):
        """Delete should fail with 500 when DB layer reports delete failure."""
        create_resp = test_client.post(
            "/api/v1/chats/presets",
            json={
                "preset_id": "db_fail_delete",
                "name": "DB Failure Delete",
                "section_order": ["identity"],
                "section_templates": {"identity": "You are {{char}}."},
            },
            headers=auth_headers,
        )
        assert create_resp.status_code == 201

        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

        monkeypatch.setattr(CharactersRAGDB, "delete_prompt_preset", lambda *args, **kwargs: False)

        resp = test_client.delete("/api/v1/chats/presets/db_fail_delete", headers=auth_headers)
        assert resp.status_code == 500


class TestSummaryRobustness:
    """Unit tests for summary parsing robustness."""

    @pytest.mark.unit
    def test_summary_matches_existing_handles_non_numeric_fields(self):
        """Malformed numeric values should not crash summary comparison."""
        existing_summary = {
            "content": "summary",
            "sourceRange": {"fromMessageId": "m1", "toMessageId": "m2"},
            "thresholdMessages": "oops",
            "windowMessages": "12",
            "compressedCount": "3",
        }
        assert _summary_matches_existing(
            existing_summary,
            content="summary",
            source_from_id="m1",
            source_to_id="m2",
            threshold=40,
            window=12,
            compressed_count=3,
        ) is False
