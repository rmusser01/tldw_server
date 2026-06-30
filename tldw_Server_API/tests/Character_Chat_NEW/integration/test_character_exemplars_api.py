"""Integration tests for character exemplar endpoints."""

import pytest

from tldw_Server_API.app.api.v1.endpoints import characters_endpoint as characters_endpoint_module

pytestmark = pytest.mark.integration


class TestCharacterExemplarEndpoints:
    def _create_character(self, test_client, auth_headers, name: str) -> int:
        response = test_client.post(
            "/api/v1/characters/",
            json={
                "name": name,
                "description": "Character for exemplar endpoint tests",
                "personality": "Confident and concise",
                "first_message": "Hello there.",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        return int(response.json()["id"])

    def test_character_exemplar_crud_and_search(self, test_client, auth_headers):
        char_id = self._create_character(test_client, auth_headers, "Exemplar API Character CRUD")

        create_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "Opening line for press challenge with measured confidence.",
                "source": {"type": "article", "url_or_id": "doc-1", "date": "2026-02-08"},
                "novelty_hint": "unknown",
                "labels": {
                    "emotion": "neutral",
                    "scenario": "press_challenge",
                    "rhetorical": ["opener", "emphasis"],
                    "register": "formal",
                },
                "safety": {"allowed": ["general"], "blocked": ["harmful"]},
                "rights": {"public_figure": True, "notes": "curated quote"},
            },
            headers=auth_headers,
        )

        assert create_response.status_code == 201
        created = create_response.json()
        exemplar_id = created["id"]
        assert created["character_id"] == char_id
        assert created["labels"]["scenario"] == "press_challenge"

        get_response = test_client.get(
            f"/api/v1/characters/{char_id}/exemplars/{exemplar_id}",
            headers=auth_headers,
        )
        assert get_response.status_code == 200
        assert get_response.json()["id"] == exemplar_id

        update_response = test_client.put(
            f"/api/v1/characters/{char_id}/exemplars/{exemplar_id}",
            json={
                "text": "Updated opener line for a media question.",
                "labels": {"emotion": "happy", "scenario": "press_challenge", "rhetorical": ["opener"]},
                "length_tokens": 10,
            },
            headers=auth_headers,
        )
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["labels"]["emotion"] == "happy"
        assert updated["length_tokens"] == 10

        search_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars/search",
            json={
                "query": "media question",
                "filter": {"emotion": "happy", "scenario": "press_challenge", "rhetorical": ["opener"]},
                "limit": 10,
                "offset": 0,
            },
            headers=auth_headers,
        )
        assert search_response.status_code == 200
        search_data = search_response.json()
        assert search_data["total"] >= 1
        assert any(item["id"] == exemplar_id for item in search_data["items"])

        delete_response = test_client.delete(
            f"/api/v1/characters/{char_id}/exemplars/{exemplar_id}",
            headers=auth_headers,
        )
        assert delete_response.status_code == 200

        missing_response = test_client.get(
            f"/api/v1/characters/{char_id}/exemplars/{exemplar_id}",
            headers=auth_headers,
        )
        assert missing_response.status_code == 404

    def test_character_exemplar_batch_create_and_debug_selection(self, test_client, auth_headers):
        char_id = self._create_character(test_client, auth_headers, "Exemplar API Character Batch")

        batch_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json=[
                {
                    "text": "Quick opener for fan banter.",
                    "labels": {
                        "emotion": "happy",
                        "scenario": "fan_banter",
                        "rhetorical": ["opener"],
                    },
                },
                {
                    "text": "Closing line that keeps tone respectful.",
                    "labels": {
                        "emotion": "neutral",
                        "scenario": "small_talk",
                        "rhetorical": ["ender"],
                    },
                },
            ],
            headers=auth_headers,
        )

        assert batch_response.status_code == 201
        batch_items = batch_response.json()
        assert isinstance(batch_items, list)
        assert len(batch_items) == 2

        debug_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars/select/debug",
            json={
                "user_turn": "Give me a quick fan-facing response.",
                "selection_config": {
                    "budget_tokens": 40,
                    "max_exemplar_tokens": 30,
                    "mmr_lambda": 0.7,
                },
            },
            headers=auth_headers,
        )

        assert debug_response.status_code == 200
        debug_data = debug_response.json()
        assert debug_data["budget_tokens"] <= 40
        assert len(debug_data["selected"]) >= 1
        assert len(debug_data["scores"]) == len(debug_data["selected"])

    def test_character_exemplar_search_filtering(self, test_client, auth_headers):
        char_id = self._create_character(test_client, auth_headers, "Exemplar API Character Filter")

        test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "Boardroom guidance with measured tone.",
                "labels": {
                    "emotion": "neutral",
                    "scenario": "boardroom",
                    "rhetorical": ["emphasis"],
                },
            },
            headers=auth_headers,
        )
        test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "Fan banter with upbeat energy.",
                "labels": {
                    "emotion": "happy",
                    "scenario": "fan_banter",
                    "rhetorical": ["opener"],
                },
            },
            headers=auth_headers,
        )

        search_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars/search",
            json={
                "filter": {
                    "emotion": "neutral",
                    "scenario": "boardroom",
                    "rhetorical": ["emphasis"],
                },
                "limit": 10,
                "offset": 0,
            },
            headers=auth_headers,
        )

        assert search_response.status_code == 200
        payload = search_response.json()
        assert payload["total"] == 1
        assert len(payload["items"]) == 1
        assert payload["items"][0]["labels"]["scenario"] == "boardroom"

    def test_character_exemplar_search_includes_canonical_pagination(self, test_client, auth_headers) -> None:
        """Search responses expose canonical offset pagination while preserving legacy total."""
        char_id = self._create_character(test_client, auth_headers, "Exemplar API Character Pagination")

        for index in range(3):
            create_response = test_client.post(
                f"/api/v1/characters/{char_id}/exemplars",
                json={
                    "text": f"Pagination exemplar {index}",
                    "labels": {
                        "emotion": "neutral",
                        "scenario": "boardroom",
                        "rhetorical": ["opener"],
                    },
                },
                headers=auth_headers,
            )
            assert create_response.status_code == 201

        search_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars/search",
            json={
                "filter": {"emotion": "neutral", "scenario": "boardroom"},
                "limit": 1,
                "offset": 1,
            },
            headers=auth_headers,
        )

        assert search_response.status_code == 200
        payload = search_response.json()
        assert payload["total"] == 3
        assert len(payload["items"]) == 1
        assert payload["pagination"] == {
            "mode": "offset",
            "limit": 1,
            "offset": 1,
            "total": 3,
            "has_more": True,
            "next_offset": 2,
        }

    def test_character_exemplar_debug_selection_can_use_embedding_scores(
        self,
        test_client,
        auth_headers,
        monkeypatch,
    ):
        char_id = self._create_character(test_client, auth_headers, "Exemplar API Character Embedding Debug")

        lexical_create = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "board meeting strategy budget plan",
                "labels": {
                    "emotion": "neutral",
                    "scenario": "boardroom",
                    "rhetorical": ["opener"],
                },
            },
            headers=auth_headers,
        )
        assert lexical_create.status_code == 201
        lexical_id = lexical_create.json()["id"]

        semantic_create = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "generic reply with sparse lexical overlap",
                "labels": {
                    "emotion": "neutral",
                    "scenario": "boardroom",
                    "rhetorical": ["emphasis"],
                },
            },
            headers=auth_headers,
        )
        assert semantic_create.status_code == 201
        semantic_id = semantic_create.json()["id"]

        observed: dict[str, str] = {}

        def _fake_embedding_scores(user_turn: str, candidates: list[dict], **kwargs):
            assert user_turn
            observed["model_id_override"] = str(kwargs.get("model_id_override"))
            observed["user_id"] = str(kwargs.get("user_id"))
            observed["character_id"] = str(kwargs.get("character_id"))
            candidate_ids = {str(item.get("id")) for item in candidates}
            assert lexical_id in candidate_ids
            assert semantic_id in candidate_ids
            return {
                lexical_id: 0.0,
                semantic_id: 1.0,
            }

        monkeypatch.setattr(
            characters_endpoint_module,
            "score_exemplars_with_embeddings",
            _fake_embedding_scores,
        )

        debug_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars/select/debug",
            json={
                "user_turn": "Need boardroom guidance on strategy and budget.",
                "selection_config": {
                    "budget_tokens": 80,
                    "max_exemplar_tokens": 60,
                    "mmr_lambda": 0.9,
                    "use_embedding_scores": True,
                    "embedding_model_id": "stub:embedding-model",
                },
            },
            headers=auth_headers,
        )

        assert debug_response.status_code == 200
        payload = debug_response.json()
        assert payload["selected"]
        assert payload["selected"][0]["id"] == semantic_id
        assert observed["model_id_override"] == "stub:embedding-model"
        assert observed["user_id"]
        assert observed["character_id"] == str(char_id)

    def test_character_exemplar_debug_selection_embedding_failure_falls_back(
        self,
        test_client,
        auth_headers,
        monkeypatch,
    ):
        char_id = self._create_character(test_client, auth_headers, "Exemplar API Character Embedding Fallback")

        create_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "press response that should still be selectable",
                "labels": {
                    "emotion": "neutral",
                    "scenario": "press_challenge",
                    "rhetorical": ["opener"],
                },
            },
            headers=auth_headers,
        )
        assert create_response.status_code == 201

        def _raise_embedding_error(user_turn: str, candidates: list[dict], **kwargs):  # noqa: ARG001
            raise RuntimeError("embedding backend unavailable")

        monkeypatch.setattr(
            characters_endpoint_module,
            "score_exemplars_with_embeddings",
            _raise_embedding_error,
        )

        debug_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars/select/debug",
            json={
                "user_turn": "How should I answer this press question?",
                "selection_config": {
                    "budget_tokens": 80,
                    "max_exemplar_tokens": 60,
                    "mmr_lambda": 0.7,
                    "use_embedding_scores": True,
                },
            },
            headers=auth_headers,
        )

        assert debug_response.status_code == 200
        payload = debug_response.json()
        assert payload["selected"]

    def test_character_exemplar_search_can_use_hybrid_embedding_rerank(
        self,
        test_client,
        auth_headers,
        monkeypatch,
    ):
        char_id = self._create_character(test_client, auth_headers, "Exemplar API Character Hybrid Search")

        lexical_create = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "board meeting strategy budget plan",
                "labels": {
                    "emotion": "neutral",
                    "scenario": "boardroom",
                    "rhetorical": ["opener"],
                },
            },
            headers=auth_headers,
        )
        assert lexical_create.status_code == 201
        lexical_id = lexical_create.json()["id"]

        semantic_create = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "generic reply with sparse lexical overlap",
                "labels": {
                    "emotion": "neutral",
                    "scenario": "boardroom",
                    "rhetorical": ["emphasis"],
                },
            },
            headers=auth_headers,
        )
        assert semantic_create.status_code == 201
        semantic_id = semantic_create.json()["id"]

        observed: dict[str, str] = {}

        def _fake_embedding_scores(user_turn: str, candidates: list[dict], **kwargs):
            assert user_turn
            observed["model_id_override"] = str(kwargs.get("model_id_override"))
            observed["user_id"] = str(kwargs.get("user_id"))
            observed["character_id"] = str(kwargs.get("character_id"))
            candidate_ids = {str(item.get("id")) for item in candidates}
            assert lexical_id in candidate_ids
            assert semantic_id in candidate_ids
            return {
                lexical_id: 0.0,
                semantic_id: 1.0,
            }

        monkeypatch.setattr(
            characters_endpoint_module,
            "score_exemplars_with_embeddings",
            _fake_embedding_scores,
        )

        search_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars/search",
            json={
                "query": "board meeting strategy budget plan",
                "use_embedding_scores": True,
                "embedding_model_id": "stub:hybrid-search",
                "limit": 10,
                "offset": 0,
            },
            headers=auth_headers,
        )

        assert search_response.status_code == 200
        payload = search_response.json()
        assert payload["items"]
        assert payload["items"][0]["id"] == semantic_id
        assert observed["model_id_override"] == "stub:hybrid-search"
        assert observed["user_id"]
        assert observed["character_id"] == str(char_id)

    def test_character_exemplar_crud_triggers_embedding_sync_hooks(
        self,
        test_client,
        auth_headers,
        monkeypatch,
    ):
        char_id = self._create_character(test_client, auth_headers, "Exemplar API Character Embedding Sync")

        observed: dict[str, list] = {"upserts": [], "deletes": []}

        def _fake_upsert(**kwargs):
            observed["upserts"].append(kwargs)
            return len(kwargs.get("exemplars") or [])

        def _fake_delete(**kwargs):
            observed["deletes"].append(kwargs)
            return len(kwargs.get("exemplar_ids") or [])

        monkeypatch.setattr(
            characters_endpoint_module,
            "upsert_character_exemplar_embeddings",
            _fake_upsert,
        )
        monkeypatch.setattr(
            characters_endpoint_module,
            "delete_character_exemplar_embeddings",
            _fake_delete,
        )

        create_response = test_client.post(
            f"/api/v1/characters/{char_id}/exemplars",
            json={
                "text": "Embedding sync create exemplar.",
                "labels": {"emotion": "neutral", "scenario": "press_challenge", "rhetorical": ["opener"]},
            },
            headers=auth_headers,
        )
        assert create_response.status_code == 201
        exemplar_id = str(create_response.json()["id"])

        update_response = test_client.put(
            f"/api/v1/characters/{char_id}/exemplars/{exemplar_id}",
            json={
                "text": "Embedding sync update exemplar.",
                "labels": {"emotion": "happy", "scenario": "fan_banter", "rhetorical": ["emphasis"]},
            },
            headers=auth_headers,
        )
        assert update_response.status_code == 200

        delete_response = test_client.delete(
            f"/api/v1/characters/{char_id}/exemplars/{exemplar_id}",
            headers=auth_headers,
        )
        assert delete_response.status_code == 200

        assert len(observed["upserts"]) >= 2
        assert len(observed["deletes"]) == 1
        assert observed["deletes"][0]["character_id"] == char_id
        assert observed["deletes"][0]["exemplar_ids"] == [exemplar_id]
