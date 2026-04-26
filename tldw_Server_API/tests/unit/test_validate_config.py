"""Tests for config.validate_config() startup validation."""

from __future__ import annotations

from unittest.mock import patch


class TestValidateConfig:
    def test_detects_fixme_placeholder(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {"API": {"openai_api_key": "FIXME"}, "good_key": "real_value"}
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert any("API.openai_api_key" in w and "FIXME" in w for w in warnings)

    def test_detects_todo_placeholder(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {"TTS-Settings": {"tts_voice": "TODO"}}
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert any("TODO" in w for w in warnings)

    def test_passes_clean_config(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {
            "embedding_config": {
                "embedding_api_url": "http://localhost:8080/v1/embeddings",
            },
            "Database": {"pg_connection_string": ""},
            "Image-Generation": {"swarmui_base_url": "http://127.0.0.1:7801"},
            "normal_key": "normal_value",
        }
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert len(warnings) == 0

    def test_detects_bad_url_scheme(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {
            "embedding_config": {
                "embedding_api_url": "ftp://user:secret@bad-scheme:8080",
            },
        }
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert any("embedding_config.embedding_api_url" in w and "unexpected URL scheme: ftp" in w for w in warnings)
        assert all("secret" not in w for w in warnings)

    def test_accepts_postgres_scheme(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {
            "Database": {
                "pg_connection_string": "postgres://user:secret@example/db",
            },
        }
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert len(warnings) == 0

    def test_empty_url_is_ok(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {
            "embedding_config": {"embedding_api_url": ""},
            "Database": {"pg_connection_string": ""},
        }
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert len(warnings) == 0
