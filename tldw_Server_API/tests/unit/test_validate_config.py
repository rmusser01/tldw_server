"""Tests for config.validate_config() startup validation."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock


class TestValidateConfig:
    def test_detects_fixme_placeholder(self):
        from tldw_Server_API.app.core.config import validate_config, loaded_config_data

        fake_data = {"some_key": "FIXME", "good_key": "real_value"}
        with patch.object(loaded_config_data, "__iter__", return_value=iter(fake_data)):
            with patch.object(loaded_config_data, "__getitem__", side_effect=fake_data.__getitem__):
                with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
                    warnings = validate_config()
        assert any("FIXME" in w for w in warnings)

    def test_detects_todo_placeholder(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {"tts_voice": "TODO"}
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert any("TODO" in w for w in warnings)

    def test_passes_clean_config(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {
            "embedding_api_url": "http://localhost:8080/v1/embeddings",
            "pg_connection_string": "",
            "swarmui_base_url": "http://127.0.0.1:7801",
            "normal_key": "normal_value",
        }
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert len(warnings) == 0

    def test_detects_bad_url_scheme(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {"embedding_api_url": "ftp://bad-scheme:8080"}
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert any("unexpected URL scheme" in w for w in warnings)

    def test_empty_url_is_ok(self):
        from tldw_Server_API.app.core.config import validate_config

        fake_data = {"embedding_api_url": "", "pg_connection_string": ""}
        with patch("tldw_Server_API.app.core.config.loaded_config_data", new=fake_data):
            warnings = validate_config()
        assert len(warnings) == 0
