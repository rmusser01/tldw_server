from __future__ import annotations

from configparser import ConfigParser

import pytest

from tldw_Server_API.app.core.config_sections import load_config_sections
from tldw_Server_API.app.core.config_sections.database import load_database_config
from tldw_Server_API.app.core.config_sections.embeddings import load_embeddings_config
from tldw_Server_API.app.core.config_sections.logging import load_logging_config
from tldw_Server_API.app.core.config_sections.server import load_server_config


pytestmark = pytest.mark.unit


def _build_parser_with_required_sections() -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("AuthNZ")
    parser.add_section("RAG")
    parser.add_section("TTS-Settings")
    parser.add_section("API")
    parser.add_section("STT-Settings")
    parser.add_section("Database")
    parser.add_section("Embeddings")
    parser.add_section("Logging")
    parser.add_section("Server")
    return parser


def test_load_config_sections_exposes_new_typed_sections() -> None:
    parser = _build_parser_with_required_sections()
    parser.set("Database", "sqlite_path", "Databases/custom.db")
    parser.set("Embeddings", "embedding_model", "test-embedding-model")
    parser.set("Logging", "log_level", "DEBUG")
    parser.set("Server", "disable_cors", "true")

    sections = load_config_sections(parser)

    assert sections.database.sqlite_path == "Databases/custom.db"
    assert sections.embeddings.embedding_model == "test-embedding-model"
    assert sections.logging.log_level == "DEBUG"
    assert sections.server.disable_cors is True


def test_database_section_loader_prefers_env_and_parses_types() -> None:
    parser = ConfigParser()
    parser.add_section("Database")
    parser.set("Database", "type", "sqlite")
    parser.set("Database", "sqlite_wal_mode", "false")
    parser.set("Database", "pg_port", "5432")
    parser.set("Database", "pg_pool_timeout", "30.0")

    cfg = load_database_config(
        parser,
        env={
            "DB_TYPE": "postgresql",
            "DB_SQLITE_WAL_MODE": "yes",
            "DB_PG_PORT": "6543",
            "DB_PG_POOL_TIMEOUT": "45.5",
        },
    )

    assert cfg.type == "postgresql"
    assert cfg.sqlite_wal_mode is True
    assert cfg.pg_port == 6543
    assert cfg.pg_pool_timeout == 45.5


def test_embeddings_section_loader_prefers_env_and_parses_bool() -> None:
    parser = ConfigParser()
    parser.add_section("Embeddings")
    parser.set("Embeddings", "embedding_provider", "config-provider")
    parser.set("Embeddings", "enable_contextual_chunking", "false")

    cfg = load_embeddings_config(
        parser,
        env={
            "EMBEDDING_PROVIDER": "env-provider",
            "EMBEDDING_MODEL": "env-model",
            "EMBEDDING_ENABLE_CONTEXTUAL_CHUNKING": "true",
        },
    )

    assert cfg.embedding_provider == "env-provider"
    assert cfg.embedding_model == "env-model"
    assert cfg.enable_contextual_chunking is True


def test_logging_and_server_section_loaders_honor_env_overrides() -> None:
    parser = ConfigParser()
    parser.add_section("Logging")
    parser.add_section("Server")
    parser.set("Logging", "log_level", "INFO")
    parser.set("Server", "disable_cors", "false")
    parser.set("Server", "cors_allow_credentials", "false")

    logging_cfg = load_logging_config(
        parser,
        env={"LOG_LEVEL": "WARNING"},
    )
    server_cfg = load_server_config(
        parser,
        env={
            "DISABLE_CORS": "1",
            "CORS_ALLOW_CREDENTIALS": "true",
        },
    )

    assert logging_cfg.log_level == "WARNING"
    assert server_cfg.disable_cors is True
    assert server_cfg.cors_allow_credentials is True
