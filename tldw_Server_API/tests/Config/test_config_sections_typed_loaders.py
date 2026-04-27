from __future__ import annotations

from configparser import ConfigParser

import pytest

import tldw_Server_API.app.core.config_sections as config_sections_mod
from tldw_Server_API.app.core.config_sections import load_config_sections
from tldw_Server_API.app.core.config_sections.audio import load_audio_config
from tldw_Server_API.app.core.config_sections.auth import load_auth_config
from tldw_Server_API.app.core.config_sections.chunking import load_chunking_config
from tldw_Server_API.app.core.config_sections.database import load_database_config
from tldw_Server_API.app.core.config_sections.embeddings import load_embeddings_config
from tldw_Server_API.app.core.config_sections.jobs import load_jobs_config
from tldw_Server_API.app.core.config_sections.logging import load_logging_config
from tldw_Server_API.app.core.config_sections.moderation import load_moderation_config
from tldw_Server_API.app.core.config_sections.providers import load_providers_config
from tldw_Server_API.app.core.config_sections.rag import load_rag_config
from tldw_Server_API.app.core.config_sections.server import load_server_config
from tldw_Server_API.app.core.config_sections.stt import load_stt_config

pytestmark = pytest.mark.unit


def _build_parser_with_required_sections() -> ConfigParser:
    parser = ConfigParser()
    parser.add_section("AuthNZ")
    parser.add_section("RAG")
    parser.add_section("TTS-Settings")
    parser.add_section("API")
    parser.add_section("STT-Settings")
    parser.add_section("Chunking")
    parser.add_section("Chat-Module")
    parser.add_section("Database")
    parser.add_section("Embeddings")
    parser.add_section("Jobs")
    parser.add_section("Logging")
    parser.add_section("Moderation")
    parser.add_section("Server")
    return parser


def test_load_config_sections_exposes_new_typed_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_key in (
        "AUTH_MODE",
        "APP_MODE",
        "SINGLE_USER_FIXED_ID",
        "RAG_VECTOR_STORE_TYPE",
        "RAG_DEFAULT_LLM_PROVIDER",
        "RAG_DEFAULT_LLM_MODEL",
        "TTS_DEFAULT_PROVIDER",
        "TTS_DEFAULT_VOICE",
        "LOCAL_TTS_DEVICE",
        "DEFAULT_API",
        "DEFAULT_PROVIDER",
        "CHAT_IMAGE_MAX_MB",
        "CHAT_STREAM_CHANNEL_MAXSIZE",
        "CHAT_STREAM_INCLUDE_METADATA",
        "CHAT_SAVE_DEFAULT",
        "DEFAULT_CHAT_SAVE",
        "ALLOW_AUTOSWITCH_TO_OPENAI",
        "CHAT_RUN_FIRST_ROLLOUT_MODE",
        "CHAT_RUN_FIRST_PROVIDER_ALLOWLIST",
        "CHAT_RUN_FIRST_PRESENTATION_VARIANT",
        "DB_SQLITE_PATH",
        "EMBEDDING_MODEL",
        "LOG_LEVEL",
        "MODERATION_USER_OVERRIDES_FILE",
        "MODERATION_CATEGORIES_ENABLED",
        "MODERATION_PII_ENABLED",
        "DISABLE_CORS",
        "STT_WS_CONTROL_V2_ENABLED",
        "STT_REDACT_CATEGORIES",
    ):
        monkeypatch.delenv(env_key, raising=False)

    parser = _build_parser_with_required_sections()
    parser.set("AuthNZ", "auth_mode", "multi_user")
    parser.set("AuthNZ", "single_user_fixed_id", "7")
    parser.set("RAG", "vector_store_type", "faiss")
    parser.set("RAG", "rag_default_llm_provider", "anthropic")
    parser.set("RAG", "rag_default_llm_model", "claude-3-5-sonnet")
    parser.set("TTS-Settings", "default_tts_provider", "kokoro")
    parser.set("TTS-Settings", "default_tts_voice", "alloy")
    parser.set("TTS-Settings", "local_tts_device", "cuda")
    parser.set("API", "default_api", "openai")
    parser.set("API", "default_provider", "openrouter")
    parser.set("Chat-Module", "enable_provider_fallback", "true")
    parser.set("Chat-Module", "max_base64_image_size_mb", "4")
    parser.set("Chat-Module", "max_text_length_per_message", "222222")
    parser.set("Chat-Module", "max_messages_per_request", "25")
    parser.set("Chat-Module", "max_images_per_request", "4")
    parser.set("Chat-Module", "chat_stream_channel_maxsize", "24")
    parser.set("Chat-Module", "chat_stream_include_metadata", "false")
    parser.set("Chat-Module", "chat_save_default", "true")
    parser.set("Chat-Module", "allow_autoswitch_to_openai", "true")
    parser.set("Chat-Module", "rate_limit_per_minute", "120")
    parser.set("Chat-Module", "rate_limit_per_conversation_per_minute", "30")
    parser.set("Chat-Module", "rate_limit_tokens_per_minute", "120000")
    parser.set("Chat-Module", "run_first_rollout_mode", "gated")
    parser.set(
        "Chat-Module",
        "run_first_provider_allowlist",
        "openai:gpt-4o-mini, anthropic:claude-3-7-sonnet",
    )
    parser.set("Chat-Module", "run_first_presentation_variant", "chat_phase2c_v1")
    parser.set("Chunking", "chunking_method", "sentences")
    parser.set("Chunking", "chunk_max_size", "512")
    parser.set("Chunking", "chunk_overlap", "128")
    parser.set("Chunking", "adaptive_chunking", "true")
    parser.set("Database", "sqlite_path", "Databases/custom.db")
    parser.set("Embeddings", "embedding_model", "test-embedding-model")
    parser.set("Jobs", "prune_enforce", "true")
    parser.set("Jobs", "prune_interval_sec", "7200")
    parser.set("Jobs", "prune_domain", "chatbooks, embeddings")
    parser.set("Jobs", "retention_days_completed", "14")
    parser.set("Logging", "log_level", "DEBUG")
    parser.set("Moderation", "enabled", "true")
    parser.set("Moderation", "output_action", "warn")
    parser.set("Moderation", "user_overrides_file", "Config_Files/mod-user-overrides.json")
    parser.set("Moderation", "categories_enabled", "pii, safety")
    parser.set("Moderation", "pii_enabled", "true")
    parser.set("Server", "disable_cors", "true")
    parser.set("STT-Settings", "ws_control_v2_enabled", "true")
    parser.set("STT-Settings", "redact_categories", '["email", "phone"]')

    sections = load_config_sections(parser)

    assert sections.auth.mode == "multi_user"
    assert sections.auth.single_user_fixed_id == 7
    assert sections.rag.vector_store_type == "faiss"
    assert sections.rag.default_llm_provider == "anthropic"
    assert sections.audio.default_tts_provider == "kokoro"
    assert sections.audio.local_tts_device == "cuda"
    assert sections.providers.default_api == "openai"
    assert sections.providers.default_provider == "openrouter"
    assert sections.chat.enable_provider_fallback is True
    assert sections.chat.max_base64_image_size_mb == 4
    assert sections.chat.max_text_length_per_message == 222222
    assert sections.chat.max_messages_per_request == 25
    assert sections.chat.max_images_per_request == 4
    assert sections.chat.chat_stream_channel_maxsize == 24
    assert sections.chat.chat_stream_include_metadata is False
    assert sections.chat.chat_save_default is True
    assert sections.chat.allow_autoswitch_to_openai is True
    assert sections.chat.rate_limit_per_minute == 120
    assert sections.chat.rate_limit_per_conversation_per_minute == 30
    assert sections.chat.rate_limit_tokens_per_minute == 120000
    assert sections.chat.run_first_rollout_mode == "gated"
    assert sections.chat.run_first_provider_allowlist == [
        "openai:gpt-4o-mini",
        "anthropic:claude-3-7-sonnet",
    ]
    assert sections.chat.run_first_presentation_variant == "chat_phase2c_v1"
    assert sections.chunking.method == "sentences"
    assert sections.chunking.max_size == 512
    assert sections.chunking.overlap == 128
    assert sections.chunking.adaptive is True
    assert sections.database.sqlite_path == "Databases/custom.db"
    assert sections.embeddings.embedding_model == "test-embedding-model"
    assert sections.jobs.prune_enforce is True
    assert sections.jobs.prune_interval_sec == 7200
    assert sections.jobs.prune_domain == ["chatbooks", "embeddings"]
    assert sections.jobs.retention_days_completed == 14
    assert sections.logging.log_level == "DEBUG"
    assert sections.moderation.enabled is True
    assert sections.moderation.output_action == "warn"
    assert sections.moderation.user_overrides_file == "Config_Files/mod-user-overrides.json"
    assert sections.moderation.categories_enabled == ["pii", "safety"]
    assert sections.moderation.pii_enabled is True
    assert sections.server.disable_cors is True
    assert sections.stt.ws_control_v2_enabled is True
    assert sections.stt.redact_categories == ["email", "phone"]


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


def test_numeric_loaders_fall_back_for_invalid_non_security_values() -> None:
    parser = ConfigParser()
    parser.add_section("Database")
    parser.add_section("Embeddings")
    parser.add_section("Logging")

    database_cfg = load_database_config(
        parser,
        env={
            "DB_PG_PORT": "not-a-port",
            "DB_PG_POOL_SIZE": "not-a-size",
            "DB_PG_MAX_OVERFLOW": "not-overflow",
            "DB_PG_POOL_TIMEOUT": "not-a-timeout",
        },
    )
    embeddings_cfg = load_embeddings_config(
        parser,
        env={
            "EMBEDDING_CHUNK_SIZE": "not-a-size",
            "EMBEDDING_OVERLAP": "not-overlap",
        },
    )
    logging_cfg = load_logging_config(
        parser,
        env={
            "LOG_BACKUP_COUNT": "not-a-count",
            "SYSTEM_LOG_FILE_MAX_ENTRIES": "not-a-limit",
        },
    )

    assert database_cfg.pg_port == 5432
    assert database_cfg.pg_pool_size == 20
    assert database_cfg.pg_max_overflow == 40
    assert database_cfg.pg_pool_timeout == 30.0
    assert embeddings_cfg.chunk_size == 400
    assert embeddings_cfg.overlap == 200
    assert logging_cfg.backup_count == 5
    assert logging_cfg.system_log_file_max_entries == 5000


def test_chunking_section_loader_parses_global_defaults() -> None:
    parser = ConfigParser()
    parser.add_section("Chunking")
    parser.set("Chunking", "chunking_method", "semantic")
    parser.set("Chunking", "chunk_max_size", "1024")
    parser.set("Chunking", "chunk_overlap", "256")
    parser.set("Chunking", "adaptive_chunking", "true")
    parser.set("Chunking", "chunking_multi_level", "yes")
    parser.set("Chunking", "chunk_language", "fr")

    cfg = load_chunking_config(parser)

    assert cfg.method == "semantic"
    assert cfg.max_size == 1024
    assert cfg.overlap == 256
    assert cfg.adaptive is True
    assert cfg.multi_level is True
    assert cfg.language == "fr"


def test_chunking_section_loader_env_overrides_parser() -> None:
    parser = ConfigParser()
    parser.add_section("Chunking")
    parser.set("Chunking", "chunking_method", "semantic")
    parser.set("Chunking", "chunk_max_size", "1024")
    parser.set("Chunking", "chunk_overlap", "256")
    parser.set("Chunking", "adaptive_chunking", "false")
    parser.set("Chunking", "chunking_multi_level", "false")
    parser.set("Chunking", "chunk_language", "fr")

    cfg = load_chunking_config(
        parser,
        env={
            "CHUNKING_METHOD": "sentences",
            "CHUNKING_MAX_SIZE": "2048",
            "CHUNKING_OVERLAP": "512",
            "CHUNKING_ADAPTIVE": "on",
            "CHUNKING_MULTI_LEVEL": "yes",
            "CHUNKING_LANGUAGE": "es",
        },
    )

    assert cfg.method == "sentences"
    assert cfg.max_size == 2048
    assert cfg.overlap == 512
    assert cfg.adaptive is True
    assert cfg.multi_level is True
    assert cfg.language == "es"


def test_chat_section_loader_honors_env_and_parses_scalar_limits() -> None:
    parser = ConfigParser()
    parser.add_section("Chat-Module")
    parser.set("Chat-Module", "enable_provider_fallback", "false")
    parser.set("Chat-Module", "max_base64_image_size_mb", "3")
    parser.set("Chat-Module", "max_text_length_per_message", "500000")
    parser.set("Chat-Module", "max_messages_per_request", "9")
    parser.set("Chat-Module", "max_images_per_request", "2")
    parser.set("Chat-Module", "chat_stream_channel_maxsize", "100")
    parser.set("Chat-Module", "chat_stream_include_metadata", "true")
    parser.set("Chat-Module", "default_save_to_db", "true")
    parser.set("Chat-Module", "allow_autoswitch_to_openai", "false")
    parser.set("Chat-Module", "rate_limit_per_minute", "60")
    parser.set("Chat-Module", "rate_limit_per_conversation_per_minute", "20")
    parser.set("Chat-Module", "rate_limit_tokens_per_minute", "100000")
    parser.set("Chat-Module", "run_first_rollout_mode", "default_on")
    parser.set(
        "Chat-Module",
        "run_first_provider_allowlist",
        "openai:gpt-4o-mini,google:gemini-2.5-flash",
    )
    parser.set("Chat-Module", "run_first_presentation_variant", "chat_phase2b_v1")

    load_chat_config = getattr(config_sections_mod, "load_chat_config", None)
    assert callable(load_chat_config)

    cfg = load_chat_config(
        parser,
        env={
            "CHAT_IMAGE_MAX_MB": "8",
            "CHAT_STREAM_CHANNEL_MAXSIZE": "42",
            "CHAT_STREAM_INCLUDE_METADATA": "false",
            "DEFAULT_CHAT_SAVE": "false",
            "ALLOW_AUTOSWITCH_TO_OPENAI": "true",
            "CHAT_RUN_FIRST_ROLLOUT_MODE": "gated",
            "CHAT_RUN_FIRST_PROVIDER_ALLOWLIST": "anthropic:claude-3-7-sonnet, openai:gpt-4o",
            "CHAT_RUN_FIRST_PRESENTATION_VARIANT": "chat_phase2d_v1",
        },
    )

    assert cfg.enable_provider_fallback is False
    assert cfg.max_base64_image_size_mb == 8
    assert cfg.max_text_length_per_message == 500000
    assert cfg.max_messages_per_request == 9
    assert cfg.max_images_per_request == 2
    assert cfg.chat_stream_channel_maxsize == 42
    assert cfg.chat_stream_include_metadata is False
    assert cfg.chat_save_default is False
    assert cfg.allow_autoswitch_to_openai is True
    assert cfg.rate_limit_per_minute == 60
    assert cfg.rate_limit_per_conversation_per_minute == 20
    assert cfg.rate_limit_tokens_per_minute == 100000
    assert cfg.run_first_rollout_mode == "gated"
    assert cfg.run_first_provider_allowlist == [
        "anthropic:claude-3-7-sonnet",
        "openai:gpt-4o",
    ]
    assert cfg.run_first_presentation_variant == "chat_phase2d_v1"


def test_jobs_section_loader_honors_env_and_parses_retention_windows() -> None:
    parser = ConfigParser()
    parser.add_section("Jobs")
    parser.set("Jobs", "prune_enforce", "false")
    parser.set("Jobs", "prune_interval_sec", "3600")
    parser.set("Jobs", "prune_domain", "config-domain")
    parser.set("Jobs", "prune_queue", "default, low")
    parser.set("Jobs", "retention_days_terminal", "45")
    parser.set("Jobs", "retention_days_nonterminal", "5")

    cfg = load_jobs_config(
        parser,
        env={
            "JOBS_PRUNE_ENFORCE": "true",
            "JOBS_PRUNE_DRY_RUN": "yes",
            "JOBS_PRUNE_DOMAIN": "chatbooks, embeddings",
            "JOBS_PRUNE_JOB_TYPE": "export, import",
            "JOBS_RETENTION_DAYS_COMPLETED": "10",
            "JOBS_RETENTION_DAYS_FAILED": "20",
            "JOBS_RETENTION_DAYS_CANCELLED": "30",
            "JOBS_RETENTION_DAYS_QUARANTINED": "40",
        },
    )

    assert cfg.prune_enforce is True
    assert cfg.prune_interval_sec == 3600
    assert cfg.prune_dry_run is True
    assert cfg.prune_domain == ["chatbooks", "embeddings"]
    assert cfg.prune_queue == ["default", "low"]
    assert cfg.prune_job_type == ["export", "import"]
    assert cfg.retention_days_terminal == 45
    assert cfg.retention_days_nonterminal == 5
    assert cfg.retention_days_completed == 10
    assert cfg.retention_days_failed == 20
    assert cfg.retention_days_cancelled == 30
    assert cfg.retention_days_quarantined == 40


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


def test_embeddings_section_loader_uses_canonical_api_url_env() -> None:
    parser = ConfigParser()
    parser.add_section("Embeddings")
    parser.set("Embeddings", "embedding_api_url", "http://config.example/v1/embeddings")

    cfg = load_embeddings_config(
        parser,
        env={
            "EMBEDDING_API_URL": " http://env.example/v1/embeddings ",
            "EMBEDDING_EMBEDDING_API_URL": "http://legacy.example/v1/embeddings",
            "EMBEDDING_ENABLE_CONTEXTUAL_CHUNKING": "on",
        },
    )

    assert cfg.embedding_api_url == "http://env.example/v1/embeddings"
    assert cfg.enable_contextual_chunking is True


def test_embeddings_section_loader_preserves_legacy_double_prefixed_api_url() -> None:
    parser = ConfigParser()
    parser.add_section("Embeddings")
    parser.set("Embeddings", "embedding_api_url", "http://config.example/v1/embeddings")

    cfg = load_embeddings_config(
        parser,
        env={"EMBEDDING_EMBEDDING_API_URL": "http://legacy.example/v1/embeddings"},
    )

    assert cfg.embedding_api_url == "http://legacy.example/v1/embeddings"


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


def test_logging_section_loader_uses_runtime_system_log_env_keys() -> None:
    parser = ConfigParser()
    parser.add_section("Logging")
    parser.set("Logging", "system_log_file_path", "Databases/config-system.jsonl")
    parser.set("Logging", "system_log_file_max_entries", "100")

    cfg = load_logging_config(
        parser,
        env={
            "SYSTEM_LOG_FILE_PATH": "Databases/env-system.jsonl",
            "SYSTEM_LOG_FILE_MAX_ENTRIES": "777",
            "LOG_SYSTEM_LOG_FILE_PATH": "Databases/log-prefixed-system.jsonl",
            "LOG_SYSTEM_LOG_FILE_MAX_ENTRIES": "333",
        },
    )

    assert cfg.system_log_file_path == "Databases/env-system.jsonl"
    assert cfg.system_log_file_max_entries == 777


def test_server_section_loader_accepts_existing_truthy_tokens() -> None:
    parser = ConfigParser()
    parser.add_section("Server")

    cfg = load_server_config(
        parser,
        env={
            "DISABLE_CORS": "on",
            "CORS_ALLOW_CREDENTIALS": "y",
        },
    )

    assert cfg.disable_cors is True
    assert cfg.cors_allow_credentials is True


def test_auth_audio_rag_and_provider_loaders_honor_env_and_fallbacks() -> None:
    parser = ConfigParser()
    parser.add_section("AuthNZ")
    parser.add_section("TTS-Settings")
    parser.add_section("RAG")
    parser.add_section("API")
    parser.set("AuthNZ", "auth_mode", "single_user")
    parser.set("AuthNZ", "single_user_fixed_id", "44")
    parser.set("TTS-Settings", "default_tts_provider", "config-tts")
    parser.set("TTS-Settings", "local_tts_device", "cpu")
    parser.set("RAG", "vector_store_type", "chromadb")
    parser.set("RAG", "rag_default_llm_provider", "config-provider")
    parser.set("API", "default_api", "config-api")
    parser.set("API", "default_provider", "config-provider")

    auth_cfg = load_auth_config(
        parser,
        env={"AUTH_MODE": "multi_user", "SINGLE_USER_FIXED_ID": "99"},
    )
    audio_cfg = load_audio_config(
        parser,
        env={"TTS_DEFAULT_PROVIDER": "env-tts", "LOCAL_TTS_DEVICE": "mps"},
    )
    rag_cfg = load_rag_config(
        parser,
        env={"RAG_VECTOR_STORE_TYPE": "faiss", "RAG_DEFAULT_LLM_PROVIDER": "env-provider"},
    )
    providers_cfg = load_providers_config(
        parser,
        env={"DEFAULT_API": "env-api", "DEFAULT_PROVIDER": "env-provider"},
    )

    assert auth_cfg.mode == "multi_user"
    assert auth_cfg.single_user_fixed_id == 99
    assert audio_cfg.default_tts_provider == "env-tts"
    assert audio_cfg.local_tts_device == "mps"
    assert rag_cfg.vector_store_type == "faiss"
    assert rag_cfg.default_llm_provider == "env-provider"
    assert providers_cfg.default_api == "env-api"
    assert providers_cfg.default_provider == "env-provider"


def test_stt_section_loader_parses_lists_and_invalid_numbers_conservatively() -> None:
    parser = ConfigParser()
    parser.add_section("STT-Settings")
    parser.set("STT-Settings", "paused_audio_queue_cap_seconds", "2.5")
    parser.set("STT-Settings", "redact_categories", "email, phone,EMAIL")

    cfg = load_stt_config(
        parser,
        env={
            "STT_WS_CONTROL_V2_ENABLED": "true",
            "STT_PAUSED_AUDIO_QUEUE_CAP_SECONDS": "-1",
            "STT_OVERFLOW_WARNING_INTERVAL_SECONDS": "bad-value",
            "STT_TRANSCRIPT_DIAGNOSTICS_ENABLED": "yes",
            "STT_DELETE_AUDIO_AFTER_SUCCESS": "0",
            "STT_AUDIO_RETENTION_HOURS": "6.5",
            "STT_REDACT_PII": "on",
            "STT_ALLOW_UNREDACTED_PARTIALS": "off",
            "STT_REDACT_CATEGORIES": '["EMAIL", "phone", "email"]',
        },
    )

    assert cfg.ws_control_v2_enabled is True
    assert cfg.paused_audio_queue_cap_seconds == 2.0
    assert cfg.overflow_warning_interval_seconds == 5.0
    assert cfg.transcript_diagnostics_enabled is True
    assert cfg.delete_audio_after_success is False
    assert cfg.audio_retention_hours == 6.5
    assert cfg.redact_pii is True
    assert cfg.allow_unredacted_partials is False
    assert cfg.redact_categories == ["email", "phone"]


def test_moderation_section_loader_honors_env_backed_fields_and_normalizes_lists() -> None:
    parser = ConfigParser()
    parser.add_section("Moderation")
    parser.set("Moderation", "enabled", "true")
    parser.set("Moderation", "input_enabled", "false")
    parser.set("Moderation", "output_action", "warn")
    parser.set("Moderation", "categories_enabled", "pii, violence,PII")
    parser.set("Moderation", "max_scan_chars", "12000")
    parser.set("Moderation", "blocklist_write_debounce_ms", "15")

    cfg = load_moderation_config(
        parser,
        env={
            "MODERATION_USER_OVERRIDES_FILE": "/tmp/mod-overrides.json",
            "MODERATION_MAX_SCAN_CHARS": "64000",
            "MODERATION_MAX_REPLACEMENTS_PER_PATTERN": "250",
            "MODERATION_MATCH_WINDOW_CHARS": "8192",
            "MODERATION_MAX_FALLBACK_SCAN_CHARS": "900000",
            "MODERATION_BLOCKLIST_WRITE_DEBOUNCE_MS": "25",
            "MODERATION_CATEGORIES_ENABLED": " safety,PII,safety ",
            "MODERATION_PII_ENABLED": "yes",
        },
    )

    assert cfg.enabled is True
    assert cfg.input_enabled is False
    assert cfg.output_action == "warn"
    assert cfg.user_overrides_file == "/tmp/mod-overrides.json"
    assert cfg.max_scan_chars == 64000
    assert cfg.max_replacements_per_pattern == 250
    assert cfg.match_window_chars == 8192
    assert cfg.max_fallback_scan_chars == 900000
    assert cfg.blocklist_write_debounce_ms == 25
    assert cfg.categories_enabled == ["safety", "pii"]
    assert cfg.pii_enabled is True


def test_moderation_section_loader_rejects_invalid_security_values() -> None:
    parser = ConfigParser()
    parser.add_section("Moderation")
    parser.set("Moderation", "enabled", "maybe")

    with pytest.raises(ValueError, match="enabled.*maybe.*true"):
        load_moderation_config(parser)

    parser.set("Moderation", "enabled", "true")
    with pytest.raises(ValueError, match="max_scan_chars.*not-an-int"):
        load_moderation_config(parser, env={"MODERATION_MAX_SCAN_CHARS": "not-an-int"})
