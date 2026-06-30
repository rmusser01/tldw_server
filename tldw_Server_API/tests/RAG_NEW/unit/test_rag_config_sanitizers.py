"""Sanitizer coverage for RAG config fallback logs."""

import pytest

from tldw_Server_API.app.core.RAG.rag_service import config as rag_config_module
from tldw_Server_API.app.core.RAG.rag_service.config import RAGConfig


pytestmark = pytest.mark.unit


_RAG_ENV_OVERRIDES = (
    "RAG_FTS_TOP_K",
    "RAG_VECTOR_TOP_K",
    "RAG_ENABLE_RERANKING",
    "RAG_ENABLE_CACHE",
    "RAG_DEFAULT_MODEL",
    "RAG_CHROMA_PERSIST_DIR",
    "RAG_NLI_MODEL",
)


class _LoggerStub:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.debugs: list[str] = []
        self.infos: list[str] = []

    def error(self, message, *args, **kwargs):
        _ = (args, kwargs)
        self.errors.append(str(message))

    def warning(self, message, *args, **kwargs):
        _ = (args, kwargs)
        self.warnings.append(str(message))

    def debug(self, message, *args, **kwargs):
        _ = (args, kwargs)
        self.debugs.append(str(message))

    def info(self, message, *args, **kwargs):
        _ = (args, kwargs)
        self.infos.append(str(message))


def _clear_rag_env(monkeypatch):
    for env_var in _RAG_ENV_OVERRIDES:
        monkeypatch.delenv(env_var, raising=False)


def test_config_file_load_fallback_logs_sanitized_path_and_exception(
    monkeypatch,
    tmp_path,
):
    _clear_rag_env(monkeypatch)
    logger_stub = _LoggerStub()
    secret_path_token = "secret-config-path-token"
    missing_config = tmp_path / f"{secret_path_token}.toml"

    monkeypatch.setattr(rag_config_module, "logger", logger_stub)

    config = RAGConfig.from_toml(missing_config)

    assert config.batch_size == 32
    assert logger_stub.errors == ["Error loading RAG config: FileNotFoundError"]
    assert logger_stub.warnings == ["Using default configuration"]
    fallback_logs = "\n".join(logger_stub.errors + logger_stub.warnings)
    assert str(missing_config) not in fallback_logs
    assert secret_path_token not in fallback_logs
    assert "No such file or directory" not in fallback_logs


def test_env_override_fallback_logs_sanitized_env_and_exception(monkeypatch):
    _clear_rag_env(monkeypatch)
    logger_stub = _LoggerStub()
    secret_value_token = "secret-env-value-token"

    monkeypatch.setattr(rag_config_module, "logger", logger_stub)
    monkeypatch.setenv("RAG_FTS_TOP_K", f"not-an-int-{secret_value_token}")

    config = RAGConfig()

    assert config.retriever.fts_top_k == 10
    assert logger_stub.warnings == ["Failed to apply RAG env override: ValueError"]
    fallback_logs = "\n".join(logger_stub.warnings)
    assert "RAG_FTS_TOP_K" not in fallback_logs
    assert secret_value_token not in fallback_logs
    assert "invalid literal" not in fallback_logs
