import pytest

from tldw_Server_API.app.core import config as config_module
from tldw_Server_API.app.core.RAG.rag_service import semantic_cache

pytestmark = pytest.mark.unit


class _LoggerCapture:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _record(self, message: str, *args: object) -> None:
        text = str(message)
        if args:
            text = text.format(*args)
        self.messages.append(text)

    def warning(self, message: str, *args: object) -> None:
        self._record(message, *args)

    def error(self, message: str, *args: object) -> None:
        self._record(message, *args)

    def exception(self, message: str, *args: object) -> None:
        self._record(message, *args)


def _joined_logs(logger_capture: _LoggerCapture) -> str:
    return "\n".join(logger_capture.messages)


def test_shared_cache_anchors_relative_persist_path(tmp_path, monkeypatch):


    base_dir = tmp_path / "cache_root"
    base_dir.mkdir()
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(base_dir))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    semantic_cache._SHARED_CACHES.clear()

    cache = semantic_cache.get_shared_cache(
        cache_cls=semantic_cache.SemanticCache,
        similarity_threshold=0.9,
        ttl=5,
        max_size=10,
        persist_path="relative_cache.json",
        namespace="tenant",
    )

    expected_path = (base_dir / "relative_cache.json").resolve()
    assert cache.persist_path == str(expected_path)


def test_long_namespace_persist_paths_include_collision_resistant_suffix(
    tmp_path,
    monkeypatch,
):
    base_dir = tmp_path / "cache_root"
    base_dir.mkdir()
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(base_dir))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)

    common_prefix = "tenant-" + ("a" * 80)
    first = semantic_cache._default_persist_path(common_prefix + "-first")
    second = semantic_cache._default_persist_path(common_prefix + "-second")

    assert first is not None
    assert second is not None
    assert first != second
    assert len(semantic_cache.Path(first).name) <= 100
    assert len(semantic_cache.Path(second).name) <= 100


@pytest.mark.parametrize(
    ("first_namespace", "second_namespace"),
    [
        ("tenant/a", "tenant?a"),
        ("Tenant-A", "tenant-a"),
    ],
)
def test_persist_paths_preserve_raw_namespace_identity_on_case_insensitive_filesystems(
    tmp_path,
    monkeypatch,
    first_namespace,
    second_namespace,
):
    base_dir = tmp_path / "cache_root"
    base_dir.mkdir()
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(base_dir))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)

    first = semantic_cache._default_persist_path(first_namespace)
    second = semantic_cache._default_persist_path(second_namespace)

    assert first is not None
    assert second is not None
    assert semantic_cache.Path(first).name.lower() != semantic_cache.Path(second).name.lower()


def test_shared_cache_rejects_absolute_persist_path_outside_base(tmp_path, monkeypatch):


    base_dir = tmp_path / "cache_root"
    base_dir.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    persist_path = outside_dir / "cache.json"
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(base_dir))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    semantic_cache._SHARED_CACHES.clear()

    cache = semantic_cache.get_shared_cache(
        cache_cls=semantic_cache.SemanticCache,
        similarity_threshold=0.91,
        ttl=5,
        max_size=10,
        persist_path=str(persist_path),
        namespace="tenant",
    )

    expected_path = semantic_cache._default_persist_path("tenant")
    assert expected_path is not None
    assert cache.persist_path == str(expected_path)


def test_resolve_default_cache_dir_sanitizes_config_exception_log(monkeypatch):
    logger_capture = _LoggerCapture()
    monkeypatch.setattr(semantic_cache, "logger", logger_capture)
    monkeypatch.delenv("RAG_SEMANTIC_CACHE_DIR", raising=False)
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)

    def fail_config_load():
        raise TypeError("config failed for /private/config/secret-token")

    monkeypatch.setattr(config_module, "load_and_log_configs", fail_config_load)

    assert semantic_cache._resolve_default_cache_dir() is None
    assert logger_capture.messages == [
        "Semantic cache: could not load config for PROJECT_ROOT: TypeError",
    ]
    joined = _joined_logs(logger_capture)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "config failed" not in joined


def test_resolve_default_cache_dir_sanitizes_project_root_resolution_log(monkeypatch):
    logger_capture = _LoggerCapture()
    monkeypatch.setattr(semantic_cache, "logger", logger_capture)
    monkeypatch.delenv("RAG_SEMANTIC_CACHE_DIR", raising=False)
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    monkeypatch.setattr(
        config_module,
        "load_and_log_configs",
        lambda: {"PROJECT_ROOT": "/private/project/secret-token"},
    )
    original_resolve = semantic_cache.Path.resolve

    def fail_project_root_cache_dir(self, *args, **kwargs):
        if "secret-token" in str(self):
            raise OSError("resolve failed for /private/project/secret-token")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(semantic_cache.Path, "resolve", fail_project_root_cache_dir)

    assert semantic_cache._resolve_default_cache_dir() is None
    assert logger_capture.messages == [
        "Semantic cache: failed to resolve cache path from PROJECT_ROOT: OSError",
    ]
    joined = _joined_logs(logger_capture)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "resolve failed" not in joined


def test_default_persist_path_resolves_failure_logs_sanitized_exception(
    tmp_path,
    monkeypatch,
):
    logger_capture = _LoggerCapture()
    base_dir = tmp_path / "cache_root"
    base_dir.mkdir()
    monkeypatch.setattr(semantic_cache, "logger", logger_capture)
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(base_dir))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    original_resolve = semantic_cache.Path.resolve
    target_name = (
        f"semantic_cache_{semantic_cache._normalize_namespace_key_for_filename('tenant')}.json"
    )

    def fail_default_persist_path(self, *args, **kwargs):
        if str(self).endswith(target_name):
            raise OSError("default path failed for /private/cache/secret-token")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(semantic_cache.Path, "resolve", fail_default_persist_path)

    assert semantic_cache._default_persist_path("tenant") is None
    assert logger_capture.messages == [
        "Semantic cache: failed to resolve default persist path: OSError",
    ]
    joined = _joined_logs(logger_capture)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "default path failed" not in joined


def test_default_persist_path_rejects_out_of_root_resolved_path_with_sanitized_log(
    tmp_path,
    monkeypatch,
):
    logger_capture = _LoggerCapture()
    base_dir = tmp_path / "cache_root"
    base_dir.mkdir()
    monkeypatch.setattr(semantic_cache, "logger", logger_capture)
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(base_dir))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    original_resolve = semantic_cache.Path.resolve
    target_name = (
        f"semantic_cache_{semantic_cache._normalize_namespace_key_for_filename('tenant')}.json"
    )

    def resolve_outside_root(self, *args, **kwargs):
        if str(self).endswith(target_name):
            return semantic_cache.Path("/private/outside/secret-token") / target_name
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(semantic_cache.Path, "resolve", resolve_outside_root)

    assert semantic_cache._default_persist_path("tenant") is None
    assert logger_capture.messages == [
        "Refusing to use out-of-root semantic cache path.",
    ]
    joined = _joined_logs(logger_capture)
    assert "/private/" not in joined
    assert "secret-token" not in joined


def test_sanitize_persist_path_resolves_failure_logs_sanitized_exception_and_falls_back(
    tmp_path,
    monkeypatch,
):
    logger_capture = _LoggerCapture()
    base_dir = tmp_path / "cache_root"
    base_dir.mkdir()
    monkeypatch.setattr(semantic_cache, "logger", logger_capture)
    monkeypatch.setenv("RAG_SEMANTIC_CACHE_DIR", str(base_dir))
    monkeypatch.delenv("RAG_CACHE_DIR", raising=False)
    monkeypatch.setattr(semantic_cache, "_DEFAULT_CACHE_DIR", None)
    original_resolve = semantic_cache.Path.resolve

    def fail_secret_persist_path(self, *args, **kwargs):
        if "secret-token" in str(self):
            raise OSError("persist path failed for /private/cache/secret-token")
        return original_resolve(self, *args, **kwargs)

    monkeypatch.setattr(semantic_cache.Path, "resolve", fail_secret_persist_path)

    result = semantic_cache._sanitize_persist_path(
        "/private/cache/secret-token/cache.json",
        "tenant",
    )

    target_name = (
        f"semantic_cache_{semantic_cache._normalize_namespace_key_for_filename('tenant')}.json"
    )
    expected_path = (base_dir / target_name).resolve()
    assert result == str(expected_path)
    assert logger_capture.messages == [
        "Semantic cache: failed to resolve persist_path: OSError",
        "Failed to resolve semantic cache persist_path; using default cache path.",
    ]
    joined = _joined_logs(logger_capture)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "persist path failed" not in joined


def test_sanitize_persist_path_resolves_base_failure_logs_sanitized_exception_and_disables(
    monkeypatch,
):
    class _UnresolvableBaseDir:
        def expanduser(self):
            return self

        def resolve(self, strict=False):
            raise OSError("base dir failed for /private/cache/secret-token")

        def __str__(self) -> str:
            return "/private/cache/secret-token"

    logger_capture = _LoggerCapture()
    monkeypatch.setattr(semantic_cache, "logger", logger_capture)
    monkeypatch.setattr(
        semantic_cache,
        "_resolve_default_cache_dir",
        lambda: _UnresolvableBaseDir(),
    )
    monkeypatch.setattr(semantic_cache, "_default_persist_path", lambda namespace_key: None)

    assert semantic_cache._sanitize_persist_path("relative_cache.json", "tenant") is None
    assert logger_capture.messages == [
        "Semantic cache: failed to resolve base cache dir in sanitize: OSError",
        "Failed to resolve base cache dir; persistence disabled.",
    ]
    joined = _joined_logs(logger_capture)
    assert "/private/" not in joined
    assert "secret-token" not in joined
    assert "base dir failed" not in joined
