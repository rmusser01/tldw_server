"""Environment isolation helpers for deterministic local CATS fuzzing runs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY

SENSITIVE_NAME_SUBSTRINGS = (
    "API_KEY",
    "ENCRYPTION_KEY",
    "PASSWORD",
    "SALT",
    "TOKEN",
    "WEBHOOK",
    "SECRET",
)

SENSITIVE_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "APIFY_API_TOKEN",
        "APHRODITE_API_KEY",
        "AUDIO_STUDIO_ACE_STEP_API_KEY",
        "BAIDU_API_KEY",
        "BEDROCK_API_KEY",
        "BING_SEARCH_API_KEY",
        "BRAVE_SEARCH_API_KEY",
        "BRAVE_AI_API_KEY",
        "BYOK_ENCRYPTION_KEY",
        "BYOK_SECONDARY_ENCRYPTION_KEY",
        "COHERE_API_KEY",
        "CUSTOM_OPENAI_API_2_API_KEY",
        "CUSTOM_OPENAI2_API_KEY",
        "CUSTOM_OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "DISCORD_BOT_TOKEN",
        "DISCORD_WEBHOOK_URL",
        "ELEVENLABS_API_KEY",
        "EMBEDDING_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_SEARCH_API_KEY",
        "GROQ_API_KEY",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_TOKEN",
        "KAGI_API_KEY",
        "KOBOLD_API_KEY",
        "LLAMA_CLOUD_API_KEY",
        "LLAMA_API_KEY",
        "MISTRAL_API_KEY",
        "MCP_API_KEY_SALT",
        "MCP_JWT_SECRET",
        "MOONSHOT_API_KEY",
        "NEXT_PUBLIC_X_API_KEY",
        "OLLAMA_API_KEY",
        "OOBA_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "QWEN_API_KEY",
        "SERPER_API_KEY",
        "SERPAPI_API_KEY",
        "SESSION_ENCRYPTION_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_WEBHOOK_URL",
        "TABBY_API_KEY",
        "TAVILY_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "VECTOR_DB_API_KEY",
        "VLLM_API_KEY",
        "VOYAGE_API_KEY",
        "WEB_SCRAPER_API_KEY",
        "XAI_API_KEY",
        "YANDEX_API_KEY",
        "ZAI_API_KEY",
    }
)

PROVIDER_RUNTIME_ENV_NAMES = frozenset(
    {
        "AUDIO_STUDIO_ACE_STEP_BASE_URL",
        "AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST",
        "BEDROCK_API_BASE_URL",
        "CUSTOM_OPENAI2_API_BASE",
        "CUSTOM_OPENAI2_API_BASE_URL",
        "CUSTOM_OPENAI2_API_IP",
        "CUSTOM_OPENAI2_API_URL",
        "CUSTOM_OPENAI_API_BASE",
        "CUSTOM_OPENAI_API_BASE_2",
        "CUSTOM_OPENAI_API_BASE_URL_2",
        "CUSTOM_OPENAI_API_IP",
        "CUSTOM_OPENAI_API_IP_1",
        "CUSTOM_OPENAI_API_IP_2",
        "CUSTOM_OPENAI_API_MODEL_37",
        "CUSTOM_OPENAI_API_URL",
        "CUSTOM_OPENAI_API_URL_2",
        "EMBEDDING_API_URL",
        "MLX_MODEL_PATH",
    }
)
PROVIDER_RUNTIME_NAME_SUFFIXES = (
    "_API_BASE",
    "_API_BASE_URL",
    "_API_IP",
    "_API_URL",
    "_BASE_URL",
    "_RUNTIME_ENDPOINT",
)
CUSTOM_OPENAI_RUNTIME_NAME_SUFFIXES = ("_BASE", "_IP", "_URL")

SANITIZE_ONLY_ENV_NAMES = frozenset(
    {
        "CODEX_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "NPM_TOKEN",
        "SINGLE_USER_API_KEY",
        "SINGLE_USER_TEST_API_KEY",
    }
)

SERVER_GUARDED_TEST_FLAGS = ("TEST_MODE", "TESTING", "TLDW_TEST_MODE")
CONFIG_ENV_NAMES = ("TLDW_CONFIG_FILE", "TLDW_CONFIG_PATH", "TLDW_CONFIG_DIR")
ROUTE_POLICY_ENV_NAMES = frozenset(
    {
        "ROUTES_STABLE_ONLY",
        "ROUTES_DISABLE",
        "ROUTES_ENABLE",
        "ROUTES_EXPERIMENTAL",
    }
)


def _is_sensitive_name(name: str) -> bool:
    """Return True when an environment name is known or shaped like a secret."""
    upper_name = name.upper()
    return upper_name in SENSITIVE_ENV_NAMES or any(substring in upper_name for substring in SENSITIVE_NAME_SUBSTRINGS)


def _should_blank_name(name: str) -> bool:
    """Return True when a name should be scrubbed from the child process environment."""
    upper_name = name.upper()
    return _is_sensitive_name(upper_name) or _is_provider_runtime_name(upper_name)


def _is_provider_runtime_name(name: str) -> bool:
    """Return True when a name can route requests to an external provider endpoint."""
    upper_name = name.upper()
    if upper_name in PROVIDER_RUNTIME_ENV_NAMES:
        return True
    if upper_name == "BASE_URL":
        return True
    if upper_name.endswith(PROVIDER_RUNTIME_NAME_SUFFIXES):
        return True
    return upper_name.startswith("CUSTOM_OPENAI") and upper_name.endswith(CUSTOM_OPENAI_RUNTIME_NAME_SUFFIXES)


def find_sensitive_values(env: Mapping[str, str]) -> dict[str, str]:
    """Return a redacted map of populated sensitive environment names."""
    return {name: "set" for name, value in env.items() if value and _is_sensitive_name(name)}


def _find_blocking_sensitive_values(env: Mapping[str, str]) -> dict[str, str]:
    """Return populated sensitive names that should block harness startup."""
    return {
        name: "set"
        for name, value in env.items()
        if value and _is_sensitive_name(name) and name.upper() not in SANITIZE_ONLY_ENV_NAMES
    }


def _write_minimal_env_file(env_file: Path, runtime_dir: Path, user_db_dir: Path) -> None:
    """Write the minimal .env consumed by the OpenAPI export subprocess."""
    config_file = runtime_dir / "config.txt"
    lines = [
        "AUTH_MODE=single_user",
        f"SINGLE_USER_API_KEY={DEFAULT_TEST_API_KEY}",
        f"SINGLE_USER_TEST_API_KEY={DEFAULT_TEST_API_KEY}",
        f"DATABASE_URL=sqlite:///{runtime_dir / 'users.db'}",
        f"USER_DB_BASE_DIR={user_db_dir}",
        f"TLDW_CONFIG_FILE={config_file}",
        f"TLDW_CONFIG_PATH={config_file}",
        f"TLDW_CONFIG_DIR={runtime_dir}",
        "MINIMAL_TEST_APP=1",
        "MINIMAL_TEST_INCLUDE_AUDIO=1",
        "TEST_MODE=true",
        "PYTHONWARNINGS=ignore",
        "LOGURU_LEVEL=ERROR",
        "PYTHON_DOTENV_DISABLED=true",
    ]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_minimal_config_file(config_file: Path, runtime_dir: Path, user_db_dir: Path) -> None:
    """Write an inert config.txt that keeps the harness route surface stable."""
    lines = [
        "[Authentication]",
        "auth_mode = single_user",
        f"single_user_api_key = {DEFAULT_TEST_API_KEY}",
        "",
        "[AuthNZ]",
        "auth_mode = single_user",
        "",
        "[Server]",
        "host = 127.0.0.1",
        "port = 0",
        "",
        "[Database]",
        f"database_url = sqlite:///{runtime_dir / 'users.db'}",
        f"user_db_base_dir = {user_db_dir}",
        "",
        "[API-Routes]",
        "stable_only = false",
        "disable = setup",
        "enable =",
        "",
        "[API]",
        "",
        "[Local-API]",
        "",
        "[external_providers]",
        "",
        "[Embeddings]",
        "",
        "[RAG]",
        "",
        "[Search-Engines]",
        "",
        "[Web-Scraper]",
        "",
    ]
    config_file.write_text("\n".join(lines), encoding="utf-8")


def _write_server_env_file(env_file: Path, server_env: Mapping[str, str]) -> None:
    """Write the server-side environment file used by the spawned uvicorn process."""
    names = (
        "AUTH_MODE",
        "SINGLE_USER_API_KEY",
        "SINGLE_USER_TEST_API_KEY",
        "DATABASE_URL",
        "USER_DB_BASE_DIR",
        "TLDW_CONFIG_FILE",
        "TLDW_CONFIG_PATH",
        "TLDW_CONFIG_DIR",
        "MINIMAL_TEST_APP",
        "MINIMAL_TEST_INCLUDE_AUDIO",
        "PYTHONWARNINGS",
        "LOGURU_LEVEL",
        "PYTHON_DOTENV_DISABLED",
    )
    lines = [f"{name}={server_env[name]}" for name in names]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_child_env(
    work_dir: Path,
    parent_env: Mapping[str, str] | None = None,
    allow_external: bool = False,
) -> dict[str, str]:
    """Build a scrubbed environment for OpenAPI export and CATS child processes."""
    source_env = dict(os.environ if parent_env is None else parent_env)
    detected = _find_blocking_sensitive_values(source_env)
    if detected and not allow_external:
        names = ", ".join(sorted(detected))
        raise ValueError(f"Refusing to build CATS fuzz env with real credentials: {names}")

    runtime_dir = work_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    user_db_dir = runtime_dir / "user_databases"
    user_db_dir.mkdir(parents=True, exist_ok=True)
    env_file = runtime_dir / ".env"
    config_file = runtime_dir / "config.txt"

    child_env = dict(source_env)
    for name in set(child_env).union(SENSITIVE_ENV_NAMES, SANITIZE_ONLY_ENV_NAMES, PROVIDER_RUNTIME_ENV_NAMES):
        if _should_blank_name(name) or name.upper() in ROUTE_POLICY_ENV_NAMES:
            child_env[name] = ""

    child_env.update(
        {
            "AUTH_MODE": "single_user",
            "SINGLE_USER_API_KEY": DEFAULT_TEST_API_KEY,
            "SINGLE_USER_TEST_API_KEY": DEFAULT_TEST_API_KEY,
            "DATABASE_URL": f"sqlite:///{runtime_dir / 'users.db'}",
            "USER_DB_BASE_DIR": str(user_db_dir),
            "TLDW_ENV_FILE": str(env_file),
            "TLDW_CONFIG_FILE": str(config_file),
            "TLDW_CONFIG_PATH": str(config_file),
            "TLDW_CONFIG_DIR": str(runtime_dir),
            "MINIMAL_TEST_APP": "1",
            "MINIMAL_TEST_INCLUDE_AUDIO": "1",
            "TEST_MODE": "true",
            "PYTHONWARNINGS": "ignore",
            "LOGURU_LEVEL": "ERROR",
            "PYTHON_DOTENV_DISABLED": "true",
        }
    )
    _write_minimal_config_file(config_file, runtime_dir, user_db_dir)
    _write_minimal_env_file(env_file, runtime_dir, user_db_dir)
    return child_env


def build_server_env(work_dir: Path, child_env: Mapping[str, str]) -> dict[str, str]:
    """Build the uvicorn server environment from the already-scrubbed child environment."""
    runtime_dir = work_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    user_db_dir = runtime_dir / "user_databases"
    user_db_dir.mkdir(parents=True, exist_ok=True)
    env_file = runtime_dir / "cats-server.env"

    server_env = dict(child_env)
    for name in SERVER_GUARDED_TEST_FLAGS:
        server_env.pop(name, None)

    server_env.update(
        {
            "AUTH_MODE": child_env.get("AUTH_MODE", "single_user"),
            "SINGLE_USER_API_KEY": child_env.get("SINGLE_USER_API_KEY", DEFAULT_TEST_API_KEY),
            "SINGLE_USER_TEST_API_KEY": child_env.get("SINGLE_USER_TEST_API_KEY", DEFAULT_TEST_API_KEY),
            "DATABASE_URL": child_env.get("DATABASE_URL", f"sqlite:///{runtime_dir / 'users.db'}"),
            "USER_DB_BASE_DIR": child_env.get("USER_DB_BASE_DIR", str(user_db_dir)),
            "TLDW_ENV_FILE": str(env_file),
            "TLDW_CONFIG_FILE": child_env["TLDW_CONFIG_FILE"],
            "TLDW_CONFIG_PATH": child_env["TLDW_CONFIG_PATH"],
            "TLDW_CONFIG_DIR": child_env["TLDW_CONFIG_DIR"],
            "MINIMAL_TEST_APP": "1",
            "MINIMAL_TEST_INCLUDE_AUDIO": "1",
            "PYTHONWARNINGS": "ignore",
            "LOGURU_LEVEL": "ERROR",
            "PYTHON_DOTENV_DISABLED": "true",
        }
    )
    _write_server_env_file(env_file, server_env)
    return server_env


__all__ = [
    "SANITIZE_ONLY_ENV_NAMES",
    "SERVER_GUARDED_TEST_FLAGS",
    "SENSITIVE_ENV_NAMES",
    "SENSITIVE_NAME_SUBSTRINGS",
    "PROVIDER_RUNTIME_ENV_NAMES",
    "build_child_env",
    "build_server_env",
    "find_sensitive_values",
]
