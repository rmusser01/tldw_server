from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY

SENSITIVE_NAME_SUBSTRINGS = (
    "API_KEY",
    "TOKEN",
    "WEBHOOK",
    "SECRET",
)

SENSITIVE_ENV_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "APIFY_API_TOKEN",
        "BRAVE_SEARCH_API_KEY",
        "COHERE_API_KEY",
        "DEEPSEEK_API_KEY",
        "DISCORD_BOT_TOKEN",
        "DISCORD_WEBHOOK_URL",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_TOKEN",
        "LLAMA_CLOUD_API_KEY",
        "MISTRAL_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "QWEN_API_KEY",
        "SERPAPI_API_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_WEBHOOK_URL",
        "TAVILY_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "VOYAGE_API_KEY",
        "XAI_API_KEY",
        "ZAI_API_KEY",
    }
)

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


def _is_sensitive_name(name: str) -> bool:
    upper_name = name.upper()
    return upper_name in SENSITIVE_ENV_NAMES or any(substring in upper_name for substring in SENSITIVE_NAME_SUBSTRINGS)


def find_sensitive_values(env: Mapping[str, str]) -> dict[str, str]:
    return {name: "set" for name, value in env.items() if value and _is_sensitive_name(name)}


def _find_blocking_sensitive_values(env: Mapping[str, str]) -> dict[str, str]:
    return {
        name: "set"
        for name, value in env.items()
        if value and _is_sensitive_name(name) and name.upper() not in SANITIZE_ONLY_ENV_NAMES
    }


def _write_minimal_env_file(env_file: Path, runtime_dir: Path, user_db_dir: Path) -> None:
    lines = [
        "AUTH_MODE=single_user",
        f"SINGLE_USER_API_KEY={DEFAULT_TEST_API_KEY}",
        f"SINGLE_USER_TEST_API_KEY={DEFAULT_TEST_API_KEY}",
        f"DATABASE_URL=sqlite:///{runtime_dir / 'users.db'}",
        f"USER_DB_BASE_DIR={user_db_dir}",
        "MINIMAL_TEST_APP=1",
        "MINIMAL_TEST_INCLUDE_AUDIO=1",
        "TEST_MODE=true",
        "PYTHONWARNINGS=ignore",
        "LOGURU_LEVEL=ERROR",
    ]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_child_env(
    work_dir: Path,
    parent_env: Mapping[str, str] | None = None,
    allow_external: bool = False,
) -> dict[str, str]:
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

    child_env = dict(source_env)
    for name in set(child_env).union(SENSITIVE_ENV_NAMES, SANITIZE_ONLY_ENV_NAMES):
        if _is_sensitive_name(name):
            child_env[name] = ""

    child_env.update(
        {
            "AUTH_MODE": "single_user",
            "SINGLE_USER_API_KEY": DEFAULT_TEST_API_KEY,
            "SINGLE_USER_TEST_API_KEY": DEFAULT_TEST_API_KEY,
            "DATABASE_URL": f"sqlite:///{runtime_dir / 'users.db'}",
            "USER_DB_BASE_DIR": str(user_db_dir),
            "TLDW_ENV_FILE": str(env_file),
            "MINIMAL_TEST_APP": "1",
            "MINIMAL_TEST_INCLUDE_AUDIO": "1",
            "TEST_MODE": "true",
            "PYTHONWARNINGS": "ignore",
            "LOGURU_LEVEL": "ERROR",
        }
    )
    _write_minimal_env_file(env_file, runtime_dir, user_db_dir)
    return child_env


__all__ = [
    "SANITIZE_ONLY_ENV_NAMES",
    "SENSITIVE_ENV_NAMES",
    "SENSITIVE_NAME_SUBSTRINGS",
    "build_child_env",
    "find_sensitive_values",
]
