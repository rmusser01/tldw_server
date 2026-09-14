# config_info.py - Endpoint for serving configuration information for documentation
"""
API endpoint to provide configuration information for auto-populating documentation.
Only exposes non-sensitive configuration suitable for documentation.
Also includes provider status and validation endpoints for first-run UX.
"""

import asyncio
import configparser
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.core import config as config_mod
from tldw_Server_API.app.core.AuthNZ.byok_config import (
    PROVIDER_APP_CONFIG_KEYS,
    is_runtime_base_url_override,
)
from tldw_Server_API.app.core.AuthNZ.byok_helpers import (
    load_server_config_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    merge_server_fallback_snapshot,
    resolve_static_server_fallback_from_snapshot,
)
from tldw_Server_API.app.core.AuthNZ.byok_testing import (
    provider_validation_public_error,
    test_provider_credentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    configured_provider_model_from_snapshot,
)
from tldw_Server_API.app.core.Chat.streaming_utils import (
    PROVIDER_STREAM_ERROR_MESSAGES,
)
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_api_key_env_keys,
    custom_openai_provider_number,
    custom_openai_section_name,
    iter_custom_openai_provider_names,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import (
    canonical_builtin_llm_provider_name,
)
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import (
    provider_requires_api_key,
)
from tldw_Server_API.app.core.testing import env_flag_enabled, is_truthy
from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

router = APIRouter()
_DOCS_API_KEY_PLACEHOLDER = "YOUR_API_KEY"
_SHIPPED_CAPABILITIES = {"hasReadingSnapshotPagesV1": True}


def get_config_path() -> Path:
    """Get the path to the config file."""
    # Try environment variable first
    config_path = os.getenv("TLDW_CONFIG_PATH")
    if config_path:
        return Path(config_path)

    # Default location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
    return Path(project_root) / "Config_Files" / "config.txt"


def load_safe_config() -> dict:
    """
    Load configuration that is safe to expose for documentation.
    Excludes sensitive information like actual API keys.
    """
    config_path = get_config_path()

    if not config_path.exists():
        logger.warning("Config file not found")
        return {
            "configured": False,
            "message": "Configuration file not found",
            "capabilities": dict(_SHIPPED_CAPABILITIES),
            "supported_features": dict(_SHIPPED_CAPABILITIES),
        }

    config = configparser.ConfigParser()
    config.read(config_path)

    # Get authentication mode
    auth_mode = config.get('Authentication', 'auth_mode', fallback='single_user')

    # Determine what to expose based on auth mode
    safe_config = {
        "configured": True,
        "auth_mode": auth_mode,
        "server": {
            "host": config.get('Server', 'host', fallback='127.0.0.1'),
            "port": config.getint('Server', 'port', fallback=8000)
        }
    }

    # Never expose a real API key via docs-info.
    if auth_mode == 'single_user':
        api_key = config.get('Authentication', 'single_user_api_key', fallback='').strip()
        placeholders = {
            "",
            "your_api_key_here",
            "YOUR_API_KEY_HERE",
            "change-me-in-production",
            "CHANGE_ME_TO_SECURE_API_KEY",
            "test-api-key-12345",
        }
        safe_config["api_key_configured"] = bool(api_key and api_key not in placeholders)
    else:
        safe_config["api_key_configured"] = False
    safe_config["api_key_for_docs"] = ""

    # Check which LLM providers are configured (without exposing keys)
    configured_providers = []
    if config.has_section('API'):
        provider_keys = {
            'openai_api_key': 'OpenAI',
            'anthropic_api_key': 'Anthropic',
            'groq_api_key': 'Groq',
            'google_api_key': 'Google',
            'cohere_api_key': 'Cohere',
            'mistral_api_key': 'Mistral',
            'deepseek_api_key': 'DeepSeek',
            'huggingface_api_key': 'HuggingFace',
            'openrouter_api_key': 'OpenRouter',
            'novita_api_key': 'Novita',
            'poe_api_key': 'Poe',
            'together_api_key': 'Together',
        }

        for key_name, provider_name in provider_keys.items():
            value = config.get('API', key_name, fallback='')
            if value and value not in ['', 'your_api_key_here', 'YOUR_API_KEY_HERE']:
                configured_providers.append(provider_name)

    safe_config["configured_llm_providers"] = configured_providers

    # FFmpeg availability (needed by clients to gate video/audio ingestion UX)
    safe_config["ffmpeg_available"] = shutil.which("ffmpeg") is not None

    # Feature flags / capabilities (safe to expose). The shipped fallback
    # remains available if dynamic derivation fails, while a successful route
    # lookup can explicitly disable Reading support.
    caps = dict(_SHIPPED_CAPABILITIES)
    try:
        derived_caps: dict[str, bool] = {
            "hasReadingSnapshotPagesV1": bool(
                config_mod.route_enabled("reading", default_stable=True)
            )
        }
        has_media_routes = bool(config_mod.route_enabled("media", default_stable=True))
        has_audio_http = bool(config_mod.route_enabled("audio", default_stable=True))
        has_audio_websocket = bool(config_mod.route_enabled("audio-websocket", default_stable=True))
        derived_caps["personalization"] = bool(
            config_mod.legacy_get("PERSONALIZATION_ENABLED", True)
        ) and bool(config_mod.route_enabled("personalization", default_stable=False))
        derived_caps["persona"] = bool(
            config_mod.legacy_get("PERSONA_ENABLED", True)
        ) and bool(config_mod.route_enabled("persona", default_stable=True))
        derived_caps["hasSlides"] = bool(
            config_mod.route_enabled("slides", default_stable=True)
        )
        derived_caps["hasPresentationStudio"] = bool(derived_caps["hasSlides"])
        derived_caps["hasPresentationRender"] = bool(
            derived_caps["hasPresentationStudio"]
        ) and bool(is_truthy(os.getenv("PRESENTATION_RENDER_ENABLED", "true")))
        # Docs-info serves as the authoritative feature gate for clients that
        # cannot infer websocket transport support from OpenAPI alone.
        derived_caps["hasStt"] = bool(has_audio_http or has_audio_websocket)
        derived_caps["hasTts"] = bool(has_audio_http or has_audio_websocket)
        derived_caps["hasVoiceChat"] = bool(has_audio_http or has_audio_websocket)
        derived_caps["hasVoiceConversationTransport"] = bool(has_audio_websocket)
        derived_caps["hasAudio"] = bool(
            derived_caps["hasStt"]
            or derived_caps["hasTts"]
            or derived_caps["hasVoiceChat"]
        )
        derived_caps["hasMediaPlaylistPreflight"] = has_media_routes
        derived_caps["hasMediaIngestJobs"] = has_media_routes
        derived_caps["hasMediaIngestJobEvents"] = has_media_routes
        derived_caps["hasMediaIngestWorker"] = bool(
            has_media_routes
            and should_start_inprocess_worker(
                "MEDIA_INGEST_JOBS_WORKER_ENABLED",
                "media",
                sidecar_mode=env_flag_enabled("TLDW_WORKERS_SIDECAR_MODE"),
                default_stable=True,
                test_mode=False,
            )
        )
        derived_caps["hasDurableMediaCollections"] = has_media_routes
        derived_caps["hasKnowledgeQaMediaScope"] = has_media_routes
    except Exception:
        logger.debug("Failed to derive safe capability flags")
    else:
        caps.update(derived_caps)
    # Expose both for backward-compat and forward-looking UI.
    safe_config["supported_features"] = caps
    safe_config["capabilities"] = caps

    return safe_config


@router.get("/config/docs-info")
async def get_documentation_config():
    """
    Get configuration information suitable for auto-populating documentation.

    This endpoint returns non-sensitive configuration that can be used to:
    - Provide a safe placeholder key for documentation snippets
    - Show which LLM providers are configured
    - Provide the correct base URL for examples

    Returns:
        Dict with configuration information safe for documentation
    """
    config = load_safe_config()

    # Build base URL
    host = config.get("server", {}).get("host", "127.0.0.1")
    port = config.get("server", {}).get("port", 8000)

    # Convert 0.0.0.0 to localhost for documentation (comparison/sanitization only).
    if host == "0.0.0.0":  # nosec B104
        host = "localhost"

    base_url = f"http://{host}:{port}"

    return {
        "configured": config.get("configured", False),
        "auth_mode": config.get("auth_mode", "single_user"),
        "api_key": _DOCS_API_KEY_PLACEHOLDER,
        "api_key_configured": bool(config.get("api_key_configured", False)),
        "base_url": base_url,
        "configured_providers": config.get("configured_llm_providers", []),
        # FFmpeg availability for client-side video/audio ingestion hints
        "ffmpeg_available": config.get("ffmpeg_available", False),
        # Surface capabilities map so WebUI can dynamically hide/show experimental tabs
        "capabilities": config.get("capabilities", config.get("supported_features", {})),
        # Keep supported_features for older clients
        "supported_features": config.get("supported_features", {}),
        "examples": {
            "python": generate_python_example(
                _DOCS_API_KEY_PLACEHOLDER,
                base_url
            ),
            "curl": generate_curl_example(
                _DOCS_API_KEY_PLACEHOLDER,
                base_url
            ),
            "javascript": generate_js_example(
                _DOCS_API_KEY_PLACEHOLDER,
                base_url
            )
        }
    }


@router.get("/config/flashcards-import-limits")
async def get_flashcards_import_limits():
    """
    Expose current flashcards import limits derived from environment or defaults.
    These reflect server-enforced caps; per-request overrides can only lower these values.
    """
    def _int_env(name: str, default: int) -> int:
        try:
            return max(1, int(os.getenv(name, str(default))))
        except Exception:
            return default

    return {
        "max_lines": _int_env('FLASHCARDS_IMPORT_MAX_LINES', 10000),
        "max_line_length": _int_env('FLASHCARDS_IMPORT_MAX_LINE_LENGTH', 32768),
        "max_field_length": _int_env('FLASHCARDS_IMPORT_MAX_FIELD_LENGTH', 8192),
        "overrides": {
            "query_params": [
                "max_lines", "max_line_length", "max_field_length"
            ],
            "note": "Query overrides can only reduce, not increase caps"
        }
    }


def generate_python_example(api_key: str, base_url: str) -> str:
    """Generate a Python code example with the provided configuration."""
    return f"""import json
from urllib.request import Request, urlopen

API_KEY = "{api_key}"
BASE_URL = "{base_url}"

# Create evaluation
eval_data = {{
    "name": "test_evaluation",
    "eval_type": "exact_match",
    "eval_spec": {{"threshold": 1.0}},
    "dataset": [
        {{"input": {{"output": "test"}}, "expected": {{"output": "test"}}}}
    ]
}}

req = Request(
    f"{{BASE_URL}}/api/v1/evaluations",
    data=json.dumps(eval_data).encode("utf-8"),
    headers={{
        "Authorization": f"Bearer {{API_KEY}}",
        "Content-Type": "application/json",
    }},
    method="POST",
)

with urlopen(req, timeout=30) as resp:
    payload = json.loads(resp.read().decode("utf-8"))
print(f"Created evaluation: {{payload['id']}}")"""


def generate_curl_example(api_key: str, base_url: str) -> str:
    """Generate a cURL example with the provided configuration."""
    return f"""curl -X POST {base_url}/api/v1/evaluations \\
  -H "Authorization: Bearer {api_key}" \\
  -H "Content-Type: application/json" \\
  -d '{{
    "name": "test_eval",
    "eval_type": "exact_match",
    "eval_spec": {{"threshold": 1.0}},
    "dataset": [
      {{"input": {{"output": "test"}}, "expected": {{"output": "test"}}}}
    ]
  }}'"""


def generate_js_example(api_key: str, base_url: str) -> str:
    """Generate a JavaScript example with the provided configuration."""
    return f"""const API_KEY = '{api_key}';
const BASE_URL = '{base_url}';

const evalData = {{
    name: 'test_evaluation',
    eval_type: 'exact_match',
    eval_spec: {{ threshold: 1.0 }},
    dataset: [
        {{ input: {{ output: 'test' }}, expected: {{ output: 'test' }} }}
    ]
}};

const response = await fetch(`${{BASE_URL}}/api/v1/evaluations`, {{
    method: 'POST',
    headers: {{
        'Authorization': `Bearer ${{API_KEY}}`,
        'Content-Type': 'application/json'
    }},
    body: JSON.stringify(evalData)
}});

const data = await response.json();
console.log('Created evaluation:', data.id);"""


# ---------------------------------------------------------------------------
# Tokenizer configuration readout and selection
# ---------------------------------------------------------------------------

class TokenizerConfig(BaseModel):
    mode: str = Field(..., description="Current tokenizer mode: whitespace|char_approx")
    divisor: int = Field(..., description="Char-based approx divisor (if applicable)")
    available_modes: list[str] = Field(default_factory=lambda: ["whitespace", "char_approx"])


class TokenizerUpdate(BaseModel):
    mode: str = Field(..., pattern="^(whitespace|char_approx)$", description="New tokenizer mode")
    divisor: int = Field(4, ge=1, description="Char-based approx divisor")


@router.get("/config/tokenizer", response_model=TokenizerConfig)
async def get_tokenizer_config() -> TokenizerConfig:
    mode = str(global_settings.get("TOKEN_ESTIMATOR_MODE", "whitespace")).lower()
    divisor = int(global_settings.get("TOKEN_CHAR_APPROX_DIVISOR", 4))
    return TokenizerConfig(mode=mode, divisor=divisor)


@router.put("/config/tokenizer", response_model=TokenizerConfig)
async def update_tokenizer_config(update: TokenizerUpdate) -> TokenizerConfig:
    # Update in-memory settings; non-persistent across restarts
    global_settings["TOKEN_ESTIMATOR_MODE"] = update.mode
    global_settings["TOKEN_CHAR_APPROX_DIVISOR"] = int(update.divisor)
    return TokenizerConfig(mode=update.mode, divisor=int(update.divisor))


@router.get("/config/jobs")
async def get_jobs_config_info():
    """Expose non-sensitive Jobs backend configuration and tuning flags.

    Returns current backend selection (sqlite|postgres) and key lease/backoff parameters. Does not expose DSN.
    """
    backend = "postgres" if (os.getenv("JOBS_DB_URL", "").startswith("postgres")) else "sqlite"
    def _to_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default
    def _to_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return default
    return {
        "backend": backend,
        "configured": bool(os.getenv("JOBS_DB_URL")) or backend == "sqlite",
        "standard_queues": ["default", "high", "low"],
        "flags": {
            "JOBS_LEASE_SECONDS": _to_int("JOBS_LEASE_SECONDS", 60),
            "JOBS_LEASE_RENEW_SECONDS": _to_int("JOBS_LEASE_RENEW_SECONDS", 30),
            "JOBS_LEASE_RENEW_JITTER_SECONDS": _to_int("JOBS_LEASE_RENEW_JITTER_SECONDS", 5),
            "JOBS_LEASE_MAX_SECONDS": _to_int("JOBS_LEASE_MAX_SECONDS", 3600),
            "JOBS_ENFORCE_LEASE_ACK": is_truthy(os.getenv("JOBS_ENFORCE_LEASE_ACK")),
        },
        "notes": "DSN is not exposed for security. Configure via the environment (PostgreSQL DSN) to use a Postgres backend."
    }


@router.get(
    "/config/quickstart",
    response_class=HTMLResponse,
    responses={
        status.HTTP_200_OK: {
            "description": "Fallback quickstart HTML page when redirect resolution fails.",
            "content": {
                "text/html": {},
            },
        },
        status.HTTP_307_TEMPORARY_REDIRECT: {
            "description": "Redirect to the configured quickstart destination.",
            "headers": {
                "location": {
                    "description": "Quickstart destination URL.",
                    "schema": {"type": "string"},
                },
            },
        },
    },
)
async def get_quickstart_redirect():
    """
    Redirect to a Quickstart URL defined in config.txt or environment.

    Precedence:
    1) Environment variable QUICKSTART_URL
    2) Config [UI] quickstart_url
    3) Config [Docs] quickstart_url
    4) Default: /docs

    Behavior:
    - If the target starts with http(s)://, redirect there.
    - If the target starts with '/', redirect to that path (same origin).
    - Otherwise, treat as same-origin path and redirect to /<target>.
    """
    try:
        # 1) Environment override
        url = os.getenv("QUICKSTART_URL")

        # 2/3) Config file
        if not url:
            try:
                cfg = config_mod.load_comprehensive_config()
                if cfg.has_section('UI') and cfg.has_option('UI', 'quickstart_url'):
                    url = cfg.get('UI', 'quickstart_url').strip()
                elif cfg.has_section('Docs') and cfg.has_option('Docs', 'quickstart_url'):
                    url = cfg.get('Docs', 'quickstart_url').strip()
            except Exception:
                logger.warning("Quickstart redirect: could not read config, using default")

        # 4) Default
        if not url:
            url = "/docs"

        # Normalize
        if url.startswith("http://") or url.startswith("https://") or url.startswith("/"):
            target = url
        else:
            # Treat as same-origin path
            target = f"/{url}"

        logger.info(f"Redirecting /api/v1/config/quickstart to: {target}")
        return RedirectResponse(url=target, status_code=307)
    except Exception:
        logger.error("Quickstart redirect failed")
        # Fallback to a minimal built-in HTML page with a link to /docs
        fallback_html = """
        <!DOCTYPE html>
        <html><head><meta charset='utf-8'><title>Quickstart</title></head>
        <body>
          <p>Quickstart is not configured. Continue to the API docs:</p>
          <a href="/docs">Open API docs</a>
        </body></nhtml>
        """
        return HTMLResponse(content=fallback_html, status_code=200)


# ---------------------------------------------------------------------------
# Provider status and validation endpoints
# ---------------------------------------------------------------------------

# Mapping of provider name -> environment variable holding its API key.
_PROVIDER_ENV_KEY_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "qwen": "QWEN_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "zai": "ZAI_API_KEY",
    "novita": "NOVITA_API_KEY",
    "poe": "POE_API_KEY",
    "together": "TOGETHER_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
}

_VALIDATION_TIMEOUT_SECONDS = 5.0


def _configured_validation_endpoint(
    provider: str,
    app_config: dict[str, object] | None,
) -> str | None:
    """Return the explicit endpoint the runtime adapter will prefer."""
    if not isinstance(app_config, dict):
        return None
    custom_number = custom_openai_provider_number(provider)
    section = (
        custom_openai_section_name(custom_number)
        if custom_number is not None
        else PROVIDER_APP_CONFIG_KEYS.get(provider)
    )
    if section is None:
        return None
    provider_config = app_config.get(section)
    if not isinstance(provider_config, dict):
        return None

    if custom_number is not None or provider in {"novita", "poe", "together"}:
        endpoint_fields = ("api_ip", "api_base_url")
    elif provider == "openai":
        endpoint_fields = ("api_base_url", "api_base", "base_url")
    elif provider == "bedrock":
        endpoint_fields = (
            "runtime_endpoint",
            "api_base_url",
            "base_url",
            "api_url",
            "endpoint",
        )
    elif provider == "google":
        endpoint_fields = ("api_base_url", "api_base")
    elif provider == "huggingface":
        use_router_value = provider_config.get("use_router_url_format")
        if use_router_value is None:
            use_router_value = provider_config.get(
                "huggingface_use_router_url_format"
            )
        use_router = str(use_router_value or "false").casefold() == "true"
        if use_router:
            runtime_override = is_runtime_base_url_override(
                provider_config.get("_runtime_base_url_override")
            )
            api_base_url = provider_config.get("api_base_url")
            if (
                runtime_override
                and isinstance(api_base_url, str)
                and api_base_url.strip()
            ):
                return api_base_url.strip()
            endpoint_fields = (
                "router_base_url",
                "huggingface_router_base_url",
            )
        else:
            endpoint_fields = ("api_base_url",)
    else:
        endpoint_fields = ("api_base_url",)

    for field in endpoint_fields:
        value = provider_config.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter for provider validation (5 calls/min per IP)
# ---------------------------------------------------------------------------
_VALIDATE_RATE_LIMIT = 5  # max calls
_VALIDATE_RATE_WINDOW = 60  # seconds
_validate_call_log: dict[str, list[float]] = defaultdict(list)


def _check_validate_rate_limit(client_ip: str) -> None:
    """Raise 429 if the caller exceeds the provider-validation rate limit."""
    now = time.monotonic()
    window_start = now - _VALIDATE_RATE_WINDOW

    # Prune expired entries
    timestamps = _validate_call_log[client_ip]
    _validate_call_log[client_ip] = [t for t in timestamps if t > window_start]

    if len(_validate_call_log[client_ip]) >= _VALIDATE_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {_VALIDATE_RATE_LIMIT} validation requests per {_VALIDATE_RATE_WINDOW}s",
        )

    _validate_call_log[client_ip].append(now)


def _key_hint(api_key: str) -> str:
    """Return a safe hint for a key: first 3 and last 4 chars, or just last 4 if short."""
    if len(api_key) <= 8:
        return "****" + api_key[-2:] if len(api_key) >= 2 else "****"
    return api_key[:3] + "..." + api_key[-4:]


def _resolve_provider_key(provider: str) -> Optional[str]:
    """Resolve the API key for a provider from env vars and config, without exposing it.

    Returns the raw key string or None.
    """
    # 1) Check environment variable
    env_var = _PROVIDER_ENV_KEY_MAP.get(provider)
    if env_var:
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    # 2) Check get_api_keys() which reads both env and config.txt
    try:
        from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys
        keys = get_api_keys()
        val = (keys.get(provider) or "").strip()
        if val:
            return val
    except Exception:
        logger.debug("Failed to load API keys for provider")

    return None


class ProviderStatusItem(BaseModel):
    name: str = Field(..., description="Provider identifier (e.g. 'openai')")
    configured: bool = Field(..., description="Whether an API key is present")
    requires_api_key: bool = Field(..., description="Whether the provider requires an API key")
    key_hint: Optional[str] = Field(None, description="Masked key hint (e.g. 'sk-...abcd')")
    key_source: Optional[str] = Field(None, description="Where the key was found (env, config)")


class ProvidersStatusResponse(BaseModel):
    providers: list[ProviderStatusItem]
    any_configured: bool = Field(..., description="True if at least one cloud provider has a key")


class ProviderValidateRequest(BaseModel):
    provider: str = Field(..., description="Provider name to validate (e.g. 'openai')")
    api_key: Optional[str] = Field(
        None,
        description="API key to validate. Must be provided; server will not fall back to configured keys.",
    )


class ProviderValidateResponse(BaseModel):
    provider: str
    valid: bool
    error: Optional[str] = None


@router.get("/config/providers", response_model=ProvidersStatusResponse)
async def list_configured_providers() -> ProvidersStatusResponse:
    """List all supported LLM providers and their configuration status.

    Returns which providers have API keys set (from environment variables or
    config.txt) without exposing the actual keys. Useful for admin dashboards
    and first-run setup wizards to show a 'no providers configured' banner.
    """
    try:
        from tldw_Server_API.app.core.LLM_Calls.provider_metadata import (
            PROVIDER_REQUIRES_KEY,
        )
    except Exception:
        PROVIDER_REQUIRES_KEY = {}

    # Ordered list of cloud providers first, then local
    cloud_providers = [
        "openai", "anthropic", "google", "cohere", "groq", "mistral",
        "deepseek", "huggingface", "openrouter", "qwen", "moonshot", "zai",
        "novita", "poe", "together", "bedrock",
    ]
    local_providers = [
        "llama.cpp", "kobold", "ooba", "tabbyapi", "vllm",
        "local-llm", "ollama", "aphrodite", "mlx",
    ]
    custom_providers = list(iter_custom_openai_provider_names())

    items: list[ProviderStatusItem] = []
    any_configured = False

    for name in cloud_providers + local_providers + custom_providers:
        requires_key = PROVIDER_REQUIRES_KEY.get(name, True)
        api_key = _resolve_provider_key(name) if requires_key else None

        configured = bool(api_key) if requires_key else True
        hint = _key_hint(api_key) if api_key else None

        # Determine source
        key_source: Optional[str] = None
        if api_key:
            env_var = _PROVIDER_ENV_KEY_MAP.get(name)
            custom_number = custom_openai_provider_number(name)
            has_custom_env_key = (
                custom_number is not None
                and any(
                    os.environ.get(env_key, "").strip()
                    for env_key in custom_openai_api_key_env_keys(custom_number)
                )
            )
            if (env_var and os.environ.get(env_var, "").strip()) or has_custom_env_key:
                key_source = "env"
            else:
                key_source = "config"

        if configured and name in cloud_providers:
            any_configured = True

        items.append(ProviderStatusItem(
            name=name,
            configured=configured,
            requires_api_key=requires_key,
            key_hint=hint,
            key_source=key_source,
        ))

    return ProvidersStatusResponse(providers=items, any_configured=any_configured)


@router.post("/config/validate-provider", response_model=ProviderValidateResponse)
async def validate_provider_key(body: ProviderValidateRequest, request: Request) -> ProviderValidateResponse:
    """Validate a caller-supplied provider key through the runtime adapter.

    The caller **must** supply ``api_key`` in the request body. The endpoint
    never falls back to server-configured keys to prevent unauthenticated
    callers from probing which providers are configured.

    Rate-limited to 5 requests per minute per client IP.

    Returns ``{valid: true}`` on success or ``{valid: false, error: "..."}``
    on failure.
    """
    # Rate-limit: 5 calls/min per IP
    client_ip = request.client.host if request.client else "unknown"
    _check_validate_rate_limit(client_ip)

    provider_input = body.provider.strip().lower()

    # Always require the caller to provide the key -- never fall back to
    # server-configured keys (prevents unauthenticated provider probing).
    api_key = (body.api_key or "").strip()
    if not api_key:
        return ProviderValidateResponse(
            provider=provider_input,
            valid=False,
            error="api_key is required. Provide the key you want to validate.",
        )

    try:
        provider = canonical_builtin_llm_provider_name(provider_input)
    except ValueError:
        return ProviderValidateResponse(
            provider=provider_input,
            valid=False,
            error=PROVIDER_STREAM_ERROR_MESSAGES["provider_configuration_invalid"],
        )

    custom_number = custom_openai_provider_number(provider)
    if not provider_requires_api_key(provider) and custom_number is None:
        return ProviderValidateResponse(
            provider=provider,
            valid=False,
            error=PROVIDER_STREAM_ERROR_MESSAGES["provider_configuration_invalid"],
        )

    try:
        server_config_snapshot = load_server_config_snapshot()
        server_fallback = resolve_static_server_fallback_from_snapshot(
            provider,
            server_config_snapshot,
        )
        caller_fallback = merge_server_fallback_snapshot(
            provider,
            server_fallback,
            api_key=api_key,
            credential_fields={},
            auth_source=None,
            provider_config={},
        )
        model = configured_provider_model_from_snapshot(
            provider,
            caller_fallback.app_config,
        )
        if not model:
            return ProviderValidateResponse(
                provider=provider,
                valid=False,
                error=PROVIDER_STREAM_ERROR_MESSAGES[
                    "provider_configuration_invalid"
                ],
            )

        authoritative_endpoint = _configured_validation_endpoint(
            provider,
            caller_fallback.app_config,
        )
        if custom_number is not None:
            if authoritative_endpoint is None:
                return ProviderValidateResponse(
                    provider=provider,
                    valid=False,
                    error=PROVIDER_STREAM_ERROR_MESSAGES[
                        "provider_configuration_invalid"
                    ],
                )

        await test_provider_credentials(
            provider=provider,
            api_key=caller_fallback.api_key,
            app_config=caller_fallback.app_config,
            model=model,
            include_override_model=False,
            timeout_seconds=_VALIDATION_TIMEOUT_SECONDS,
            enforce_egress_policy=authoritative_endpoint is not None,
            authoritative_endpoint=authoritative_endpoint,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        public_error = provider_validation_public_error(exc)
        del exc
        logger.warning("Provider validation failed")
        return ProviderValidateResponse(
            provider=provider,
            valid=False,
            error=public_error.message,
        )
    return ProviderValidateResponse(provider=provider, valid=True, error=None)
