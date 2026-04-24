"""
Main MCP Server implementation for unified module

Handles WebSocket and HTTP connections with production-ready features.
"""

import asyncio
import ipaddress
import json
import os
import re
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import InvalidTokenError, TokenExpiredError
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Streaming.streams import WebSocketStream
from tldw_Server_API.app.core.feature_flags import is_mcp_hub_policy_enforcement_enabled
from tldw_Server_API.app.core.testing import (
    env_flag_enabled as _env_flag_enabled,
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
    is_test_mode as _is_test_mode,
    is_truthy as _is_truthy,
)
from tldw_Server_API.app.services.app_lifecycle import assert_may_start_work
from tldw_Server_API.app.services.shutdown_transport_registry import (
    register_shutdown_transport_family,
)

from .auth.authnz_rbac import get_rbac_policy
from .auth.jwt_manager import JWTManager, get_jwt_manager
from .auth.rate_limiter import RateLimitExceeded, get_rate_limiter
from .config import get_config, validate_config
from .modules.registry import get_module_registry
from .monitoring.metrics import get_metrics_collector
from .protocol import MCPProtocol, MCPRequest, MCPResponse, RequestContext
from .security.ip_filter import get_ip_access_controller
from .security.request_guards import enforce_client_certificate_headers

_MCP_SERVER_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    HTTPException,
    WebSocketDisconnect,
    InvalidTokenError,
    TokenExpiredError,
    RateLimitExceeded,
)

_ENV_PLACEHOLDER_RE = re.compile(r"^\$\{(?P<name>[A-Z0-9_]+)(?::-(?P<default>.*))?\}$")


def _is_authnz_access_token(token: str) -> bool:
    """Return True when the token verifies as an AuthNZ access token."""
    try:
        jwt_service = get_jwt_service()
        jwt_service.decode_access_token(token)
        return True
    except TokenExpiredError:
        return True
    except InvalidTokenError:
        return False
    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
        return False


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if not isinstance(value, str):
        return value

    match = _ENV_PLACEHOLDER_RE.match(value.strip())
    if not match:
        return value
    env_name = match.group("name")
    default = match.group("default")
    return os.getenv(env_name, default if default is not None else "")


def _extract_api_key_permissions(info: Optional[dict[str, Any]]) -> list[str]:
    """Normalize API key scopes into MCP permissions."""
    if not info:
        return []
    try:
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import normalize_scope
    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
        return []

    raw_scopes = info.get("scopes")
    if raw_scopes is None:
        raw_scopes = info.get("scope")

    try:
        scopes = normalize_scope(raw_scopes)
    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
        scopes = set()

    return sorted(scopes) if scopes else []


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


class _JWTManagerProxy:
    """Proxy for JWT manager to allow per-server monkeypatching without global side effects."""

    def __init__(self, manager: JWTManager):
        self._manager = manager

    def __getattr__(self, name: str):
        return getattr(self._manager, name)


class WebSocketConnection:
    """Manages a single WebSocket connection"""

    def __init__(
        self,
        websocket: WebSocket,
        connection_id: str,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ):
        self.websocket = websocket
        self.connection_id = connection_id
        self.client_id = client_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self.session_id = session_id or connection_id
        self.workspace_id = workspace_id
        self.cwd = cwd
        self.connected_at = datetime.now(timezone.utc)
        self.last_activity = self.connected_at
        self.message_count = 0
        self.error_count = 0
        self.request_times: deque[float] = deque(maxlen=1000)

    async def send_json(self, data: dict[str, Any]):
        """Send JSON data to client"""
        try:
            await self.websocket.send_json(data)
            self.last_activity = datetime.now(timezone.utc)
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(connection_id=self.connection_id).error(f"Error sending to WebSocket {self.connection_id}: {e}")
            self.error_count += 1
            raise

    async def receive_json(self) -> dict[str, Any]:
        """Receive JSON data from client"""
        try:
            data = await self.websocket.receive_json()
            self.last_activity = datetime.now(timezone.utc)
            self.message_count += 1
            return data
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(connection_id=self.connection_id).error(f"Error receiving from WebSocket {self.connection_id}: {e}")
            self.error_count += 1
            raise

    async def close(self, code: int = 1000, reason: str = ""):
        """Close the WebSocket connection"""
        try:
            await self.websocket.close(code=code, reason=reason)
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(connection_id=self.connection_id).error(f"Error closing WebSocket {self.connection_id}: {e}")


@dataclass
class SessionData:
    """Lightweight in-memory session state for HTTP/WS MCP sessions."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    uris_seen: deque[str] = field(default_factory=lambda: deque(maxlen=500))
    uris_index: set[str] = field(default_factory=set)
    client_info: dict[str, Any] = field(default_factory=dict)
    safe_config: dict[str, Any] = field(default_factory=dict)
    workspace_id: Optional[str] = None
    cwd: Optional[str] = None

    def touch(self):
        self.last_activity = datetime.now(timezone.utc)

    def add_seen_uri(self, uri: str, max_len: int):
        if uri in self.uris_index:
            return
        # Enforce bounded size
        if len(self.uris_seen) >= max_len:
            oldest = self.uris_seen.popleft()
            self.uris_index.discard(oldest)
        self.uris_seen.append(uri)
        self.uris_index.add(uri)


class MCPServer:
    """
    Production-ready MCP Server with WebSocket and HTTP support.

    Features:
    - WebSocket connection management
    - HTTP request handling
    - Connection pooling
    - Graceful shutdown
    - Health monitoring
    - Metrics collection
    """

    def __init__(self):
        self.config = get_config()
        self.protocol = MCPProtocol()
        self.module_registry = get_module_registry()
        self.jwt_manager = _JWTManagerProxy(get_jwt_manager())
        self.rbac_policy = get_rbac_policy()
        self.rate_limiter = get_rate_limiter()
        self._ws_auth_required_initial = self.config.ws_auth_required

        # Connection management
        self.connections: dict[str, WebSocketConnection] = {}
        self.connection_lock = asyncio.Lock()
        self._ip_connection_counts: dict[str, int] = {}

        # Session management (HTTP/WS)
        self.sessions: dict[str, SessionData] = {}
        self.session_lock = asyncio.Lock()

        # Server state
        self.initialized = False
        self._initialize_lock = asyncio.Lock()
        self.startup_time = datetime.now(timezone.utc)
        self.shutdown_event = asyncio.Event()

        # Background tasks
        self.background_tasks: set[asyncio.Task] = set()
        register_shutdown_transport_family(
            "mcp.websocket",
            active_count=self.get_active_connection_count,
            drain=self.drain_connections,
        )

        logger.info("MCP Server created")

    def get_active_connection_count(self) -> int:
        return len(self.connections)

    async def drain_connections(self, timeout_s: float | None = None) -> None:
        await self._close_all_connections()

    async def _guard_websocket_start(self, websocket: WebSocket) -> bool:
        app = getattr(websocket, "app", None)
        if app is None:
            return True
        try:
            assert_may_start_work(app, "mcp.websocket")
            return True
        except HTTPException:
            await websocket.close(code=1013, reason="shutdown_draining")
            return False

    @staticmethod
    def _mask_secrets(text: str) -> str:
        """Best-effort masking of bearer/API keys in strings."""
        try:
            if not text:
                return text
            import re as _re
            text = _re.sub(r"(Bearer)\s+[A-Za-z0-9._\-~+/=]+", r"\1 ****", text, flags=_re.IGNORECASE)
            patterns = [
                r"(api[_-]?key)\s*[:=]\s*([^\s,;]+)",
                r"(token)\s*[:=]\s*([^\s,;]+)",
                r"(access[_-]?token)\s*[:=]\s*([^\s,;]+)",
                r"(refresh[_-]?token)\s*[:=]\s*([^\s,;]+)",
            ]
            for p in patterns:
                text = _re.sub(p, lambda m: f"{m.group(1)}=****", text, flags=_re.IGNORECASE)
            return text
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            return text

    async def initialize(self):
        """Initialize the server and all modules"""
        if self.initialized:
            logger.warning("Server already initialized")
            return
        async with self._initialize_lock:
            if self.initialized:
                logger.warning("Server already initialized")
                return
            logger.info("Initializing MCP Server")

            try:
                # Fail fast on insecure production configurations
                try:
                    _test_mode = _is_test_mode() or _is_explicit_pytest_runtime()
                    if not self.config.debug_mode and not _test_mode:
                        ok = validate_config()
                        if not ok:
                            raise RuntimeError("MCP configuration validation failed; refusing to start in production")
                except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                    # If validation fails, propagate to abort startup
                    raise
                # Warn if demo auth is enabled in a non-debug environment
                try:
                    if _env_flag_enabled("MCP_ENABLE_DEMO_AUTH") and not self.config.debug_mode:
                        logger.warning("MCP_ENABLE_DEMO_AUTH is enabled - for development only; DO NOT USE IN PRODUCTION")
                except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                    pass
                # Start module health monitoring
                await self.module_registry.start_health_monitoring()

                # Start metrics collection
                if self.config.metrics_enabled:
                    try:
                        await get_metrics_collector().start_collection()
                    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(f"MCP metrics collector start failed: {self._mask_secrets(str(e))}")

                # Register default modules (will be implemented when migrating modules)
                await self._register_default_modules()

                # Ensure default tool permissions exist (wildcard)
                await self._ensure_default_tool_permissions()

                # Start background tasks
                self._start_background_tasks()

                self.initialized = True
                logger.info("MCP Server initialized successfully")

            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Server initialization failed: {self._mask_secrets(str(e))}")
                raise

    async def _ensure_default_tool_permissions(self):
        """Seed wildcard tool permission tools.execute:* if missing."""
        try:
            pool = await get_db_pool()
            name = 'tools.execute:*'
            desc = 'Wildcard tool execution'
            if pool.pool:
                await pool.execute(
                    "INSERT INTO permissions (name, description, category) VALUES ($1, $2, $3) ON CONFLICT (name) DO NOTHING",
                    name, desc, 'tools'
                )
            else:
                row = await pool.fetchone("SELECT 1 FROM permissions WHERE name = ?", name)
                if not row:
                    await pool.execute(
                        "INSERT INTO permissions (name, description, category) VALUES (?, ?, ?)",
                        name, desc, 'tools'
                    )
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Seed wildcard tool permission failed: {self._mask_secrets(str(e))}")

    async def shutdown(self):
        """Gracefully shutdown the server"""
        logger.info("Shutting down MCP Server")

        # Signal shutdown
        self.shutdown_event.set()

        # Close all WebSocket connections
        await self._close_all_connections()

        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Shutdown modules
        await self.module_registry.shutdown_all()

        self.initialized = False
        logger.info("MCP Server shutdown complete")

    async def _register_default_modules(self):
        """Register default modules via config/env-driven loader"""
        # Autoload modules from YAML config and/or MCP_MODULES env var
        try:
            import importlib
            # Lazy import yaml to avoid hard dependency during tests if not installed
            try:
                import yaml  # type: ignore
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                yaml = None  # type: ignore

            modules_to_load = []

            # 1) YAML configuration
            cfg_path = os.getenv(
                "MCP_MODULES_CONFIG",
                "tldw_Server_API/Config_Files/mcp_modules.yaml",
            )
            if os.path.exists(cfg_path) and yaml is not None:
                try:
                    with open(cfg_path) as f:
                        data = yaml.safe_load(f) or {}
                    modules_cfg = data.get("modules", [])
                    if isinstance(modules_cfg, list):
                        modules_to_load.extend(modules_cfg)
                    logger.info(f"Loaded {len(modules_cfg)} MCP modules from {cfg_path}")
                except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Failed to read MCP modules YAML {cfg_path}: {e}")
            elif os.path.exists(cfg_path) and yaml is None:
                logger.warning(
                    f"MCP modules config found at {cfg_path} but PyYAML not installed; skipping"
                )

            # 2) Environment variable list (comma-separated)
            # Example: MCP_MODULES="media=tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module:MediaModule"
            env_spec = os.getenv("MCP_MODULES", "").strip()
            if env_spec:
                for item in [s for s in env_spec.split(",") if s.strip()]:
                    try:
                        mod_id, class_ref = item.split("=", 1)
                        modules_to_load.append({
                            "id": mod_id.strip(),
                            "class": class_ref.strip(),
                            "enabled": True,
                        })
                    except ValueError:
                        logger.warning(f"Invalid MCP_MODULES item format: '{item}'")

            # 3) Optional default: enable media module if flag is set and nothing else specified
            enable_media_flag = _env_flag_enabled("MCP_ENABLE_MEDIA_MODULE")
            test_mode = _is_test_mode()
            if enable_media_flag and not modules_to_load:
                default_media_path = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
                modules_to_load.append({
                    "id": "media",
                    "class": "tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module:MediaModule",
                    "enabled": True,
                    "name": "Media",
                    "version": "1.0.0",
                    "department": "media",
                    "settings": {
                        "db_path": default_media_path,
                        "cache_ttl": 300,
                    },
                })
                logger.info("MCP_ENABLE_MEDIA_MODULE=true; queuing MediaModule for registration")

            # 4) Test convenience: default-enable media module when TEST_MODE unless explicitly disabled
            if test_mode and not any(m.get("id") == "media" for m in modules_to_load):
                if os.getenv("MCP_ENABLE_MEDIA_MODULE", "").strip().lower() not in {"0", "false", "off", "no", "n"}:
                    default_media_path = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
                    modules_to_load.append({
                        "id": "media",
                        "class": "tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module:MediaModule",
                        "enabled": True,
                        "name": "Media",
                        "version": "1.0.0",
                        "department": "media",
                        "settings": {
                            "db_path": default_media_path,
                            "cache_ttl": 300,
                        },
                    })
                    logger.info("TEST_MODE auto-enabled MediaModule for deterministic tool catalogs")

            # 5) Filesystem module is enabled by default for workspace-bounded fs primitives.
            if os.getenv("MCP_ENABLE_FILESYSTEM_MODULE", "true").strip().lower() not in {"0", "false", "off", "no", "n"}:
                if not any(m.get("id") == "filesystem" for m in modules_to_load if isinstance(m, dict)):
                    modules_to_load.append({
                        "id": "filesystem",
                        "class": "tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module:FilesystemModule",
                        "enabled": True,
                        "name": "Filesystem",
                        "version": "1.0.0",
                        "department": "management",
                        "settings": {},
                    })
                    logger.info("MCP filesystem module enabled by default; queuing FilesystemModule for registration")

            # 6) Optional: Sandbox module (code interpreter) - disabled by default
            if _env_flag_enabled("MCP_ENABLE_SANDBOX_MODULE"):
                modules_to_load.append({
                    "id": "sandbox",
                    "class": "tldw_Server_API.app.core.MCP_unified.modules.implementations.sandbox_module:SandboxModule",
                    "enabled": True,
                    "name": "Sandbox Engine",
                    "version": "1.0.0",
                    "department": "management",
                    "settings": {},
                })
                logger.info("MCP_ENABLE_SANDBOX_MODULE=true; queuing SandboxModule for registration")

            # Register all specified modules
            from .modules.base import ModuleConfig  # Local import to avoid cycles
            for m in modules_to_load:
                if not m or not isinstance(m, dict):
                    continue
                if not m.get("enabled", True):
                    logger.info(f"Skipping disabled module: {m.get('id')}")
                    continue
                try:
                    module_id = m["id"]
                    class_ref = m["class"]
                    module_path, class_name = class_ref.split(":", 1)
                    # Restrict module autoload to allowed namespace for safety
                    allowed_prefixes = (
                        "tldw_Server_API.app.core.MCP_unified.modules.implementations",
                    )
                    if not any(module_path.startswith(p) for p in allowed_prefixes):
                        logger.warning(
                            f"Blocked module autoload for '{class_ref}': outside allowed namespace"
                        )
                        continue
                    cls = getattr(importlib.import_module(module_path), class_name)

                    mc = ModuleConfig(
                        name=m.get("name", module_id),
                        version=m.get("version", "1.0.0"),
                        description=m.get("description", ""),
                        department=m.get("department", "general"),
                        enabled=True,
                        timeout_seconds=m.get("timeout_seconds", self.config.module_timeout),
                        max_retries=m.get("max_retries", self.config.module_max_retries),
                        circuit_breaker_threshold=m.get("circuit_breaker_threshold", 5),
                        circuit_breaker_timeout=m.get("circuit_breaker_timeout", 60),
                        max_concurrent=m.get("max_concurrent", 20),
                        circuit_breaker_backoff_factor=m.get("circuit_breaker_backoff_factor", 2.0),
                        circuit_breaker_max_timeout=m.get("circuit_breaker_max_timeout", 300),
                        settings=_resolve_env_placeholders(m.get("settings", {})),
                    )
                    await self.module_registry.register_module(module_id, cls, mc)
                    logger.info(f"Registered MCP module: {module_id} ({class_ref})")
                except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Failed to register module {m}: {self._mask_secrets(str(e))}")

        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Default modules registration failed: {self._mask_secrets(str(e))}")

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Connection cleanup task
        task = asyncio.create_task(self._connection_cleanup_loop())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

        # Metrics collection task (optional)
        if self.config.metrics_enabled:
            task = asyncio.create_task(self._metrics_collection_loop())
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

        # Session cleanup task
        task = asyncio.create_task(self._session_cleanup_loop())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

    async def _connection_cleanup_loop(self):
        """Periodically clean up stale connections"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                raise
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error in connection cleanup: {e}")

    async def _cleanup_stale_connections(self):
        """Remove stale WebSocket connections"""
        async with self.connection_lock:
            stale_connections = []
            current_time = datetime.now(timezone.utc)

            for conn_id, connection in self.connections.items():
                # Check for stale connections (no activity for 5 minutes)
                if (current_time - connection.last_activity).total_seconds() > 300 or connection.error_count > 10:
                    stale_connections.append(conn_id)

            # Close stale connections
            for conn_id in stale_connections:
                logger.info(f"Closing stale connection: {conn_id}")
                connection = self.connections[conn_id]
                await connection.close(code=1001, reason="Connection timeout")
                del self.connections[conn_id]
            # Update connection gauge
            with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                get_metrics_collector().update_connection_count("websocket", len(self.connections))

    async def _metrics_collection_loop(self):
        """Periodically collect and log metrics"""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._log_metrics()
            except asyncio.CancelledError:
                raise
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error in metrics collection: {e}")

    async def _log_metrics(self):
        """Log server metrics"""
        metrics = await self.get_metrics()
        logger.info(f"Server metrics: {metrics}")

    # ------------------------
    # Session helpers
    # ------------------------

    async def _session_cleanup_loop(self):
        """Periodically evict stale sessions."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(60)
                await self._cleanup_stale_sessions()
            except asyncio.CancelledError:
                raise
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error in session cleanup: {e}")

    async def _cleanup_stale_sessions(self):
        ttl = timedelta(minutes=max(1, int(self.config.session_ttl_minutes)))
        cutoff = datetime.now(timezone.utc) - ttl
        async with self.session_lock:
            to_delete = [sid for sid, s in self.sessions.items() if s.last_activity < cutoff]
            for sid in to_delete:
                self.sessions.pop(sid, None)

    async def _get_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> SessionData:
        workspace_key = _normalize_optional_text(workspace_id)
        cwd_key = _normalize_optional_text(cwd)
        async with self.session_lock:
            s = self.sessions.get(session_id)
            if s is None:
                # Enforce global cap
                if len(self.sessions) >= max(1, int(self.config.max_sessions)):
                    # Evict oldest
                    oldest_id = min(self.sessions, key=lambda k: self.sessions[k].last_activity)
                    self.sessions.pop(oldest_id, None)
                s = SessionData(
                    session_id=session_id,
                    user_id=user_id,
                    workspace_id=workspace_key,
                    cwd=cwd_key,
                )
                # Respect configured max URIs per session
                s.uris_seen = deque(maxlen=max(1, int(self.config.max_session_uris)))
                self.sessions[session_id] = s
            else:
                # Enforce session ownership binding
                if s.user_id is None:
                    if user_id is not None:
                        s.user_id = user_id
                else:
                    if user_id is None or str(s.user_id) != str(user_id):
                        raise PermissionError("Session is bound to a different user")
                existing_context_bound = s.workspace_id is not None or s.cwd is not None
                incoming_context_bound = workspace_key is not None or cwd_key is not None
                if not existing_context_bound and incoming_context_bound:
                    s.workspace_id = workspace_key
                    s.cwd = cwd_key
                elif existing_context_bound:
                    if s.workspace_id != workspace_key or s.cwd != cwd_key:
                        raise PermissionError("Workspace context mismatch")
            s.touch()
            return s

    def _merge_safe_config(self, current: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        """Merge allowlisted safe config keys with clamping."""
        if not incoming:
            return current
        out = dict(current)
        # Allowlist keys
        allow_int = {
            "snippet_length": (1, 2000),
            "max_tokens": (500, 200000),
            "sibling_window": (0, 10),
            "chars_per_token": (1, 20),
            "maxSessionUris": (10, 5000),
        }
        allow_bool = {"aliasMode", "compactShape"}
        allow_str = {"order_by"}
        for k, v in incoming.items():
            if k in allow_bool and isinstance(v, bool) or k in allow_str and isinstance(v, str):
                out[k] = v
            elif k in allow_int and isinstance(v, (int, float)):
                lo, hi = allow_int[k]
                out[k] = int(max(lo, min(int(v), hi)))
        return out

    async def handle_websocket(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        api_key: Optional[str] = None,
        mcp_session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        cwd: Optional[str] = None,
    ):
        """
        Handle a WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            client_id: Optional client identifier
            auth_token: Optional authentication token
        """
        connection_id = f"ws_{client_id or 'anonymous'}_{datetime.now().timestamp()}"
        user_id = None
        stable_session_id = _normalize_optional_text(mcp_session_id)
        workspace_key = _normalize_optional_text(workspace_id)
        cwd_key = _normalize_optional_text(cwd)

        controller = get_ip_access_controller()
        metadata: dict[str, Any] = {}
        forwarded_for = websocket.headers.get("x-forwarded-for") or websocket.headers.get("X-Forwarded-For")
        real_ip = websocket.headers.get("x-real-ip") or websocket.headers.get("X-Real-IP")
        raw_remote_ip = None
        try:
            raw_remote_ip = getattr(websocket.client, "host", None) or (
                websocket.client[0] if isinstance(websocket.client, (list, tuple)) else None
            )
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            raw_remote_ip = None

        resolved_ip = controller.resolve_client_ip(raw_remote_ip, forwarded_for, real_ip)
        # Test harness mapping and bypass: allow WS in pytest/TEST_MODE and map 'testclient' to loopback
        try:
            _is_test_env = bool(_is_explicit_pytest_runtime() or _is_test_mode())
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            _is_test_env = False
        if resolved_ip == "testclient" or resolved_ip is None and _is_test_env:
            resolved_ip = "127.0.0.1"

        if not controller.is_allowed(resolved_ip) and not _is_test_env:
            with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                logger.warning(
                    "Rejecting MCP WebSocket connection from disallowed IP",
                    extra={"audit": True, "ip": resolved_ip or "unknown", "client_id": client_id},
                )
            await websocket.close(code=1008, reason="IP not allowed")
            return

        client_ip = resolved_ip or "unknown"

        try:
            enforce_client_certificate_headers(websocket.headers, remote_addr=raw_remote_ip)
        except HTTPException:
            await websocket.close(code=1008, reason="Client certificate required")
            return

        # Origin validation (enforce when ws_allowed_origins configured)
        try:
            allowed = list(self.config.ws_allowed_origins or [])
            if allowed:
                # Allow wildcard '*' if explicitly configured
                origin = websocket.headers.get("origin") or websocket.headers.get("Origin") or ""
                if "*" not in allowed:
                    # If no Origin header provided (e.g., non-browser TestClient), allow by default
                    if origin and origin not in allowed:
                        await websocket.close(code=1008, reason="Origin not allowed")
                        return
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            # Fail-safe: do not block if config parsing fails
            pass

        # Gate query parameter auth tokens if disabled by config
        if (auth_token or api_key) and not self.config.ws_allow_query_auth:
            try:
                # Emit a deprecation warning; ignore query tokens unless explicitly allowed
                logger.warning("WS query-parameter authentication is disabled; pass Authorization bearer token or X-API-KEY header instead")
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                pass
            auth_token = None
            api_key = None

        # Prefer headers/subprotocol for auth if present
        try:
            # Authorization: Bearer <token>
            _authz = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
            if _authz and _authz.lower().startswith("bearer "):
                auth_token = _authz.split(" ", 1)[1].strip()
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            # X-API-KEY header
            _xkey = websocket.headers.get("x-api-key") or websocket.headers.get("X-API-KEY")
            if _xkey:
                api_key = _xkey.strip()
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            # Sec-WebSocket-Protocol: bearer,<token>
            _proto = websocket.headers.get("sec-websocket-protocol") or websocket.headers.get("Sec-WebSocket-Protocol")
            if _proto and not auth_token:
                # pick first value that looks like bearer,<token>
                parts = [p.strip() for p in _proto.split(",")]
                if len(parts) >= 2 and parts[0].lower() == "bearer" and parts[1]:
                    auth_token = parts[1]
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            pass

        # Authenticate if token provided (prefer AuthNZ JWT, then MCP JWT)
        if auth_token:
            ok = False
            authnz_token_failed = False
            try:
                # Try AuthNZ JWT first for consistency with HTTP endpoints
                from starlette.requests import Request as _Request

                from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import verify_jwt_and_fetch_user

                scope = {
                    "type": "http",
                    "method": "GET",
                    "path": "/api/v1/mcp/ws",
                    "headers": [
                        (k.encode("latin-1"), v.encode("latin-1"))
                        for k, v in websocket.headers.items()
                    ],
                }
                try:
                    client = websocket.client
                    if isinstance(client, (list, tuple)) and len(client) >= 2:
                        scope["client"] = (client[0], client[1])
                    elif client is not None and getattr(client, "host", None) is not None:
                        scope["client"] = (client.host, getattr(client, "port", 0))
                except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                    pass

                req = _Request(scope)
                user = await verify_jwt_and_fetch_user(req, auth_token)
                user_id = str(getattr(user, "id", None) or "")
                ok = bool(user_id)
                if ok:
                    logger.info(f"WebSocket authenticated for user (AuthNZ JWT): {user_id}")
                    roles = list(getattr(user, "roles", []) or [])
                    perms = list(getattr(user, "permissions", []) or [])
                    if roles:
                        metadata["roles"] = roles
                    if perms:
                        metadata["permissions"] = perms
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"AuthNZ JWT auth failed: {self._mask_secrets(str(e))}")
                if _is_authnz_access_token(auth_token):
                    authnz_token_failed = True
                    if not api_key:
                        await websocket.close(code=1008, reason="Authentication failed")
                        return
            # Try MCP JWT only when the token is not an AuthNZ access token
            if not ok and not authnz_token_failed:
                try:
                    token_data = self.jwt_manager.verify_token(auth_token)
                    user_id = token_data.sub
                    ok = True
                    logger.info(f"WebSocket authenticated for user (MCP JWT): {user_id}")
                    if token_data.roles:
                        metadata["roles"] = token_data.roles
                    if token_data.permissions:
                        metadata["permissions"] = token_data.permissions
                except HTTPException as _e:
                    logger.debug(f"MCP JWT auth failed: {self._mask_secrets(str(_e))}")
            if auth_token and not ok and not api_key:
                await websocket.close(code=1008, reason="Authentication failed")
                return

        # API key auth (optional)
        if api_key and not user_id:
            try:
                from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
                mgr = await get_api_key_manager()
                # Enforce allowed IPs for API keys by forwarding resolved client IP
                info = await mgr.validate_api_key(api_key, ip_address=client_ip)
                if info and info.get('user_id'):
                    user_id = str(info['user_id'])
                    # Attach org/team context
                    if info.get('org_id') is not None:
                        metadata['org_id'] = info.get('org_id')
                    if info.get('team_id') is not None:
                        metadata['team_id'] = info.get('team_id')
                    roles = metadata.setdefault('roles', [])
                    if 'api_client' not in roles:
                        roles.append('api_client')
                    try:
                        scopes = _extract_api_key_permissions(info)
                        if scopes:
                            metadata["api_key_scopes"] = list(scopes)
                            metadata["auth_via"] = "api_key"
                            perms = metadata.setdefault('permissions', [])
                            for scope in scopes:
                                if scope not in perms:
                                    perms.append(scope)
                    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                        pass
                    logger.info(f"WebSocket authenticated via API key for user: {user_id}")
                else:
                    await websocket.close(code=1008, reason="Authentication failed")
                    return
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"WebSocket API key authentication failed: {self._mask_secrets(str(e))}")
                await websocket.close(code=1008, reason="Authentication failed")
                return

        # Optionally require authentication for WS (production hardening)
        ws_auth_required = self.config.ws_auth_required
        try:
            if _is_test_env:
                # Honor test env override to avoid stale cached config in pytest.
                import os as _os
                override = _os.getenv("MCP_WS_AUTH_REQUIRED")
                if override is not None:
                    override_val = _is_truthy(override)
                    if self.config.ws_auth_required == self._ws_auth_required_initial:
                        ws_auth_required = override_val
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            pass
        if ws_auth_required and not user_id:
            await websocket.close(code=1008, reason="Authentication required")
            return

        if stable_session_id:
            metadata["session_id"] = stable_session_id
        if workspace_key:
            metadata["workspace_id"] = workspace_key
        if cwd_key:
            metadata["cwd"] = cwd_key

        if not await self._guard_websocket_start(websocket):
            return

        try:
            if stable_session_id:
                await self._get_or_create_session(
                    stable_session_id,
                    user_id=user_id,
                    workspace_id=workspace_key,
                    cwd=cwd_key,
                )
        except PermissionError as exc:
            await websocket.close(code=1008, reason=str(exc))
            return

        # Check connection limits (global and per-IP)
        async with self.connection_lock:
            if len(self.connections) >= self.config.ws_max_connections:
                logger.warning("Maximum WebSocket connections reached")
                await websocket.close(code=1013, reason="Server at capacity")
                return
            # Enforce per-IP cap if configured
            if self.config.ws_max_connections_per_ip > 0:
                count = self._ip_connection_counts.get(client_ip, 0)
                if count >= self.config.ws_max_connections_per_ip:
                    # Record rejection metric
                    try:
                        bucket = self._ip_bucket(client_ip)
                        get_metrics_collector().record_ws_rejection("per_ip_cap", bucket)
                    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                        pass
                    await websocket.close(code=1013, reason="Too many connections from IP")
                    return
            # Reserve a slot for this IP before accepting to avoid race conditions
            self._ip_connection_counts[client_ip] = self._ip_connection_counts.get(client_ip, 0) + 1
            # Do not call websocket.accept() here; WebSocketStream.start() will accept after we finish checks.

            # Create connection object
            connection = WebSocketConnection(
                websocket=websocket,
                connection_id=connection_id,
                client_id=client_id,
                user_id=user_id,
                metadata=metadata,
                session_id=stable_session_id or connection_id,
                workspace_id=workspace_key,
                cwd=cwd_key,
            )

            self.connections[connection_id] = connection
            # per-IP count already reserved; nothing to do here
            # Update connection gauge
            with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                get_metrics_collector().update_connection_count("websocket", len(self.connections))

        logger.bind(connection_id=connection_id, user_id=user_id, client_id=client_id, client_ip=client_ip).info(
            f"WebSocket connected: {connection_id} (client={client_id}, user={user_id}, ip={client_ip})"
        )

        # Initialize unified WS lifecycle (ping/idle/error) and accept the socket
        stream: Optional[WebSocketStream] = WebSocketStream(
            websocket,
            heartbeat_interval_s=float(self.config.ws_ping_interval) if self.config.ws_ping_interval else None,
            idle_timeout_s=float(self.config.ws_idle_timeout_seconds) if self.config.ws_idle_timeout_seconds else None,
            close_on_done=True,
            labels={"component": "mcp", "endpoint": "mcp_ws"},
        )
        try:
            await stream.start()
            # Handle messages (domain JSON-RPC payloads go through send_json; no event-wrapping)
            await self._handle_websocket_messages(connection, stream)

        except WebSocketDisconnect:
            logger.bind(connection_id=connection_id).info(f"WebSocket disconnected: {connection_id}")
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(connection_id=connection_id).error(f"WebSocket error for {connection_id}: {e}")
            # Preserve JSON-RPC transport semantics: do not emit non-JSON-RPC error frames here.
            # Close the socket with 1011 (internal error).
            with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                await connection.close(code=1011, reason="Internal error")
            with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                get_metrics_collector().record_connection_error("websocket", "exception")
        finally:
            # Stop WS background tasks (ping/idle loops) to avoid leaks
            if stream is not None:
                with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                    await stream.stop()
            # Remove connection
            async with self.connection_lock:
                if connection_id in self.connections:
                    del self.connections[connection_id]
                # Decrement per-IP count
                try:
                    if client_ip in self._ip_connection_counts and self._ip_connection_counts[client_ip] > 0:
                        self._ip_connection_counts[client_ip] -= 1
                        if self._ip_connection_counts[client_ip] == 0:
                            del self._ip_connection_counts[client_ip]
                except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                    pass

            logger.bind(connection_id=connection_id).info(f"WebSocket cleanup complete: {connection_id}")
            # Update connection gauge
            with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                get_metrics_collector().update_connection_count("websocket", len(self.connections))

    async def _handle_websocket_messages(self, connection: WebSocketConnection, stream: WebSocketStream):
        """Handle incoming WebSocket messages"""
        while True:
            sess: Optional[SessionData] = None
            # Receive message
            try:
                data = await connection.receive_json()
                # Mark activity for idle timer on receive
                with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                    stream.mark_activity()
            except json.JSONDecodeError as e:
                await stream.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                })
                continue

            # Check message size
            message_size = len(json.dumps(data))
            if message_size > self.config.ws_max_message_size:
                await stream.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Message too large"
                    },
                    "id": data.get("id") if isinstance(data, dict) else None
                })
                continue

            # Handle ping/pong
            if isinstance(data, dict) and data.get("type") == "pong":
                continue

            # Per-session rate limit: requests per window
            try:
                now_ts = datetime.now(timezone.utc).timestamp()
                connection.request_times.append(now_ts)
                window = max(1, int(self.config.ws_session_rate_limit_window_seconds))
                threshold = max(1, int(self.config.ws_session_rate_limit_count))
                # Prune
                while connection.request_times and (now_ts - connection.request_times[0]) > window:
                    connection.request_times.popleft()
                if len(connection.request_times) > threshold:
                    await stream.send_json({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32002,
                            "message": "Session rate limit exceeded"
                        },
                        "id": data.get("id") if isinstance(data, dict) else None
                    })
                    with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                        get_metrics_collector().record_ws_session_closure("session_rate")
                    # Close with 1013 (try again later), matching prior behavior
                    try:
                        await stream.ws.close(code=1013, reason="Session rate limit exceeded")
                    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                        with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                            await connection.close(code=1013, reason="Session rate limit exceeded")
                    break
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                pass

            # Ensure session exists and update with client/safe config if applicable
            try:
                sess = await self._get_or_create_session(
                    connection.session_id,
                    connection.user_id,
                    workspace_id=connection.workspace_id,
                    cwd=connection.cwd,
                )
                # If this is initialize, capture clientInfo and optional config
                if isinstance(data, dict) and data.get("method") == "initialize":
                    try:
                        params = data.get("params") or {}
                        client_info = params.get("clientInfo") or {}
                        if isinstance(client_info, dict):
                            sess.client_info.update(client_info)
                        # Optional config param for WS (either dict or base64-encoded JSON)
                        cfg = params.get("config")
                        safe_incoming: dict[str, Any] = {}
                        if isinstance(cfg, dict):
                            safe_incoming = cfg
                        elif isinstance(cfg, str):
                            import base64
                            import json as _json
                            try:
                                decoded = base64.b64decode(cfg).decode("utf-8")
                                safe_incoming = _json.loads(decoded)
                            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                                safe_incoming = {}
                        if safe_incoming:
                            sess.safe_config = self._merge_safe_config(sess.safe_config, safe_incoming)
                    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                        pass
                sess.touch()
            except PermissionError:
                # Session ownership mismatch - return authorization error and close
                with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                    await stream.send_json({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32001,
                            "message": "Session is bound to a different user"
                        },
                        "id": data.get("id") if isinstance(data, dict) else None
                    })
                with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
                    await stream.ws.close(code=1008, reason="Session ownership mismatch")
                break
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                pass

            # Create request context
            context_metadata = dict(connection.metadata)
            try:
                if sess and sess.workspace_id:
                    context_metadata["workspace_id"] = sess.workspace_id
                if sess and sess.cwd:
                    context_metadata["cwd"] = sess.cwd
                context_metadata["mcp_policy_context_enabled"] = is_mcp_hub_policy_enforcement_enabled()
                if sess and sess.safe_config:
                    context_metadata["safe_config"] = dict(sess.safe_config)
                if sess:
                    context_metadata["seen_uris"] = list(sess.uris_seen)
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                pass
            context = RequestContext(
                request_id=data.get("id", "unknown") if isinstance(data, dict) else "unknown",
                user_id=connection.user_id,
                client_id=connection.client_id,
                session_id=connection.session_id,
                metadata=context_metadata
            )

            # Process MCP request (supports single, notification, and batch)
            try:
                response = await self.protocol.process_request(data, context)
                # Persist seen URIs back into session (if updated by tools)
                try:
                    if sess and isinstance(context.metadata, dict):
                        seen = context.metadata.get("seen_uris")
                        if isinstance(seen, list):
                            max_len = max(1, int(self.config.max_session_uris))
                            for uri in seen:
                                if isinstance(uri, str) and uri:
                                    sess.add_seen_uri(uri, max_len)
                except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                    pass
                if response is None:
                    # Notification - no reply
                    continue
                if isinstance(response, list):
                    await stream.send_json([r.model_dump() for r in response])
                else:
                    await stream.send_json(response.model_dump())
            except RateLimitExceeded as e:
                await stream.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32002,
                        "message": f"Rate limit exceeded. Retry after {e.retry_after} seconds",
                        "data": {
                            "hint": "Reduce request frequency or wait before retrying."
                        }
                    },
                    "id": data.get("id") if isinstance(data, dict) else None
                })
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Error processing WebSocket message: {self._mask_secrets(str(e))}")
                await stream.send_json({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": "Internal error"
                    },
                    "id": data.get("id") if isinstance(data, dict) else None
                })

    def _ip_bucket(self, ip: str) -> str:
        """Return a coarse bucket label for an IP to avoid high-cardinality metrics."""
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_loopback:
                return "loopback"
            if ip_obj.is_private:
                return "private"
            if ip_obj.is_link_local:
                return "link_local"
            if ip_obj.is_reserved:
                return "reserved"
            return "public"
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            return "unknown"

    async def handle_http_request(
        self,
        request: MCPRequest,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> MCPResponse:
        """
        Handle an HTTP MCP request.

        Args:
            request: MCP request
            client_id: Optional client identifier
            user_id: Optional user identifier (from auth)

        Returns:
            MCP response
        """
        # Pull session_id and safe_config from metadata when present
        session_id: Optional[str] = None
        safe_cfg: dict[str, Any] = {}
        try:
            if metadata:
                raw_sid = metadata.get("session_id")
                if isinstance(raw_sid, str) and raw_sid:
                    session_id = raw_sid
                sc = metadata.get("safe_config")
                if isinstance(sc, dict):
                    safe_cfg = sc
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            session_id = None
            safe_cfg = {}
        # Always clamp safe_config with allowlist, even without a session.
        try:
            if safe_cfg:
                safe_cfg = self._merge_safe_config({}, safe_cfg)
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            safe_cfg = {}

        # If we have a session id, ensure session exists and merge safe config
        sess: Optional[SessionData] = None
        try:
            if session_id:
                sess = await self._get_or_create_session(session_id, user_id)
                if safe_cfg:
                    sess.safe_config = self._merge_safe_config(sess.safe_config, safe_cfg)
                # If initialize, capture clientInfo
                if request.method == "initialize" and isinstance(request.params, dict):
                    ci = (request.params or {}).get("clientInfo")
                    if isinstance(ci, dict):
                        sess.client_info.update(ci)
                sess.touch()
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            pass

        # Create request context
        metadata_map = dict(metadata or {})
        try:
            if sess and sess.safe_config:
                metadata_map["safe_config"] = dict(sess.safe_config)
            elif safe_cfg:
                metadata_map["safe_config"] = safe_cfg
            if sess:
                metadata_map["seen_uris"] = list(sess.uris_seen)
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            pass
        context = RequestContext(
            request_id=request.id or "http_request",
            user_id=user_id,
            client_id=client_id,
            session_id=session_id,
            metadata=metadata_map
        )

        # Process request
        try:
            response = await self.protocol.process_request(request, context)
            try:
                if sess and isinstance(context.metadata, dict):
                    seen = context.metadata.get("seen_uris")
                    if isinstance(seen, list):
                        max_len = max(1, int(self.config.max_session_uris))
                        for uri in seen:
                            if isinstance(uri, str) and uri:
                                sess.add_seen_uri(uri, max_len)
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                pass
            return response
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": f"Rate limit exceeded. Retry after {e.retry_after} seconds",
                    "hint": "Throttle tool calls or wait for the cooldown before retrying."
                }
            ) from e
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error processing HTTP request: {self._mask_secrets(str(e))}")
            raise HTTPException(status_code=500, detail="Internal server error") from e

    async def handle_http_batch(
        self,
        requests: list[MCPRequest],
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None
    ) -> Optional[list[MCPResponse]]:
        """
        Handle a batch of HTTP MCP requests with consistent session semantics.
        """
        # Pull session_id and safe_config from metadata when present
        session_id: Optional[str] = None
        safe_cfg: dict[str, Any] = {}
        try:
            if metadata:
                raw_sid = metadata.get("session_id")
                if isinstance(raw_sid, str) and raw_sid:
                    session_id = raw_sid
                sc = metadata.get("safe_config")
                if isinstance(sc, dict):
                    safe_cfg = sc
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            session_id = None
            safe_cfg = {}
        # Clamp safe_config with allowlist
        try:
            if safe_cfg:
                safe_cfg = self._merge_safe_config({}, safe_cfg)
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            safe_cfg = {}

        # If we have a session id, ensure session exists and merge safe config
        sess: Optional[SessionData] = None
        try:
            if session_id:
                sess = await self._get_or_create_session(session_id, user_id)
                if safe_cfg:
                    sess.safe_config = self._merge_safe_config(sess.safe_config, safe_cfg)
                # Capture clientInfo from any initialize request in the batch
                for req in requests:
                    try:
                        if req.method == "initialize" and isinstance(req.params, dict):
                            ci = (req.params or {}).get("clientInfo")
                            if isinstance(ci, dict):
                                sess.client_info.update(ci)
                    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                        continue
                sess.touch()
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            pass

        # Create request context
        metadata_map = dict(metadata or {})
        try:
            if sess and sess.safe_config:
                metadata_map["safe_config"] = dict(sess.safe_config)
            elif safe_cfg:
                metadata_map["safe_config"] = safe_cfg
            if sess:
                metadata_map["seen_uris"] = list(sess.uris_seen)
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
            pass
        context = RequestContext(
            request_id="http_batch",
            user_id=user_id,
            client_id=client_id,
            session_id=session_id,
            metadata=metadata_map
        )

        # Process batch request
        try:
            response = await self.protocol.process_request(requests, context)
            try:
                if sess and isinstance(context.metadata, dict):
                    seen = context.metadata.get("seen_uris")
                    if isinstance(seen, list):
                        max_len = max(1, int(self.config.max_session_uris))
                        for uri in seen:
                            if isinstance(uri, str) and uri:
                                sess.add_seen_uri(uri, max_len)
            except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
                pass
            if response is None:
                return None
            if isinstance(response, MCPResponse):
                return [response]
            return response
        except RateLimitExceeded as e:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": f"Rate limit exceeded. Retry after {e.retry_after} seconds",
                    "hint": "Throttle tool calls or wait for the cooldown before retrying."
                }
            ) from e
        except _MCP_SERVER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error processing HTTP batch request: {self._mask_secrets(str(e))}")
            raise HTTPException(status_code=500, detail="Internal server error") from e

    async def _close_all_connections(self):
        """Close all WebSocket connections"""
        async with self.connection_lock:
            tasks = []
            for connection in self.connections.values():
                tasks.append(
                    connection.close(code=1001, reason="Server shutdown")
                )

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            self.connections.clear()

    async def get_status(self) -> dict[str, Any]:
        """Get server status"""
        uptime = (datetime.now(timezone.utc) - self.startup_time).total_seconds()

        # Get module health
        health_results = await self.module_registry.check_all_health()

        # Get connection stats
        connection_stats = {
            "total": len(self.connections),
            "authenticated": sum(1 for c in self.connections.values() if c.user_id),
            "anonymous": sum(1 for c in self.connections.values() if not c.user_id)
        }

        return {
            "status": "healthy" if self.initialized else "initializing",
            "version": "3.0.0",
            "uptime_seconds": uptime,
            "connections": connection_stats,
            "modules": {
                "total": len(health_results),
                "healthy": sum(1 for h in health_results.values() if h.is_healthy),
                "degraded": sum(1 for h in health_results.values() if h.is_operational and not h.is_healthy),
                "unhealthy": sum(1 for h in health_results.values() if not h.is_operational)
            }
        }

    async def get_metrics(self) -> dict[str, Any]:
        """Get server metrics"""
        # Collect module metrics
        module_metrics = {}
        modules = await self.module_registry.get_all_modules()

        for module_id, module in modules.items():
            metrics = module.get_metrics()
            module_metrics[module_id] = {
                "requests": metrics.total_requests,
                "errors": metrics.failed_requests,
                "error_rate": metrics.error_rate,
                "avg_latency_ms": metrics.avg_latency_ms
            }

        # Connection metrics
        total_messages = sum(c.message_count for c in self.connections.values())
        total_errors = sum(c.error_count for c in self.connections.values())

        return {
            "connections": {
                "active": len(self.connections),
                "total_messages": total_messages,
                "total_errors": total_errors
            },
            "modules": module_metrics
        }


# Singleton instance management
_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get or create MCP server singleton"""
    global _server
    if _server is None:
        _server = MCPServer()
    return _server


async def reset_mcp_server() -> None:
    """Reset MCP server singleton for test environments."""
    global _server
    if _server is not None:
        with suppress(_MCP_SERVER_NONCRITICAL_EXCEPTIONS):
            await _server.shutdown()
    _server = None
    try:
        from .modules.registry import reset_module_registry
        await reset_module_registry()
    except _MCP_SERVER_NONCRITICAL_EXCEPTIONS:
        pass


@asynccontextmanager
async def lifespan(app):
    """
    FastAPI lifespan manager for server initialization and shutdown.

    Usage in main.py:
    ```python
    from tldw_Server_API.app.core.MCP_unified.server import lifespan

    app = FastAPI(lifespan=lifespan)
    ```
    """
    # Startup
    server = get_mcp_server()
    await server.initialize()
    logger.info("MCP Server started")

    yield

    # Shutdown
    await server.shutdown()
    logger.info("MCP Server stopped")
