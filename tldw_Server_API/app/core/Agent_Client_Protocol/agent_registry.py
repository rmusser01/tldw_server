"""YAML-based global agent registry for ACP.

Loads agent definitions from Config_Files/agents.yaml and provides
runtime availability detection (binary on PATH, API keys set).
Supports dynamic registration via REST API with SQLite persistence.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


ACP_COMPATIBILITY_DOCS_URL = "/docs-static/Development/ACP_Compatibility_Matrix.md"
AgentEntrypointStrategy = Literal[
    "native_acp",
    "adapter_acp",
    "documented_candidate",
    "custom_template",
]
AgentProbeState = Literal["ready_to_probe", "blocked", "custom_template", "documented_only"]
_SHELL_BUILTIN_COMMANDS = frozenset({"alias", "cd", "export", "set", "source", "unset"})


def _coerce_entrypoint_strategy(value: Any) -> AgentEntrypointStrategy:
    """Return a valid entrypoint strategy, defaulting unknown input conservatively."""
    if value in {"native_acp", "adapter_acp", "documented_candidate", "custom_template"}:
        return value
    return "documented_candidate"


@dataclass(frozen=True)
class AgentEntrypointClassification:
    """Deterministic ACP entrypoint readiness without launching the agent."""
    profile_key: str
    entrypoint_strategy: AgentEntrypointStrategy
    probe_state: AgentProbeState
    acp_command: str
    acp_args: tuple[str, ...]
    primary_blocker: str | None
    blockers: tuple[str, ...]
    status_message: str
    docs_url: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "acp_args", tuple(self.acp_args))
        object.__setattr__(self, "blockers", tuple(self.blockers))

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_key": self.profile_key,
            "entrypoint_strategy": self.entrypoint_strategy,
            "probe_state": self.probe_state,
            "acp_command": self.acp_command,
            "acp_args": list(self.acp_args),
            "primary_blocker": self.primary_blocker,
            "blockers": list(self.blockers),
            "status_message": self.status_message,
            "docs_url": self.docs_url,
        }


@dataclass
class AgentRegistryEntry:
    """A single agent entry from the registry."""
    type: str
    name: str
    description: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    requires_api_key: str | None = None
    default: bool = False
    install_instructions: list[str] = field(default_factory=list)
    docs_url: str | None = None
    support_state: Literal[
        "supported",
        "supported_with_caveats",
        "experimental",
        "documented_unverified",
        "unsupported",
    ] = "documented_unverified"
    verification_level: Literal[
        "documented_only",
        "stub_smoke_tested",
        "live_e2e_tested",
        "sandbox_tested",
        "production_supported",
    ] = "documented_only"
    compatibility_notes: str = "Configured locally; live-agent ACP compatibility has not been certified."
    compatibility_docs_url: str | None = ACP_COMPATIBILITY_DOCS_URL
    entrypoint_strategy: AgentEntrypointStrategy = "documented_candidate"
    acp_command: str = ""
    acp_args: list[str] = field(default_factory=list)
    adapter_source: str | None = None
    adapter_docs_url: str | None = None
    certification_blocker: str | None = None

    # Protocol adapter fields (new for agent workspace harness)
    protocol: Literal["stdio", "mcp", "openai_tool_use"] = "stdio"
    tool_execution_mode: Literal["agent_side", "server_side", "hybrid"] = "agent_side"
    mcp_transport: Literal["stdio", "sse", "streamable_http"] = "stdio"
    api_base_url: str | None = None
    model: str | None = None
    tools_from: Literal["auto", "static", "none"] = "auto"
    sandbox: Literal["required", "optional", "none"] = "none"
    trust_level: Literal["untrusted", "standard", "trusted"] = "standard"

    # MCP orchestration fields (Phase B)
    mcp_orchestration: Literal["agent_driven", "llm_driven"] = "agent_driven"
    mcp_entry_tool: str = "execute"
    mcp_structured_response: bool = False
    mcp_llm_provider: str | None = None
    mcp_llm_model: str | None = None
    mcp_max_iterations: int = 20
    mcp_refresh_tools: bool = False

    def check_availability(self) -> dict[str, Any]:
        """Check runtime availability of this agent."""
        result: dict[str, Any] = {
            "type": self.type,
            "name": self.name,
            "description": self.description,
            "support_state": self.support_state,
            "verification_level": self.verification_level,
            "compatibility_notes": self.compatibility_notes,
            "compatibility_docs_url": self.compatibility_docs_url,
        }

        # Check binary
        if self.command:
            which_result = shutil.which(self.command)
            result["binary_found"] = which_result is not None
            if which_result:
                result["binary_path"] = which_result
        else:
            result["binary_found"] = True  # No binary required (e.g., "custom")

        # Check API key
        if self.requires_api_key:
            result["api_key_set"] = bool(os.getenv(self.requires_api_key))
            if not result["api_key_set"]:
                result["missing_api_key"] = self.requires_api_key
        else:
            result["api_key_set"] = True

        # Overall status
        if not result.get("binary_found"):
            result["status"] = "unavailable"
        elif not result.get("api_key_set"):
            result["status"] = "requires_setup"
        else:
            result["status"] = "available"

        result["is_configured"] = result["status"] == "available"
        return result


def classify_agent_entrypoint(
    entry: AgentRegistryEntry,
    *,
    command_resolver: Callable[[str], str | None] = shutil.which,
    env_getter: Callable[[str], str | None] = os.getenv,
) -> AgentEntrypointClassification:
    """Classify ACP entrypoint readiness without starting the agent."""
    strategy = entry.entrypoint_strategy
    acp_command = entry.acp_command
    acp_args = list(entry.acp_args)
    docs_url = entry.compatibility_docs_url or entry.docs_url

    def classification(
        probe_state: AgentProbeState,
        *,
        blockers: tuple[str, ...] = (),
        status_message: str,
        command: str = acp_command,
        args: list[str] | None = None,
    ) -> AgentEntrypointClassification:
        normalized_blockers = tuple(dict.fromkeys(str(blocker) for blocker in blockers if blocker))
        primary_blocker = normalized_blockers[0] if normalized_blockers else None
        return AgentEntrypointClassification(
            profile_key=entry.type,
            entrypoint_strategy=strategy,
            probe_state=probe_state,
            acp_command=command,
            acp_args=tuple(args if args is not None else acp_args),
            primary_blocker=primary_blocker,
            blockers=normalized_blockers,
            status_message=status_message,
            docs_url=docs_url,
        )

    def blocked_status(blockers: list[str]) -> str:
        """Return a readable status for one or more deterministic blockers."""
        messages = {
            "entrypoint_strategy_missing": "Registry entry has no explicit ACP stdio command.",
            "shell_builtin_collision": "Configured ACP command matches a shell builtin or alias-like value.",
            "credentials_missing": "Required API key or credential environment variable is missing.",
            "adapter_missing": "Configured ACP adapter command is not available on PATH.",
            "binary_missing": "Configured ACP entrypoint command is not available on PATH.",
        }
        if not blockers:
            return "ACP entrypoint readiness is blocked."
        status_message = messages.get(blockers[0], "ACP entrypoint readiness is blocked.")
        if len(blockers) > 1:
            status_message += " Additional blockers: " + ", ".join(blockers[1:]) + "."
        return status_message

    if strategy == "custom_template":
        return classification(
            "custom_template",
            blockers=("custom_template",),
            status_message=(
                "Create a named custom ACP profile with command, args, env, "
                "workspace policy, and evidence bundle."
            ),
            command="",
            args=[],
        )

    if strategy == "documented_candidate":
        return classification(
            "documented_only",
            blockers=(entry.certification_blocker,) if entry.certification_blocker else (),
            status_message="Agent is documented as a candidate and is not eligible for live ACP probing yet.",
            command="",
            args=[],
        )

    blockers: list[str] = []
    if not acp_command:
        blockers.append("entrypoint_strategy_missing")

    shell_builtin_collision = bool(acp_command and acp_command in _SHELL_BUILTIN_COMMANDS)
    if shell_builtin_collision:
        blockers.append("shell_builtin_collision")

    if entry.requires_api_key and not env_getter(entry.requires_api_key):
        blockers.append("credentials_missing")

    if acp_command and not shell_builtin_collision and not command_resolver(acp_command):
        blocker = "adapter_missing" if strategy == "adapter_acp" else "binary_missing"
        blockers.append(blocker)

    if blockers:
        return classification(
            "blocked",
            blockers=tuple(blockers),
            status_message=blocked_status(blockers),
        )

    return classification(
        "ready_to_probe",
        status_message="Configured ACP entrypoint is ready for a bounded initialize probe.",
    )


class AgentRegistry:
    """Loads and caches agent entries from agents.yaml.

    Supports dynamic registration via ``register_agent`` / ``deregister_agent``
    backed by an optional ``ACPSessionsDB`` instance for persistence.
    """

    def __init__(self, yaml_path: str | None = None, db: Any = None) -> None:
        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "..", "Config_Files", "agents.yaml",
            )
        self._yaml_path = os.path.abspath(yaml_path)
        self._entries: list[AgentRegistryEntry] = []
        self._api_entries: list[AgentRegistryEntry] = []
        self._db = db
        self._lock = threading.RLock()
        self._default_type: str = "custom"
        self._last_load_time: float = 0
        self._last_mtime: float = 0
        self._reload_interval: float = 30.0  # seconds

    def load(self) -> None:
        """Load or reload the registry from YAML."""
        if yaml is None:
            logger.warning("PyYAML not installed — agent registry unavailable")
            self._entries = []
            return

        if not os.path.isfile(self._yaml_path):
            logger.warning("Agent registry file not found: {}", self._yaml_path)
            self._entries = []
            return

        try:
            with open(self._yaml_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            logger.error("Failed to load agent registry: {}", exc)
            return

        if not isinstance(data, dict):
            logger.error("Agent registry is not a valid YAML mapping")
            return

        entries: list[AgentRegistryEntry] = []
        default_type = "custom"

        for item in data.get("agents", []):
            if not isinstance(item, dict):
                continue
            agent_type = item.get("type")
            name = item.get("name")
            if not agent_type or not name:
                continue
            entry = AgentRegistryEntry(
                type=str(agent_type),
                name=str(name),
                description=str(item.get("description", "")),
                command=str(item.get("command", "")),
                args=list(item.get("args", [])),
                env=dict(item.get("env", {})),
                requires_api_key=item.get("requires_api_key"),
                default=bool(item.get("default", False)),
                install_instructions=list(item.get("install_instructions", [])),
                docs_url=item.get("docs_url"),
                support_state=item.get("support_state", "documented_unverified"),
                verification_level=item.get("verification_level", "documented_only"),
                compatibility_notes=str(
                    item.get(
                        "compatibility_notes",
                        "Configured locally; live-agent ACP compatibility has not been certified.",
                    )
                ),
                compatibility_docs_url=item.get("compatibility_docs_url", ACP_COMPATIBILITY_DOCS_URL),
                entrypoint_strategy=_coerce_entrypoint_strategy(item.get("entrypoint_strategy")),
                acp_command=str(item.get("acp_command", "")),
                acp_args=list(item.get("acp_args", [])),
                adapter_source=item.get("adapter_source"),
                adapter_docs_url=item.get("adapter_docs_url"),
                certification_blocker=item.get("certification_blocker"),
                mcp_orchestration=item.get("mcp_orchestration", "agent_driven"),
                mcp_entry_tool=str(item.get("mcp_entry_tool", "execute")),
                mcp_structured_response=bool(item.get("mcp_structured_response", False)),
                mcp_llm_provider=item.get("mcp_llm_provider"),
                mcp_llm_model=item.get("mcp_llm_model"),
                mcp_max_iterations=int(item.get("mcp_max_iterations", 20)),
                mcp_refresh_tools=bool(item.get("mcp_refresh_tools", False)),
            )
            entries.append(entry)
            if entry.default:
                default_type = entry.type

        self._entries = entries
        self._default_type = default_type
        self._last_load_time = time.time()
        try:
            self._last_mtime = os.path.getmtime(self._yaml_path)
        except OSError:
            pass
        self._load_api_entries()
        logger.debug("Loaded {} agents from registry ({} YAML, {} API)",
                      len(entries) + len(self._api_entries),
                      len(entries), len(self._api_entries))

    def _maybe_reload(self) -> None:
        """Reload if the file has changed."""
        now = time.time()
        if now - self._last_load_time < self._reload_interval:
            return
        try:
            current_mtime = os.path.getmtime(self._yaml_path)
        except OSError:
            return
        if current_mtime != self._last_mtime:
            logger.info("Agent registry file changed, reloading")
            self.load()

    @staticmethod
    def _load_json(val: Any, default: Any) -> Any:
        """Parse a JSON string value, returning *default* on failure or None."""
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, TypeError):
                return default
        return default if val is None else val

    def _load_api_entries(self) -> None:
        """Load dynamically registered agents from the DB (if available)."""
        if self._db is None:
            return
        with self._lock:
            try:
                rows = self._db.list_agent_entries(source="api")
            except Exception as exc:
                logger.warning("Failed to load API agent entries from DB: {}", exc)
                return
            entries: list[AgentRegistryEntry] = []
            for row in rows:
                entries.append(AgentRegistryEntry(
                    type=row["agent_type"],
                    name=row["name"],
                    description=row.get("description", ""),
                    command=row.get("command", ""),
                    args=self._load_json(row.get("args"), []),
                    env=self._load_json(row.get("env"), {}),
                    requires_api_key=row.get("requires_api_key"),
                    default=bool(row.get("is_default", 0)),
                    install_instructions=self._load_json(row.get("install_instructions"), []),
                    docs_url=row.get("docs_url"),
                    support_state="documented_unverified",
                    verification_level="documented_only",
                    compatibility_notes="Registered dynamically; live-agent ACP compatibility has not been certified.",
                    compatibility_docs_url=ACP_COMPATIBILITY_DOCS_URL,
                    entrypoint_strategy=_coerce_entrypoint_strategy(
                        row.get("entrypoint_strategy")
                    ),
                    acp_command=row.get("acp_command", ""),
                    acp_args=self._load_json(row.get("acp_args"), []),
                    adapter_source=row.get("adapter_source"),
                    adapter_docs_url=row.get("adapter_docs_url"),
                    certification_blocker=row.get("certification_blocker"),
                    mcp_orchestration=row.get("mcp_orchestration", "agent_driven"),
                    mcp_entry_tool=row.get("mcp_entry_tool", "execute"),
                    mcp_structured_response=bool(row.get("mcp_structured_response", 0)),
                    mcp_llm_provider=row.get("mcp_llm_provider"),
                    mcp_llm_model=row.get("mcp_llm_model"),
                    mcp_max_iterations=int(row.get("mcp_max_iterations", 20)),
                    mcp_refresh_tools=bool(row.get("mcp_refresh_tools", 0)),
                ))
            self._api_entries = entries

    @property
    def entries(self) -> list[AgentRegistryEntry]:
        """Get all registry entries, reloading if needed.

        API-registered entries override YAML entries with the same type.
        """
        with self._lock:
            if not self._entries:
                self.load()
            else:
                self._maybe_reload()
            api_types = {e.type for e in self._api_entries}
            merged = [e for e in self._entries if e.type not in api_types]
            merged.extend(self._api_entries)
        return merged

    @property
    def default_type(self) -> str:
        if not self._entries:
            self.load()
        return self._default_type

    def get_entry(self, agent_type: str) -> AgentRegistryEntry | None:
        """Look up an entry by type."""
        for entry in self.entries:
            if entry.type == agent_type:
                return entry
        return None

    def get_available_agents(self) -> list[dict[str, Any]]:
        """Get all agents with runtime availability info."""
        return [entry.check_availability() for entry in self.entries]

    # ------------------------------------------------------------------
    # Dynamic registration
    # ------------------------------------------------------------------

    def register_agent(
        self,
        type: str,
        name: str,
        command: str = "",
        description: str = "",
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        requires_api_key: str | None = None,
        install_instructions: list[str] | None = None,
        docs_url: str | None = None,
        mcp_orchestration: Literal["agent_driven", "llm_driven"] = "agent_driven",
        mcp_entry_tool: str = "execute",
        mcp_structured_response: bool = False,
        mcp_llm_provider: str | None = None,
        mcp_llm_model: str | None = None,
        mcp_max_iterations: int = 20,
        mcp_refresh_tools: bool = False,
        entrypoint_strategy: AgentEntrypointStrategy = "documented_candidate",
        acp_command: str = "",
        acp_args: list[str] | None = None,
        adapter_source: str | None = None,
        adapter_docs_url: str | None = None,
        certification_blocker: str | None = None,
    ) -> AgentRegistryEntry:
        """Register or update a dynamic agent entry."""
        with self._lock:
            normalized_entrypoint_strategy = _coerce_entrypoint_strategy(entrypoint_strategy)
            entry = AgentRegistryEntry(
                type=type,
                name=name,
                command=command,
                description=description,
                args=args or [],
                env=env or {},
                requires_api_key=requires_api_key,
                install_instructions=install_instructions or [],
                docs_url=docs_url,
                mcp_orchestration=mcp_orchestration,
                mcp_entry_tool=mcp_entry_tool,
                mcp_structured_response=mcp_structured_response,
                mcp_llm_provider=mcp_llm_provider,
                mcp_llm_model=mcp_llm_model,
                mcp_max_iterations=mcp_max_iterations,
                mcp_refresh_tools=mcp_refresh_tools,
                entrypoint_strategy=normalized_entrypoint_strategy,
                acp_command=acp_command,
                acp_args=acp_args or [],
                adapter_source=adapter_source,
                adapter_docs_url=adapter_docs_url,
                certification_blocker=certification_blocker,
            )
            if self._db is not None:
                self._db.save_agent_entry({
                    "agent_type": type,
                    "name": name,
                    "command": command,
                    "description": description,
                    "args": json.dumps(args or []),
                    "env": json.dumps(env or {}),
                    "requires_api_key": requires_api_key,
                    "install_instructions": json.dumps(install_instructions or []),
                    "docs_url": docs_url,
                    "mcp_orchestration": mcp_orchestration,
                    "mcp_entry_tool": mcp_entry_tool,
                    "mcp_structured_response": mcp_structured_response,
                    "mcp_llm_provider": mcp_llm_provider,
                    "mcp_llm_model": mcp_llm_model,
                    "mcp_max_iterations": mcp_max_iterations,
                    "mcp_refresh_tools": mcp_refresh_tools,
                    "entrypoint_strategy": normalized_entrypoint_strategy,
                    "acp_command": acp_command,
                    "acp_args": json.dumps(acp_args or []),
                    "adapter_source": adapter_source,
                    "adapter_docs_url": adapter_docs_url,
                    "certification_blocker": certification_blocker,
                    "source": "api",
                })
            self._api_entries = [e for e in self._api_entries if e.type != type]
            self._api_entries.append(entry)
            return entry

    def deregister_agent(self, agent_type: str) -> bool:
        """Remove a dynamically registered agent. Cannot remove YAML entries."""
        with self._lock:
            before = len(self._api_entries)
            self._api_entries = [e for e in self._api_entries if e.type != agent_type]
            removed = len(self._api_entries) < before
            if removed and self._db is not None:
                self._db.delete_agent_entry(agent_type)
            return removed

    _UPDATABLE_FIELDS = frozenset({
        "name", "description", "command", "args", "env",
        "requires_api_key", "install_instructions", "docs_url",
        "entrypoint_strategy", "acp_command", "acp_args", "adapter_source",
        "adapter_docs_url", "certification_blocker",
        "mcp_orchestration", "mcp_entry_tool", "mcp_structured_response",
        "mcp_llm_provider", "mcp_llm_model", "mcp_max_iterations", "mcp_refresh_tools",
    })

    # Defaults for fields that must never be None at runtime
    _FIELD_DEFAULT_FACTORIES: dict[str, Callable[[], Any]] = {
        "args": list,
        "env": dict,
        "install_instructions": list,
        "acp_args": list,
    }
    _NON_NULLABLE_SCALAR_FIELDS = frozenset({
        "name",
        "description",
        "command",
        "acp_command",
        "mcp_orchestration",
        "mcp_entry_tool",
        "mcp_structured_response",
        "mcp_max_iterations",
        "mcp_refresh_tools",
    })

    def update_agent(self, agent_type: str, **kwargs: Any) -> AgentRegistryEntry | None:
        """Update fields on an existing dynamic agent entry."""
        with self._lock:
            existing = None
            for e in self._api_entries:
                if e.type == agent_type:
                    existing = e
                    break
            if existing is None:
                return None
            for key, value in kwargs.items():
                if key in self._UPDATABLE_FIELDS:
                    # Normalize None → safe default for collection fields
                    if value is None and key in self._FIELD_DEFAULT_FACTORIES:
                        value = self._FIELD_DEFAULT_FACTORIES[key]()
                    elif value is None and key in self._NON_NULLABLE_SCALAR_FIELDS:
                        continue
                    if key == "entrypoint_strategy":
                        value = _coerce_entrypoint_strategy(value)
                    setattr(existing, key, value)
            if self._db is not None:
                self._db.save_agent_entry({
                    "agent_type": existing.type,
                    "name": existing.name,
                    "command": existing.command,
                    "description": existing.description,
                    "args": json.dumps(existing.args),
                    "env": json.dumps(existing.env),
                    "requires_api_key": existing.requires_api_key,
                    "install_instructions": json.dumps(existing.install_instructions),
                    "docs_url": existing.docs_url,
                    "mcp_orchestration": existing.mcp_orchestration,
                    "mcp_entry_tool": existing.mcp_entry_tool,
                    "mcp_structured_response": existing.mcp_structured_response,
                    "mcp_llm_provider": existing.mcp_llm_provider,
                    "mcp_llm_model": existing.mcp_llm_model,
                    "mcp_max_iterations": existing.mcp_max_iterations,
                    "mcp_refresh_tools": existing.mcp_refresh_tools,
                    "entrypoint_strategy": existing.entrypoint_strategy,
                    "acp_command": existing.acp_command,
                    "acp_args": json.dumps(existing.acp_args),
                    "adapter_source": existing.adapter_source,
                    "adapter_docs_url": existing.adapter_docs_url,
                    "certification_blocker": existing.certification_blocker,
                    "source": "api",
                })
            return existing


# Module-level singleton
_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


def set_registry_db(db: Any) -> None:
    """Wire the singleton registry with a DB backend for persistence.

    Call this once at application startup (e.g., in ``main.py`` or router init)
    after the ``ACPSessionsDB`` instance is available.
    """
    registry = get_agent_registry()
    registry._db = db
    registry._load_api_entries()
