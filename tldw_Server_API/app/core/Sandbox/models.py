from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class RuntimeType(str, Enum):
    docker = "docker"
    firecracker = "firecracker"
    lima = "lima"
    vz_linux = "vz_linux"
    vz_macos = "vz_macos"
    seatbelt = "seatbelt"
    worktree = "worktree"


class TrustLevel(str, Enum):
    """Risk-based isolation profiles for sandbox execution.

    - trusted: Relaxed restrictions, more resources, some egress allowed
    - standard: Default behavior, balanced security/usability
    - untrusted: Maximum isolation, strict limits, no network, minimal resources
    """
    trusted = "trusted"
    standard = "standard"
    untrusted = "untrusted"


@dataclass
class SessionSpec:
    runtime: RuntimeType | None = None
    base_image: str | None = None
    cpu_limit: float | None = None
    memory_mb: int | None = None
    timeout_sec: int = 300
    network_policy: str = "deny_all"
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    trust_level: TrustLevel | None = None  # Defaults to standard if not specified
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None


@dataclass
class Session:
    id: str
    runtime: RuntimeType
    base_image: str | None
    expires_at: datetime | None
    cpu_limit: float | None = None
    memory_mb: int | None = None
    timeout_sec: int = 300
    network_policy: str = "deny_all"
    env: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    trust_level: TrustLevel | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None


@dataclass
class RunSpec:
    session_id: str | None
    runtime: RuntimeType | None
    base_image: str | None
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    startup_timeout_sec: int | None = None
    timeout_sec: int = 300
    cpu: float | None = None
    memory_mb: int | None = None
    network_policy: str | None = None
    files_inline: list[tuple[str, bytes]] = field(default_factory=list)
    capture_patterns: list[str] = field(default_factory=list)
    # Spec 1.1 interactive settings (stdin over WS)
    interactive: bool | None = None
    stdin_max_bytes: int | None = None
    stdin_max_frame_bytes: int | None = None
    stdin_bps: int | None = None
    stdin_idle_timeout_sec: int | None = None
    # Trust level for risk-based isolation profiles
    trust_level: TrustLevel | None = None  # Defaults to standard if not specified
    # Optional port mappings for runtimes that support it (e.g., docker)
    # Each mapping: {"host_ip": "127.0.0.1", "host_port": 2222, "container_port": 22}
    port_mappings: list[dict[str, str | int]] = field(default_factory=list)
    # When true, skip setting a random non-root user in the container
    run_as_root: bool | None = None
    # When false, skip --read-only on the root filesystem (runtime-specific)
    read_only_root: bool | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None


class RunPhase(str, Enum):
    queued = "queued"
    starting = "starting"
    running = "running"
    completed = "completed"
    failed = "failed"
    killed = "killed"
    timed_out = "timed_out"


@dataclass
class RunStatus:
    id: str
    phase: RunPhase
    spec_version: str | None = None
    runtime: RuntimeType | None = None
    runtime_version: str | None = None
    base_image: str | None = None
    image_digest: str | None = None
    policy_hash: str | None = None
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str | None = None
    resource_usage: dict[str, int] | None = None
    artifacts: dict[str, bytes] | None = None
    estimated_start_time: datetime | None = None
    session_id: str | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None
    # Optional execution-claim metadata for durable claim fencing.
    claim_owner: str | None = None
    claim_expires_at: datetime | None = None
