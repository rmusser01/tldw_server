from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _str_field(payload: dict[str, Any], key: str, default: str = "") -> str:
    """Return a string field only when the payload value is already a string."""
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return default


def _bool_field(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    """Parse helper booleans without promoting malformed values to trusted truthy values."""
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return default


def _dict_field(payload: dict[str, Any], key: str = "details") -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return dict(value)
    return {}


def _list_field(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


@dataclass(slots=True)
class HelperVMMetadata:
    """Ownership metadata returned by the helper for a live VM record."""

    owner: str = "unknown"
    runtime: str = ""
    run_id: str = ""
    session_id: str = ""
    session_mode: bool = False
    template_id: str = ""
    template_path: str = ""
    run_manifest_path: str = ""
    planning_source: str = ""
    workspace_path: str = ""
    network_policy: str = ""
    created_at: str = ""

    @property
    def has_tldw_owner(self) -> bool:
        return self.owner == "tldw" and self.runtime == "vz_linux"


def _metadata_field(payload: dict[str, Any]) -> HelperVMMetadata:
    """Parse helper VM metadata, downgrading malformed payloads to unknown ownership."""
    raw = payload.get("metadata")
    if not isinstance(raw, dict):
        return HelperVMMetadata()
    string_keys = (
        "owner",
        "runtime",
        "run_id",
        "session_id",
        "template_id",
        "template_path",
        "run_manifest_path",
        "planning_source",
        "workspace_path",
        "network_policy",
        "created_at",
    )
    for key in string_keys:
        value = raw.get(key)
        if value is not None and not isinstance(value, str):
            return HelperVMMetadata()
    session_mode_value = raw.get("session_mode")
    if session_mode_value is not None and not isinstance(session_mode_value, (bool, str)):
        return HelperVMMetadata()
    return HelperVMMetadata(
        owner=_str_field(raw, "owner", "unknown").strip() or "unknown",
        runtime=_str_field(raw, "runtime").strip(),
        run_id=_str_field(raw, "run_id").strip(),
        session_id=_str_field(raw, "session_id").strip(),
        session_mode=_bool_field(raw, "session_mode"),
        template_id=_str_field(raw, "template_id").strip(),
        template_path=_str_field(raw, "template_path").strip(),
        run_manifest_path=_str_field(raw, "run_manifest_path").strip(),
        planning_source=_str_field(raw, "planning_source").strip(),
        workspace_path=_str_field(raw, "workspace_path").strip(),
        network_policy=_str_field(raw, "network_policy").strip(),
        created_at=_str_field(raw, "created_at").strip(),
    )


@dataclass(slots=True)
class HelperVMReply:
    vm_id: str
    state: str
    metadata: HelperVMMetadata = field(default_factory=HelperVMMetadata)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HelperExecReply:
    exit_code: int
    stdout: bytes = b""
    stderr: bytes = b""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HelperPingReply:
    protocol_version: str
    helper_version: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HelperHostValidationReply:
    protocol_version: str
    helper_version: str
    available: bool
    execution_mode: str
    transport: str | None = None
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HelperVMStatusReply:
    protocol_version: str
    helper_version: str
    vm_id: str
    state: str
    healthy: bool
    metadata: HelperVMMetadata = field(default_factory=HelperVMMetadata)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HelperVMListReply:
    protocol_version: str
    helper_version: str
    vms: list[HelperVMStatusReply] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


def parse_helper_ping(payload: dict[str, Any]) -> HelperPingReply:
    return HelperPingReply(
        protocol_version=_str_field(payload, "protocol_version"),
        helper_version=_str_field(payload, "helper_version"),
        status=_str_field(payload, "status"),
        details=_dict_field(payload),
    )


def parse_helper_host_validation(payload: dict[str, Any]) -> HelperHostValidationReply:
    raw_transport = payload.get("transport")
    transport = str(raw_transport) if raw_transport is not None else None
    return HelperHostValidationReply(
        protocol_version=_str_field(payload, "protocol_version"),
        helper_version=_str_field(payload, "helper_version"),
        available=_bool_field(payload, "available"),
        execution_mode=_str_field(payload, "execution_mode"),
        transport=transport,
        reasons=_list_field(payload, "reasons"),
        details=_dict_field(payload),
    )


def parse_helper_vm_reply(payload: dict[str, Any]) -> HelperVMReply:
    """Parse a `create_vm` helper reply into the normalized Python response model."""
    return HelperVMReply(
        vm_id=_str_field(payload, "vm_id").strip(),
        state=_str_field(payload, "state").strip(),
        metadata=_metadata_field(payload),
        details=_dict_field(payload),
    )


def parse_helper_vm_status(payload: dict[str, Any]) -> HelperVMStatusReply:
    return HelperVMStatusReply(
        protocol_version=_str_field(payload, "protocol_version"),
        helper_version=_str_field(payload, "helper_version"),
        vm_id=_str_field(payload, "vm_id"),
        state=_str_field(payload, "state"),
        healthy=_bool_field(payload, "healthy"),
        metadata=_metadata_field(payload),
        details=_dict_field(payload),
    )


def parse_helper_vm_list(payload: dict[str, Any]) -> HelperVMListReply:
    raw_vms = payload.get("vms")
    vms = [parse_helper_vm_status(item) for item in raw_vms] if isinstance(raw_vms, list) else []
    return HelperVMListReply(
        protocol_version=_str_field(payload, "protocol_version"),
        helper_version=_str_field(payload, "helper_version"),
        vms=vms,
        details=_dict_field(payload),
    )
