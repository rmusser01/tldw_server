from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _str_field(payload: dict[str, Any], key: str, default: str = "") -> str:
    value = payload.get(key)
    if value is None:
        return default
    return str(value)


def _bool_field(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    return bool(value)


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
class HelperVMReply:
    vm_id: str
    state: str
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


def parse_helper_vm_status(payload: dict[str, Any]) -> HelperVMStatusReply:
    return HelperVMStatusReply(
        protocol_version=_str_field(payload, "protocol_version"),
        helper_version=_str_field(payload, "helper_version"),
        vm_id=_str_field(payload, "vm_id"),
        state=_str_field(payload, "state"),
        healthy=_bool_field(payload, "healthy"),
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
