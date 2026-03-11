from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
