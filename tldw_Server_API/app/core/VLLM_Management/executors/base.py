"""Base contracts for managed vLLM executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord


@dataclass
class LifecycleResult:
    status: str
    base_url: str | None = None
    handle: dict[str, Any] = field(default_factory=dict)
    log_handle: dict[str, str] = field(default_factory=dict)


@dataclass
class StopResult:
    status: str
    forced: bool = False
    detail: str | None = None


@dataclass
class ProbeResult:
    status: str
    reachable: bool
    base_url: str | None = None
    detail: str | None = None
    capabilities: dict[str, bool] = field(default_factory=dict)


class VLLMExecutor(Protocol):
    def start(self, instance: VLLMInstanceRecord) -> LifecycleResult: ...

    def stop(self, instance: VLLMInstanceRecord, handle: dict[str, Any]) -> StopResult: ...

    def probe(self, instance: VLLMInstanceRecord) -> ProbeResult: ...
