"""Agent executor placeholder for managed vLLM instances."""

from __future__ import annotations

from tldw_Server_API.app.core.VLLM_Management.models import VLLMInstanceRecord


class AgentVLLMExecutor:
    def start(self, instance: VLLMInstanceRecord):
        raise NotImplementedError("Managed vLLM agent execution is not implemented yet")

    def stop(self, instance: VLLMInstanceRecord, handle: dict[str, object]):
        raise NotImplementedError("Managed vLLM agent execution is not implemented yet")

    def probe(self, instance: VLLMInstanceRecord):
        raise NotImplementedError("Managed vLLM agent execution is not implemented yet")
