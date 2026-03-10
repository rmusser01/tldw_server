"""Repository interfaces for managed vLLM instance storage."""

from __future__ import annotations

from typing import Protocol

from .models import VLLMInstanceCreate, VLLMInstanceRecord


class VLLMInstanceRepository(Protocol):
    """Storage contract for managed vLLM instances."""

    def create_instance(self, payload: VLLMInstanceCreate) -> VLLMInstanceRecord:
        """Persist and return a new managed instance."""

    def get_instance(self, instance_id: str) -> VLLMInstanceRecord | None:
        """Fetch a single managed instance by identifier."""

    def list_instances(self) -> list[VLLMInstanceRecord]:
        """Return all managed instances."""

    def set_default_instance(self, instance_id: str | None) -> None:
        """Set or clear the default managed vLLM instance."""

    def get_default_instance_id(self) -> str | None:
        """Return the default managed vLLM instance identifier."""
