"""Repository interfaces for managed vLLM instance storage."""

from __future__ import annotations

from typing import Any
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

    def update_instance(self, instance_id: str, patch: dict[str, Any]) -> VLLMInstanceRecord:
        """Update mutable instance spec fields and return the persisted record."""

    def update_instance_runtime(self, instance_id: str, patch: dict[str, Any]) -> VLLMInstanceRecord:
        """Persist observed runtime metadata and return the persisted record."""

    def delete_instance(self, instance_id: str) -> bool:
        """Delete an instance and return whether a record was removed."""

    def set_default_instance(self, instance_id: str | None) -> None:
        """Set or clear the default managed vLLM instance."""

    def get_default_instance_id(self) -> str | None:
        """Return the default managed vLLM instance identifier."""
