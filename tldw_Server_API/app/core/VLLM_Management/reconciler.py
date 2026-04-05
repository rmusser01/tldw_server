"""Background reconciler for managed vLLM instance health state."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from .service import VLLMManagementService


class VLLMReconciler:
    def __init__(
        self,
        *,
        service: VLLMManagementService | None = None,
        repository=None,
        executors=None,
        interval_seconds: int = 30,
    ) -> None:
        self.service = service or VLLMManagementService(repository=repository, executors=executors)
        self.interval_seconds = int(max(1, interval_seconds))

    def reconcile_once(self) -> dict[str, Any]:
        reconciled = 0
        for instance in self.service.repository.list_instances():
            if instance.desired_state != "running" and instance.observed_state == "stopped":
                continue
            self.service.probe_instance(instance.instance_id)
            reconciled += 1
        return {"reconciled": reconciled}

    async def _reconcile_once_async(self, stop_event: asyncio.Event) -> dict[str, Any]:
        reconciled = 0
        instances = await asyncio.to_thread(self.service.repository.list_instances)
        for instance in instances:
            if stop_event.is_set():
                break
            if instance.desired_state != "running" and instance.observed_state == "stopped":
                continue
            try:
                await asyncio.to_thread(self.service.probe_instance, instance.instance_id)
                reconciled += 1
            except Exception as exc:
                logger.warning(
                    "Managed vLLM reconciler probe failed for {}: {}",
                    instance.instance_id,
                    exc,
                )
        return {"reconciled": reconciled}

    async def run_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self._reconcile_once_async(stop_event)
            except Exception as exc:
                logger.warning("Managed vLLM reconciler pass failed: {}", exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                continue
