"""Startup and shutdown reconciliation for managed llama.cpp profiles."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppProfile,
    LlamaCppRuntime,
    LlamaCppRuntimeState,
)
from tldw_Server_API.app.core.Local_LLM.llamacpp_supervisor_service import LlamaCppSupervisor

SleepCallable = Callable[[float], Awaitable[Any]]


class LlamaCppRuntimeReconciler:
    """Reconcile durable llama.cpp profile definitions with owned runners."""

    def __init__(
        self,
        supervisor: LlamaCppSupervisor,
        *,
        sleep: SleepCallable = asyncio.sleep,
    ) -> None:
        self.supervisor = supervisor
        self.sleep = sleep

    async def reconcile_startup(self) -> list[LlamaCppRuntime]:
        """Run one startup reconciliation pass for autostart profiles."""
        return await self.reconcile_once()

    async def reconcile_once(self) -> list[LlamaCppRuntime]:
        """Start eligible autostart profiles and persist bounded failures."""
        runtimes: list[LlamaCppRuntime] = []
        for profile in self.supervisor.list_profiles():
            if not self._should_reconcile(profile):
                continue
            runtime = self.supervisor.get_runtime(profile.profile_id)
            if runtime.state in {LlamaCppRuntimeState.STARTING, LlamaCppRuntimeState.RUNNING}:
                continue
            if self._restart_limit_reached(profile):
                runtimes.append(self.supervisor.runtime_from_last_failure(profile))
                continue
            runtimes.append(await self._start_or_record_failure(profile))
        return runtimes

    async def shutdown(self) -> None:
        """Stop supervisor-owned runners without adopting external PIDs."""
        await self.supervisor.shutdown()

    def _should_reconcile(self, profile: LlamaCppProfile) -> bool:
        if not profile.enabled or not profile.autostart:
            return False
        return not self.supervisor.is_profile_paused(profile.profile_id)

    def _restart_limit_reached(self, profile: LlamaCppProfile) -> bool:
        restart_count = _restart_count(profile)
        if restart_count <= 0:
            return False
        max_restarts = _max_restarts(profile)
        return restart_count >= max_restarts

    async def _start_or_record_failure(self, profile: LlamaCppProfile) -> LlamaCppRuntime:
        try:
            runtime = await self.supervisor.start_profile(profile.profile_id)
        except Exception as exc:  # noqa: BLE001 - reconciler must isolate per-profile startup failures.
            logger.warning("Failed to reconcile llama.cpp profile {}: {}", profile.profile_id, exc)
            runtime = self._current_runtime_or_none(profile.profile_id)
            return await self.supervisor.record_runtime_failure(
                profile.profile_id,
                runtime=runtime,
                error=exc,
                restart_count=_restart_count(profile) + 1,
            )
        await self.supervisor.clear_runtime_failure(profile.profile_id)
        return runtime

    def _current_runtime_or_none(self, profile_id: str) -> LlamaCppRuntime | None:
        try:
            return self.supervisor.get_runtime(profile_id)
        except Exception:  # noqa: BLE001 - failure recording should still preserve the original error.
            return None


def _restart_count(profile: LlamaCppProfile) -> int:
    try:
        return max(0, int(profile.last_runtime_failure.get("restart_count") or 0))
    except (TypeError, ValueError):
        return 0


def _max_restarts(profile: LlamaCppProfile) -> int:
    try:
        return max(0, int(profile.restart_policy.get("max_restarts") or 0))
    except (TypeError, ValueError):
        return 0


__all__ = ["LlamaCppRuntimeReconciler"]
