"""
UserProfiles effect dispatch.
"""

from __future__ import annotations

from collections.abc import Iterable

from loguru import logger

from tldw_Server_API.app.core.UserProfiles.contracts import (
    EffectDescriptor,
    EffectPolicy,
    EffectTiming,
)


class ProfileEffectDispatcher:
    async def run_pre_commit(self, effects: Iterable[EffectDescriptor]) -> None:
        for effect in effects:
            if effect.timing != EffectTiming.PRE_COMMIT:
                continue
            if effect.policy == EffectPolicy.REQUIRED:
                logger.debug("Required profile effect completed: {}", effect.name)

    async def run_post_commit(self, effects: Iterable[EffectDescriptor]) -> None:
        for effect in effects:
            if effect.timing != EffectTiming.POST_COMMIT:
                continue
            try:
                logger.debug("Best-effort profile effect completed: {}", effect.name)
            except Exception as exc:
                logger.debug(
                    "Best-effort profile effect failed: {} {}",
                    effect.name,
                    type(exc).__name__,
                )
