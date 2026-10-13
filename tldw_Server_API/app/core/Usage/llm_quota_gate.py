"""Check the LLM quota at the call, for work no request boundary admitted (LLM usage accounting spec §4b, §5).

``user_to_check`` decides whether a call is checked at all. ``check_user_sync``
and ``check_user_async`` run the decision on the accounting loop, wherever the
call runs, and raise ``LLMQuotaExceededError`` when it refuses. Anything else
that goes wrong admits the call: a check must never break an LLM call.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict

from loguru import logger

from tldw_Server_API.app.core.config import usage_quotas_enabled
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage.llm_accounting import get_accounting_loop
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    current_llm_user_id,
    in_accounted_call,
)
from tldw_Server_API.app.core.Usage.quota_checks import (
    LLM_COST_PER_DAY_KEY,
    LLM_COST_PER_MONTH_KEY,
    LLM_TOKENS_PER_MONTH_KEY,
    LLMQuotaRefusal,
    as_quota_user_id,
    llm_quota_refusal,
)
from tldw_Server_API.app.core.Usage.quota_resolver import cached_limits

# How long a check handed to the accounting loop waits for its answer before admitting the call.
CHECK_TIMEOUT_SECONDS = 2.0
_LLM_LIMIT_KEYS = (LLM_TOKENS_PER_MONTH_KEY, LLM_COST_PER_DAY_KEY, LLM_COST_PER_MONTH_KEY)
_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Log a warning the first time ``key`` happens in this process, and at debug after that."""
    if key in _WARNED:
        logger.debug(message)
        return
    _WARNED.add(key)
    logger.warning(message)


def user_to_check() -> int | None:
    """The user this call is checked for, or None when it is not checked.

    Not checked: no user, an exempt or admitted (``prechecked``) context, a call
    nested in an accounted call, quotas off, or a user whose cached limits set
    no LLM key.
    """
    try:
        if in_accounted_call():
            return None
        context = current_llm_call_context()
        if context.exempt or context.prechecked:
            return None
        user_id = as_quota_user_id(current_llm_user_id())
        if user_id is None or not usage_quotas_enabled():
            return None
        limits = cached_limits(user_id)
        if limits is not None and all(limits.get(key) is None for key in _LLM_LIMIT_KEYS):
            return None
        return user_id
    except Exception as error:  # noqa: BLE001 - a check must never break an LLM call
        logger.bind(exception_type=type(error).__name__).warning("LLM quota check could not start; call admitted")
        return None


def _raise_if_refused(refusal: LLMQuotaRefusal | None) -> None:
    """Raise the 402 for a refusal."""
    if refusal is not None:
        raise LLMQuotaExceededError(**asdict(refusal))


def _running_loop() -> asyncio.AbstractEventLoop | None:
    """The loop running on this thread, or None."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def check_user_sync(user_id: int) -> bool:
    """Check ``user_id`` from synchronous code; raise ``LLMQuotaExceededError`` when refused.

    Returns False when the check cannot run here, and the call goes ahead
    unchecked: on the accounting loop's own thread, where waiting would
    deadlock, or with no accounting loop. A check that times out or fails
    admits the call and returns True.
    """
    loop = get_accounting_loop()
    if loop is None:
        _warn_once(
            "no_loop",
            "No accounting loop is set, so synchronous LLM calls are not quota-checked "
            "(run job workers inside the server process)",
        )
        return False
    if _running_loop() is loop:
        _warn_once("on_loop", "A synchronous LLM call on the event loop's thread is not quota-checked")
        return False
    coro = llm_quota_refusal(user_id)
    try:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as error:  # noqa: BLE001 - the loop closed between the two lines above
        coro.close()
        logger.bind(exception_type=type(error).__name__).warning("LLM quota check failed; call admitted")
        return True
    try:
        refusal = future.result(timeout=CHECK_TIMEOUT_SECONDS)
    except TimeoutError:
        future.cancel()
        logger.warning("LLM quota check timed out after {}s; call admitted", CHECK_TIMEOUT_SECONDS)
        return True
    except Exception as error:  # noqa: BLE001 - a check must never break an LLM call
        logger.bind(exception_type=type(error).__name__).warning("LLM quota check failed; call admitted")
        return True
    _raise_if_refused(refusal)
    return True


async def check_user_async(user_id: int) -> None:
    """Check ``user_id`` from a coroutine; raise ``LLMQuotaExceededError`` when refused.

    On the accounting loop, or with none captured, the decision is awaited
    here. From another loop it is handed to the accounting loop. A check that
    times out or fails admits the call.
    """
    try:
        running = asyncio.get_running_loop()
        target = get_accounting_loop() or running
        if target is running:
            refusal = await llm_quota_refusal(user_id)
        else:
            coro = llm_quota_refusal(user_id)
            try:
                future = asyncio.run_coroutine_threadsafe(coro, target)
            except Exception:
                coro.close()
                raise
            try:
                refusal = await asyncio.wait_for(asyncio.wrap_future(future), CHECK_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning("LLM quota check timed out after {}s; call admitted", CHECK_TIMEOUT_SECONDS)
                return
    except Exception as error:  # noqa: BLE001 - a check must never break an LLM call
        logger.bind(exception_type=type(error).__name__).warning("LLM quota check failed; call admitted")
        return
    _raise_if_refused(refusal)


def check_llm_call_sync() -> bool:
    """Check the current call from synchronous code. False when it could not be checked here."""
    user_id = user_to_check()
    return True if user_id is None else check_user_sync(user_id)


async def check_llm_call_async() -> None:
    """Check the current call from a coroutine."""
    user_id = user_to_check()
    if user_id is not None:
        await check_user_async(user_id)
