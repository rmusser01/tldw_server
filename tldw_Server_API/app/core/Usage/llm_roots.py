"""Give background work a user for LLM usage accounting, and keep it across executor hand-offs (spec §1).

A job, a workflow run and a scheduled task start with no HTTP request, so
nothing tells the recorder who they are for. ``llm_call_root`` does. An
executor hand-off made with ``loop.run_in_executor(None, ...)`` does not copy
context variables, so the default executor installed here carries the LLM call
state itself.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from tldw_Server_API.app.core.Usage.llm_call_context import (
    capture_llm_call_state,
    llm_call_context,
    run_with_llm_call_state,
)
from tldw_Server_API.app.core.Usage.quota_checks import as_quota_user_id

# Loops whose default executor carries the LLM call state. asyncio has no getter for the default
# executor, and uvloop, which the server runs on, keeps it where Python cannot read it.
# ponytail: a later set_default_executor on a listed loop goes unnoticed; nothing in the app calls it.
_INSTALLED: weakref.WeakSet[asyncio.AbstractEventLoop] = weakref.WeakSet()


class LLMStateThreadPoolExecutor(ThreadPoolExecutor):
    """A default executor whose worker threads see the submitter's LLM call state, and nothing else of its context."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        """Capture the state where the work is submitted and apply it where the work runs."""
        return super().submit(run_with_llm_call_state, capture_llm_call_state(), fn, *args, **kwargs)


def ensure_llm_state_executor(loop: asyncio.AbstractEventLoop | None = None) -> bool:
    """Make ``loop``'s default executor carry the LLM call state. Idempotent; False with no running loop."""
    try:
        target = loop or asyncio.get_running_loop()
    except RuntimeError:
        return False
    if target in _INSTALLED:
        return True
    # Same sizing and thread names as the executor asyncio creates on first use.
    target.set_default_executor(LLMStateThreadPoolExecutor(thread_name_prefix="asyncio"))
    _INSTALLED.add(target)
    return True


@contextmanager
def llm_call_root(user_id: Any, *, operation: str | None = None) -> Iterator[None]:
    """Run one unit of background work for ``user_id``: a job, a workflow run, a scheduled task.

    An owner that is not a number sets no user, and never clears one that is
    already set. A root is never admitted by the request that started it: its
    calls are quota-checked when they run (LLM usage accounting spec §4a, §4b).
    A media ingest job that asks for analysis checks once when it starts, and
    then marks itself admitted until it ends (TASK-13864).
    """
    overrides: dict[str, Any] = {"prechecked": False}
    resolved = as_quota_user_id(user_id)
    if resolved is not None:
        overrides["user_id"] = resolved
    if operation:
        overrides["operation"] = operation
    ensure_llm_state_executor()
    with llm_call_context(**overrides):
        yield
