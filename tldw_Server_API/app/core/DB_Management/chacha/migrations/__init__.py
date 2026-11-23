"""ChaCha package migrations (TASK-13950.22): the step modules and the glue that binds a step to a context."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.chacha.context import ChaChaContext


def bind(step: Callable[[ChaChaContext, Any], None], ctx: ChaChaContext) -> Callable[[Any], None]:
    """Return ``step`` as a one-argument callable over ``conn``, keeping its ``__name__``, ``__qualname__``, ``__doc__`` and ``__wrapped__``."""

    def bound(conn: Any) -> None:
        step(ctx, conn)

    return functools.update_wrapper(bound, step)
