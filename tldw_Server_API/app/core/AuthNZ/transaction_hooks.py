"""Request-scoped hooks that run only after a successful AuthNZ commit."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar, Token

AfterCommitHook = Callable[[], Awaitable[None]]

_AFTER_COMMIT_HOOKS: ContextVar[list[AfterCommitHook] | None] = ContextVar(
    "authnz_after_commit_hooks",
    default=None,
)


def begin_after_commit_scope() -> Token[list[AfterCommitHook] | None]:
    """Start one request-local after-commit hook scope."""
    return _AFTER_COMMIT_HOOKS.set([])


def defer_until_after_commit(hook: AfterCommitHook) -> bool:
    """Queue a hook, returning False when no request transaction owns a scope."""
    hooks = _AFTER_COMMIT_HOOKS.get()
    if hooks is None:
        return False
    hooks.append(hook)
    return True


async def finish_after_commit_scope(
    token: Token[list[AfterCommitHook] | None],
    *,
    committed: bool,
) -> None:
    """Discard hooks on rollback or execute them after a successful commit."""
    hooks = tuple(_AFTER_COMMIT_HOOKS.get() or ())
    _AFTER_COMMIT_HOOKS.reset(token)
    if committed:
        for hook in hooks:
            await hook()
