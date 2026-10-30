"""Fixtures that give a test its own, empty app-service container.

``fresh_services`` snapshots the process container, empties every slot, and on
teardown closes whatever the test put there and restores the snapshot.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import MISSING, fields

import pytest_asyncio
from loguru import logger

from tldw_Server_API.app.core.Utils.app_services import AppServices, app_services


async def close_app_service(slot: str, *, restore: object = None) -> None:
    """Put ``restore`` in ``slot`` and close the object that was there.

    Does nothing if the slot already holds ``restore``. Closes with ``aclose()``
    if the object has one, else ``shutdown()``. Errors are logged and swallowed.
    """
    services = app_services()
    taken = getattr(services, slot)
    if taken is restore:
        return
    setattr(services, slot, restore)
    try:
        for name in ("aclose", "shutdown", "close"):
            closer = getattr(taken, name, None)
            if callable(closer):
                result = closer()
                if inspect.isawaitable(result):
                    await result
                return
        logger.debug("close_app_service({}): {!r} has no aclose/shutdown/close", slot, taken)
    except Exception as exc:  # teardown must not mask the test's own failure
        logger.debug("close_app_service({}): ignoring close error: {}", slot, exc)


def _set_defaults(services: AppServices) -> None:
    for f in fields(services):
        setattr(services, f.name, f.default_factory() if f.default_factory is not MISSING else f.default)


async def discard_app_services() -> None:
    """Close what every slot holds, empty it, and reset the caches built on the pool."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    try:
        await reset_db_pool()  # pool, MCP server, API-key manager, UsersDB, RBAC and MCP caches
    except Exception as exc:
        logger.debug("discard_app_services: ignoring reset_db_pool error: {}", exc)
    _set_defaults(app_services())


@asynccontextmanager
async def fresh_app_services() -> AsyncIterator[AppServices]:
    """Run with every slot empty; close what the block created and restore the rest."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    services = app_services()
    snapshot = {f.name: getattr(services, f.name) for f in fields(services)}
    _set_defaults(services)
    try:
        await reset_db_pool()  # slot is empty: this only drops the caches
        yield services
    finally:
        try:
            await discard_app_services()
        finally:
            for name, value in snapshot.items():
                setattr(services, name, value)


@pytest_asyncio.fixture
async def fresh_services() -> AsyncIterator[AppServices]:
    async with fresh_app_services() as services:
        yield services
