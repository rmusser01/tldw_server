"""Prepare the always-served API schema before startup admits requests."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI


async def warm_openapi_schema(app: FastAPI) -> None:
    """Populate the existing cache off-loop and drain admitted work on cancellation."""
    if app.openapi_schema:
        return
    build = asyncio.create_task(asyncio.to_thread(app.openapi))
    # Retain a completion receipt even if the builder fails during cancellation.
    completion = asyncio.gather(build, return_exceptions=True)
    cancelled: asyncio.CancelledError | None = None
    while not completion.done():
        try:
            await asyncio.shield(completion)
        except asyncio.CancelledError as error:
            if cancelled is None:
                cancelled = error
    if cancelled is not None:
        raise cancelled
    build.result()
