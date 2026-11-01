"""Barrier-controlled page execution; result and physical drain are independent."""

import asyncio
from dataclasses import dataclass

from tldw_Server_API.app.core.Web_Scraping.contracts.execution import DiscoveryResult, PageData
from tldw_Server_API.app.core.Web_Scraping.jobs.models import ScrapingJob
from tldw_Server_API.app.core.Web_Scraping.runtime.controls import WorkHandle


def job(name="job", **kwargs):
    return ScrapingJob(name, f"https://example.com/{name}", "auto", **kwargs)


def page(**article):
    return PageData(
        {"content": "body", "extraction_successful": True, **article},
        DiscoveryResult("https://example.com", "https://example.com/final", ()),
    )


@dataclass
class Attempt:
    payload: object
    controls: object
    result: asyncio.Future
    drained: asyncio.Future
    cancellations: int = 0

    def cancel(self):
        self.cancellations += 1

    def finish(self, value=None):
        if not self.result.done():
            self.result.set_result(value or page())
        if not self.drained.done():
            self.drained.set_result(None)


class Executor:
    def __init__(self):
        self.started = asyncio.Queue()
        self.attempts = []

    def __call__(self, payload, controls):
        loop = asyncio.get_running_loop()
        attempt = Attempt(payload, controls, loop.create_future(), loop.create_future())
        self.attempts.append(attempt)
        self.started.put_nowait(attempt)
        return WorkHandle(attempt.result, attempt.drained, attempt.cancel)


async def settle(queue):
    """Wait for currently owned executions, without timing-based ordering."""
    while queue._tasks:
        await asyncio.gather(*tuple(queue._tasks))
        await turn()


async def turn():
    """A FIFO event-loop barrier, not an elapsed-time race heuristic."""
    future = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_soon(future.set_result, None)
    await future
