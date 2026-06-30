from __future__ import annotations

import asyncio
import contextlib
import os

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.testing import is_truthy


async def run_jobs_integrity_sweeper(stop_event: asyncio.Event | None = None) -> None:
    """Periodically validate and optionally repair Jobs invariants.

    Controlled by env flags:
      - JOBS_INTEGRITY_SWEEP_ENABLED=true|false
      - JOBS_INTEGRITY_SWEEP_INTERVAL_SEC=60
      - JOBS_INTEGRITY_SWEEP_FIX=true|false (when true, attempts self-heal)
    """
    if not is_truthy(os.getenv("JOBS_INTEGRITY_SWEEP_ENABLED")):
        return

    interval = float(os.getenv("JOBS_INTEGRITY_SWEEP_INTERVAL_SEC", "60") or "60")
    do_fix = is_truthy(os.getenv("JOBS_INTEGRITY_SWEEP_FIX"))
    jm = JobManager()
    logger.info(f"Starting Jobs integrity sweeper (every {interval}s, fix={do_fix})")
    while True:
        try:
            if stop_event and stop_event.is_set():
                logger.info("Stopping Jobs integrity sweeper on shutdown signal")
                return
            stats = jm.integrity_sweep(fix=do_fix)
            with contextlib.suppress(Exception):
                logger.debug(f"Jobs integrity sweep: {stats}")
        except Exception as e:
            logger.bind(error_type=type(e).__name__).debug("Jobs integrity sweep error")
        await asyncio.sleep(interval)
