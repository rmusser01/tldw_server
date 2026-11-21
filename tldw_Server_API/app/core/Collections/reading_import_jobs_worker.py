"""
Reading import Jobs worker.

Consumes core Jobs entries for reading list imports and delegates to
`handle_reading_import_job`.
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from tldw_Server_API.app.core.Collections.reading_import_jobs import (
    READING_IMPORT_DOMAIN,
    handle_reading_import_job,
    reading_import_queue,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK


async def run_reading_import_jobs_worker() -> None:
    worker_id = (os.getenv("READING_IMPORT_JOBS_WORKER_ID") or f"reading-import-{os.getpid()}").strip()
    queue = reading_import_queue()
    cfg = WorkerConfig(
        domain=READING_IMPORT_DOMAIN,
        queue=queue,
        worker_id=worker_id,
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)
    logger.info(f"Reading import worker starting: queue={queue} worker_id={worker_id}")
    try:
        await sdk.run(handler=handle_reading_import_job)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception(f"Reading import worker crashed: queue={queue} worker_id={worker_id}")
        raise


async def main() -> None:
    """Load the provider-override store, then take jobs: LLM claims calls refuse until it is loaded (TASK-15581)."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import standalone_llm_provider_overrides

    async with standalone_llm_provider_overrides():
        await run_reading_import_jobs_worker()


if __name__ == "__main__":
    asyncio.run(main())
