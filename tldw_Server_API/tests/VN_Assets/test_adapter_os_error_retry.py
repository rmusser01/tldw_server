"""Adapter-origin OS failures retain the real Jobs retry lifecycle."""

from __future__ import annotations

import errno
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Image_Generation.exceptions import ImageGenerationError
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeImageAdapter, FakeImageRegistry
from tldw_Server_API.tests.VN_Assets.test_pr3071_worker_retry_lifecycle import deliver
from tldw_Server_API.tests.VN_Assets.test_pr3071_worker_retry_lifecycle import lifecycle as lifecycle

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_adapter_os_error_retries_v1_before_terminalizing(
    lifecycle: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operational adapter error remains planned until the actual SDK retry succeeds."""

    class OnceFailingAdapter(FakeImageAdapter):
        error = BlockingIOError(errno.EAGAIN, "finite adapter resource unavailable")

        def generate(self, request: Any) -> Any:
            if not self.requests:
                self.requests.append(request)
                raise self.error
            return super().generate(request)

    adapter = OnceFailingAdapter()
    lifecycle.worker.image_registry = FakeImageRegistry(adapter)
    handle_job = lifecycle.worker.handle_job_async
    failures: list[Exception] = []

    async def observe_failure(job: dict[str, Any]) -> dict[str, Any]:
        try:
            return await handle_job(job)
        except (OSError, ImageGenerationError) as exc:
            failures.append(exc)
            raise

    monkeypatch.setattr(lifecycle.worker, "handle_job_async", observe_failure)
    batch_id = lifecycle.start()
    first = await deliver(lifecycle)
    stored = lifecycle.jobs.get_job(first["id"])
    assert first["retry_count"] == 0 and first["max_retries"] == 1
    assert stored["status"] == "queued" and stored["retry_count"] == 1
    outcome = lifecycle.repo.get_variant_outcome(batch_id, lifecycle.slot.id, 0)
    assert outcome["outcome_status"] == "planned"
    assert len(failures) == 1
    assert isinstance(failures[0], ImageGenerationError)
    assert failures[0].__cause__ is adapter.error
    assert adapter.error.errno == errno.EAGAIN
    assert lifecycle.storage.usage == 0
    second = await deliver(lifecycle)
    assert second["id"] == first["id"] and second["retry_count"] == 1
    assert lifecycle.jobs.get_job(second["id"])["status"] == "completed"
    assert lifecycle.repo.get_batch(batch_id)["status"] == "completed"
    completed = lifecycle.repo.get_variant_outcome(batch_id, lifecycle.slot.id, 0)
    assert completed["outcome_status"] == "completed"
    assert completed["item_id"] == outcome["item_id"]
    assert len(adapter.requests) == 2
    assert len(failures) == 1
    assert len(lifecycle.storage.records) == 1
    assert lifecycle.storage.usage == len(b"fake-png")
    assert [path.read_bytes() for path in lifecycle.outputs.glob("generated-*.png")] == [b"fake-png"]
