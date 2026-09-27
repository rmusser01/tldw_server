"""Native inline sibling overlap must retain the live delivery's slot display."""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management import VNAssetPacks_DB
from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import ChargedStorage
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import handoff as handoff
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeGenerationGate, FakeImageAdapter, FakeImageRegistry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_finishing_inline_delivery_preserves_running_sibling(handoff: SimpleNamespace) -> None:
    """Force preemption between cleanup's counter read/write without mocking state.

    Args:
        handoff: Real file-backed VN/Jobs state and output storage; cancel its V1
            row to retain mixed-version history, then run two public V0 deliveries.

    Returns:
        None: Observe generating while the sibling adapter is held, both real
        completions/counters, distinct registration bytes and the live owner handle.

    Standard thread tracing holds the first cleanup at its read/write boundary.
    A foreign observer releases it after the sibling reaches its adapter, or
    after a bounded wait when serialization prevents that interleaving. The trace
    reads no private map/connection pool and replaces no repository/SQL algorithm.
    """
    repo, payload = handoff.service.repo, dict(handoff.payload)
    repo.cancel_batch(payload["batch_id"])
    batch = repo.create_batch(
        pack_id=payload["pack_id"], requested_by_user_id=1, status="enqueued",
        total_slots=1, total_variants=2, planned_count=2,
    )
    payload["batch_id"] = batch["id"]
    owner = repo.db.get_connection()
    paused, resume, sibling_started, sibling_release = (threading.Event() for _ in range(4))
    source = Path(VNAssetPacks_DB.__file__).resolve()
    tree = ast.parse(source.read_text())
    finish = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == "finish_legacy_display")
    stop_line = next(node.lineno for node in ast.walk(finish) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "remaining > 0")

    def trace(frame: FrameType, event: str, _argument: Any) -> Any:
        """Pause only the first native cleanup after its local counter read."""
        if (event == "line" and frame.f_code.co_filename == str(source)
                and frame.f_lineno == stop_line and not paused.is_set()):
            paused.set()
            assert resume.wait(8), "cleanup interleaving observer did not release"
        return trace

    class SiblingAdapter(FakeImageAdapter):
        """Hold the second actual model call until the first delivery finishes."""

        def generate(self, request: ImageGenRequest) -> ImageGenResult:
            """Keep a real sibling active, then use the deterministic image result."""
            if request.request_id.endswith(":1"):
                sibling_started.set()
                assert sibling_release.wait(8), "sibling adapter was not released"
            return super().generate(request)

    def release_cleanup() -> None:
        """Allow the old interleaving, but release an actually serialized begin."""
        sibling_started.wait(3)
        resume.set()

    storage = ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(SiblingAdapter()),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
    )
    previous_trace = threading.gettrace()
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    observer = threading.Thread(target=release_cleanup, name="inline-overlap-observer")
    threading.settrace(trace)
    try:
        first = asyncio.create_task(worker.handle_generate_variant({**payload, "variant_index": 0}))
        tasks.append(first)
        assert await asyncio.to_thread(paused.wait, 5), "first cleanup did not reach preemption boundary"
        observer.start()
        sibling = asyncio.create_task(worker.handle_generate_variant({**payload, "variant_index": 1}))
        tasks.append(sibling)
        first_result = await first
        assert await asyncio.to_thread(sibling_started.wait, 3), "sibling did not reach adapter"
        assert not sibling.done()
        assert repo.get_slot(payload["slot_id"])["status"] == "generating"
        sibling_release.set()
        sibling_result = await sibling
        assert first_result["item_id"] != sibling_result["item_id"]
        completed = repo.get_batch(batch["id"])
        assert (completed["completed_count"], completed["failed_count"]) == (2, 0)
        assert len(storage.records) == 2
        assert all((handoff.outputs / record["storage_path"]).read_bytes() == b"fake-png"
                   for record in storage.records.values())
        assert owner.execute("SELECT 1").fetchone()[0] == 1
    finally:
        resume.set()
        sibling_release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        if observer.ident is not None:
            observer.join(timeout=5)
            assert not observer.is_alive()
        threading.settrace(previous_trace)
