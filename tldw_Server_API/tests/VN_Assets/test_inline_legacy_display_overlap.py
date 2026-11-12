"""Native inline sibling overlap must retain the live delivery's slot display."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import ChargedStorage
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import handoff as handoff
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeGenerationGate, FakeImageAdapter, FakeImageRegistry


@pytest.mark.integration
@pytest.mark.asyncio
async def test_finishing_inline_delivery_preserves_running_sibling(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish a real inline delivery while a same-slot sibling is still generating.

    Args:
        handoff: Real file-backed VN/Jobs state and output storage; cancel its V1
            row to retain mixed-version history, then run two public V0 deliveries.
        monkeypatch: Observe the public connection getter without replacing native
            database operations or cleanup algorithms.

    Returns:
        None: Observe generating while the sibling adapter is held, both real
        completions/counters, distinct registration bytes and the live owner handle.

    Raises:
        AssertionError: A bounded dependency barrier, public outcome, byte identity
            or owned connection/thread cleanup contract is violated.

    Supported storage and adapter dependencies coordinate the overlap with Events.
    This covers the observable sibling display, not the historical counter read/write
    interleaving; it depends on no production source, AST, line or private counter.
    """
    repo, payload = handoff.service.repo, dict(handoff.payload)
    historical_batch = repo.get_batch(payload["batch_id"])
    assert historical_batch["recipe_version"] == 1
    repo.cancel_batch(payload["batch_id"])
    historical_outcome = repo.get_variant_outcome(payload["batch_id"], payload["slot_id"], 0)
    assert historical_outcome["outcome_status"] == "cancelled"
    batch = repo.create_batch(
        pack_id=payload["pack_id"], requested_by_user_id=1, status="enqueued",
        total_slots=1, total_variants=2, planned_count=2,
    )
    assert batch["recipe_version"] == 0
    payload["batch_id"] = batch["id"]
    owner = repo.db.get_connection()
    owner_thread = threading.get_ident()
    owned: list[tuple[threading.Thread, sqlite3.Connection]] = []
    get_connection = repo.db.get_connection
    first_saved, first_release, sibling_started, sibling_release = (threading.Event() for _ in range(4))
    storage = ChargedStorage(handoff.outputs)

    def observe_connection() -> sqlite3.Connection:
        """Forward native public acquisition and retain only foreign owned handles."""
        connection = get_connection()
        if threading.get_ident() != owner_thread and all(connection is not saved for _, saved in owned):
            owned.append((threading.current_thread(), connection))
        return connection

    monkeypatch.setattr(repo.db, "get_connection", observe_connection)

    async def save_then_hold(**kwargs: Any) -> dict[str, Any]:
        """Persist actual registration bytes, holding only the first delivery's save."""
        record = await storage(**kwargs)
        if kwargs["image_bytes"] == b"inline-variant-0":
            first_saved.set()
            assert await asyncio.to_thread(first_release.wait, 8), "first storage save was not released"
        return record

    class SiblingAdapter(FakeImageAdapter):
        """Hold the sibling call and return distinct bytes per requested variant."""

        def generate(self, request: ImageGenRequest) -> ImageGenResult:
            """Keep the sibling active until released, without altering worker logic."""
            if request.request_id.endswith(":1"):
                sibling_started.set()
                assert sibling_release.wait(8), "sibling adapter was not released"
            content = f"inline-variant-{request.request_id.rsplit(':', 1)[1]}".encode("ascii")
            return ImageGenResult(content=content, content_type="image/png", bytes_len=len(content))

    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(SiblingAdapter()),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=save_then_hold, generated_files_repo=storage,
    )
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    try:
        first = asyncio.create_task(worker.handle_generate_variant({**payload, "variant_index": 0}))
        tasks.append(first)
        assert await asyncio.to_thread(first_saved.wait, 5), "first registration did not reach storage barrier"
        sibling = asyncio.create_task(worker.handle_generate_variant({**payload, "variant_index": 1}))
        tasks.append(sibling)
        assert await asyncio.to_thread(sibling_started.wait, 5), "sibling did not reach adapter"
        first_release.set()
        first_result = await asyncio.wait_for(first, 8)
        assert not sibling.done()
        assert repo.get_slot(payload["slot_id"])["status"] == "generating"
        sibling_release.set()
        sibling_result = await asyncio.wait_for(sibling, 8)
        assert first_result["item_id"] != sibling_result["item_id"]
        completed = repo.get_batch(batch["id"])
        assert (completed["completed_count"], completed["failed_count"]) == (2, 0)
        assert completed["status"] == "completed"
        assert repo.get_slot(payload["slot_id"])["status"] == "reviewing"
        assert len(storage.records) == 2
        for result, content in [(first_result, b"inline-variant-0"), (sibling_result, b"inline-variant-1")]:
            item = repo.get_item(result["item_id"])
            record = storage.records[item["generated_file_id"]]
            assert record["source_ref"] == f"vn_asset_item:{item['id']}"
            assert record["file_size_bytes"] == len(content)
            assert (handoff.outputs / record["storage_path"]).read_bytes() == content
        assert storage.usage == len(b"inline-variant-0") + len(b"inline-variant-1")
        assert repo.get_variant_outcome(historical_batch["id"], payload["slot_id"], 0) == historical_outcome
        assert owned, "public delivery did not expose owned off-thread database work"
        for thread, connection in owned:
            assert not thread.is_alive()
            with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
                connection.execute("SELECT 1")
        assert owner.execute("SELECT 1").fetchone()[0] == 1
    finally:
        first_release.set()
        sibling_release.set()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
