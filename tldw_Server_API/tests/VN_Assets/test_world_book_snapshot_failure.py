"""Configured world-book outages abort new snapshots without changing V0 fallback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import VNAssetGenerationRequest
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import ChargedStorage
from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import handoff as handoff
from tldw_Server_API.tests.VN_Assets.test_generation_jobs import FakeGenerationGate, FakeImageAdapter, FakeImageRegistry


def configure_books(handoff: SimpleNamespace, *, with_entries: bool = True) -> list[int]:
    """Select two real books with distinct enabled context through the public repo."""
    books = WorldBookService(handoff.service.repo.db)
    ids = [books.create_world_book(f"Archive {index}") for index in range(2)]
    if with_entries:
        for index, book_id in enumerate(ids):
            books.add_entry(
                world_book_id=book_id, keywords=["archive"], content=f"Frozen archive lore {index}.",
            )
    handoff.service.repo.update_pack(handoff.payload["pack_id"], {
        "source_world_book_ids": ids, "style_prompt": "archive",
    })
    return ids


@pytest.mark.integration
@pytest.mark.parametrize("failing_book", [0, 1])
def test_configured_book_failure_rolls_back_new_submission(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, failing_book: int,
) -> None:
    """A first/partial read outage cannot freeze an incomplete V1 recipe or enqueue.

    Args:
        handoff: Real file-backed VN and Jobs with an existing immutable batch.
        monkeypatch: Fail one configured public book query, leaving other reads real.
        failing_book: First or second selected book, including a partial-read case.

    Returns:
        None: Verify safe error, atomic rollback and unchanged old recipes/Jobs rows.
    """
    ids = configure_books(handoff)
    repo, payload = handoff.service.repo, handoff.payload
    batches = repo.list_batches(payload["pack_id"])
    recipes = repo.list_batch_recipes(payload["batch_id"])
    jobs = handoff.jobs.list_jobs(domain="vn_assets", owner_user_id="1")
    original = WorldBookService.get_entries
    private_detail = "unpublished provider context should not escape"

    def fail_selected_book(self: WorldBookService, *, world_book_id: int, **kwargs: Any) -> list[Any]:
        """Raise a native configured-read failure without substituting other queries."""
        if world_book_id == ids[failing_book]:
            raise OSError(private_detail)
        return original(self, world_book_id=world_book_id, **kwargs)

    monkeypatch.setattr(WorldBookService, "get_entries", fail_selected_book)
    diagnostics: list[str] = []
    sink = logger.add(lambda message: diagnostics.append(str(message)), level="WARNING")
    try:
        with pytest.raises(VNAssetGenerationError, match="vn_asset_world_book_context_unavailable") as caught:
            handoff.service.start_generation(
                payload["pack_id"], user_id=1,
                request=VNAssetGenerationRequest(slot_ids=[payload["slot_id"]]),
            )
    finally:
        logger.remove(sink)
    assert caught.value.retryable
    assert caught.value.context["pack_id"] == payload["pack_id"]
    assert private_detail not in str(caught.value)
    assert private_detail not in repr(dict(caught.value.context))
    assert all(private_detail not in message for message in diagnostics)
    assert repo.list_batches(payload["pack_id"]) == batches
    assert repo.list_batch_recipes(payload["batch_id"]) == recipes
    assert handoff.jobs.list_jobs(domain="vn_assets", owner_user_id="1") == jobs
    assert not repo.db.get_connection().in_transaction


@pytest.mark.integration
@pytest.mark.parametrize("configuration", ["unconfigured", "empty", "valid"])
def test_snapshot_context_controls_remain_valid(handoff: SimpleNamespace, configuration: str) -> None:
    """Unconfigured/empty books remain valid and real selected content stays frozen."""
    repo, payload = handoff.service.repo, handoff.payload
    if configuration != "unconfigured":
        configure_books(handoff, with_entries=configuration == "valid")
    batch = handoff.service.start_generation(
        payload["pack_id"], user_id=1, request=VNAssetGenerationRequest(slot_ids=[payload["slot_id"]]),
    )
    recipe = repo.get_batch_recipe(batch.batch_id, payload["slot_id"], 0)
    assert recipe is not None
    for index in range(2):
        assert (f"Frozen archive lore {index}." in recipe["prompt"]) is (configuration == "valid")
    repo.update_pack(payload["pack_id"], {"source_world_book_ids": []})
    assert repo.get_batch_recipe(batch.batch_id, payload["slot_id"], 0) == recipe


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_generation_keeps_optional_context_fallback(
    handoff: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A V0 public delivery still generates with its legacy optional-book fallback."""
    configure_books(handoff)
    repo, payload = handoff.service.repo, dict(handoff.payload)
    batch = repo.create_batch(
        pack_id=payload["pack_id"], requested_by_user_id=1, status="enqueued",
        total_slots=1, total_variants=1, planned_count=1,
    )
    payload["batch_id"] = batch["id"]

    def unavailable_book(self: WorldBookService, **_kwargs: Any) -> list[Any]:
        """Represent a configured outage through the public query boundary."""
        raise OSError("legacy configured book unavailable")

    monkeypatch.setattr(WorldBookService, "get_entries", unavailable_book)
    adapter = FakeImageAdapter()
    storage = ChargedStorage(handoff.outputs)
    worker = VNAssetGenerationWorker(
        repo=repo, jobs_manager=handoff.jobs, image_registry=FakeImageRegistry(adapter),
        backend_gate=FakeGenerationGate(), save_vn_asset_image=storage, generated_files_repo=storage,
    )
    result = await worker.handle_generate_variant(payload)
    assert result["item_id"] > 0
    assert len(adapter.requests) == 1
    assert "Frozen archive lore" not in adapter.requests[0].prompt
    assert repo.get_batch(batch["id"])["completed_count"] == 1
