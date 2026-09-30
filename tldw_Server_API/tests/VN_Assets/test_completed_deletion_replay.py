"""Native completed Jobs replay after deliberate VN draft cleanup."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import _run_runtime, _runtime_env

NATIVE_SCRIPT = r'''
import asyncio
import contextlib
import json
import os
import runpy
import traceback
from pathlib import Path
root = Path.cwd()
with (root / 'native.stdout.log').open('w') as stdout, (root / 'native.stderr.log').open('w') as stderr:
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            test = runpy.run_path(os.environ['VN_DELETION_TEST_FILE'])
            result = asyncio.run(test['exercise_completed_deletion'](root))
        except BaseException:
            traceback.print_exc()
            raise
print('RUNTIME_RESULT=' + json.dumps(result))
'''


async def exercise_completed_deletion(root: Path) -> dict[str, Any]:
    """Generate, deliberately clean up, and redeliver with native persistence.

    Args:
        root: Isolated runtime containing databases and physical output files.

    Returns:
        Safe observations after asserting replay, accounting and corruption guards.
    """
    from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
        VNAssetCleanupRequest,
        VNAssetGenerationRequest,
        VNAssetPackCreate,
        VNAssetReviewRequest,
        VNAssetSlotCreate,
    )
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile, setup_database
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Storage import generated_file_helpers as helpers
    from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime, is_test_mode
    from tldw_Server_API.app.core.VN_Assets.jobs import vn_asset_generation_jobs_queue
    from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
    from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
    from tldw_Server_API.tests.DB_Management.vn_asset_corruption import (
        drop_deletion_receipt_column,
        set_deletion_receipt,
        set_recipe_item,
    )
    from tldw_Server_API.tests.VN_Assets.test_generation_jobs import (
        FakeGenerationGate,
        FakeImageAdapter,
        FakeImageRegistry,
    )

    assert not is_test_mode() and not is_explicit_pytest_runtime()
    db = None
    try:
        assert await setup_database()
        await bootstrap_single_user_profile()
        pool = await get_db_pool()
        assert pool.backend_type == "sqlite"
        owner = get_settings().SINGLE_USER_FIXED_ID
        storage = StorageQuotaService(pool)
        await storage.initialize()
        files = await storage.get_generated_files_repo()
        users = AuthnzUsersRepo(pool)
        db = CharactersRAGDB(str(root / "vn.db"), client_id="completed-deletion-test")
        jobs = JobManager(db_path=root / "jobs.db")
        service = VNAssetPackService(db, owner_user_id=owner, jobs_manager=jobs)
        repo = service.repo
        character = db.add_character_card({"name": "Mira", "description": "Archivist"})
        pack = service.create_pack(VNAssetPackCreate(title="Deletion replay", primary_character_id=character))
        slot = service.create_slot(pack.id, VNAssetSlotCreate(
            asset_type="sprite", slot_key="portrait", width=64, height=64, variant_count=2,
        ))
        outputs = root / "outputs"
        adapter = FakeImageAdapter()
        saved: list[dict[str, Any]] = []

        async def get_storage() -> StorageQuotaService:
            """Return the initialized native storage service.

            Returns:
                Real quota and generated-file service, not a fake registry.
            """
            return storage

        async def save_image(**kwargs: Any) -> dict[str, Any]:
            """Count calls while forwarding the public native saver.

            Args:
                kwargs: Original worker saver arguments.

            Returns:
                Actual committed generated-file registration.
            """
            record = await helpers.save_and_register_vn_asset_image(**kwargs)
            saved.append(record)
            return record

        with patch.object(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: outputs)), \
                patch.object(helpers, "get_storage_service", get_storage):
            batch = service.start_generation(pack.id, user_id=owner, request=VNAssetGenerationRequest(slot_ids=[slot.id]))
            worker = VNAssetGenerationWorker(
                repo=repo, jobs_manager=jobs, image_registry=FakeImageRegistry(adapter),
                backend_gate=FakeGenerationGate(), save_vn_asset_image=save_image,
                generated_files_repo=files, unregister_generated_file=storage.unregister_generated_file,
            )
            worker.handle_enqueue_batch({"user_id": owner, "pack_id": pack.id, "batch_id": batch.batch_id})
            deliveries: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for index in range(2):
                job = jobs.acquire_next_job(
                    domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id="deletion-worker",
                    lease_seconds=120, job_type="vn_asset_generate_variant",
                )
                assert job is not None and job["payload"]["variant_index"] == index
                result = await worker.handle_generate_variant(job["payload"], job=job)
                assert jobs.complete_job(job["id"], result=result, worker_id="deletion-worker",
                                         lease_id=job["lease_id"], enforce=True)
                deliveries.append((job, result))
            target_job, original = deliveries[0]
            sibling = deliveries[1][1]
            service.review_item(sibling["item_id"], VNAssetReviewRequest(review_status="approved", preferred=True))
            approved = repo.get_item(sibling["item_id"])
            slot_before = repo.get_slot(slot.id)
            batch_before = repo.get_batch(batch.batch_id)
            recipe_before = repo.get_batch_recipe(batch.batch_id, slot.id, 0)
            original_job = jobs.get_job(target_job["id"])
            assert batch_before["status"] == "completed"
            assert [batch_before[k] for k in ("completed_count", "failed_count", "cancelled_count")] == [2, 0, 0]

            # Unmarked missing state is not a deliberate deletion receipt.
            set_recipe_item(repo, batch.batch_id, slot.id, None)
            drop_deletion_receipt_column(repo)
            repo.initialize_schema()
            assert repo.get_variant_outcome(batch.batch_id, slot.id, 0)["deleted_item_json"] is None
            with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
                await worker.handle_generate_variant(target_job["payload"], job=target_job)
            set_recipe_item(repo, batch.batch_id, slot.id, original["item_id"])

            target = repo.get_item(original["item_id"])
            target_path = outputs / target["storage_ref"]
            assert target_path.read_bytes() == adapter.content
            outcome_before = repo.get_variant_outcome(batch.batch_id, slot.id, 0)
            with pytest.raises(RuntimeError, match="rollback deletion"):
                with repo.generation_transaction():
                    assert repo.delete_item(original["item_id"])
                    assert repo.get_item(original["item_id"]) is None
                    assert repo.get_variant_outcome(batch.batch_id, slot.id, 0)["deleted_item_json"] is not None
                    raise RuntimeError("rollback deletion")
            assert repo.get_item(original["item_id"]) == target
            assert repo.get_variant_outcome(batch.batch_id, slot.id, 0) == outcome_before
            assert target_path.read_bytes() == adapter.content
            cleaned = await service.cleanup_pack(
                pack.id, VNAssetCleanupRequest(dry_run=False, statuses=["draft"], item_ids=[original["item_id"]]),
                files_repo=files, unregister_generated_file=storage.unregister_generated_file,
            )
            assert cleaned.removed_item_ids == [original["item_id"]]
            assert repo.get_item(original["item_id"]) is None
            assert await files.get_file_by_id(original["generated_file_id"]) is None
            assert not target_path.exists()
            usage_after_cleanup = await files.get_user_storage_usage(owner)
            user_after_cleanup = await users.get_user_by_id(owner)
            physical = {str(p.relative_to(outputs)): p.read_bytes() for p in outputs.rglob("*.png")}

            assert await worker.handle_generate_variant(target_job["payload"], job=target_job) == original
            outcome = repo.get_variant_outcome(batch.batch_id, slot.id, 0)
            receipt = outcome["deleted_item_json"]
            identity = json.loads(receipt)
            for key in ("owner_user_id", "pack_id", "slot_id", "variant_index", "batch_id"):
                set_deletion_receipt(repo, batch.batch_id, slot.id, json.dumps({**identity, key: identity[key] + 1}))
                with pytest.raises(VNAssetGenerationError, match="vn_asset_recipe_item_missing"):
                    await worker.handle_generate_variant(target_job["payload"], job=target_job)
            for invalid in ("[" * 1200 + "]" * 1200, '{"id": ' + "9" * 5000 + "}",
                            "{", "[]", "{}", json.dumps({**identity, "id": 0}),
                            json.dumps({**identity, "generated_file_id": 0}), json.dumps({**identity, "id": True})):
                set_deletion_receipt(repo, batch.batch_id, slot.id, invalid)
                with pytest.raises(VNAssetGenerationError, match="vn_asset_recipe_item_missing"):
                    await worker.handle_generate_variant(target_job["payload"], job=target_job)
            set_deletion_receipt(repo, batch.batch_id, slot.id, None)
            with pytest.raises(VNAssetGenerationError, match="vn_asset_batch_terminal"):
                await worker.handle_generate_variant(target_job["payload"], job=target_job)
            set_deletion_receipt(repo, batch.batch_id, slot.id, receipt)
            for _ in range(2):
                assert await worker.handle_generate_variant(target_job["payload"], job=target_job) == original
            assert jobs.get_job(target_job["id"]) == original_job
            assert repo.get_batch(batch.batch_id) == batch_before
            assert repo.get_batch_recipe(batch.batch_id, slot.id, 0) == recipe_before
            assert repo.get_item(sibling["item_id"]) == approved
            assert repo.get_slot(slot.id) == slot_before
            assert await files.get_user_storage_usage(owner) == usage_after_cleanup
            assert await users.get_user_by_id(owner) == user_after_cleanup
            assert {str(p.relative_to(outputs)): p.read_bytes() for p in outputs.rglob("*.png")} == physical
            assert len(adapter.requests) == len(saved) == 2
            assert repo.get_item(original["item_id"]) is None
            assert await files.get_file_by_id(original["generated_file_id"]) is None
            return {"backend": "sqlite", "replayed_original": True, "approved_preserved": True,
                    "adapter_calls": 2, "saver_calls": 2, "counters": [2, 0, 0]}
    finally:
        if db is not None:
            db.close_connection()
        await reset_db_pool()


@pytest.mark.integration
def test_native_completed_deletion_replays_without_resurrection(tmp_path: Path) -> None:
    """Retain original Jobs outcomes after cleanup while rejecting corrupt receipts.

    Args:
        tmp_path: Private subprocess databases and output directory.

    Returns:
        None; asserts native replay/accounting observations without leaking logs.
    """
    env = _runtime_env(tmp_path, f"sqlite:///{tmp_path / 'users.db'}", backend="sqlite")
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_path),
        "USER_DB_BASE_DIR": str(tmp_path / "user-databases"),
        "TLDW_DATABASE_DIR": str(tmp_path / "databases"),
        "SYSTEM_LOG_FILE_PATH": str(tmp_path / "system-logs.jsonl"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "VN_DELETION_TEST_FILE": str(Path(__file__).resolve()),
    })
    try:
        assert _run_runtime(tmp_path, env, NATIVE_SCRIPT) == {
            "backend": "sqlite", "replayed_original": True, "approved_preserved": True,
            "adapter_calls": 2, "saver_calls": 2, "counters": [2, 0, 0],
        }
    finally:
        if evidence := os.environ.get("TASK63_EVIDENCE"):
            shutil.copytree(tmp_path, Path(evidence) / "native")
