"""Characterize post-registration lease takeover with native SQLite accounting.

Each case runs in the isolated AuthNZ runtime used by storage integration tests.
Only public saver/registry callbacks schedule real Jobs release and acquisition;
no worker helper, cleanup implementation, or quota ledger is replaced.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import _run_runtime, _runtime_env

if TYPE_CHECKING:
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker

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
            test = runpy.run_path(os.environ['VN_LEASE_TEST_FILE'])
            result = asyncio.run(test['characterize_takeover'](
                root, revoke_discovery=os.environ['VN_REVOKE_DISCOVERY'] == '1',
            ))
        except BaseException:
            traceback.print_exc()
            raise
print('RUNTIME_RESULT=' + json.dumps(result))
'''


async def characterize_takeover(
    root: Path, *, revoke_discovery: bool, worker_type: type[VNAssetGenerationWorker] | None = None,
) -> dict[str, Any]:
    """Exercise native recovery and return safe observations after all assertions.

    Args:
        root: Isolated runtime directory containing all database/output files.
        revoke_discovery: Revoke the successor after its real source lookup.
        worker_type: Optional copied consumer for evidence-only sensitivity runs.

    Returns:
        Safe counts and identities, never credentials or raw startup diagnostics.
    """
    from tldw_Server_API.app.api.v1.schemas.vn_asset_schemas import (
        VNAssetGenerationRequest,
        VNAssetPackCreate,
        VNAssetReviewRequest,
        VNAssetSlotCreate,
    )
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import bootstrap_single_user_profile, setup_database
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.exceptions import VNAssetGenerationError
    from tldw_Server_API.app.core.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Storage import generated_file_helpers as helpers
    from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime, is_test_mode
    from tldw_Server_API.app.core.VN_Assets.concurrency import BackendGenerationLease
    from tldw_Server_API.app.core.VN_Assets.jobs import vn_asset_generation_jobs_queue
    from tldw_Server_API.app.core.VN_Assets.service import VNAssetPackService
    from tldw_Server_API.app.core.VN_Assets.worker import VNAssetGenerationWorker
    from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

    assert not is_test_mode() and not is_explicit_pytest_runtime()
    worker_type = worker_type or VNAssetGenerationWorker
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
        orgs = AuthnzOrgsTeamsRepo(pool)
        organization = await orgs.create_organization(name="lease-storage", owner_user_id=owner)
        team = await orgs.create_team(org_id=organization["id"], name="lease-storage-team")
        org_id, team_id = int(organization["id"]), int(team["id"])
        quotas = AuthnzStorageQuotasRepo(pool)
        await quotas.upsert_org_quota(org_id, quota_mb=100)
        await quotas.upsert_team_quota(team_id, quota_mb=100)
        outputs = root / "outputs"
        db = CharactersRAGDB(str(root / "vn.db"), client_id="lease-storage-characterization")
        jobs = JobManager(db_path=root / "jobs.db")
        service = VNAssetPackService(db, owner_user_id=owner, jobs_manager=jobs)
        repo = service.repo
        character = db.add_character_card({"name": "Mira", "description": "Archivist"})
        pack = service.create_pack(VNAssetPackCreate(title="Lease recovery", primary_character_id=character))
        slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="recover"))
        approved_slot = service.create_slot(pack.id, VNAssetSlotCreate(asset_type="sprite", slot_key="approved"))

        async def get_storage() -> StorageQuotaService:
            """Supply the real initialized service at the helper's public boundary."""
            return storage

        async def accounting() -> dict[str, Any]:
            """Read actual user/org/team charges and profile version, without caches."""
            user = await users.get_user_by_id(owner)
            org = await quotas.get_org_quota(org_id)
            current_team = await quotas.get_team_quota(team_id)
            assert user is not None and org is not None and current_team is not None
            return {
                "usage": [float(user["storage_used_mb"]), float(org["used_mb"]), float(current_team["used_mb"])],
                "version": user["profile_version"],
            }

        def acquire(worker_id: str) -> dict[str, Any]:
            """Acquire the original variant row using the real Jobs lease facade."""
            job = jobs.acquire_next_job(
                domain="vn_assets", queue=vn_asset_generation_jobs_queue(), worker_id=worker_id,
                lease_seconds=120, job_type="vn_asset_generate_variant",
            )
            assert job is not None
            return job

        def replace(job: dict[str, Any], worker_id: str, successor_id: str) -> dict[str, Any]:
            """Release the current native lease and acquire a distinct successor lease."""
            assert jobs.release_job(job["id"], worker_id=worker_id, lease_id=job["lease_id"], enforce=True)
            successor = acquire(successor_id)
            assert successor["id"] == job["id"]
            assert successor["lease_id"] != job["lease_id"]
            return successor

        class ImageBoundary:
            """Supply deterministic provider bytes; native storage remains unchanged."""

            def __init__(self) -> None:
                """Record public provider invocations for no-regeneration assertions."""
                self.calls = 0

            def resolve_backend(self, requested: str | None) -> str:
                """Resolve the requested provider without invoking a model."""
                return requested or "stable_diffusion_cpp"

            def get_adapter(self, _backend: str) -> ImageBoundary:
                """Return the deterministic external-provider boundary."""
                return self

            def generate(self, _request: ImageGenRequest) -> ImageGenResult:
                """Return stable bytes and count actual model-boundary invocations."""
                self.calls += 1
                return ImageGenResult(content=b"native-lease-image", content_type="image/png", bytes_len=18)

            def try_acquire(self, backend: str, *, model: str | None = None) -> BackendGenerationLease:
                """Admit the provider through the supported generation gate interface."""
                return BackendGenerationLease(acquired=True, backend=backend, model=model)

        class DiscoveryBoundary:
            """Forward real registry reads and optionally revoke after discovery."""

            def __init__(self, revoke: Callable[[], None] | None = None) -> None:
                """Keep an optional one-shot public Jobs callback."""
                self.revoke = revoke
                self.discovered: list[int] = []

            async def get_file_by_source_ref(self, **identity: Any) -> dict[str, Any] | None:
                """Return the real owned registration before scheduling lease loss."""
                record = await files.get_file_by_source_ref(**identity)
                if record is not None:
                    self.discovered.append(int(record["id"]))
                    if self.revoke is not None:
                        callback, self.revoke = self.revoke, None
                        callback()
                return record

            async def get_file_by_id(self, file_id: int) -> dict[str, Any] | None:
                """Forward attached-file reads to the real native registry."""
                return await files.get_file_by_id(file_id)

        with patch.object(DatabasePaths, "get_user_outputs_dir", staticmethod(lambda _user_id: outputs)), \
                patch.object(helpers, "get_storage_service", get_storage):
            approved = repo.create_item(pack_id=pack.id, slot_id=approved_slot.id)
            approved_file = await helpers.save_and_register_vn_asset_image(
                user_id=owner, pack_id=pack.id, item_id=approved["id"], asset_type="sprite",
                image_bytes=b"approved-history", org_id=org_id, team_id=team_id,
            )
            repo.update_item_storage(
                approved["id"], generated_file_id=approved_file["id"], storage_ref=approved_file["storage_path"],
                mime_type="image/png", width=None, height=None, bytes=len(b"approved-history"),
            )
            service.review_item(approved["id"], VNAssetReviewRequest(review_status="approved", preferred=True))
            approved_before = repo.get_item(approved["id"])
            approved_slot_before = repo.get_slot(approved_slot.id)
            baseline = await accounting()
            baseline_files, baseline_count = await files.list_files(user_id=owner)
            baseline_bytes = await files.get_user_storage_usage(owner)
            baseline_physical = {str(path.relative_to(outputs)): path.read_bytes() for path in outputs.rglob("*.png")}
            batch = service.start_generation(
                pack.id, user_id=owner, request=VNAssetGenerationRequest(slot_ids=[slot.id]),
            )
            worker_type(repo=repo, jobs_manager=jobs).handle_enqueue_batch({
                "user_id": owner, "pack_id": pack.id, "batch_id": batch.batch_id,
            })
            old_job = acquire("old-worker")
            payload = old_job["payload"]
            recipe_before = repo.get_batch_recipe(batch.batch_id, slot.id, 0)
            image = ImageBoundary()
            saved: list[dict[str, Any]] = []
            deliveries: list[dict[str, Any]] = []

            async def save_then_revoke(**kwargs: Any) -> dict[str, Any]:
                """Commit real bytes/registration/accounting, then replace the Jobs lease."""
                record = await helpers.save_and_register_vn_asset_image(**kwargs, org_id=org_id, team_id=team_id)
                saved.append(record)
                item = repo.get_item(kwargs["item_id"])
                assert item is not None and item["generated_file_id"] is None
                assert (outputs / record["storage_path"]).read_bytes() == b"native-lease-image"
                deliveries.append(replace(old_job, "old-worker", "successor"))
                return record

            old_worker = worker_type(
                repo=repo, jobs_manager=jobs, image_registry=image, backend_gate=image,
                save_vn_asset_image=save_then_revoke, generated_files_repo=files,
                unregister_generated_file=storage.unregister_generated_file,
            )
            with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost") as error:
                await old_worker.handle_generate_variant(payload, job=old_job)
            assert error.value.retryable
            assert len(saved) == 1 and image.calls == 1
            record = saved[0]
            outcome = repo.get_variant_outcome(batch.batch_id, slot.id, 0)
            assert outcome is not None and outcome["outcome_status"] == "planned"
            item_id = outcome["item_id"]
            hidden = repo.get_item(item_id)
            assert hidden is not None and hidden["generated_file_id"] is None and hidden["storage_ref"] is None
            assert outcome["claim_lease_id"] == old_job["lease_id"]
            assert record["source_ref"] == f"vn_asset_item:{item_id}"
            assert record["user_id"] == owner and record["source_feature"] == "vn_assets"
            charged = await accounting()
            assert charged["usage"] == [used + 18 / 1048576 for used in baseline["usage"]]
            assert charged["version"] > baseline["version"]
            after_loss = repo.get_batch(batch.batch_id)
            assert after_loss is not None
            assert [after_loss[key] for key in ("completed_count", "failed_count", "cancelled_count")] == [0, 0, 0]
            assert all(item.id != item_id for item in service.list_items(pack.id))

            for identity in (
                {"user_id": owner + 1, "source_feature": "vn_assets", "source_ref": record["source_ref"]},
                {"user_id": owner, "source_feature": "chat", "source_ref": record["source_ref"]},
                {"user_id": owner, "source_feature": "vn_assets", "source_ref": f"vn_asset_item:{item_id + 1000}"},
            ):
                assert await files.get_file_by_source_ref(**identity) is None
            assert jobs.get_job(old_job["id"], owner_user_id=str(owner + 1)) is None
            with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost"):
                await old_worker.handle_generate_variant(payload, job=old_job)
            with pytest.raises(VNAssetGenerationError, match="vn_asset_job_owner_mismatch"):
                await old_worker.handle_generate_variant({**payload, "user_id": owner + 1}, job=deliveries[0])
            assert repo.get_variant_outcome(batch.batch_id, slot.id, 0) == outcome
            assert repo.get_item(item_id) == hidden
            assert repo.get_batch(batch.batch_id) == after_loss
            assert await accounting() == charged

            async def successor_save(**kwargs: Any) -> dict[str, Any]:
                """Forward to real idempotent storage while detecting unwanted regeneration."""
                saved.append(await helpers.save_and_register_vn_asset_image(**kwargs, org_id=org_id, team_id=team_id))
                return saved[-1]

            def revoke_lookup() -> None:
                """Revoke after a real registry read, before authoritative attachment."""
                deliveries.append(replace(deliveries[0], "successor", "final-successor"))

            discovery = DiscoveryBoundary(revoke_lookup if revoke_discovery else None)
            successor = worker_type(
                repo=repo, jobs_manager=jobs, image_registry=image, backend_gate=image,
                save_vn_asset_image=successor_save, generated_files_repo=discovery,
                unregister_generated_file=storage.unregister_generated_file,
            )
            if revoke_discovery:
                with pytest.raises(VNAssetGenerationError, match="vn_asset_job_lease_lost") as error:
                    await successor.handle_generate_variant(payload, job=deliveries[0])
                assert error.value.retryable
                pending = repo.get_variant_outcome(batch.batch_id, slot.id, 0)
                assert pending is not None and pending["item_id"] == item_id and pending["outcome_status"] == "planned"
                assert repo.get_item(item_id) == hidden
                assert repo.get_batch(batch.batch_id) == after_loss
                assert await accounting() == charged
                assert await files.get_file_by_id(record["id"]) == record
                assert (outputs / record["storage_path"]).read_bytes() == b"native-lease-image"
            result = await successor.handle_generate_variant(payload, job=deliveries[-1])
            observed_files, observed_count = await files.list_files(user_id=owner)
            observations = {
                "adapter_calls": image.calls, "saver_calls": len(saved),
                "target_live_files": len([row for row in observed_files if row["source_ref"] == record["source_ref"]]),
                "live_count_delta": observed_count - baseline_count,
                "physical_count_delta": len(list(outputs.rglob("*.png"))) - len(baseline_physical),
                "same_item": result["item_id"] == item_id,
                "target_file_id": record["id"], "target_item_id": item_id,
                "usage_delta_mb": [used - previous for used, previous in zip(charged["usage"], baseline["usage"])],
                "accounting_unchanged_on_replay": await accounting() == charged,
            }
            (root / "takeover-observations.json").write_text(json.dumps(observations, indent=2) + "\n")
            assert image.calls == 1, "takeover must discover storage before invoking the image adapter"
            assert len(saved) == 1, "takeover must not call the saver again"
            assert discovery.discovered == [record["id"]] * (2 if revoke_discovery else 1)
            assert result["item_id"] == item_id
            attached = repo.get_item(item_id)
            assert attached is not None and attached["generated_file_id"] == record["id"]
            assert attached["storage_ref"] == record["storage_path"] and attached["bytes"] == 18
            completed = repo.get_variant_outcome(batch.batch_id, slot.id, 0)
            assert completed is not None and completed["outcome_status"] == "completed" and completed["item_id"] == item_id
            assert completed["claim_lease_id"] == deliveries[-1]["lease_id"]
            assert completed["claim_token"] != outcome["claim_token"]
            final_batch = repo.get_batch(batch.batch_id)
            assert final_batch is not None and final_batch["status"] == "completed"
            assert [final_batch[key] for key in ("completed_count", "failed_count", "cancelled_count")] == [1, 0, 0]
            assert repo.get_batch_recipe(batch.batch_id, slot.id, 0) == recipe_before
            assert repo.get_item(approved["id"]) == approved_before
            assert repo.get_slot(approved_slot.id) == approved_slot_before
            assert await files.get_file_by_id(approved_file["id"]) == baseline_files[0]
            live_files, live_count = await files.list_files(user_id=owner)
            assert live_count == baseline_count + 1
            target_files = [row for row in live_files if row["source_ref"] == record["source_ref"]]
            assert target_files == [record]
            usage = await files.get_user_storage_usage(owner)
            assert usage["total_bytes"] == baseline_bytes["total_bytes"] + 18
            assert usage["trash_bytes"] == baseline_bytes["trash_bytes"] == 0
            physical = {str(path.relative_to(outputs)): path.read_bytes() for path in outputs.rglob("*.png")}
            assert physical == {**baseline_physical, record["storage_path"]: b"native-lease-image"}
            assert await accounting() == charged
            assert [item.id for item in service.list_items(pack.id)].count(item_id) == 1
            # A completed redelivery is observational; it cannot double-settle counters.
            assert await successor.handle_generate_variant(payload, job=deliveries[-1]) == result
            assert repo.get_batch(batch.batch_id) == final_batch
            assert await accounting() == charged
            assert image.calls == 1 and len(saved) == 1
            return {
                "backend": pool.backend_type, "revoke_discovery": revoke_discovery,
                "target_live_files": len(target_files), "target_bytes": 18,
                "adapter_calls": image.calls, "saver_calls": len(saved),
                "same_item": result["item_id"] == item_id,
                "same_file": attached["generated_file_id"] == record["id"],
                "charged_once_all_scopes": await accounting() == charged,
                "counters": [final_batch[key] for key in ("completed_count", "failed_count", "cancelled_count")],
                "approved_preserved": repo.get_item(approved["id"]) == approved_before,
            }
    finally:
        if db is not None:
            db.close_connection()
        await reset_db_pool()


@pytest.mark.integration
@pytest.mark.parametrize("revoke_discovery", [False, True], ids=["takeover", "discovery-lease-fence"])
def test_native_post_registration_takeover_reuses_storage(tmp_path: Path, revoke_discovery: bool) -> None:
    """A real successor reuses the charged file; stale and foreign authority cannot attach it."""
    env = _runtime_env(tmp_path, f"sqlite:///{tmp_path / 'users.db'}", backend="sqlite")
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1", "TMPDIR": str(tmp_path),
        "USER_DB_BASE_DIR": str(tmp_path / "user-databases"),
        "TLDW_DATABASE_DIR": str(tmp_path / "databases"),
        "SYSTEM_LOG_FILE_PATH": str(tmp_path / "system-logs.jsonl"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "VN_LEASE_TEST_FILE": str(Path(__file__).resolve()),
        "VN_REVOKE_DISCOVERY": str(int(revoke_discovery)),
    })
    assert _run_runtime(tmp_path, env, NATIVE_SCRIPT) == {
        "backend": "sqlite", "revoke_discovery": revoke_discovery,
        "target_live_files": 1, "target_bytes": 18, "adapter_calls": 1, "saver_calls": 1,
        "same_item": True, "same_file": True, "charged_once_all_scopes": True,
        "counters": [1, 0, 0], "approved_preserved": True,
    }
