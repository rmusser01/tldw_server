"""Generated optimization candidates must remain readable by public Prompt APIs."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import UUID

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.prompt_studio.prompt_studio_prompts import get_prompt
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import ensure_prompt_studio_rls
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories.prompts import PromptsRepository
from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import PromptStudioDatabase
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import MCTSOptimizer
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_engine import (
    BootstrapOptimizer,
    MIPROOptimizer,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.optimization_strategies import (
    GeneticOptimizer,
    IterativeRefinementOptimizer,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["iterative", "genetic", "mipro", "bootstrap", "mcts"])
async def test_generated_candidate_is_publicly_readable(isolated_db, strategy):
    db = isolated_db
    project = db.create_project(name="Public generated candidates", user_id="1")
    base = db.create_prompt(
        project_id=project["id"], name="Director", system_prompt="Use the source.", user_prompt="{{source}}"
    )
    if strategy == "iterative":
        candidate_id = await IterativeRefinementOptimizer(db, None)._create_refined_prompt(
            base["id"], "Keep the full name."
        )
    elif strategy == "genetic":
        candidate_id = await GeneticOptimizer(db, None)._individual_to_prompt(
            {"system_prompt": "Keep the full name.", "user_prompt": "{{source}}"}, base["id"]
        )
    elif strategy == "mipro":
        candidate_id = await MIPROOptimizer(db, None)._create_prompt_variant(base["id"], "Keep the full name.")
    elif strategy == "bootstrap":
        candidate_id = await BootstrapOptimizer(db, None)._create_prompt_with_examples(
            base["id"], [{"inputs": {"source": "Rowan"}, "actual_output": {"response": "Dr. Mira Vale"}}]
        )
    else:
        candidate_id = MCTSOptimizer(db, None)._create_ephemeral_prompt_version(
            base_prompt=base, system_text="Keep the full name.", user_text="{{source}}"
        )
    public = await get_prompt(candidate_id, db=db, user_context={"user_id": "1", "is_admin": False})
    assert public.success and public.data.id == candidate_id  # nosec B101 - Public candidate roundtrip regression.
    assert public.data.parent_version_id == base["id"]  # nosec B101 - Preserve immutable lineage.
    assert db.get_prompt(base["id"]) == base  # nosec B101 - Generation must not overwrite the original.


@pytest.fixture(params=["sqlite", "postgresql"])
def repair_database(request, tmp_path):
    backend = request.getfixturevalue("pg_restricted_backend") if request.param == "postgresql" else None
    db = PromptStudioDatabase(tmp_path / "repair.db", client_id="audit-client", tenant_user_id="1", backend=backend)
    try:
        yield db, tmp_path / "repair.db", backend
    finally:
        db.close()


@pytest.mark.asyncio
async def test_existing_generated_uuid_upgrade_preserves_records(repair_database):
    db, path, backend = repair_database
    project = db.create_project(name="Legacy candidates", user_id="1")
    original = db.create_prompt(project["id"], "Original", user_prompt="{{source}}")
    legacy = [
        "refined-1792351268.528022",
        "genetic-1792351268.611687",
        "mcts-1792351268.653234",
        "opt-12345678-1234-4567-8123-123456789abc",
        "bootstrap-12345678-1234-4567-8123-123456789abc",
    ]
    untouched = ["refined-not-a-timestamp", "opt-not-a-uuid", "genetic-12x", "custom-1792351268.5"]
    records = []
    for index, old in enumerate(legacy + untouched):
        record = db.create_prompt(
            project["id"], f"Old candidate {index}", parent_version_id=original["id"], user_prompt="Keep this text."
        )
        with db.transaction() as conn:
            db._cursor_exec(conn, "UPDATE prompt_studio_prompts SET uuid = ? WHERE id = ?", (old, record["id"]))
        records.append(db.get_prompt(record["id"]))
    case = db.create_test_case(project["id"], "Retained case", inputs={}, expected_outputs={"response": "kept"})
    run = db.create_test_run(
        project_id=project["id"],
        prompt_id=records[0]["id"],
        test_case_id=case["id"],
        model_name="recorded-model",
        inputs={},
        outputs={"response": "kept"},
    )
    before_run = db.get_test_run(run["id"])
    db.close()  # Release the read before initialization takes schema locks.
    reopened = PromptStudioDatabase(path, client_id="audit-client", tenant_user_id="1", backend=backend)
    try:
        repaired = []
        for record in records[:5]:
            public = await get_prompt(record["id"], db=reopened, user_context={"user_id": "1", "is_admin": False})
            current = reopened.get_prompt(record["id"])
            UUID(current["uuid"])
            assert public.success and current["uuid"] != record["uuid"]  # nosec B101 - Causal legacy identity repair.
            assert {k: v for k, v in current.items() if k != "uuid"} == {k: v for k, v in record.items() if k != "uuid"}  # nosec B101 - Repair changes identity only.
            repaired.append(current)
        for record in records[5:]:
            assert reopened.get_prompt(record["id"]) == record  # nosec B101 - Never normalize unknown malformed identifiers.
        assert reopened.get_prompt(original["id"]) == original  # nosec B101 - Valid original remains byte-for-field unchanged.
        assert reopened.get_test_run(run["id"]) == before_run  # nosec B101 - Retain saved output, provenance and links.
        reopened.close()
        again = PromptStudioDatabase(path, client_id="second-initializer", tenant_user_id="1", backend=backend)
        try:
            assert [again.get_prompt(p["id"]) for p in repaired] == repaired  # nosec B101 - Repeat initializer is idempotent.
        finally:
            again.close()
    finally:
        reopened.close()


def test_competing_generated_uuid_repairs_keep_one_identity(repair_database):
    db, path, backend = repair_database
    peer = PromptStudioDatabase(path, client_id="competing-initializer", tenant_user_id="1", backend=backend)
    try:
        project = db.create_project(name="Concurrent repair", user_id="1")
        record = db.create_prompt(project["id"], "Old concurrent candidate", user_prompt="Keep this.")
        with db.transaction() as conn:
            db._cursor_exec(
                conn, "UPDATE prompt_studio_prompts SET uuid = ? WHERE id = ?", ("refined-1792351268.5", record["id"])
            )
        db.close()
        peer.close()
        barrier = Barrier(2)

        def repair(instance):
            try:
                barrier.wait(timeout=5)
                PromptsRepository(instance._impl).repair_generated_uuids()
                return instance.get_prompt(record["id"])
            finally:
                instance.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(repair, [db, peer]))
        assert results[0] == results[1]  # nosec B101 - Both competing repairs observe the same committed identity.
        UUID(results[0]["uuid"])
        assert {k: v for k, v in results[0].items() if k != "uuid"} == {k: v for k, v in record.items() if k != "uuid"}  # nosec B101 - No competing content or version changes.
    finally:
        peer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("policies", [False, True])
async def test_generated_uuid_repair_respects_postgres_tenant(pg_restricted_backend, tmp_path, policies):
    path = tmp_path / "tenant-repair.db"
    alice = PromptStudioDatabase(path, client_id="alice-audit", tenant_user_id="1", backend=pg_restricted_backend)
    bob = None
    try:
        alice.close()
        if policies:
            assert ensure_prompt_studio_rls(pg_restricted_backend) is True  # nosec B101 - Install actual supported RLS policies.
        project = alice.create_project(name="Alice repair", user_id="1")
        record = alice.create_prompt(project["id"], "Alice old candidate", user_prompt="Private source")
        old = "refined-1792351268.5"
        with alice.transaction() as conn:
            alice._cursor_exec(conn, "UPDATE prompt_studio_prompts SET uuid = ? WHERE id = ?", (old, record["id"]))
        alice.close()
        bob = PromptStudioDatabase(path, client_id="bob-audit", tenant_user_id="2", backend=pg_restricted_backend)
        if policies:
            assert bob.get_prompt(record["id"]) is None  # nosec B101 - Actual policy filters foreign candidates.
        with pytest.raises(HTTPException) as denied:
            await get_prompt(record["id"], db=bob, user_context={"user_id": "2", "is_admin": False})
        assert denied.value.status_code == (404 if policies else 403)  # nosec B101 - Public ownership remains enforced before/after RLS.
        bob.close()
        assert alice.get_prompt(record["id"])["uuid"] == old  # nosec B101 - Bob's initializer cannot repair Alice's row.
        alice.close()
        reopened = PromptStudioDatabase(
            path, client_id="alice-audit", tenant_user_id="1", backend=pg_restricted_backend
        )
        try:
            UUID(reopened.get_prompt(record["id"])["uuid"])
        finally:
            reopened.close()
    finally:
        alice.close()
        if bob is not None:
            bob.close()
