"""Actual default-dataset creation must fence prior local review authority."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_legacy_mutations import (
    local_product_db as local_product_db,
)
from tldw_Server_API.tests.Sync.test_sync_v2_profile_bootstrap import _service

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("entry", ["profile", "personal-context", "personal-context-supplied"])
def test_default_creation_reserves_only_real_canonical_authority(local_product_db, tmp_path, entry):
    db = local_product_db
    service, sync_store = _service(tmp_path)
    service.materializers["notes.note"] = NotesMaterializer(db)
    if entry == "profile":
        result = service.bootstrap_profile(
            user_id=db.client_id,
            mode="offline_sync",
            device_id="local-transition-device",
            requested_domains=["notes.note"],
        )
        dataset_id = result.dataset.dataset_id
    else:
        supplied = (
            sync_store.get_or_create_default_personal_dataset(db.client_id)
            if entry == "personal-context-supplied"
            else None
        )
        result = service._profile_manager()._bind_personal_context_dataset(
            user_id=db.client_id,
            manifest=SimpleNamespace(profile_id="local-transition-profile"),
            authority_id="local-transition-authority",
            integrity_key_id="local-transition-key",
            purge_generation=0,
            dataset=supplied,
        )
        dataset_id = result.dataset_id
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT owner_user_id,dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound "
            "FROM note_task_scope_authority WHERE owner_user_id=?",
            (db.client_id,),
        ).fetchall()
    assert [dict(row) for row in rows] == [
        {
            "owner_user_id": db.client_id,
            "dataset_id": dataset_id,
            "task_graph_bound": False,
            "moodboard_graph_bound": False,
            "studio_graph_bound": False,
        }
    ]
