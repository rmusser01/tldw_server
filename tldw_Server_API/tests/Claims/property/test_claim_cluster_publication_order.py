"""Real-store publication permutations preserve high watermarks and receipts."""

from dataclasses import replace
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterCalculation
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import request, state

pytestmark = pytest.mark.property


@settings(max_examples=20, deadline=None)
@given(
    order=st.lists(st.integers(min_value=1, max_value=6), min_size=1, max_size=15),
    skips=st.sets(st.integers(min_value=1, max_value=6)),
)
def test_arbitrary_publication_order_never_regresses_head_or_rewrites_replays(order, skips):
    with TemporaryDirectory() as directory:
        db = MediaDatabase(db_path=f"{directory}/order.db", client_id="42")
        try:
            req = request(db)
            skip = request(db, method="embeddings", authority=req["identity"].authority_uuid)
            skip["calculation"] = ClusterCalculation((), "skipped", "no_embeddings", 1)
            requests = {
                generation: {
                    **(skip if generation in skips else req),
                    "identity": replace(
                        (skip if generation in skips else req)["identity"],
                        generation=generation,
                        job_uuid=f"00000000-0000-0000-0000-{generation:012d}",
                    ),
                }
                for generation in set(order)
            }
            highest = 0
            committed = {}
            for generation in order:
                before = state(db)
                result = db.publish_claim_cluster_rebuild(**requests[generation])
                if generation in committed:
                    assert not result.newly_published and result.result == committed[generation]
                    assert state(db) == before
                elif generation < highest:
                    assert not result.newly_published and result.result["reason"] == "superseded"
                    assert state(db) == before
                else:
                    assert result.newly_published
                    committed[generation] = dict(result.result)
                    highest = generation
                    if generation in skips:
                        after = state(db)
                        for table in ("claim_clusters", "claim_cluster_membership", "Claims"):
                            assert after[table] == before[table]
                head = db.execute_query("SELECT generation FROM claims_cluster_publication_heads").fetchone()[0]
                assert head == highest
            assert len(db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall()) == len(
                committed
            )
        finally:
            db.close_connection()
