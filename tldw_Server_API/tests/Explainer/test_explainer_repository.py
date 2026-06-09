from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.Explainer_DB import ExplainerDatabase
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository

pytestmark = pytest.mark.unit


def test_create_goal_session_persists_root_node(tmp_path):
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)

    session = repo.create_session(
        owner_user_id="7",
        title="Learn attention",
        mode="goal",
        output_intent="explain",
        grounding="open",
        depth_preset="standard",
        selected_sources=[],
        root_prompt="Explain transformer attention",
    )

    loaded = repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.root_node_ids
    assert loaded.nodes[loaded.root_node_ids[0]].title == "Explain transformer attention"


def test_repository_rejects_cross_user_session_access(tmp_path):
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)
    session = repo.create_session(
        owner_user_id="7",
        title="Private",
        mode="goal",
        output_intent="plan",
        grounding="open",
        depth_preset="quick",
        selected_sources=[],
        root_prompt="Private topic",
    )

    assert repo.get_session(session.id, owner_user_id="8") is None


def test_create_node_persists_citation_snapshots(tmp_path):
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)
    session = repo.create_session(
        owner_user_id="7",
        title="Learn attention",
        mode="goal",
        output_intent="explain",
        grounding="source_led",
        depth_preset="standard",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
            }
        ],
        root_prompt="Explain transformer attention",
    )

    node = repo.create_node(
        session.id,
        owner_user_id="7",
        parent_id=session.root_node_ids[0],
        title="Scaled dot product attention",
        body="Attention compares every token against every other token.",
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "excerpt": "Attention weights are computed from query-key similarity.",
                "location_label": "chunk 3",
                "start_offset": 120,
                "end_offset": 178,
                "url": "https://example.test/attention",
                "snapshot_hash": "sha256:abc123",
            }
        ],
    )

    loaded = repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert node is not None
    [citation] = loaded.nodes[node.id].citations
    assert citation.source_id == "media-42"
    assert citation.excerpt == "Attention weights are computed from query-key similarity."
    assert citation.location_label == "chunk 3"
    assert citation.snapshot_hash == "sha256:abc123"


def test_update_node_replaces_citation_snapshots(tmp_path):
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)
    session = repo.create_session(
        owner_user_id="7",
        title="Learn attention",
        mode="goal",
        output_intent="explain",
        grounding="source_led",
        depth_preset="standard",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
            }
        ],
        root_prompt="Explain transformer attention",
    )
    node = repo.create_node(
        session.id,
        owner_user_id="7",
        parent_id=session.root_node_ids[0],
        title="Scaled dot product attention",
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "excerpt": "Initial citation.",
            }
        ],
    )
    assert node is not None

    updated = repo.update_node(
        session.id,
        node.id,
        owner_user_id="7",
        citations=[
            {
                "source_id": "media-99",
                "source_type": "note",
                "title": "Updated note",
                "excerpt": "Replacement citation.",
                "location_label": "paragraph 2",
            }
        ],
    )

    assert updated is not None
    [citation] = updated.citations
    assert citation.source_id == "media-99"
    assert citation.excerpt == "Replacement citation."
    assert citation.location_label == "paragraph 2"
