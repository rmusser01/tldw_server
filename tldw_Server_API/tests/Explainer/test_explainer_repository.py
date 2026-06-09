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
