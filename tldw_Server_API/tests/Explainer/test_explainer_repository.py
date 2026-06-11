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


def test_update_node_can_clear_nullable_fields(tmp_path):
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
    node = repo.create_node(
        session.id,
        owner_user_id="7",
        parent_id=session.root_node_ids[0],
        title="Scaled dot product attention",
        body="Body to clear",
    )
    assert node is not None
    updated = repo.update_node(
        session.id,
        node.id,
        owner_user_id="7",
        selected_custom_answer="Answer to clear",
        generation_metadata={"model": "test"},
    )
    assert updated is not None
    assert updated.body == "Body to clear"
    assert updated.selected_custom_answer == "Answer to clear"
    assert updated.generation_metadata == {"model": "test"}

    cleared = repo.update_node(
        session.id,
        node.id,
        owner_user_id="7",
        body=None,
        selected_custom_answer=None,
        generation_metadata=None,
    )

    assert cleared is not None
    assert cleared.body is None
    assert cleared.selected_custom_answer is None
    assert cleared.generation_metadata is None


def test_delete_node_soft_deletes_descendant_subtree_and_citations(tmp_path):
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
    parent = repo.create_node(
        session.id,
        owner_user_id="7",
        parent_id=session.root_node_ids[0],
        title="Parent branch",
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "excerpt": "Parent citation.",
            }
        ],
    )
    assert parent is not None
    child = repo.create_node(
        session.id,
        owner_user_id="7",
        parent_id=parent.id,
        title="Child branch",
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "excerpt": "Child citation.",
            }
        ],
    )
    assert child is not None
    grandchild = repo.create_node(
        session.id,
        owner_user_id="7",
        parent_id=child.id,
        title="Grandchild branch",
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "excerpt": "Grandchild citation.",
            }
        ],
    )
    assert grandchild is not None

    assert repo.delete_node(session.id, parent.id, owner_user_id="7") is True

    loaded = repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert parent.id not in loaded.nodes
    assert child.id not in loaded.nodes
    assert grandchild.id not in loaded.nodes
    active_citation_count = db.get_connection().execute(
        """
        SELECT COUNT(*) AS count
        FROM explainer_citations
        WHERE session_id = ? AND deleted_at IS NULL
        """,
        (session.id,),
    ).fetchone()["count"]
    assert active_citation_count == 0


def test_list_session_summaries_returns_lightweight_page(tmp_path):
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)
    first = repo.create_session(
        owner_user_id="7",
        title="First",
        mode="goal",
        output_intent="explain",
        grounding="open",
        depth_preset="quick",
        selected_sources=[],
        root_prompt="First topic",
    )
    second = repo.create_session(
        owner_user_id="7",
        title="Second",
        mode="goal",
        output_intent="plan",
        grounding="open",
        depth_preset="standard",
        selected_sources=[],
        root_prompt="Second topic",
    )
    repo.create_node(
        second.id,
        owner_user_id="7",
        parent_id=second.root_node_ids[0],
        title="Second detail",
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "excerpt": "List summaries must not expose this excerpt.",
            }
        ],
    )
    repo.create_session(
        owner_user_id="8",
        title="Other user",
        mode="goal",
        output_intent="explain",
        grounding="open",
        depth_preset="quick",
        selected_sources=[],
        root_prompt="Other topic",
    )

    summaries, total = repo.list_session_summaries(owner_user_id="7", limit=1, offset=0)

    assert total == 2
    assert len(summaries) == 1
    assert summaries[0].id in {first.id, second.id}
    assert summaries[0].owner_user_id == "7"
    assert not hasattr(summaries[0], "nodes")


def _make_session(repo: ExplainerRepository, owner_user_id: str = "7"):
    return repo.create_session(
        owner_user_id=owner_user_id,
        title="Learn attention",
        mode="goal",
        output_intent="explain",
        grounding="open",
        depth_preset="standard",
        selected_sources=[],
        root_prompt="Explain transformer attention",
    )


def test_create_node_persists_generation_metadata(tmp_path):
    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    session = _make_session(repo)

    node = repo.create_node(
        session.id,
        owner_user_id="7",
        parent_id=session.root_node_ids[0],
        title="Child with metadata",
        generation_metadata={"provider": "fake", "model": "fake-model"},
    )

    assert node is not None
    assert node.generation_metadata == {"provider": "fake", "model": "fake-model"}


def test_create_child_nodes_persists_batch_with_citations_and_metadata(tmp_path):
    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    session = _make_session(repo)
    root_id = session.root_node_ids[0]

    created = repo.create_child_nodes(
        session.id,
        owner_user_id="7",
        parent_id=root_id,
        children=[
            {
                "title": "First child",
                "body": "Body one",
                "kind": "explanation",
                "intent": "explain",
                "status": "complete",
                "evidence_state": "supported",
                "outside_knowledge_used": False,
                "citations": [
                    {
                        "source_id": "media-1",
                        "source_type": "media",
                        "title": "Source",
                        "excerpt": "Cited text.",
                    }
                ],
                "generation_metadata": {"batch": "abc"},
            },
            {
                "title": "Second child",
                "kind": "step",
                "intent": "plan",
                "status": "complete",
                "evidence_state": "uncited",
                "outside_knowledge_used": True,
                "generation_metadata": {"batch": "abc"},
            },
        ],
    )

    assert created is not None
    assert [node.title for node in created] == ["First child", "Second child"]
    loaded = repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    child_ids = loaded.nodes[root_id].child_node_ids
    assert len(child_ids) == 2
    first = loaded.nodes[child_ids[0]]
    assert first.generation_metadata == {"batch": "abc"}
    assert first.citations[0].excerpt == "Cited text."


def test_create_child_nodes_is_atomic_when_one_child_is_invalid(tmp_path):
    from tldw_Server_API.app.core.DB_Management.Explainer_DB import InputError

    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    session = _make_session(repo)
    root_id = session.root_node_ids[0]

    with pytest.raises(InputError):
        repo.create_child_nodes(
            session.id,
            owner_user_id="7",
            parent_id=root_id,
            children=[
                {"title": "Valid child", "kind": "explanation", "intent": "explain"},
                {"title": "Broken child", "kind": "not-a-kind", "intent": "explain"},
            ],
        )

    loaded = repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[root_id].child_node_ids == []


def test_transition_node_status_requires_allowed_from_status(tmp_path):
    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    session = _make_session(repo)
    node_id = session.root_node_ids[0]

    moved = repo.transition_node_status(
        session.id,
        node_id,
        owner_user_id="7",
        to_status="queued",
        from_statuses=["idle", "error", "complete"],
    )
    assert moved is True

    repo.update_node(session.id, node_id, owner_user_id="7", status="generating")
    blocked = repo.transition_node_status(
        session.id,
        node_id,
        owner_user_id="7",
        to_status="queued",
        from_statuses=["idle", "error", "complete"],
    )
    assert blocked is False
    loaded = repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].status == "generating"


def test_delete_session_hard_deletes_all_session_rows(tmp_path):
    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    session = _make_session(repo)
    repo.create_node(
        session.id,
        owner_user_id="7",
        parent_id=session.root_node_ids[0],
        title="Child",
        citations=[
            {
                "source_id": "media-1",
                "source_type": "media",
                "title": "Source",
                "excerpt": "Cited text.",
            }
        ],
    )

    deleted = repo.delete_session(session.id, owner_user_id="7")

    assert deleted is True
    assert repo.get_session(session.id, owner_user_id="7", include_archived=True) is None
    conn = repo.db.get_connection()
    for table in ("explainer_sessions", "explainer_nodes", "explainer_selected_sources", "explainer_citations"):
        count = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        assert count == 0, table


def test_delete_session_enforces_ownership(tmp_path):
    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    session = _make_session(repo)

    assert repo.delete_session(session.id, owner_user_id="8") is False
    assert repo.get_session(session.id, owner_user_id="7") is not None


def test_list_sessions_returns_full_unarchived_sessions_most_recent_first(tmp_path):
    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    first = _make_session(repo)
    second = _make_session(repo)
    archived = _make_session(repo)
    repo.create_node(
        second.id,
        owner_user_id="7",
        parent_id=second.root_node_ids[0],
        title="Child of second",
        citations=[
            {
                "source_id": "media-1",
                "source_type": "media",
                "title": "Source",
                "excerpt": "Cited text.",
            }
        ],
    )
    repo.archive_session(archived.id, owner_user_id="7")

    listed = repo.list_sessions(owner_user_id="7")

    assert {session.id for session in listed} == {first.id, second.id}
    listed_second = next(session for session in listed if session.id == second.id)
    expected_second = repo.get_session(second.id, owner_user_id="7")
    assert expected_second is not None
    assert listed_second.nodes.keys() == expected_second.nodes.keys()
    child_id = expected_second.nodes[expected_second.root_node_ids[0]].child_node_ids[0]
    assert listed_second.nodes[child_id].citations[0].excerpt == "Cited text."
