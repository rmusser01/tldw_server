"""Which notes have links, from the server (NL-11 review item 3, #3101).

The WebUI's "Linked notes only" decided from the edges in the response. The
server caps each node's degree (40 by default), so a hub note linking 60
notes kept 40 of those edges and the filter hid 20 notes that are linked.
``include_link_status=true`` adds ``has_note_links`` to each note node: a
manual link or a [[link]], either way, with another live note anywhere in
the library. Semantic similarity is not a link. Without the flag the field
is absent, so older clients that parse nodes strictly keep working.
"""

import pytest

from tldw_Server_API.tests.Notes_Graph.integration.test_graph_endpoint import (  # noqa: F401
    _headers,
    client_and_db,
)

pytestmark = pytest.mark.integration


def _library(db):
    spokes = [db.add_note(title=f"Spoke {index}", content="spoke") for index in range(60)]
    hub = db.add_note(title="Hub", content=" ".join(f"[[id:{spoke}]]" for spoke in spokes))
    orphans = [db.add_note(title=f"Orphan {index}", content="alone") for index in range(3)]
    keyword_id = db.add_keyword("loose")
    for orphan in orphans:
        db.link_note_to_keyword(orphan, keyword_id)
    return hub, spokes, orphans


def _graph(client, **params):
    response = client.get(
        "/api/v1/notes/graph",
        params={"radius": 1, "max_nodes": 1000, "max_edges": 4000, **params},
        headers=_headers(),
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_link_status_counts_links_the_degree_cap_dropped(client_and_db) -> None:
    client, db = client_and_db
    hub, spokes, orphans = _library(db)

    data = _graph(client, include_link_status="true")

    status = {node["id"]: node.get("has_note_links") for node in data["nodes"] if node["type"] == "note"}
    drawn_partners = {
        edge["target"] if edge["source"] == hub else edge["source"]
        for edge in data["edges"]
        if hub in (edge["source"], edge["target"]) and edge["type"] == "wikilink"
    }
    assert len(drawn_partners) < len(spokes)
    assert status[hub] is True
    assert all(status[spoke] is True for spoke in spokes)
    assert all(status[orphan] is False for orphan in orphans)
    assert all("has_note_links" not in node for node in data["nodes"] if node["type"] == "tag")


def test_a_manual_link_counts_and_a_trashed_partner_does_not(client_and_db) -> None:
    client, db = client_and_db
    first = db.add_note(title="First", content="one")
    second = db.add_note(title="Second", content="two")
    lonely = db.add_note(title="Lonely", content="three")
    trashed = db.add_note(title="Trashed", content="gone")
    db.update_note(lonely, {"content": f"[[id:{trashed}]]"}, expected_version=1)
    link = client.post(
        f"/api/v1/notes/{first}/links",
        json={"to_note_id": second, "directed": False},
        headers=_headers(),
    )
    assert link.status_code == 200, link.text
    db.soft_delete_note(trashed, expected_version=1)

    data = _graph(client, include_link_status="true")

    status = {node["id"]: node.get("has_note_links") for node in data["nodes"] if node["type"] == "note"}
    assert status == {first: True, second: True, lonely: False}


def test_link_status_is_absent_unless_asked_for(client_and_db) -> None:
    client, db = client_and_db
    _library(db)

    data = _graph(client)

    assert all("has_note_links" not in node for node in data["nodes"])
