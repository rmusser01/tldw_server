"""Library graphs against a real SQLite library (NL-11 review, #3101).

The All-notes graph holds up to 500 notes. These run the endpoint on a real
library larger than the focused graph's 250-note cap, with real [[links]],
and a tag filter whose notes link to untagged ones.
"""

import pytest

from tldw_Server_API.tests.Notes_Graph.integration.test_graph_endpoint import (  # noqa: F401
    _headers,
    client_and_db,
)

pytestmark = pytest.mark.integration


def _link_chain(db, count: int, links_each: int) -> list[str]:
    """Notes where each one links to the ``links_each`` notes before it."""
    note_ids: list[str] = []
    for index in range(count):
        targets = note_ids[-links_each:]
        content = " ".join(f"[[id:{target}]]" for target in targets) or "first"
        note_ids.append(db.add_note(title=f"Library note {index}", content=content))
    return note_ids


def test_all_notes_shows_a_300_note_library_with_its_links(client_and_db) -> None:
    client, db = client_and_db
    note_ids = _link_chain(db, 300, 3)

    response = client.get(
        "/api/v1/notes/graph",
        params={
            "radius": 1,
            "max_nodes": 1000,
            "max_edges": 4000,
            "edge_types": "wikilink,backlink",
        },
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert {node["id"] for node in data["nodes"] if node["type"] == "note"} == set(note_ids)
    links = 3 * 300 - (3 + 2 + 1)
    assert len([edge for edge in data["edges"] if edge["type"] == "wikilink"]) == links
    assert len([edge for edge in data["edges"] if edge["type"] == "backlink"]) == links
    assert data["truncated"] is False
    assert data["all_notes_note_cap"] == 500
    assert data["all_notes_eligible"] is True


def test_a_tag_filter_shows_only_the_tagged_notes_of_a_real_library(client_and_db) -> None:
    client, db = client_and_db
    keyword_id = db.add_keyword("reading")
    tagged: list[str] = []
    for index in range(20):
        untagged = db.add_note(title=f"Untagged {index}", content="plain")
        note_id = db.add_note(title=f"Tagged {index}", content=f"See [[id:{untagged}]]")
        db.link_note_to_keyword(note_id, keyword_id)
        tagged.append(note_id)

    response = client.get(
        "/api/v1/notes/graph",
        params={"radius": 1, "tag": "reading", "max_nodes": 1000, "max_edges": 4000},
        headers=_headers(),
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert {node["id"] for node in data["nodes"] if node["type"] == "note"} == set(tagged)
    assert [node["label"] for node in data["nodes"] if node["type"] == "tag"] == ["reading"]
