"""The transport FakeAPI belongs only to its test application's lifespan."""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import notes_graph_suggestions as endpoint
from tldw_Server_API.app.core.Notes_Graph.suggestion_api import build_notes_graph_suggestions_api
from tldw_Server_API.tests.Notes_Graph.integration.test_suggestion_endpoints import (
    NOTE_ID,
    FakeAPI,
    _app,
    _base_permissions,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def real_factory(monkeypatch):
    # Start from the real boundary even when a preceding baseline test leaked.
    monkeypatch.setattr(endpoint, "build_notes_graph_suggestions_api", build_notes_graph_suggestions_api)
    return build_notes_graph_suggestions_api


def _read(client):
    response = client.get(f"/api/v1/notes/{NOTE_ID}/graph/suggestions/capabilities")
    assert response.status_code == 200


def test_helper_construction_and_openapi_leave_factory_unchanged(real_factory):
    app = _app(FakeAPI(), _base_permissions())
    assert app.openapi()["paths"]
    assert endpoint.build_notes_graph_suggestions_api is real_factory


@pytest.mark.parametrize("exceptional", [False, True], ids=["normal-exit", "exceptional-exit"])
def test_client_exit_restores_exact_previous_factory(real_factory, exceptional):
    fake = FakeAPI()
    app = _app(fake, _base_permissions())

    class ExpectedExit(Exception):
        pass

    def use_client():
        with TestClient(app) as client:
            assert endpoint.build_notes_graph_suggestions_api is not real_factory
            _read(client)
            if exceptional:
                raise ExpectedExit("synthetic client exit")

    if exceptional:
        with pytest.raises(ExpectedExit, match="synthetic client exit"):
            use_client()
    else:
        use_client()
    assert [name for name, _kwargs in fake.calls] == ["capabilities"]
    assert endpoint.build_notes_graph_suggestions_api is real_factory


def test_nested_client_exit_restores_outer_factory_before_real_factory(real_factory):
    outer = FakeAPI()
    inner = FakeAPI()
    with TestClient(_app(outer, _base_permissions())) as outer_client:
        outer_factory = endpoint.build_notes_graph_suggestions_api
        _read(outer_client)
        with TestClient(_app(inner, _base_permissions())) as inner_client:
            _read(inner_client)
        assert endpoint.build_notes_graph_suggestions_api is outer_factory
        _read(outer_client)
    assert [name for name, _kwargs in outer.calls] == ["capabilities", "capabilities"]
    assert [name for name, _kwargs in inner.calls] == ["capabilities"]
    assert endpoint.build_notes_graph_suggestions_api is real_factory
