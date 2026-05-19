from datetime import datetime

from tldw_Server_API.app.api.v1.schemas.writing_manuscript_schemas import (
    ManuscriptProjectResponse,
    ManuscriptProjectSettings,
    ManuscriptResearchResponse,
    ManuscriptResearchResult,
)


def test_manuscript_project_response_settings_uses_typed_model():
    annotation = ManuscriptProjectResponse.model_fields["settings"].annotation

    assert annotation is ManuscriptProjectSettings

    payload = ManuscriptProjectResponse(
        id="proj-1",
        title="Novel",
        status="draft",
        settings={"theme": "dark", "goal": 90000},
        created_at=datetime.utcnow(),
        last_modified=datetime.utcnow(),
        client_id="test-client",
        version=1,
    )
    assert isinstance(payload.settings, ManuscriptProjectSettings)
    assert payload.settings.model_dump() == {"theme": "dark", "goal": 90000}


def test_manuscript_research_response_results_uses_typed_items():
    annotation = ManuscriptResearchResponse.model_fields["results"].annotation

    assert "ManuscriptResearchResult" in str(annotation)

    payload = ManuscriptResearchResponse(
        query="dragon myths",
        results=[{"title": "Source", "excerpt": "Excerpt", "source_type": "note"}],
    )
    assert isinstance(payload.results[0], ManuscriptResearchResult)
