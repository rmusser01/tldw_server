from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
from tldw_Server_API.app.core.exceptions import (
    ResearchDiscoveryBadRequestError,
    ResearchDiscoveryValidationError,
)

pytestmark = pytest.mark.unit


def _form(**overrides) -> AddMediaForm:
    values = {
        "media_type": "pdf",
        "research_discovery_id": "rd_example",
        "research_discovery_selections": json.dumps([{"result_id": "result-1", "candidate_id": "candidate-1"}]),
    }
    values.update(overrides)
    return AddMediaForm(**values)


def test_parses_ordered_discovery_selection_pairs():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.research_discovery_handoff import (
        validate_research_discovery_handoff,
    )

    form = _form(
        research_discovery_selections=json.dumps(
            [
                {"result_id": " result-2 ", "candidate_id": " candidate-2 "},
                {"result_id": "result-1", "candidate_id": "candidate-1"},
            ]
        )
    )

    selections = validate_research_discovery_handoff(form_data=form, files=None)

    assert selections == (("result-2", "candidate-2"), ("result-1", "candidate-1"))


def test_discovery_mode_preserves_existing_pdf_processing_controls():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.research_discovery_handoff import (
        validate_research_discovery_handoff,
    )

    form = _form(
        pdf_parsing_engine="docling",
        enable_ocr=True,
        ocr_mode="always",
        perform_chunking=True,
        chunk_method="sentences",
        perform_analysis=False,
        media_collection_id=42,
        generate_embeddings=True,
        embedding_dispatch_mode="background",
    )

    validate_research_discovery_handoff(form_data=form, files=None)

    assert form.pdf_parsing_engine == "docling"
    assert form.enable_ocr is True
    assert form.chunk_method == "sentences"
    assert form.perform_analysis is False
    assert form.media_collection_id == 42
    assert form.generate_embeddings is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"research_discovery_id": None},
        {"research_discovery_selections": None},
        {"research_discovery_id": ""},
        {"research_discovery_selections": ""},
    ],
)
def test_rejects_unpaired_or_empty_discovery_fields(overrides):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.research_discovery_handoff import (
        validate_research_discovery_handoff,
    )

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        validate_research_discovery_handoff(form_data=_form(**overrides), files=None)

    assert exc_info.value.public_detail == "research_discovery_fields_must_be_paired"


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        json.dumps({"result_id": "r1", "candidate_id": "c1"}),
        json.dumps([]),
        json.dumps(["r1", "c1"]),
        json.dumps([{"result_id": "r1"}]),
        json.dumps([{"result_id": "r1", "candidate_id": "c1", "title": "client metadata"}]),
        json.dumps([{"result_id": " ", "candidate_id": "c1"}]),
    ],
)
def test_rejects_malformed_selection_json(payload):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.research_discovery_handoff import (
        validate_research_discovery_handoff,
    )

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        validate_research_discovery_handoff(
            form_data=_form(research_discovery_selections=payload),
            files=None,
        )

    assert exc_info.value.public_detail == "research_discovery_selections_malformed"


def test_rejects_duplicate_selection_pairs():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.research_discovery_handoff import (
        validate_research_discovery_handoff,
    )

    pair = {"result_id": "r1", "candidate_id": "c1"}
    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        validate_research_discovery_handoff(
            form_data=_form(research_discovery_selections=json.dumps([pair, pair])),
            files=None,
        )

    assert exc_info.value.public_detail == "research_discovery_duplicate_selection"


def test_rejects_more_than_five_selections():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.research_discovery_handoff import (
        validate_research_discovery_handoff,
    )

    payload = [{"result_id": f"r{index}", "candidate_id": f"c{index}"} for index in range(6)]
    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        validate_research_discovery_handoff(
            form_data=_form(research_discovery_selections=json.dumps(payload)),
            files=None,
        )

    assert exc_info.value.public_detail == "research_discovery_selection_limit_exceeded"


def test_rejects_non_pdf_media_type():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.research_discovery_handoff import (
        validate_research_discovery_handoff,
    )

    with pytest.raises(ResearchDiscoveryValidationError) as exc_info:
        validate_research_discovery_handoff(form_data=_form(media_type="document"), files=None)

    assert exc_info.value.public_detail == "research_discovery_media_type_must_be_pdf"


@pytest.mark.parametrize(
    ("form_overrides", "files", "error"),
    [
        ({"urls": ["https://client.example/paper.pdf"]}, None, "research_discovery_conflicting_input_sources"),
        ({}, [object()], "research_discovery_conflicting_input_sources"),
        ({"use_cookies": True, "cookies": "session=secret"}, None, "research_discovery_credentials_not_allowed"),
        ({"cookies": "session=secret"}, None, "research_discovery_credentials_not_allowed"),
    ],
)
def test_rejects_competing_sources_and_credentials(form_overrides, files, error):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.research_discovery_handoff import (
        validate_research_discovery_handoff,
    )

    with pytest.raises(ResearchDiscoveryBadRequestError) as exc_info:
        validate_research_discovery_handoff(form_data=_form(**form_overrides), files=files)

    assert exc_info.value.public_detail == error
