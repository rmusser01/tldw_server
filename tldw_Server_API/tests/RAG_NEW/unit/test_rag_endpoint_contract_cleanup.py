import pytest

import tldw_Server_API.app.api.v1.endpoints.rag_unified as rag_ep


pytestmark = pytest.mark.unit


def test_rag_endpoint_no_longer_exports_transitional_shim_helpers() -> None:
    shim_names = (
        "_apply_search_agent_defaults",
        "_build_agentic_request_context",
        "_resolve_standard_request",
        "_build_agentic_execution_payload",
        "_build_agentic_config",
        "_coordinate_standard_result_evidence",
        "convert_result_to_response",
    )
    for shim_name in shim_names:
        assert not hasattr(rag_ep, shim_name), f"transitional shim still exported: {shim_name}"
