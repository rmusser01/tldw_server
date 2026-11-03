"""The test suite's offline model-inventory default (tests/conftest.py, review M5)."""

import pytest

from tldw_Server_API.app.core.http_client import fetch
from tldw_Server_API.app.core.LLM_Calls import gateway, provider_model_inventory

pytestmark = pytest.mark.unit


def test_unmarked_tests_cannot_reach_live_provider_discovery():
    with pytest.raises(AssertionError, match="unmarked test reached live provider discovery"):
        provider_model_inventory._http_fetch(method="GET", url="https://api.openai.com/v1/models")


@pytest.mark.provider_inventory
def test_marked_tests_keep_the_real_egress_checked_fetch():
    assert provider_model_inventory._http_fetch is fetch


def test_unmarked_tests_get_the_stubbed_model_check_where_the_call_path_reads_it():
    """The conftest finds the check by module name; naming a module that no longer holds it skips the stub."""
    assert gateway._validate_provider_model_selection.__module__.endswith("conftest")
