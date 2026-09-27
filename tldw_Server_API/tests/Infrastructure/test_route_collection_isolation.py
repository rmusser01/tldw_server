"""Collecting feature tests must not disable routes used by other suites."""

import os
import runpy
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.integration
@pytest.mark.parametrize(
    "relative_path",
    [
        "Chat_NEW/conftest.py",
        "Chat_Macros/integration/test_chat_macros_api.py",
        "Skills/integration/test_skills_api.py",
    ],
)
def test_collection_preserves_selected_route_policy(relative_path: str) -> None:
    """Feature-specific import optimizations must not change shared routing."""
    source = Path(__file__).parents[1] / relative_path
    with patch.dict(os.environ, {"ROUTES_DISABLE": "sentinel-route"}):
        runpy.run_path(str(source))
        assert os.environ["ROUTES_DISABLE"] == "sentinel-route"  # nosec B101
