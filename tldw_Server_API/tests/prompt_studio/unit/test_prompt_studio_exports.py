import pytest

from tldw_Server_API.app.core.Prompt_Management import prompt_studio


pytestmark = pytest.mark.unit


def test_prompt_studio_does_not_export_deprecated_auth_permissions():
    assert "PermissionManager" not in prompt_studio.__all__
    assert "Permission" not in prompt_studio.__all__
    assert not hasattr(prompt_studio, "PermissionManager")
    assert not hasattr(prompt_studio, "Permission")
