import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import ModerationService

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "name",
    [
        "_split_action_directive",
        "_resolve_categories_override",
        "_parse_categories_override",
        "_compile_user_rule",
        "_coalesce_bool",
    ],
)
def test_service_omits_unused_compiler_delegate(name: str) -> None:
    """Keep unused compiler forwarding methods out of the service facade."""
    assert name not in ModerationService.__dict__
