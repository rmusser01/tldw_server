import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.rpg_schemas import (
    RPGAuthoritySettingsReplaceRequest,
    RPGCampaignUpdateRequest,
    RPGSessionUpdateRequest,
)

pytestmark = pytest.mark.unit


def test_campaign_update_requires_one_mutable_field():
    with pytest.raises(ValidationError):
        RPGCampaignUpdateRequest(expected_version=1)
    ok = RPGCampaignUpdateRequest(title="New", expected_version=1)
    assert ok.title == "New"  # nosec B101


def test_campaign_update_rejects_unknown_status_and_extra_fields():
    with pytest.raises(ValidationError):
        RPGCampaignUpdateRequest(status="deleted", expected_version=1)
    with pytest.raises(ValidationError):
        RPGCampaignUpdateRequest(title="New", expected_version=1, owner_user_id=7)


def test_session_update_bounds():
    with pytest.raises(ValidationError):
        RPGSessionUpdateRequest(title="", expected_version=1)
    with pytest.raises(ValidationError):
        RPGSessionUpdateRequest(title="x" * 201, expected_version=1)
    assert RPGSessionUpdateRequest(status="archived", expected_version=3).status == "archived"  # nosec B101


def test_authority_settings_request_strict_shape():
    base = {"import_auto_commit": False, "mcp_auto_commit": False, "model_auto_commit": True, "model_auto_commit_event_types": ["note.added"]}
    assert RPGAuthoritySettingsReplaceRequest(**base, expected_version=2).model_auto_commit is True  # nosec B101
    with pytest.raises(ValidationError):  # extra field
        RPGAuthoritySettingsReplaceRequest(**base, expected_version=2, surprise=1)
    with pytest.raises(ValidationError):  # missing field
        RPGAuthoritySettingsReplaceRequest(import_auto_commit=False, mcp_auto_commit=False, model_auto_commit=False, expected_version=2)
    with pytest.raises(ValidationError):  # integer 1 must not coerce to bool
        RPGAuthoritySettingsReplaceRequest(**{**base, "mcp_auto_commit": 1}, expected_version=2)
