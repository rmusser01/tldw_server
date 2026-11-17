from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.api.v1.schemas.org_team_schemas import OrgMemberListItem


@pytest.mark.parametrize(
    ("added_at", "expected"),
    [
        (datetime(2026, 1, 1, tzinfo=timezone.utc), "2026-01-01T00:00:00+00:00"),
        ("2026-01-01 00:00:00", "2026-01-01 00:00:00"),
        (None, None),
    ],
)
def test_org_member_response_preserves_string_timestamp_contract(added_at, expected):
    member = OrgMemberListItem(user_id=1, role="member", added_at=added_at)
    assert member.model_dump(mode="json")["added_at"] == expected
