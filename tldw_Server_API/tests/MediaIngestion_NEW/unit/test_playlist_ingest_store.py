import pytest
from pydantic import ValidationError


def test_duplicate_policy_choices_are_explicit():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import DuplicatePolicy

    assert {policy.value for policy in DuplicatePolicy} == {
        "skip",
        "include_existing",
        "update_metadata_only",
        "overwrite",
    }


def test_review_override_requires_explicit_duplicate_policy():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    with pytest.raises(ValidationError):
        ReviewOverride()


def test_run_state_rejects_client_only_file_reattach_state():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id="occ-1",
            ordinal=1,
            state="file_reattach_required",
        )
