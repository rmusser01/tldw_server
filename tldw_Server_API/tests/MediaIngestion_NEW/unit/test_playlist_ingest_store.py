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


@pytest.mark.parametrize(
    ("state", "outcome"),
    [
        pytest.param("terminal", None, id="terminal-without-outcome"),
        pytest.param("running", "completed", id="nonterminal-with-outcome"),
    ],
)
def test_run_snapshot_requires_outcome_exactly_when_terminal(state, outcome):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id="occ-1",
            ordinal=1,
            state=state,
            outcome=outcome,
        )


@pytest.mark.parametrize("duplicate_policy", ["update_metadata_only", "overwrite"])
def test_mutating_policies_accept_and_normalize_review_patch(duplicate_policy):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    override = ReviewOverride(
        duplicate_policy=duplicate_policy,
        metadata_patch={
            "title": "  Reviewed title  ",
            "author": "  Reviewed author  ",
            "keywords_add": ["  alpha  ", "beta"],
        },
    )

    assert override.model_dump()["metadata_patch"] == {
        "title": "Reviewed title",
        "author": "Reviewed author",
        "keywords_add": ["alpha", "beta"],
    }


@pytest.mark.parametrize(
    "metadata_patch",
    [
        pytest.param({}, id="empty"),
        pytest.param({"content": "forbidden"}, id="forbidden-content"),
        pytest.param({"title": "   "}, id="blank-title"),
        pytest.param({"author": "   "}, id="blank-author"),
        pytest.param({"keywords_add": []}, id="empty-keywords"),
        pytest.param({"keywords_add": ["   "]}, id="blank-keyword"),
        pytest.param({"title": "x" * 501}, id="oversize-title"),
        pytest.param({"author": "x" * 501}, id="oversize-author"),
        pytest.param({"keywords_add": ["x" * 129]}, id="oversize-keyword"),
        pytest.param({"keywords_add": ["tag"] * 101}, id="too-many-keywords"),
        pytest.param({"title": {"nested": "value"}}, id="deep-title"),
        pytest.param({"keywords_add": ("   ",)}, id="tuple-blank-keyword"),
        pytest.param({"keywords_add": ("x" * 129,)}, id="tuple-oversize-keyword"),
        pytest.param({"keywords_add": ("alpha",)}, id="tuple-keywords"),
        pytest.param({"title": b"   "}, id="bytes-title"),
        pytest.param({"author": b"Reviewed author"}, id="bytes-author"),
    ],
)
def test_review_patch_rejects_invalid_shape_and_values(metadata_patch):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    with pytest.raises(ValidationError):
        ReviewOverride(
            duplicate_policy="overwrite",
            metadata_patch=metadata_patch,
        )


def test_update_metadata_policy_requires_patch():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    with pytest.raises(ValidationError):
        ReviewOverride(duplicate_policy="update_metadata_only")


@pytest.mark.parametrize("duplicate_policy", ["skip", "include_existing"])
def test_nonmutating_duplicate_policies_reject_patch(duplicate_policy):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import ReviewOverride

    with pytest.raises(ValidationError):
        ReviewOverride(
            duplicate_policy=duplicate_policy,
            metadata_patch={"title": "Reviewed title"},
        )


def test_run_snapshot_normalizes_occurrence_id():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    snapshot = RunItemSnapshot(
        occurrence_id="  occ-1  ",
        ordinal=1,
        state="running",
    )

    assert snapshot.occurrence_id == "occ-1"


def test_run_snapshot_rejects_blank_occurrence_id():
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id="   ",
            ordinal=1,
            state="running",
        )


@pytest.mark.parametrize("occurrence_id", [b"   ", b"occ-1"])
def test_run_snapshot_rejects_bytes_occurrence_id(occurrence_id):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id=occurrence_id,
            ordinal=1,
            state="running",
        )


@pytest.mark.parametrize(("field", "value"), [("job_id", 0), ("media_id", -1)])
def test_run_snapshot_requires_positive_persisted_ids(field, value):
    from tldw_Server_API.app.api.v1.schemas.media_playlist_ingest import RunItemSnapshot

    with pytest.raises(ValidationError):
        RunItemSnapshot(
            occurrence_id="occ-1",
            ordinal=1,
            state="running",
            **{field: value},
        )
