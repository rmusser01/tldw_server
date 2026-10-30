"""Versioned Research dispatch identity stays separate from Jobs lease credentials."""

import pytest

from tldw_Server_API.app.core.Research.phase_dispatch import new_phase_dispatch, read_phase_dispatch

pytestmark = pytest.mark.unit


def test_dispatch_retains_checkpoint_policy_and_outline_authority():
    value = new_phase_dispatch(
        "synthesizing",
        checkpoint_id="cp_original",
        policy_version=7,
        approved_outline_locked=True,
        predecessor_job_uuid="parent-job",
    )
    assert read_phase_dispatch(value) == value  # nosec B101 -- Durable replay must retain the exact authorized fields.
    assert (
        value["checkpoint_id"] == "cp_original"
        and value["policy_version"] == 7
        and value["approved_outline_locked"] is True
    )  # nosec B101 -- Resume must not reconstruct these values from phase alone.
    assert value["job_uuid"] is None and "lease_id" not in value  # nosec B101 -- Lease credentials never become durable phase authority.


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("authorization", {}),
        ("authorization", "untrusted"),
        ("authorization", True),
        ("schema_version", True),
        ("schema_version", 2),
        ("phase", "completed"),
        ("dispatch_id", "not-a-uuid"),
        ("checkpoint_id", {}),
        ("checkpoint_id", ""),
        ("policy_version", True),
        ("policy_version", 0),
        ("approved_outline_locked", "yes"),
        ("predecessor_job_uuid", "../parent"),
        ("job_uuid", ""),
        ("lease_id", "credential"),
    ],
)
def test_malformed_dispatch_authority_is_refused(key, value):
    dispatch = new_phase_dispatch("collecting")
    dispatch[key] = value
    with pytest.raises(ValueError, match="research_phase_dispatch_invalid"):
        read_phase_dispatch(dispatch)


def test_dispatch_reader_does_not_alias_mutable_caller_data():
    original = new_phase_dispatch("packaging")
    result = read_phase_dispatch(original)
    result["checkpoint_id"] = "changed"
    assert original["checkpoint_id"] is None  # nosec B101 -- Later edits must not mutate an admitted descriptor.


def test_dispatch_refuses_unhashable_phase_with_field_error():
    dispatch = new_phase_dispatch("collecting")
    dispatch["phase"] = {}
    with pytest.raises(ValueError, match="research_phase_dispatch_invalid"):
        read_phase_dispatch(dispatch)
