"""Shared resource contracts retain Jobs identity and canonical JSON semantics."""

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.Jobs.resource_contracts import (
    ResourceAdmission,
    ResourceAdmissionRequest,
    ResourceAttempt,
    ResourceBinding,
    ResourceScope,
    canonical_settings_json,
    settings_fingerprint,
)
from tldw_Server_API.app.core.Jobs.resource_policy import resource_requirement_for

AUTHORITY = "58978735-69ad-47d1-a324-04157989d9a8"
JOB_UUID = "a9dfb098-7959-419a-b017-7884f87eb782"


def test_resource_requirement_is_queue_and_version_independent_but_aliases_fail_closed():
    assert resource_requirement_for("claims", "claims_rebuild_clusters") == "cluster-rebuild"
    for alias in ("claims_rebuild_clusters.v1", "claims_rebuild_clusters.v2", "claims_rebuild_clusters.v123"):
        with pytest.raises(ValueError, match="unsupported"):
            resource_requirement_for("claims", alias)
    assert resource_requirement_for("other", "claims_rebuild_clusters") is None
    assert resource_requirement_for("claims", "claims_rebuild_media") is None


def test_scope_keeps_generic_jobs_owner_strings_and_excludes_queue():
    scope = ResourceScope("claims", "u1", "cluster-rebuild")
    assert scope.owner_user_id == "u1"
    assert not hasattr(scope, "queue")
    with pytest.raises(FrozenInstanceError):
        scope.resource = "other"


@pytest.mark.parametrize("field", ["domain", "owner_user_id", "resource"])
@pytest.mark.parametrize("value", [None, True, 1, "", "  "])
def test_scope_rejects_non_string_or_blank_identity(field, value):
    with pytest.raises(ValueError):
        replace(ResourceScope("claims", "u1", "cluster-rebuild"), **{field: value})


@pytest.mark.parametrize("generation", [True, False, 0, -1, 1.0, "1"])
def test_binding_requires_positive_exact_integer_generation(generation):
    with pytest.raises(ValueError):
        ResourceBinding(ResourceScope("claims", "u1", "cluster-rebuild"), AUTHORITY, generation)


def test_binding_and_attempt_pin_uuid_and_lease_identity():
    binding = ResourceBinding(ResourceScope("claims", "u1", "cluster-rebuild"), AUTHORITY, 1)
    attempt = ResourceAttempt(1, JOB_UUID, "worker", "lease", binding)
    with pytest.raises(FrozenInstanceError):
        binding.generation = 2
    for field, value in (("job_id", True), ("job_uuid", "bad"), ("worker_id", ""), ("lease_id", "")):
        with pytest.raises(ValueError):
            replace(attempt, **{field: value})
    with pytest.raises(ValueError):
        replace(binding, authority_uuid="bad")


def test_mappings_are_defensively_copied_at_contract_boundaries():
    settings = {"version": 1, "nested": {"values": [1]}}
    request = ResourceAdmissionRequest(ResourceScope("claims", "u1", "cluster-rebuild"), settings)
    settings["nested"]["values"].append(2)
    assert request.canonical_settings == {"version": 1, "nested": {"values": [1]}}
    job = {"payload": {"version": 1}}
    admission = ResourceAdmission("created", job, None)
    job["payload"]["version"] = 2
    assert admission.job == {"payload": {"version": 1}}


@pytest.mark.parametrize("key", ["", " x", "x ", "a" * 129, "a\nb", "caf\u00e9", True])
def test_request_key_validation(key):
    with pytest.raises(ValueError):
        ResourceAdmissionRequest(ResourceScope("claims", "u1", "cluster-rebuild"), {}, key)


def test_canonical_json_is_utf8_sorted_compact_and_normalizes_negative_zero():
    value = {"z": [-0.0], "a": "caf\u00e9", "version": 1}
    encoded = '{"a":"caf\u00e9","version":1,"z":[0.0]}'
    assert canonical_settings_json(value) == encoded
    assert settings_fingerprint(value) == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    for invalid in ({"x": float("nan")}, {"x": float("inf")}, {1: "key"}, {"x": object()}):
        with pytest.raises(ValueError):
            canonical_settings_json(invalid)


def test_canonical_settings_reject_strings_that_cannot_be_utf8_encoded():
    with pytest.raises(ValueError):
        canonical_settings_json({"text": "\ud800"})


@given(st.dictionaries(st.text(), st.integers(), max_size=10))
def test_fingerprint_is_independent_of_mapping_insertion_order(settings):
    assert settings_fingerprint(settings) == settings_fingerprint(dict(reversed(list(settings.items()))))
