"""Contracts for authoritative provider scope outcomes."""

from dataclasses import FrozenInstanceError
from importlib import import_module, util
from types import MappingProxyType

import pytest


def _contracts():
    name = "tldw_Server_API.app.core.AuthNZ.repos.provider_scope_result"
    assert util.find_spec(name) is not None, "provider scope result contract is missing"
    module = import_module(name)
    return module.ProviderScopeStatus, module.ProviderScopeResult


def test_scope_status_values_are_string_enum_members():
    status, _ = _contracts()
    assert {member.value for member in status} == {"resolved", "authorized_absent", "unauthorized", "unavailable"}
    assert all(isinstance(member, str) for member in status)


def test_scope_result_is_frozen_and_redacts_entire_record():
    status, result_type = _contracts()
    result = result_type(status.RESOLVED, {"encrypted_blob": "private-blob", "key_hint": "private-hint"})
    assert "private" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.status = status.UNAUTHORIZED


def test_scope_result_accepts_and_copies_normalized_mapping():
    status, result_type = _contracts()
    record = {"provider": "openai", "encrypted_blob": "private-blob"}
    result = result_type(status.RESOLVED, MappingProxyType(record))
    record["provider"] = "changed"
    assert dict(result.record) == {"provider": "openai", "encrypted_blob": "private-blob"}


def test_resolved_record_is_top_level_read_only():
    status, result_type = _contracts()
    result = result_type(status.RESOLVED, {"provider": "openai"})
    with pytest.raises(TypeError):
        result.record["provider"] = "changed"


@pytest.mark.parametrize("record", [None, "private-blob", [("provider", "openai")]])
def test_resolved_requires_normalized_mapping(record):
    status, result_type = _contracts()
    with pytest.raises(ValueError, match="record"):
        result_type(status.RESOLVED, record)


@pytest.mark.parametrize("state", ["AUTHORIZED_ABSENT", "UNAUTHORIZED", "UNAVAILABLE"])
def test_nonresolved_states_cannot_carry_records(state):
    status, result_type = _contracts()
    with pytest.raises(ValueError, match="record"):
        result_type(getattr(status, state), {"encrypted_blob": "private-blob"})


def test_scope_result_rejects_untyped_status():
    _, result_type = _contracts()
    with pytest.raises(ValueError, match="status"):
        result_type("authorized_absent")
