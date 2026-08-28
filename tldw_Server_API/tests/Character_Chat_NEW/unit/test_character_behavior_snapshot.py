"""Unit tests for the version-1 character behavior snapshot contract."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import pytest

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    BehaviorSnapshotV1,
    build_behavior_snapshot,
)


def _participant(*, participant_id: str, name: str) -> dict[str, object]:
    return {
        "source": {"kind": "character", "id": participant_id, "version": 3},
        "identity": {"name": name, "aliases": [name]},
        "prompt": {
            "system_prompt": "Stay in character.\r\nUse the saved voice.\rAlways.",
            "description": f"{name} description",
            "personality": f"{name} personality",
            "scenario": "A rainy station",
            "message_example": f"{name}: Hello",
            "post_history_instructions": "Keep continuity",
            "prompt_relevant_extensions": {
                "nested": {"text": "line one\r\nline two"},
                "unicode": "café",
            },
        },
        "greeting": {
            "content": f"Hello from {name}\rWelcome",
            "source": "default",
            "source_index": 0,
        },
        "generation_defaults": {},
        "exemplars": [],
        "world_books": [],
        "default_memory": None,
    }


def _source() -> dict[str, object]:
    return {
        "schema_version": 1,
        "participants": [
            _participant(participant_id="7", name="Ari"),
            _participant(participant_id="8", name="Bea"),
        ],
        "routing_defaults": {"turn_taking_mode": "single"},
    }


def _reverse_mapping_order(value):
    if isinstance(value, dict):
        return {
            key: _reverse_mapping_order(item)
            for key, item in reversed(list(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mapping_order(item) for item in value]
    return value


@pytest.mark.unit
def test_build_behavior_snapshot_is_canonical_and_covers_every_participant():
    source = _source()

    snapshot = build_behavior_snapshot(source)
    reordered = build_behavior_snapshot(_reverse_mapping_order(source))

    assert isinstance(snapshot, BehaviorSnapshotV1)
    assert snapshot.schema_version == 1
    assert snapshot.digest == reordered.digest
    assert snapshot.canonical_bytes == reordered.canonical_bytes
    assert snapshot.size_bytes == len(snapshot.canonical_bytes)
    assert snapshot.payload == json.loads(snapshot.canonical_bytes)
    expected_bytes = json.dumps(
        snapshot.payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert snapshot.canonical_bytes == expected_bytes
    assert b"caf\xc3\xa9" in snapshot.canonical_bytes
    assert snapshot.digest == f"sha256:{hashlib.sha256(expected_bytes).hexdigest()}"
    assert [item["identity"]["name"] for item in snapshot.payload["participants"]] == [
        "Ari",
        "Bea",
    ]
    assert snapshot.payload["participants"][0] == {
        "source": {"kind": "character", "id": "7", "version": 3},
        "identity": {"name": "Ari", "aliases": ["Ari"]},
        "prompt": {
            "system_prompt": "Stay in character.\nUse the saved voice.\nAlways.",
            "description": "Ari description",
            "personality": "Ari personality",
            "scenario": "A rainy station",
            "message_example": "Ari: Hello",
            "post_history_instructions": "Keep continuity",
            "prompt_relevant_extensions": {
                "nested": {"text": "line one\nline two"},
                "unicode": "café",
            },
        },
        "greeting": {
            "content": "Hello from Ari\nWelcome",
            "source": "default",
            "source_index": 0,
        },
        "generation_defaults": {},
        "exemplars": [],
        "world_books": [],
        "default_memory": None,
    }
    assert snapshot.payload["routing_defaults"] == {"turn_taking_mode": "single"}


@pytest.mark.unit
def test_build_behavior_snapshot_is_isolated_from_source_mutation():
    source = _source()
    snapshot = build_behavior_snapshot(source)
    original_digest = snapshot.digest

    source["participants"][0]["identity"]["aliases"].append("Changed")
    source["participants"][0]["prompt"]["system_prompt"] = "Changed"
    source["participants"].pop()

    assert snapshot.digest == original_digest
    assert snapshot.payload["participants"][0]["identity"]["aliases"] == ["Ari"]
    assert len(snapshot.payload["participants"]) == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.digest = "sha256:changed"


@pytest.mark.unit
def test_behavior_snapshot_payload_access_is_defensively_isolated():
    snapshot = build_behavior_snapshot(_source())
    mutable_view = snapshot.payload

    mutable_view["participants"][0]["identity"]["aliases"].append("Changed")
    mutable_view["participants"].pop()

    fresh_payload = snapshot.payload
    assert fresh_payload["participants"][0]["identity"]["aliases"] == ["Ari"]
    assert len(fresh_payload["participants"]) == 2
    assert fresh_payload == json.loads(snapshot.canonical_bytes)
    assert snapshot.size_bytes == len(snapshot.canonical_bytes)
    assert snapshot.digest == (
        f"sha256:{hashlib.sha256(snapshot.canonical_bytes).hexdigest()}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    [
        (),
        ("participants", 0),
        ("participants", 0, "source"),
        ("participants", 0, "identity"),
        ("participants", 0, "prompt"),
        ("participants", 0, "greeting"),
        ("routing_defaults",),
    ],
    ids=["snapshot", "participant", "source", "identity", "prompt", "greeting", "routing"],
)
def test_build_behavior_snapshot_rejects_unclassified_fixed_schema_keys(path):
    source = _source()
    target = source
    for part in path:
        target = target[part]
    target["unclassified"] = "value"

    with pytest.raises(ValueError, match="unexpected keys"):
        build_behavior_snapshot(source)


@pytest.mark.unit
def test_fixed_source_classification_has_no_credential_bearing_fields():
    source = _source()
    source["participants"][0]["source"]["api_key"] = "must-not-be-stored"

    with pytest.raises(ValueError, match="source.*unexpected keys"):
        build_behavior_snapshot(source)

    source = _source()
    source["participants"][0]["source"]["kind"] = "provider"
    with pytest.raises(ValueError, match="source.kind"):
        build_behavior_snapshot(source)


@pytest.mark.unit
@pytest.mark.parametrize("kind", [["character"], {"value": "character"}])
def test_build_behavior_snapshot_rejects_non_string_source_kind(kind):
    source = _source()
    source["participants"][0]["source"]["kind"] = kind

    with pytest.raises(ValueError, match="source.kind"):
        build_behavior_snapshot(source)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.pop("participants"), "snapshot.*missing keys"),
        (lambda value: value.__setitem__("participants", []), "at least one participant"),
        (lambda value: value.__setitem__("schema_version", 2), "schema_version"),
        (
            lambda value: value["participants"][0]["source"].__setitem__("version", "3"),
            "source.version",
        ),
        (
            lambda value: value["participants"][0]["identity"].__setitem__("name", 7),
            "identity.name",
        ),
        (
            lambda value: value["participants"][0]["identity"].__setitem__("aliases", "Ari"),
            "identity.aliases",
        ),
        (
            lambda value: value["participants"][0]["prompt"].__setitem__("scenario", None),
            "prompt.scenario",
        ),
        (
            lambda value: value["participants"][0]["greeting"].__setitem__("source_index", "0"),
            "greeting.source_index",
        ),
        (
            lambda value: value["participants"][0].__setitem__("generation_defaults", []),
            "generation_defaults",
        ),
        (
            lambda value: value["participants"][0].__setitem__("exemplars", {}),
            "exemplars",
        ),
        (
            lambda value: value["participants"][0].__setitem__("world_books", {}),
            "world_books",
        ),
        (
            lambda value: value["participants"][0].__setitem__("default_memory", "memory"),
            "default_memory",
        ),
        (
            lambda value: value["routing_defaults"].__setitem__("turn_taking_mode", "random"),
            "turn_taking_mode",
        ),
    ],
)
def test_build_behavior_snapshot_rejects_invalid_closed_schema_types(mutator, message):
    source = _source()
    mutator(source)

    with pytest.raises(ValueError, match=message):
        build_behavior_snapshot(source)


@pytest.mark.unit
def test_build_behavior_snapshot_rejects_duplicate_participant_identity():
    source = _source()
    duplicate = copy.deepcopy(source["participants"][0])
    duplicate["identity"]["name"] = "Ari copy"
    source["participants"].append(duplicate)

    with pytest.raises(ValueError, match="duplicate participant source"):
        build_behavior_snapshot(source)


@pytest.mark.unit
@pytest.mark.parametrize("value", [b"binary", bytearray(b"binary"), memoryview(b"binary")])
def test_build_behavior_snapshot_rejects_binary_values(value):
    source = _source()
    source["participants"][0]["prompt"]["prompt_relevant_extensions"] = {
        "content": value,
    }

    with pytest.raises(ValueError, match="binary"):
        build_behavior_snapshot(source)


@pytest.mark.unit
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_build_behavior_snapshot_rejects_non_finite_floats(value):
    source = _source()
    source["participants"][0]["generation_defaults"] = {"temperature": value}

    with pytest.raises(ValueError, match="finite"):
        build_behavior_snapshot(source)


@pytest.mark.unit
def test_build_behavior_snapshot_rejects_unsupported_json_values():
    source = _source()
    source["participants"][0]["generation_defaults"] = {"unsupported": object()}

    with pytest.raises(ValueError, match="JSON-compatible"):
        build_behavior_snapshot(source)


@pytest.mark.unit
@pytest.mark.parametrize(
    "extensible_field",
    ["prompt_relevant_extensions", "generation_defaults", "exemplars", "world_books", "default_memory"],
)
def test_build_behavior_snapshot_rejects_credentials_in_extensible_maps(extensible_field):
    source = _source()
    participant = source["participants"][0]
    secret_map = {"nested": {"api_key": "must-not-be-stored"}}
    if extensible_field == "prompt_relevant_extensions":
        participant["prompt"][extensible_field] = secret_map
    elif extensible_field in {"exemplars", "world_books"}:
        participant[extensible_field] = [secret_map]
    else:
        participant[extensible_field] = secret_map

    with pytest.raises(ValueError, match="credential-like key"):
        build_behavior_snapshot(source)


@pytest.mark.unit
@pytest.mark.parametrize(
    "credential_key",
    [
        "api.key",
        "api/key",
        "api\tkey",
        "api--key",
        "api__key",
        "API KEY",
        "x-api-key",
        "ＡＰＩ．ＫＥＹ",
        "accessToken",
        "authToken",
        "bearerToken",
        "clientSecret",
        "privateKey",
        "refreshToken",
        "xApiKey",
        "secretKey",
        "awsSecretAccessKey",
        "openaiApiKey",
        "apiToken",
        "consumerSecret",
        "signingSecret",
        "vendorAccessToken",
        "vendorAuthToken",
        "vendorBearerToken",
        "vendorRefreshToken",
        "vendorClientSecret",
        "vendorApiKey",
        "vendorApiToken",
        "vendorPrivateKey",
        "vendorXApiKey",
        "oauthToken",
        "sessionToken",
        "csrfToken",
        "idToken",
        "oauthAccessToken",
    ],
)
def test_build_behavior_snapshot_rejects_credential_key_separator_variants(
    credential_key,
):
    source = _source()
    source["participants"][0]["prompt"]["prompt_relevant_extensions"] = {
        credential_key: "must-not-be-stored",
    }

    with pytest.raises(ValueError, match="credential-like key"):
        build_behavior_snapshot(source)


@pytest.mark.unit
def test_build_behavior_snapshot_accepts_token_as_legitimate_extension_content():
    source = _source()
    source["participants"][0]["prompt"]["prompt_relevant_extensions"] = {
        "token": "The word token is legitimate prompt content.",
        "nested": {
            "token_budget": 512,
            "tokenBudget": 256,
            "max_tokens": 1024,
            "description": "Keep token references.",
        },
    }

    snapshot = build_behavior_snapshot(source)

    assert snapshot.payload["participants"][0]["prompt"]["prompt_relevant_extensions"][
        "token"
    ].startswith("The word token")


@pytest.mark.unit
def test_build_behavior_snapshot_rejects_configured_size_overflow():
    source = _source()
    unconstrained = build_behavior_snapshot(source)

    with pytest.raises(ValueError, match="exceeds maximum"):
        build_behavior_snapshot(source, max_bytes=unconstrained.size_bytes - 1)
