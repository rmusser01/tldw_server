import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Local_LLM.llamacpp_profile_store import JsonLlamaCppProfileStore
from tldw_Server_API.app.core.Local_LLM.llamacpp_runtime_models import (
    LlamaCppPortPolicy,
    LlamaCppProfile,
    LlamaCppProfileConflictError,
    LlamaCppProfileMode,
    LlamaCppProfileStoreError,
)


def profile(
    profile_id: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    enabled: bool = True,
    port_policy: LlamaCppPortPolicy = LlamaCppPortPolicy.EXPLICIT,
) -> LlamaCppProfile:
    return LlamaCppProfile(
        profile_id=profile_id,
        name=f"Profile {profile_id}",
        enabled=enabled,
        mode=LlamaCppProfileMode.CHAT,
        model_id=f"gguf:{profile_id}",
        model_path=f"/models/{profile_id}.gguf",
        mmproj_model_id=None,
        host=host,
        port=port,
        port_policy=port_policy,
        server_args={"ctx-size": 4096},
        autostart=False,
        restart_policy={"max_restarts": 1},
        provider_alias=f"llamacpp-{profile_id}",
        tags=["test"],
    )


def test_profile_store_bootstraps_default_profile_from_config(tmp_path: Path):
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")

    default_profile = store.ensure_default_profile(
        model_id="gguf:abc",
        model_path="/models/abc.gguf",
        host="127.0.0.1",
        port=8080,
        server_args={"ctx-size": 2048},
    )

    assert default_profile.profile_id == "default"
    assert default_profile.name == "Default llama.cpp server"
    assert default_profile.enabled is True
    assert default_profile.mode == LlamaCppProfileMode.CHAT
    assert default_profile.model_id == "gguf:abc"
    assert default_profile.model_path == "/models/abc.gguf"
    assert default_profile.host == "127.0.0.1"
    assert default_profile.port == 8080
    assert default_profile.port_policy == LlamaCppPortPolicy.EXPLICIT
    assert default_profile.server_args == {"ctx-size": 2048}
    assert store.get("default") == default_profile


def test_profile_store_rejects_duplicate_enabled_explicit_ports(tmp_path: Path):
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    store.upsert(profile("one", host="127.0.0.1", port=8181, enabled=True))

    with pytest.raises(LlamaCppProfileConflictError, match="host/port"):
        store.upsert(profile("two", host="127.0.0.1", port=8181, enabled=True))


def test_profile_store_allows_disabled_profile_on_duplicate_explicit_port(tmp_path: Path):
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    store.upsert(profile("one", host="127.0.0.1", port=8181, enabled=True))

    disabled = store.upsert(profile("two", host="127.0.0.1", port=8181, enabled=False))

    assert disabled.profile_id == "two"
    assert {item.profile_id for item in store.list_profiles()} == {"one", "two"}


def test_profile_store_allows_autoselect_profile_on_duplicate_port(tmp_path: Path):
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    store.upsert(profile("one", host="127.0.0.1", port=8181, enabled=True))

    autoselect = store.upsert(
        profile("two", host="127.0.0.1", port=8181, enabled=True, port_policy=LlamaCppPortPolicy.AUTOSELECT)
    )

    assert autoselect.profile_id == "two"
    assert {item.profile_id for item in store.list_profiles()} == {"one", "two"}


def test_profile_store_round_trips_updates_and_deletes(tmp_path: Path):
    store_path = tmp_path / "profiles.json"
    store = JsonLlamaCppProfileStore(store_path)
    store.upsert(profile("one", port=8181))

    reloaded = JsonLlamaCppProfileStore(store_path)
    assert reloaded.get("one") == profile("one", port=8181)
    assert reloaded.get("missing") is None

    reloaded.upsert(profile("one", port=8282))
    assert JsonLlamaCppProfileStore(store_path).get("one") == profile("one", port=8282)

    assert reloaded.delete("one") is True
    assert reloaded.delete("one") is False
    assert JsonLlamaCppProfileStore(store_path).list_profiles() == []


def test_profile_store_rejects_invalid_dict_store_without_overwriting(tmp_path: Path):
    store_path = tmp_path / "profiles.json"
    original_payload = {"unexpected": "preserve me"}
    store_path.write_text(json.dumps(original_payload), encoding="utf-8")
    store = JsonLlamaCppProfileStore(store_path)

    with pytest.raises(LlamaCppProfileStoreError, match="profiles"):
        store.upsert(profile("one"))

    assert json.loads(store_path.read_text(encoding="utf-8")) == original_payload


@pytest.mark.parametrize("wildcard_host", ["0.0.0.0", "::"])
def test_profile_store_rejects_wildcard_host_port_conflicts(tmp_path: Path, wildcard_host: str):
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    store.upsert(profile("one", host=wildcard_host, port=8181, enabled=True))

    with pytest.raises(LlamaCppProfileConflictError, match="host/port"):
        store.upsert(profile("two", host="127.0.0.1", port=8181, enabled=True))


def test_profile_store_rejects_concrete_host_when_wildcard_port_already_exists(tmp_path: Path):
    store = JsonLlamaCppProfileStore(tmp_path / "profiles.json")
    store.upsert(profile("one", host="127.0.0.1", port=8181, enabled=True))

    with pytest.raises(LlamaCppProfileConflictError, match="host/port"):
        store.upsert(profile("two", host="0.0.0.0", port=8181, enabled=True))
