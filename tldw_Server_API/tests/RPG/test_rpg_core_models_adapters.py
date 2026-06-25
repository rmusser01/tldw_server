import pytest

from tldw_Server_API.app.core.RPG.constants import (
    RPG_ADAPTER_DND5E_SRD,
    RPG_ADAPTER_FATE,
    RPG_ADAPTER_PF2E,
    RPG_ADAPTER_VERSION_V1,
)
from tldw_Server_API.app.core.RPG.errors import RPGNotFoundError
from tldw_Server_API.app.core.RPG.models import RPGSnapshotState
from tldw_Server_API.app.core.RPG.rules.adapters import build_default_adapter_registry


def test_default_adapter_registry_lists_supported_rulesets_with_license_summaries():
    registry = build_default_adapter_registry()

    infos = registry.list_infos()
    keys = [info.adapter_key for info in infos]

    assert keys == [RPG_ADAPTER_DND5E_SRD, RPG_ADAPTER_FATE, RPG_ADAPTER_PF2E]  # nosec B101
    assert {info.adapter_version for info in infos} == {RPG_ADAPTER_VERSION_V1}  # nosec B101
    assert infos[0].license_summary.license_name == "CC-BY-4.0"  # nosec B101
    assert infos[1].license_summary.license_name == "CC-BY-3.0"  # nosec B101
    assert infos[2].license_summary.license_name == "ORC"  # nosec B101


def test_default_adapter_registry_exposes_mechanics_and_defensive_schema_copies():
    registry = build_default_adapter_registry()
    fate = registry.get(RPG_ADAPTER_FATE)
    pf2e = registry.get(RPG_ADAPTER_PF2E)

    assert fate.mechanics_tags["resolution_family"] == "fate"  # nosec B101
    assert pf2e.mechanics_tags["resolution_family"] == "d20"  # nosec B101
    assert "note.added" in fate.supported_event_types()  # nosec B101
    assert "actor_id" in fate.actor_schema()["properties"]  # nosec B101

    actor_schema = fate.actor_schema()
    actor_schema["properties"]["actor_id"]["minLength"] = 99

    assert fate.actor_schema()["properties"]["actor_id"]["minLength"] == 1  # nosec B101


def test_default_adapter_registry_rejects_unknown_adapter_key():
    registry = build_default_adapter_registry()

    with pytest.raises(RPGNotFoundError, match="unknown RPG rules adapter"):
        registry.get("not-a-system")


def test_snapshot_state_defaults_do_not_share_mutable_collections():
    first = RPGSnapshotState()
    second = RPGSnapshotState()

    first.notes.append({"note_id": "n1", "text": "private"})
    first.actors["pc-1"] = {"name": "Ada"}

    assert second.notes == []  # nosec B101
    assert second.actors == {}  # nosec B101
