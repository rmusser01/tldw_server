from pathlib import Path

import pytest
import yaml

from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
from tldw_Server_API.app.api.v1.router_groups.minimal import iter_minimal_optional_router_specs
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig


pytestmark = pytest.mark.rate_limit


def _policy_path() -> Path:
    return Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml"


def test_realtime_router_specs_are_gated_by_audio_realtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")

    content_specs = [spec for spec in iter_content_router_specs() if spec.route_key == "audio-realtime"]
    minimal_specs = [spec for spec in iter_minimal_optional_router_specs() if spec.route_key == "audio-realtime"]

    assert {(spec.prefix, spec.name) for spec in content_specs} >= {
        ("/api/v1/audio", "audio_realtime"),
        ("/api/v1/audio", "audio_realtime_websocket"),
        ("/v1", "realtime_compat"),
    }
    assert {(spec.prefix, spec.name) for spec in minimal_specs} >= {
        ("/api/v1/audio", "audio_realtime"),
        ("/api/v1/audio", "audio_realtime_websocket"),
        ("/v1", "realtime_compat"),
    }
    assert {
        spec.tags for spec in content_specs if spec.name == "realtime_compat"
    } == {("audio-realtime",)}
    assert {
        spec.tags for spec in minimal_specs if spec.name == "realtime_compat"
    } == {("audio-realtime",)}


def test_realtime_policy_yaml_maps_route_key_and_compat_path() -> None:
    data = yaml.safe_load(_policy_path().read_text(encoding="utf-8"))
    route_map = data["route_map"]

    assert route_map["by_route"]["audio-realtime"] == "audio.default"
    assert route_map["by_path"]["/v1/realtime"] == "audio.default"
    assert route_map["by_path"]["/api/v1/audio*"] == "audio.default"


@pytest.mark.asyncio
async def test_policy_loader_exposes_realtime_route_mapping() -> None:
    loader = PolicyLoader(str(_policy_path()), PolicyReloadConfig(enabled=False))
    snap = await loader.load_once()

    assert snap.route_map["by_route"]["audio-realtime"] == "audio.default"
    assert snap.route_map["by_path"]["/v1/realtime"] == "audio.default"
