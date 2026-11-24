"""Contract for the reviewed, declaration-only outbound inventory."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from Helper_Scripts.ci.outbound_inventory import AssetIdentity, Inventory, InventoryError, origin_only

ROOT = Path(__file__).resolve().parents[3]
KOKORO = AssetIdentity(
    "huggingface", "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped", "onnx/model.onnx", None,
)


@pytest.fixture
def declaration():
    return {
        "id": "setup.kokoro", "kind": "builtin", "component": "setup",
        "source_path": "tldw_Server_API/app/core/Setup/install_manager.py",
        "purpose": "Download model", "phase": "setup", "enabled_when": "Operator opts in",
        "origins": ["https://huggingface.co"],
        "assets": [{
            "id": "kokoro.onnx", "evidence": "declared",
            "identity": {"source": KOKORO.source, "repository": KOKORO.repository,
                         "resource": KOKORO.resource, "revision": None},
        }],
    }


@pytest.fixture
def load(tmp_path, declaration):
    def load_inventory(capabilities=None, exceptions=()):
        path = tmp_path / "inventory.json"
        path.write_text(json.dumps({
            "version": 1,
            "capabilities": [declaration] if capabilities is None else capabilities,
            "exceptions": list(exceptions),
        }))
        return Inventory.load(path)

    return load_inventory


def test_shipped_inventory_anchors_and_kokoro_identity():
    inventory = Inventory.load(ROOT / "Helper_Scripts/ci/outbound_inventory.json")
    assert inventory.require_asset("setup.kokoro", "kokoro.onnx", KOKORO).evidence == "declared"
    assert all((ROOT / item.source_path).is_file() for item in inventory.capabilities.values())
    inventory.check_origin("provider.openai", "https://api.openai.com/v1/chat/completions")
    for item in inventory.capabilities.values():
        assert all(".invalid" not in origin for origin in item.origins)


def test_shipped_local_provider_discovery_matches_the_reviewed_loopback_origins():
    from Helper_Scripts.ci.outbound_backend_smoke import LOCAL_PROVIDER_FIXTURE_PORTS

    inventory = Inventory.load(ROOT / "Helper_Scripts/ci/outbound_inventory.json")
    expected = {f"http://127.0.0.1:{port}" for port in LOCAL_PROVIDER_FIXTURE_PORTS}
    assert inventory.capabilities["webui.local-provider-discovery"].origins == expected


@pytest.mark.parametrize("field,value,reason", [
    ("extra", True, "invalid_schema"),
    ("enabled_when", "", "invalid_schema"),
    ("source_path", "a/../other.py", "invalid_source_anchor"),
    ("origins", ["https://huggingface.co", "https://HUGGINGFACE.CO.:443/"], "duplicate_origin"),
    ("kind", "user-directed", "invalid_schema"),
])
def test_refuses_malformed_capability(load, declaration, field, value, reason):
    declaration[field] = value
    with pytest.raises(InventoryError, match=reason):
        load()


@pytest.mark.parametrize("origin", [
    "https://huggingface.co/path", "https://huggingface.co?x", "https://huggingface.co#",
    "https://user:secret@huggingface.co", "https://*.example.org", "https://hf.co:0",
    "https://[::1]extra", "https://hf.co:", "https://hf.co\\evil",
])
def test_origins_are_exact_and_errors_redacted(load, declaration, origin):
    declaration["origins"] = [origin]
    with pytest.raises(InventoryError, match=r"^invalid_origin$"):
        load()


def test_duplicate_json_keys_fail(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"version": 0, "version": 1, "capabilities": [], "exceptions": []}')
    with pytest.raises(InventoryError, match="duplicate_json_key"):
        Inventory.load(path)


def test_duplicate_capability_fails(load, declaration):
    with pytest.raises(InventoryError, match="duplicate_capability"):
        load([declaration, declaration])


def test_duplicate_asset_fails(load, declaration):
    declaration["assets"] *= 2
    with pytest.raises(InventoryError, match="duplicate_asset"):
        load()


def test_origin_normalizes_port_and_host_but_distinguishes_scheme_and_nondefault_port(load):
    inventory = load()
    inventory.check_origin("setup.kokoro", "https://HUGGINGFACE.CO.:443/model?secret=123")
    for url in ("http://huggingface.co/model", "https://huggingface.co:444/model"):
        with pytest.raises(InventoryError, match=r"^undeclared_origin$"):
            inventory.check_origin("setup.kokoro", url)
    assert origin_only("https://bücher.example:443/model") == "https://xn--bcher-kva.example"
    assert origin_only("http://[0:0:0:0:0:0:0:1]:8080/") == "http://[::1]:8080"
    with pytest.raises(InventoryError, match=r"^invalid_origin$"):
        origin_only("https://user:secret@huggingface.co/file")


@pytest.mark.parametrize("field,change", [
    ("source", "test"), ("repository", "other/Model"), ("resource", "onnx/other.onnx"),
    ("revision", "replacement"),
])
def test_asset_requires_full_identity(load, field, change):
    inventory = load()
    assert inventory.require_asset("setup.kokoro", "kokoro.onnx", KOKORO).identity == KOKORO
    with pytest.raises(InventoryError, match="asset_identity_mismatch"):
        inventory.require_asset("setup.kokoro", "kokoro.onnx", replace(KOKORO, **{field: change}))
    with pytest.raises(InventoryError, match="undeclared_asset"):
        inventory.require_asset("setup.kokoro", "kokoro.other", KOKORO)


def test_exception_has_exact_owned_anchor_and_grants_no_destination_permission(load):
    exception = {
        "id": "legacy.calls", "source_path": "app/relative.py", "symbol": "Client.download",
        "owner": "setup", "follow_up": "TASK-15634",
    }
    inventory = load(exceptions=[exception])
    assert inventory.exceptions["legacy.calls"].symbol == "Client.download"
    with pytest.raises(InventoryError, match="undeclared_origin"):
        inventory.check_origin("legacy.calls", "https://fixture.invalid")
    with pytest.raises(InventoryError, match="duplicate_exception"):
        load(exceptions=[exception, exception | {"id": "legacy.other"}])
    with pytest.raises(InventoryError, match="invalid_source_anchor"):
        load(exceptions=[exception | {"symbol": "*"}])
