"""Every policy ID the code or the route map refers to exists in the shipped YAML.

At runtime an unknown ID falls back to ``default``. CI still fails, so a typo is
caught before it silently loosens a strict policy.
"""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]

ROOT = Path(__file__).resolve().parents[3]
YAML = ROOT / "tldw_Server_API" / "Config_Files" / "resource_governor_policies.yaml"
APP = ROOT / "tldw_Server_API" / "app"
_LITERAL = re.compile(r"""policy_id\s*=\s*["']([a-z_]+(?:\.[a-z_]+)+)["']""")
_ENV_DEFAULT = re.compile(r"""(?:getenv|environ\.get)\(\s*["']RG_[A-Z_]+_POLICY_ID["']\s*,\s*["']([a-z_.]+)["']""")
# Referenced on purpose without a shipped policy; each entry says why.
OPTIONAL = {
    "authnz.federation.login": "federation is opt-in; endpoint skips RG when the policy is undefined",
    "authnz.federation.callback": "federation is opt-in; endpoint skips RG when the policy is undefined",
}


def _defined():
    return set(yaml.safe_load(YAML.read_text(encoding="utf-8"))["policies"])


def _referenced():
    data = yaml.safe_load(YAML.read_text(encoding="utf-8"))
    refs = set(str(v) for v in (data["route_map"].get("by_path") or {}).values())
    refs |= set(str(v) for v in (data["route_map"].get("by_tag") or {}).values())
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        refs |= set(_LITERAL.findall(text)) | set(_ENV_DEFAULT.findall(text))
    return refs


def test_every_referenced_policy_is_defined():
    missing = sorted(_referenced() - _defined() - set(OPTIONAL))
    assert missing == [], f"undefined policy IDs: {missing}"


def test_startup_logs_undefined_route_map_targets():
    from types import SimpleNamespace

    from tldw_Server_API.app.services.startup_resource_governor import log_undefined_policy_references

    snap = SimpleNamespace(route_map={"by_path": {"/a": "known", "/b": "typo"}}, policies={"known": {}})
    loader = SimpleNamespace(get_snapshot=lambda: snap)
    assert log_undefined_policy_references(loader) == ["typo"]
