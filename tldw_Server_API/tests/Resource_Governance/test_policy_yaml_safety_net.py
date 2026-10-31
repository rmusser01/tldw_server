"""The shipped policy file is a safety net: per-entity buckets, generous limits."""

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit]

YAML = Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml"
EMAIL_SENDERS = {"authnz.forgot_password", "authnz.resend_verification", "authnz.magic_link.request"}


def _policies():
    return yaml.safe_load(YAML.read_text(encoding="utf-8"))["policies"]


def test_only_email_senders_keep_a_server_wide_bucket():
    with_global = {pid for pid, pol in _policies().items() if "global" in (pol.get("scopes") or ["global"])}
    assert with_global == EMAIL_SENDERS


def test_default_policy_is_the_safety_net():
    default = _policies()["default"]
    assert default["requests"] == {"rpm": 600, "burst": 2.0}
    assert default["scopes"] == ["user", "api_key", "ip"]


@pytest.mark.parametrize(
    ("policy_id", "rpm", "burst"),
    [
        ("core.default", 600, 2.0),
        ("chat.default", 300, 2.0),
        ("character_chat.default", 300, 2.0),
        ("embeddings.default", 300, 2.0),
        ("workflows.default", 300, 2.0),
        ("watchlists.default", 300, 2.0),
        ("mcp.default", 300, 2.0),
        ("mcp.ingestion", 300, 2.0),
        ("mcp.read", 600, 2.0),
        ("mcp.search", 600, 2.0),
        ("authnz.default", 300, 2.0),
        ("research.default", 120, 2.0),
        ("evals.default", 120, 2.0),
        ("rag.default", 300, 2.0),
    ],
)
def test_interactive_policies_sit_above_normal_use(policy_id, rpm, burst):
    req = _policies()[policy_id]["requests"]
    assert (req["rpm"], req["burst"]) == (rpm, burst)


def test_chat_token_budget_is_a_per_user_runaway_guard():
    chat = _policies()["chat.default"]
    assert chat["tokens"] == {"per_min": 1_000_000, "burst": 1.5}
    assert "global" not in chat["scopes"]
