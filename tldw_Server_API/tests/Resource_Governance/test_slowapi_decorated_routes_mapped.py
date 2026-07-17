from pathlib import Path

import pytest

from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = pytest.mark.rate_limit


def _repo_policy_path() -> str:


     # tldw_Server_API/tests/Resource_Governance → tldw_Server_API
    return str(Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml")


def _resolve_policy_id_for_path(by_path: dict[str, str], path: str) -> str | None:
    for pat, pol in by_path.items():
        pat = str(pat)
        if pat.endswith("*"):
            if path.startswith(pat[:-1]):
                return str(pol)
        else:
            if path == pat:
                return str(pol)
    return None


@pytest.mark.asyncio
async def test_rg_route_map_covers_rate_limited_paths():
    """
    Ensure every ingress-limited endpoint is covered by RG route_map so
    request caps remain enforced via Resource Governor.
    """
    loader = PolicyLoader(_repo_policy_path(), PolicyReloadConfig(enabled=False, interval_sec=999))
    await loader.load_once()
    snap = loader.get_snapshot()
    by_path = dict((snap.route_map or {}).get("by_path") or {})
    assert loader.get_policy("flashcards.default")["fail_mode"] == "fallback_memory"
    assert loader.get_policy("quizzes.default")["fail_mode"] == "fallback_memory"

    # Minimal representative set of ingress-limited routes:
    # - audio.py (multiple endpoints) -> /api/v1/audio/*
    # - chatbooks.py export/import/preview/download -> explicit mappings
    # - media/listing.py search -> /api/v1/media/*
    decorated_paths = [
        "/api/v1/audio/speech",
        "/api/v1/audio/transcriptions",
        "/api/v1/health",
        "/api/v1/flashcards/review",
        "/api/v1/quizzes",
        "/api/v1/chatbooks/export",
        "/api/v1/chatbooks/import",
        "/api/v1/chatbooks/preview",
        "/api/v1/chatbooks/download/test-id",
        "/api/v1/media/search",
        "/api/v1/skills/trash",
    ]

    for path in decorated_paths:
        policy_id = _resolve_policy_id_for_path(by_path, path)
        assert policy_id is not None, f"RG route_map missing coverage for ingress path: {path}"
        assert loader.get_policy(policy_id) is not None, f"RG policy_id referenced by route_map is missing: {policy_id}"
