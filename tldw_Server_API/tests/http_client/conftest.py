import pytest

# CI sets a job-wide WORKFLOWS_EGRESS_ALLOWLIST; an allowlist applies even under the
# permissive profile, so any host a test uses outside it was denied by the environment
# rather than by the policy under test. Tests that exercise allowlists set them.
_AMBIENT_EGRESS_LISTS = (
    "EGRESS_ALLOWLIST",
    "EGRESS_DENYLIST",
    "WORKFLOWS_EGRESS_ALLOWLIST",
    "WORKFLOWS_EGRESS_DENYLIST",
)


@pytest.fixture(autouse=True)
def _http_client_permissive_egress(monkeypatch):
    """Keep http_client unit tests independent of external egress profiles."""
    monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "permissive")
    for name in _AMBIENT_EGRESS_LISTS:
        monkeypatch.delenv(name, raising=False)
