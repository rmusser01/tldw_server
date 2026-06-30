from pathlib import Path


def test_admin_runs_page_exposes_placeholder_with_watchlists_cta():
    """
    Smoke-test the Admin Runs page wrapper so route placeholder copy remains
    wired while the dedicated admin surface is not implemented.
    """
    p = Path("apps/tldw-frontend/pages/admin/watchlists-runs.tsx")
    assert p.exists(), "watchlists-runs.tsx not found"
    text = p.read_text(encoding="utf-8")
    assert "RoutePlaceholder" in text
    assert 'plannedPath="/admin/watchlists-runs"' in text
    assert 'primaryCtaHref="/watchlists"' in text
