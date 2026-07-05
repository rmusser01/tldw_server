"""Would-have-caught regression tests for the test-isolation defect class.

Each test is a minimized reproducer of a specific process-global leak from
audits/2026-07-04-test-suite-audit-round2.md (RA5). They assert that the
*reset hook* restores isolation — so removing/weakening that hook makes the
test fail (the "would have caught it" property).

* #2580 — service/DB singleton caches registered against the wrong DB
* #2581 — Embeddings drain singletons bleeding across suites
* #2585 — reload_app_main() swapping the app-main module identity (STILL OPEN;
          its meta-test is xfail-with-issue-link until the fix ships)
"""
from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# #2581 — Embeddings connection-pool manager drain singleton
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_embeddings_pool_manager_reset_yields_fresh_instance() -> None:
    """The global pool manager must be nulled by its reset hook, so a suite
    that drained it does not hand a drained manager to the next suite (#2581).

    Reproducer: without ``cleanup_connection_pools()`` nulling ``_pool_manager``
    the second ``get_pool_manager()`` returns the same (drained) object.
    """
    from tldw_Server_API.app.core.Embeddings import connection_pool as cp

    first = cp.get_pool_manager()
    assert cp._pool_manager is first
    try:
        await cp.cleanup_connection_pools()  # the reset hook
        assert cp._pool_manager is None, "cleanup did not null the global manager"
        second = cp.get_pool_manager()
        assert second is not first, "drained manager leaked into the next acquisition"
    finally:
        await cp.cleanup_connection_pools()


# --------------------------------------------------------------------------- #
# #2580 — per-user ChaCha DB cache bound to USER_DB_BASE_DIR
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_chacha_db_cache_rebinds_to_new_base_dir_after_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A per-user DB handle cached under base dir A must not be reused after the
    base dir changes to B; the reset hook clears the cache so the handle rebinds
    to the new DB (#2580 — wrong-DB handle leak).
    """
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as dep

    base_a = tmp_path / "base_a"
    base_b = tmp_path / "base_b"
    base_a.mkdir()
    base_b.mkdir()

    await dep.shutdown_chacha_resources()
    dep.reset_chacha_shutdown_state()
    assert dep._get_chacha_cached_instance_count() == 0

    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_a))
    db_a = await dep.get_chacha_db_for_user_id(1)
    path_a = db_a.db_path_str
    assert dep._get_chacha_cached_instance_count() == 1
    assert str(base_a) in path_a

    # the reset hook — clears the per-user cache
    await dep.shutdown_chacha_resources()
    dep.reset_chacha_shutdown_state()
    assert dep._get_chacha_cached_instance_count() == 0, "reset did not clear the DB cache"

    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_b))
    db_b = await dep.get_chacha_db_for_user_id(1)
    path_b = db_b.db_path_str
    try:
        assert str(base_b) in path_b, "handle did not rebind to the new base dir"
        assert path_a != path_b, "same DB handle leaked across a base-dir change (#2580)"
    finally:
        await dep.shutdown_chacha_resources()
        dep.reset_chacha_shutdown_state()


# --------------------------------------------------------------------------- #
# #2585 — reload_app_main() swaps sys.modules['...app.main'] identity (OPEN)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.xfail(
    reason="#2585 open: reload_app_main() swaps the app-main module identity and "
    "there is no autouse restore, so references held before a reload go stale. "
    "Flip to strict (remove xfail) when #2585 ships.",
    strict=False,
)
def test_app_main_identity_is_stable_across_a_reload() -> None:
    """DESIRED invariant (currently failing → xfail): a test that reloads the app
    main module should not leave a different module identity behind for code that
    imported it earlier. Restores app.main in ``finally`` so the meta-test never
    pollutes the rest of the session.
    """
    from tldw_Server_API.tests.helpers.app_main_state import (
        reload_app_main,
        restore_app_main,
        snapshot_app_main,
    )

    original = snapshot_app_main()
    try:
        reloaded = reload_app_main()
        # The hazard: identity diverges. Asserting stability documents the fix
        # target; while #2585 is open this fails and is xfail'd.
        assert id(reloaded) == id(original), (
            "reload_app_main swapped the app-main module identity (#2585)"
        )
    finally:
        restore_app_main(original)
        # sanity: session is left with the original module in place
        assert sys.modules.get("tldw_Server_API.app.main") is original
