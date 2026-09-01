"""Would-have-caught regression tests for the test-isolation defect class.

Each test is a minimized reproducer of a specific process-global leak from
audits/2026-07-04-test-suite-audit-round2.md (RA5). They assert that the
*reset hook* restores isolation — so removing/weakening that hook makes the
test fail (the "would have caught it" property).

* #2580 — service/DB singleton caches registered against the wrong DB
* #2581 — Embeddings drain singletons bleeding across suites
* #2585 — reload_app_main() swapping the app-main module identity
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# #2581 — Embeddings connection-pool manager drain singleton
# --------------------------------------------------------------------------- #
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
async def test_chacha_db_cache_rebinds_to_new_base_dir_after_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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

    # Whole body is guarded: any failed assertion still tears down cached handles.
    try:
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
        assert str(base_b) in path_b, "handle did not rebind to the new base dir"
        assert path_a != path_b, "same DB handle leaked across a base-dir change (#2580)"
    finally:
        await dep.shutdown_chacha_resources()
        dep.reset_chacha_shutdown_state()


# --------------------------------------------------------------------------- #
# #2585 — reload_app_main() swaps sys.modules['...app.main'] identity
# --------------------------------------------------------------------------- #
def test_a_reload_does_not_outlive_the_block_that_asked_for_it() -> None:
    """A reload is legitimate; leaving it behind for everyone else is not.

    The earlier version of this test asserted ``reloaded is original``, which a
    working reload can never satisfy -- reloading is *supposed* to produce a new
    module. The invariant that actually protects the session is that the swap is
    undone afterwards, so no later code sees a module identity it did not ask
    for.
    """
    from tldw_Server_API.tests.helpers.app_main_state import (
        app_main_isolated,
        import_app_main,
        reload_app_main,
        snapshot_app_main,
    )

    original = import_app_main()
    assert original is not None

    with app_main_isolated():
        reloaded = reload_app_main()
        assert reloaded is not original, "reload_app_main() did not reload anything"
        assert snapshot_app_main() is reloaded, "the reload was not published"

    assert snapshot_app_main() is original, (
        "a reload outlived its block, so every later import of app.main resolves "
        "to a module the rest of the session never saw (#2585)"
    )


PROBE = """\
from tldw_Server_API.tests.helpers.app_main_state import (
    import_app_main,
    reload_app_main,
    snapshot_app_main,
)

PINNED = import_app_main()


def test_one_reloads_app_main():
    assert reload_app_main() is not PINNED, "reload_app_main() did not reload"


def test_two_still_sees_the_module_it_pinned():
    assert snapshot_app_main() is PINNED, (
        "a reload in an earlier test leaked into this one (#2585)"
    )
"""


def test_a_reload_in_one_test_does_not_leak_into_the_next() -> None:
    """The suite-wide property, proved by running it.

    Asserting that the conftest fixture calls a particular helper would pass a
    rename and fail a harmless refactor, and would say nothing about whether the
    fixture is autouse -- dropping that decorator disables isolation everywhere
    while every structural check still passes.

    So this runs two tests in a real pytest session instead. The probe module
    lives under tests/ so the root conftest applies to it, and is named with a
    leading underscore so the outer run does not collect it.
    """
    import subprocess
    import sys

    probe = Path(__file__).parent / "_app_main_leak_probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    try:
        result = subprocess.run(
            # no:randomly is load-bearing: pytest-randomly shuffles collection,
            # and this probe means nothing unless the reload runs before the
            # test that checks for it.
            [
                sys.executable,
                "-m",
                "pytest",
                str(probe),
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:randomly",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
            timeout=600,
        )
    finally:
        probe.unlink(missing_ok=True)

    assert result.returncode == 0, (
        "a reload in one test leaked into the next, so app.main means different "
        "things to different tests (#2585). Check that tests/conftest.py still "
        "wraps every test in app_main_isolated() and that the fixture is "
        f"autouse.\n\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}"
    )
