"""Request scopes restore Jobs tenant state across nesting, concurrency and cancellation."""

import asyncio

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


def state():
    return (JobManager._RLS_IS_ADMIN.get(), JobManager._RLS_DOMAIN_ALLOWLIST.get(), JobManager._RLS_OWNER_USER_ID.get())


@pytest.fixture(autouse=True)
def restore_context():
    previous = state()
    try:
        yield
    finally:
        JobManager.set_rls_context(is_admin=previous[0], domain_allowlist=previous[1], owner_user_id=previous[2])


def test_scoped_rls_restores_nested_parent_after_error():
    JobManager.set_rls_context(is_admin=True, domain_allowlist="outer", owner_user_id="outer-user")
    with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="media_ingest", owner_user_id="11"):
        with pytest.raises(ValueError, match="planned failure"):
            with JobManager.scoped_rls_context(is_admin=True, domain_allowlist="other", owner_user_id="22"):
                raise ValueError("planned failure")
        assert state() == (False, "media_ingest", "11")  # nosec B101 # Nested failure must restore the request actor.
    assert state() == (True, "outer", "outer-user")  # nosec B101 # Exact parent scope survives.


@pytest.mark.asyncio
async def test_concurrent_rls_scopes_do_not_publish_another_actor():
    previous = state()
    barrier = asyncio.Barrier(2)

    async def actor(owner):
        with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="media_ingest", owner_user_id=owner):
            await barrier.wait()
            return state()

    async with asyncio.TaskGroup() as tasks:
        first = tasks.create_task(actor("11"))
        second = tasks.create_task(actor("22"))
    result = [first.result(), second.result()]
    assert result == [
        (False, "media_ingest", "11"),
        (False, "media_ingest", "22"),
    ]  # nosec B101 # Real interleaved task contexts remain isolated.
    assert state() == previous  # nosec B101 # Parent task is unchanged.


@pytest.mark.asyncio
async def test_cancelled_rls_scope_restores_its_parent():
    JobManager.set_rls_context(is_admin=False, domain_allowlist="parent", owner_user_id="parent-user")
    entered = asyncio.Event()
    waiting = asyncio.Event()
    restored = []

    async def work():
        try:
            with JobManager.scoped_rls_context(is_admin=True, domain_allowlist="media_ingest", owner_user_id="11"):
                entered.set()
                await waiting.wait()
        except asyncio.CancelledError:
            restored.append(state())
            raise

    async with asyncio.TaskGroup() as tasks:
        task = tasks.create_task(work())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert restored == [(False, "parent", "parent-user")]  # nosec B101 # Cancellation resets tokens before unwinding.
    assert state() == (False, "parent", "parent-user")  # nosec B101 # Cancelling another task grants no authority.


def test_nested_scope_restores_nonempty_prior_values():
    JobManager.set_rls_context(is_admin=True, domain_allowlist="prior", owner_user_id="prior-owner")
    try:
        with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="1"):
            assert state() == (False, "claims", "1")
            with JobManager.scoped_rls_context(is_admin=False, domain_allowlist="notes", owner_user_id="2"):
                assert state() == (False, "notes", "2")
            assert state() == (False, "claims", "1")
        assert state() == (True, "prior", "prior-owner")
    finally:
        JobManager.clear_rls_context()


def test_exception_restores_all_previous_values():
    JobManager.set_rls_context(is_admin=False, domain_allowlist="previous", owner_user_id="owner")
    try:
        with pytest.raises(RuntimeError, match="test exit"):
            with JobManager.scoped_rls_context(is_admin=True, domain_allowlist=None, owner_user_id=None):
                raise RuntimeError("test exit")
        assert state() == (False, "previous", "owner")
    finally:
        JobManager.clear_rls_context()
