"""Batch status must not be readable by whoever knows a batch id.

get_batch_status short-circuited on a process-global dict keyed by batch id
alone, returning before the owner-scoped database lookup directly below it.
One handler, two lookup paths, and only the slower one checked ownership.
"""

import pytest

from tldw_Server_API.app.api.v1.endpoints import vector_stores_openai as vs

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_registry():
    vs._BATCH_STATUS.clear()
    yield
    vs._BATCH_STATUS.clear()


def _user(uid):
    from types import SimpleNamespace

    return SimpleNamespace(id=uid, is_superuser=False)


async def test_owner_reads_its_own_cached_batch(monkeypatch):
    monkeypatch.setattr(vs, "resolve_user_id_for_request", lambda user, **kw: str(user.id))
    vs._BATCH_STATUS["vsb_1"] = {
        "id": "vsb_1", "status": "processing", "upserted": 0, "error": None, "user_id": "7",
    }

    out = await vs.get_batch_status(store_id="vs_x", batch_id="vsb_1", current_user=_user("7"))

    assert out["status"] == "processing"


async def test_another_account_does_not_read_the_cached_batch(monkeypatch):
    """The regression: any caller knowing the id got the entry back."""
    monkeypatch.setattr(vs, "resolve_user_id_for_request", lambda user, **kw: str(user.id))
    monkeypatch.setattr(vs, "db_get_batch", lambda batch_id, user_id: None)
    vs._BATCH_STATUS["vsb_1"] = {
        "id": "vsb_1", "status": "processing", "upserted": 0, "error": "secret", "user_id": "7",
    }

    with pytest.raises(Exception) as exc:
        await vs.get_batch_status(store_id="vs_x", batch_id="vsb_1", current_user=_user("9"))

    assert getattr(exc.value, "status_code", None) == 404


async def test_an_unstamped_entry_is_not_served_to_anyone(monkeypatch):
    """Entries predating the owner stamp must fall through, not leak."""
    monkeypatch.setattr(vs, "resolve_user_id_for_request", lambda user, **kw: str(user.id))
    monkeypatch.setattr(vs, "db_get_batch", lambda batch_id, user_id: None)
    vs._BATCH_STATUS["vsb_1"] = {"id": "vsb_1", "status": "processing"}

    with pytest.raises(Exception) as exc:
        await vs.get_batch_status(store_id="vs_x", batch_id="vsb_1", current_user=_user("7"))

    assert getattr(exc.value, "status_code", None) == 404
