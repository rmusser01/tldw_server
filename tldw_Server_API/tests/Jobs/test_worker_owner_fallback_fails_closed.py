"""Background workers must not substitute user 1 for a missing job owner.

Five workers returned DatabasePaths.get_single_user_id() when a job carried no
owner_user_id, with no AUTH_MODE gate. A job enqueued, replayed, migrated or
hand-inserted without an owner therefore opened user 1's databases: a chatbooks
export with a null owner exported user 1's conversations into the requesting
account's download.

Single-user mode legitimately resolves to the fixed id. The defect was reaching
for it unconditionally instead of through resolve_user_id_value, which gates on
the auth mode. core_jobs_worker and audio_jobs_worker already failed the job on
a missing owner; these five now agree with them.
"""

import pytest

from tldw_Server_API.app.core.AuthNZ import User_DB_Handling
from tldw_Server_API.app.core.Chatbooks.services import jobs_worker as chatbooks_worker
from tldw_Server_API.app.core.Data_Tables import jobs_worker as data_tables_worker
from tldw_Server_API.app.core.Embeddings.services import jobs_worker as embeddings_worker
from tldw_Server_API.app.core.Evaluations import (
    embeddings_abtest_jobs_worker as abtest_worker,
)
from tldw_Server_API.app.core.Personalization import companion_reflection_jobs

pytestmark = pytest.mark.unit


@pytest.fixture()
def multi_user(monkeypatch):
    monkeypatch.setattr(User_DB_Handling, "is_single_user_mode", lambda: False)


@pytest.fixture()
def single_user(monkeypatch):
    monkeypatch.setattr(User_DB_Handling, "is_single_user_mode", lambda: True)


# (label, callable taking no args) for a job that names no owner at all
OWNERLESS = [
    ("chatbooks", lambda: chatbooks_worker._normalize_user_id(None)),
    ("data_tables", lambda: data_tables_worker._normalize_user_id({}, {})),
    ("embeddings", lambda: embeddings_worker._get_user_id({}, {})),
    ("personalization", lambda: companion_reflection_jobs._resolve_user_id({}, {})),
    ("evaluations_abtest", lambda: abtest_worker._normalize_user_id(None)),
]


@pytest.mark.parametrize(("label", "resolve"), OWNERLESS, ids=[c[0] for c in OWNERLESS])
def test_ownerless_job_raises_in_multi_user_mode(label, resolve, multi_user):
    """The regression: each of these used to return user 1."""
    with pytest.raises(ValueError):
        resolve()


@pytest.mark.parametrize(("label", "resolve"), OWNERLESS, ids=[c[0] for c in OWNERLESS])
def test_ownerless_job_still_resolves_in_single_user_mode(label, resolve, single_user):
    """Self-hosted single-user installs keep working unchanged."""
    assert str(resolve()).strip()


def test_a_named_owner_is_honoured_and_not_replaced():
    assert chatbooks_worker._normalize_user_id("77") == "77"
    assert data_tables_worker._normalize_user_id({"owner_user_id": "77"}, {}) == "77"
    assert embeddings_worker._get_user_id({"owner_user_id": "77"}, {}) == "77"
