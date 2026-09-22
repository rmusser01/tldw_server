"""ChaChaNotes client_id is the PostgreSQL tenant key, not a free-form label.

CharactersRAGDB pushes its client_id into app.current_user_id, and every ChaCha
row-level security policy compares client_id against that setting. So whatever
a caller passes as client_id *is* the account boundary on PostgreSQL.

Callers had been passing descriptive strings. "voice_assistant" is a constant
shared by every user, which pooled all of them into one tenant.
"chat-macro-worker-<id>" is per-user but does not match the user's normal
tenant, so rows written under it were invisible to them afterwards.

The instance cache compounds both: its key is the user directory alone, with no
client_id component, so whichever caller initialised a given user first fixed
that user's tenant for the life of the process.
"""

import pytest

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import _tenant_client_id

pytestmark = pytest.mark.unit


def test_absent_client_id_resolves_to_the_user():
    assert _tenant_client_id(7, None) == "7"


def test_matching_client_id_is_kept():
    assert _tenant_client_id(7, "7") == "7"


def test_the_shared_voice_assistant_label_cannot_become_a_tenant():
    """The regression: every user resolved to the same tenant string."""
    assert _tenant_client_id(7, "voice_assistant") == "7"
    assert _tenant_client_id(9, "voice_assistant") == "9"


def test_two_users_never_share_a_tenant_whatever_label_is_passed():
    labels = [None, "voice_assistant", "chat-macro-worker-7", "", "admin"]

    for label in labels:
        assert _tenant_client_id(7, label) != _tenant_client_id(9, label)


def test_per_worker_labels_do_not_fork_a_users_tenant():
    """These were per-user but not the user's own tenant, hiding their rows."""
    assert _tenant_client_id(7, "chat-macro-worker-7") == "7"
    assert _tenant_client_id(7, "chat-macro-cancel-7") == "7"
