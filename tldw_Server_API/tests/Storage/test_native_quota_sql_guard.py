"""Native reservation SQL must survive the managed AuthNZ guard on both engines."""

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql
from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import _TransactionBoundPool
from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import (
    NativeQuotaOwner,
    NativeQuotaReservationStore,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("postgres", [False, True])
async def test_native_reservation_insert_uses_guard_compatible_bound_sql(postgres):
    """Exercise the real repository and placeholder adapter before guarded DB I/O."""
    received = []

    class GuardedConnection:
        async def execute(self, query, *args):
            received.append(
                (
                    _guard_sql(
                        query,
                        backend="postgres" if postgres else "sqlite",
                        connection_identity=self,
                        operation="execute",
                    ),
                    args,
                )
            )

    connection = GuardedConnection()
    store = NativeQuotaReservationStore(_TransactionBoundPool(connection, postgres=postgres))
    owner = NativeQuotaOwner(1, "native:guard-fixture", "guard-fixture-user")
    reservation = await store.insert(owner, "candidate-one", 7)

    assert (reservation.state, len(received), received[0][1] if postgres else received[0][1][0]) == (
        "active",
        1,
        (1, "guard-fixture-user", "native:guard-fixture", "candidate-one", "global", None, None, None, 7),
    )
