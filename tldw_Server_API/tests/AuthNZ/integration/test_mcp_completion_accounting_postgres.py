"""Composed accounting parity through the canonical PostgreSQL fixture."""

import pytest

from tldw_Server_API.tests.AuthNZ.integration.test_provider_usage_reservations_postgres import (
    reservation_pool as reservation_pool,
)
from tldw_Server_API.tests.AuthNZ_Unit.test_mcp_completion_accounting_storage import (
    AccountingStorageContract,
)

pytestmark = pytest.mark.integration


class TestPostgresAccountingStorage(AccountingStorageContract):
    pass
