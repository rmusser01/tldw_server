"""Public safe display-error identity shared by the VN repository and worker."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core import exceptions
from tldw_Server_API.app.core.DB_Management import VNAssetPacks_DB
from tldw_Server_API.app.core.VN_Assets import worker

pytestmark = pytest.mark.unit


def test_legacy_display_error_is_centralized_without_changing_diagnostics() -> None:
    """Moving the class must preserve RuntimeError, safe args and original frames."""
    error_class = getattr(exceptions, "LegacyDisplayReconciliationError", None)
    assert error_class is not None, "display exception belongs in core.exceptions"
    assert VNAssetPacks_DB.LegacyDisplayReconciliationError is worker.LegacyDisplayReconciliationError is error_class
    try:
        raise OSError("PRIVATE_READER_MESSAGE")
    except OSError as original:
        wrapped = error_class(original)
        assert wrapped.error_type is OSError
        assert wrapped.error_traceback is original.__traceback__
    assert isinstance(wrapped, RuntimeError)
    assert wrapped.args == ("VN legacy display reconciliation failed",)
    assert wrapped.__cause__ is None
    assert "PRIVATE_READER_MESSAGE" not in repr(wrapped)
