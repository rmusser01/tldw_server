from __future__ import annotations

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import chat_documents as chat_document_endpoints


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_generation_statistics_sanitizes_unexpected_service_error():
    class RuntimeErrorStubService:
        def __init__(self, db):
            self._db = db

        def get_generated_documents(self, *args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("stats backend unavailable")

    with pytest.raises(HTTPException) as excinfo:
        await chat_document_endpoints.get_generation_statistics(
            db=object(),
            service_cls=RuntimeErrorStubService,
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to get generation statistics"
