from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import llamacpp


def llamacpp_test_client(*, headers: dict[str, str]) -> TestClient:
    app = FastAPI()
    app.include_router(llamacpp.router, prefix="/api/v1")
    app.include_router(llamacpp.public_router)
    return TestClient(app, headers=headers)
