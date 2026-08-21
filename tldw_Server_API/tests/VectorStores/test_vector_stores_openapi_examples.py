from __future__ import annotations

from typing import Any

import pytest


VECTOR_LIST_PATH = "/api/v1/vector_stores/{store_id}/vectors"


def _openapi_spec(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("MINIMAL_TEST_INCLUDE_AUDIO", "1")
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")

    from fastapi import FastAPI
    from tldw_Server_API.app.api.v1.endpoints.vector_stores_openai import router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.openapi_schema = None
    return app.openapi()


@pytest.mark.unit
def test_vector_list_query_examples_are_cats_validate_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _openapi_spec(monkeypatch)
    params = {param["name"]: param for param in spec["paths"][VECTOR_LIST_PATH]["get"]["parameters"]}

    for name in ("filter", "order_by", "order_dir"):
        schema = params[name].get("schema", {})
        assert "examples" not in schema, f"{name} must use parameter-level examples"
        examples = params[name].get("examples")
        assert isinstance(examples, dict), f"{name} must expose parameter-level examples"
        assert examples, f"{name} must expose non-empty parameter-level examples"
        for example_name, example in examples.items():
            assert isinstance(example_name, str)
            assert isinstance(example, dict)
            assert "value" in example
