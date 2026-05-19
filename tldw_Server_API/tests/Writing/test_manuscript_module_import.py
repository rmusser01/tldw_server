import importlib


def test_writing_manuscripts_endpoint_module_imports():
    module = importlib.import_module("tldw_Server_API.app.api.v1.endpoints.writing_manuscripts")

    assert module is not None
    assert hasattr(module, "_enforce_rate_limit")
