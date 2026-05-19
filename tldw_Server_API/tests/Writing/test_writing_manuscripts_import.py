"""Import smoke tests for the manuscript router and app bootstrap."""

import importlib
import sys


def test_writing_manuscripts_module_imports_cleanly() -> None:
    """The manuscript router module imports without missing dependencies."""
    module_name = "tldw_Server_API.app.api.v1.endpoints.writing_manuscripts"
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)

    assert module.router is not None


def test_app_main_imports_cleanly_with_writing_manuscripts_router() -> None:
    """App bootstrap registers the manuscript routes without import-time failures."""
    main_module_name = "tldw_Server_API.app.main"
    manuscripts_module_name = "tldw_Server_API.app.api.v1.endpoints.writing_manuscripts"
    sys.modules.pop(main_module_name, None)
    sys.modules.pop(manuscripts_module_name, None)

    module = importlib.import_module(main_module_name)

    assert module.app is not None
    assert any(
        route.path.startswith("/api/v1/writing/manuscripts")
        for route in module.app.routes
    )
