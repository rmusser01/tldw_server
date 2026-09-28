"""Keep hand-written Prompt Studio DB stubs honest (TASK-13290).

Endpoint tests substitute small stub DBs. A stub that defines a method the real
PromptStudioDatabase cannot serve on every backend hides the bug instead of testing
around it: list_optimizations once existed only on the PostgreSQL implementation, and
the stubs kept the SQLite 500 invisible.
"""

from __future__ import annotations

import inspect

from tldw_Server_API.app.core.DB_Management import PromptStudioDatabase as psd


def unservable_stub_methods(stub_cls: type) -> list[str]:
    """Public stub methods the real DB does not serve on both backends.

    The facade serves a name if it implements it itself. A name it forwards to
    self._impl -- explicitly or via __getattr__ -- must exist on both implementations.
    """
    missing = []
    for name, value in vars(stub_cls).items():
        if name.startswith("_") or not callable(value):
            continue
        facade_method = vars(psd.PromptStudioDatabase).get(name)
        if facade_method is not None and f"self._impl.{name}(" not in inspect.getsource(facade_method):
            continue
        if hasattr(psd._SQLitePromptStudioDatabase, name) and hasattr(psd._BackendPromptStudioDatabase, name):
            continue
        missing.append(name)
    return missing
