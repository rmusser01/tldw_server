from __future__ import annotations

import contextlib
import importlib
import sys
from collections.abc import Iterator
from types import ModuleType

APP_PACKAGE_NAME = "tldw_Server_API.app"
APP_MAIN_MODULE_NAME = "tldw_Server_API.app.main"


def _get_app_package():
    return importlib.import_module(APP_PACKAGE_NAME)


def snapshot_app_main() -> ModuleType | None:
    module = sys.modules.get(APP_MAIN_MODULE_NAME)
    return module if isinstance(module, ModuleType) else None


def clear_app_main() -> None:
    sys.modules.pop(APP_MAIN_MODULE_NAME, None)
    package = sys.modules.get(APP_PACKAGE_NAME)
    if package is None:
        return
    if getattr(package, "main", None) is not None:
        try:
            delattr(package, "main")
        except AttributeError:
            return


def set_app_main(module: ModuleType) -> ModuleType:
    sys.modules[APP_MAIN_MODULE_NAME] = module
    try:
        setattr(_get_app_package(), "main", module)
    except Exception:
        return module
    return module


def restore_app_main(module: ModuleType | None) -> None:
    clear_app_main()
    if module is not None:
        set_app_main(module)


def import_app_main() -> ModuleType:
    current = snapshot_app_main()
    package = sys.modules.get(APP_PACKAGE_NAME)
    package_attr = getattr(package, "main", None) if package is not None else None

    if current is None:
        if isinstance(package_attr, ModuleType):
            try:
                delattr(package, "main")
            except AttributeError:
                pass
        imported = importlib.import_module(APP_MAIN_MODULE_NAME)
        return set_app_main(imported)

    if isinstance(package_attr, ModuleType) and package_attr is not current:
        try:
            setattr(package, "main", current)
        except Exception:
            return current

    return current


def reload_app_main() -> ModuleType:
    clear_app_main()
    importlib.invalidate_caches()
    imported = importlib.import_module(APP_MAIN_MODULE_NAME)
    return set_app_main(imported)


@contextlib.contextmanager
def app_main_isolated() -> Iterator[None]:
    """Confine an app-main reload to this block.

    :func:`reload_app_main` replaces ``sys.modules[APP_MAIN_MODULE_NAME]`` with a
    new module object holding a new FastAPI app. Modules that already ran
    ``from ...main import app`` keep the object they pinned at import time, so
    leaving the swap in place hands the rest of the session a split view: the
    name resolves to one app while earlier importers hold another.

    That is not hypothetical. A ``TestClient`` lifespan exit drains whichever app
    object it was given, which tends to be a pinned one, while anything looking
    at ``app.main`` sees the new app and reports nothing wrong -- and
    ``DrainGateMiddleware`` answers 503 to every request through the drained one
    for the rest of the process (#2585).

    Restoring the original module on exit keeps a reload local to the code that
    asked for it. Callers that want a reloaded app still get one; they just do
    not leave it behind. Restoring is skipped when nothing swapped, which is the
    overwhelmingly common case.

    Yields:
        None. The block runs with whatever module is current; on exit the module
        that was current on entry is put back, or removed again if there was
        none.
    """
    snapshot = snapshot_app_main()
    try:
        yield
    finally:
        if snapshot_app_main() is not snapshot:
            restore_app_main(snapshot)
