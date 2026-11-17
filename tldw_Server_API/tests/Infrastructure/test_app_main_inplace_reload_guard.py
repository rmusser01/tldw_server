"""The app-main guard must undo ``importlib.reload(app_main)``, not only a module swap."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from tldw_Server_API.app import main as app_main
from tldw_Server_API.tests.helpers.app_main_state import (
    APP_MAIN_MODULE_NAME,
    app_main_isolated,
)

pytestmark = pytest.mark.unit

_APP_AT_IMPORT = app_main.app


def test_guard_restores_app_after_in_place_reload(monkeypatch):
    fake = ModuleType(APP_MAIN_MODULE_NAME)
    original_app = fake.app = object()
    fake.helper = lambda: "original"
    monkeypatch.setitem(sys.modules, APP_MAIN_MODULE_NAME, fake)

    with app_main_isolated():
        # What importlib.reload does: same module object, globals rebound.
        fake.app = object()
        fake.helper = lambda: "rebuilt"
        fake.added_by_reload = True

    assert sys.modules[APP_MAIN_MODULE_NAME] is fake
    assert fake.app is original_app
    assert fake.helper() == "original"
    assert not hasattr(fake, "added_by_reload")


def test_guard_leaves_an_untouched_module_alone(monkeypatch):
    fake = ModuleType(APP_MAIN_MODULE_NAME)
    fake.app = object()
    monkeypatch.setitem(sys.modules, APP_MAIN_MODULE_NAME, fake)

    with app_main_isolated():
        fake.patched_by_test = 1

    assert fake.patched_by_test == 1


# Polluter then victim, in this order (as in TASK-15426 PR D). The autouse guard
# in tests/conftest.py is what makes the victim pass.
def test_zz_polluter_reloads_app_main_in_place():
    importlib.reload(app_main)
    assert app_main.app is not _APP_AT_IMPORT


def test_zz_victim_sees_the_original_app():
    assert app_main.app is _APP_AT_IMPORT
