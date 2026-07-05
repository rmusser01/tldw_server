from __future__ import annotations

from pathlib import Path
import tomllib


AUTHNZ_CONFTEST_PLUGIN = "tldw_Server_API.tests.AuthNZ.conftest"
AUTHNZ_FULL_FIXTURES_PLUGIN = "tldw_Server_API.tests._plugins.authnz_full_fixtures"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _pytest_ini_options() -> dict:
    pyproject_path = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return data["tool"]["pytest"]["ini_options"]


def test_chat_fixtures_plugin_not_globally_registered():
    plugins = _pytest_ini_options()["plugins"]
    assert "tldw_Server_API.tests._plugins.chat_fixtures" not in plugins

    root_conftest = (_repo_root() / "conftest.py").read_text(encoding="utf-8")
    assert "tldw_Server_API.tests._plugins.chat_fixtures" not in root_conftest


def test_default_pytest_import_mode_uses_importlib():
    ini_options = _pytest_ini_options()

    assert "--import-mode=importlib" in ini_options["addopts"]
    assert "import_mode" not in ini_options


def test_authnz_conftest_not_registered_as_pytest_plugin():
    plugins = _pytest_ini_options()["plugins"]

    assert AUTHNZ_CONFTEST_PLUGIN not in plugins
    assert AUTHNZ_FULL_FIXTURES_PLUGIN in plugins


def test_authnz_conftest_not_referenced_by_pytest_plugins():
    offenders: list[str] = []
    tests_root = _repo_root() / "tldw_Server_API" / "tests"
    for path in tests_root.rglob("*.py"):
        if path == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        if "pytest_plugins" in text and AUTHNZ_CONFTEST_PLUGIN in text:
            offenders.append(str(path.relative_to(_repo_root())))

    assert not offenders


def test_chat_suite_conftest_opt_in_for_chat_fixtures():
    chat_conftest = _repo_root() / "tldw_Server_API" / "tests" / "Chat" / "conftest.py"
    text = chat_conftest.read_text(encoding="utf-8")
    assert "tests._plugins.chat_fixtures" in text


def test_chat_fixtures_plugin_not_loaded_for_logging_run(pytestconfig):
    plugin_names = {name for name, _plugin in pytestconfig.pluginmanager.list_name_plugin()}
    assert "tldw_Server_API.tests._plugins.chat_fixtures" not in plugin_names
