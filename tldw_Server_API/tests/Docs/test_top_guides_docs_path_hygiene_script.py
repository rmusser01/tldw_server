"""Local documentation checks must not resolve external URLs in this repository."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit


def _load_checker() -> ModuleType:
    """Load the documentation checker without executing its CLI entry point."""
    script = Path("Helper_Scripts/docs/check_top_guides_docs_path_hygiene.py")
    spec = spec_from_file_location("check_top_guides_docs_path_hygiene", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_top_guides_docs_path_hygiene_script_passes() -> None:
    """Check the actual guides for broken repository references."""
    module = _load_checker()
    assert module.main() == 0


@pytest.mark.parametrize(
    "reference",
    [
        "[Chatbook](https://github.com/example/chatbook/blob/dev/Docs/User_Guide/index.md)",
        "<https://example.org/Docs/User_Guide/index.md>",
        "http://example.org/Docs/User_Guide/index.md",
        "[Docs](//example.org/Docs/User_Guide/index.md)",
    ],
)
def test_external_urls_do_not_hide_missing_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reference: str
) -> None:
    """Ignore remote targets while retaining local references on the same line."""
    module = _load_checker()
    guide = tmp_path / "guide.md"
    guide.write_text(f"{reference} and [local](Docs/missing.md)\n", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "GUIDE_DIRS", ())
    monkeypatch.setattr(module, "GUIDE_FILES", (guide,))
    assert module._collect_missing_paths() == [("guide.md", "Docs/missing.md")]


def test_repeated_local_slashes_are_not_external_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid local path with repeated separators still needs existence checks."""
    module = _load_checker()
    (tmp_path / "Docs" / "User_Guides").mkdir(parents=True)
    guide = tmp_path / "guide.md"
    guide.write_text("[local](Docs/User_Guides//missing.md)", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "GUIDE_DIRS", ())
    monkeypatch.setattr(module, "GUIDE_FILES", (guide,))
    assert module._collect_missing_paths() == [("guide.md", "Docs/User_Guides//missing.md")]
