"""Private nonmutating replay of the exact pre-UAT251 repository source."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PACKET = Path(__file__).resolve().parent
NAME = "tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo"
SOURCE = PACKET / "baseline/mcp_hub_repo.py"


def pytest_configure(config):
    spec = importlib.util.spec_from_file_location(NAME, SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
    (PACKET / "baseline-replay-source.json").write_text(json.dumps({
        "module": NAME,
        "source": str(SOURCE),
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "loadedSource": module.__file__,
        "productionFilesModified": False,
    }, indent=2) + "\n")
