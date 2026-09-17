"""Load only the three frozen pre-UAT168 Python modules without source edits."""
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

_PACKET = Path(__file__).resolve().parent
_ROWS = json.loads((_PACKET / "baseline-manifest.json").read_text())
_MODULES = {
    row["path"].removesuffix(".py").replace("/", "."): row
    for row in _ROWS if row["path"].endswith(".py")
}


class FrozenModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        row = _MODULES.get(fullname)
        if row is None:
            return None
        source = _PACKET / "baseline" / row["path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Frozen UAT168 baseline hash mismatch")
        return importlib.util.spec_from_file_location(fullname, source)


if any(name in sys.modules for name in _MODULES):
    raise RuntimeError("UAT168 baseline modules were already loaded")
sys.meta_path.insert(0, FrozenModules())
