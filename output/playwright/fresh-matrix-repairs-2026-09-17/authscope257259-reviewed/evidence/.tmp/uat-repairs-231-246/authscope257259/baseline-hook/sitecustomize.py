"""Read-only baseline module selection for TASK13260.199/.201 diagnosis."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_NAMES = {"tldw_Server_API.app.core.AuthNZ." + n: _ROOT / (n + ".py") for n in ("User_DB_Handling", "auth_principal_resolver")}

class _BaselineAuthFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        source = _NAMES.get(fullname)
        if source is None:
            return None
        with (_ROOT / "module-loads.jsonl").open("a") as receipt:
            receipt.write(json.dumps({"module": fullname, "source": str(source)}) + "\n")
        return importlib.util.spec_from_file_location(fullname, source)

sys.meta_path.insert(0, _BaselineAuthFinder())
