"""Read-only baseline substitution retaining original module resource paths."""
import importlib.abc
import importlib.util
from pathlib import Path
import sys

ORIGINAL = '/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/AuthNZ/database.py'


class BaselineLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = ORIGINAL
        source = Path('/private/tmp/cycle5_uat144_baseline_database.py').read_text()
        exec(compile(source, ORIGINAL, 'exec'), module.__dict__)


class BaselineFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'tldw_Server_API.app.core.AuthNZ.database':
            return importlib.util.spec_from_loader(fullname, BaselineLoader(), origin=ORIGINAL)
        return None


sys.meta_path.insert(0, BaselineFinder())
