"""Replay retained pre-repair source in memory; never replace checkout files."""
import importlib
from pathlib import Path
import sys
import types


def pytest_configure(config):
    packet = Path(__file__).resolve().parent
    root = packet.parents[2]
    for name, relative, saved in (
        ("tldw_Server_API.app.core.Ingestion_Media_Processing.persistence",
         "tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py", "persistence.py"),
        ("tldw_Server_API.app.services.media_ingest_jobs_worker",
         "tldw_Server_API/app/services/media_ingest_jobs_worker.py", "media_ingest_jobs_worker.py"),
    ):
        parent_name, attr = name.rsplit(".", 1)
        parent = importlib.import_module(parent_name)
        module = types.ModuleType(name)
        module.__file__ = str(root / relative)
        module.__package__ = parent_name
        sys.modules[name] = module
        exec(compile((packet / "baseline" / saved).read_text(), module.__file__, "exec"), module.__dict__)
        setattr(parent, attr, module)
