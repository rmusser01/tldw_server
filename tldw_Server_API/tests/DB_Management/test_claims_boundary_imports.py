"""Persistence imports must not initialize their higher-level consumers."""

import os
import subprocess  # nosec B404 -- current interpreter only; no shell or untrusted arguments.
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module,forbidden",
    [
        ("DB_Management.chroma_existing", "Embeddings.chroma_storage"),
        ("DB_Management.chroma_existing_state", "Embeddings.chroma_storage"),
        ("DB_Management.chroma_read_validation", "Embeddings.chroma_storage"),
        ("DB_Management.media_db.runtime.claims_cluster_rebuild_snapshot", "claims_analytics_export_contract"),
        ("DB_Management.media_db.runtime.claims_cluster_rebuild_publication", "claims_cluster_rebuild_contract"),
        ("Jobs.owned_status", "api.v1.schemas.jobs_owned"),
    ],
)
def test_lower_layer_import_does_not_load_consumer(module, forbidden):
    prefix = "tldw_Server_API.app."
    target = prefix + "core." + module
    blocked = prefix + (forbidden if forbidden.startswith("api.") else "core." + forbidden)
    script = f"""
import importlib
import sys

class RejectConsumer:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {blocked!r}:
            raise RuntimeError("lower layer imported consumer: " + fullname)

sys.meta_path.insert(0, RejectConsumer())
importlib.import_module({target!r})
"""
    result = subprocess.run(  # nosec B603 -- literal module cases in a fresh trusted interpreter.
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
