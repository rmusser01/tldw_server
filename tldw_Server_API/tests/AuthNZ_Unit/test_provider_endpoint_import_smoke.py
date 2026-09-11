"""Fresh-interpreter smoke coverage for provider credential endpoint imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_provider_endpoint_modules_import_without_cross_domain_cycle() -> None:
    """Credential consumers remain importable before the Chat endpoint module."""
    repository_root = Path(__file__).resolve().parents[3]
    modules = (
        "tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced",
        "tldw_Server_API.app.api.v1.endpoints.rag_unified",
        "tldw_Server_API.app.api.v1.endpoints.messages",
        "tldw_Server_API.app.api.v1.endpoints.chat",
    )
    code = "; ".join(f"import {module}" for module in modules)
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(repository_root),
            "TEST_MODE": "1",
            "TLDW_TEST_MODE": "1",
        }
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-4000:]
