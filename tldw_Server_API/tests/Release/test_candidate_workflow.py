"""Build inputs must dispatch exact-source paired qualification on its branch."""

from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/verify-app-bundle.yml"


@pytest.mark.parametrize(
    "changed_path",
    [
        "apps/tldw-frontend/pages/api/_tldw-webui/session.ts",
        "apps/tldw-frontend/pages/api/_tldw-webui/runtime-config.ts",
        "apps/tldw-frontend/lib/api.ts",
        "apps/tldw-frontend/next.config.js",
        "apps/tldw-frontend/scripts/build-with-profile.mjs",
        "apps/packages/ui/src/services/tldw/request-core.ts",
        "apps/packages/ui/src/services/tldw/browser-networking.ts",
        "apps/packages/ui/package.json",
        "apps/package.json",
        "apps/bun.lock",
        "apps/scripts/postinstall.mjs",
        "apps/extension/package.json",
        "apps/extension/scripts/wxt-prepare.mjs",
        "apps/packages/voice-assistant-sdk/package.json",
        "apps/mcp-unified/src/server.ts",
        "packages/tldw_profile_core/pyproject.toml",
        "tldw_Server_API/app/core/AuthNZ/config.py",
        "tldw_Server_API/app/main.py",
        "pyproject.toml",
        "Dockerfiles/Dockerfile.gateway",
        "Dockerfiles/gateway/bun.lock",
        "Dockerfiles/entrypoints/tldw-app-first-run.sh",
        ".dockerignore",
        "Docs/Published/API-related/AuthNZ-API-Guide.md",
        "Docs/Documentation.md",
        "README.md",
        "LICENSE",
        "LICENSES/CC-BY-4.0.txt",
        "THIRD_PARTY_NOTICES.txt",
    ],
)
def test_runtime_build_input_dispatches_paired_qualification(changed_path: str) -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    push = workflow["on"]["push"]
    assert "codex/complete-app-wp1" in push["branches"]
    assert any(fnmatchcase(changed_path, pattern) for pattern in push["paths"])
