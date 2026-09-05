"""Behavior contracts for supported web dependency release baselines."""

from __future__ import annotations

import json
import re
import shutil
import subprocess  # nosec B404
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]
UV_IMAGE = (
    "ghcr.io/astral-sh/uv:0.12.7@"
    "sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945"
)
PYTHON_PRODUCTION_PROFILES = {
    "app": ("Dockerfiles/Dockerfile.prod", None),
    "worker": ("Dockerfiles/Dockerfile.worker", "multiplayer"),
    "audio-worker": ("Dockerfiles/Dockerfile.audio_gpu_worker", None),
}
WEB_REQUIRED_OVERRIDES = {
    "@playwright/test": "1.58.0",
    "@xmldom/xmldom": "0.8.13",
    "antd": "6.2.1",
    "linkify-it": "5.0.2",
    "playwright": "1.58.0",
    "tmp": "0.2.7",
    "vitest": "4.0.18",
    "wxt": "0.20.27",
}
ADMIN_REQUIRED_OVERRIDES = {
    "baseline-browser-mapping": "2.9.19",
    "eslint-plugin-react-hooks": "7.0.1",
}
KNOWN_VULNERABLE_BUN_RELEASES = {
    "@xmldom/xmldom@0.7.13",
    "brace-expansion@1.1.12",
    "brace-expansion@2.0.2",
    "brace-expansion@5.0.0",
    "brace-expansion@5.0.5",
    "browserslist@4.26.3",
    "browserslist@4.28.1",
    "fast-uri@3.1.0",
    "js-yaml@4.2.0",
    "linkify-it@5.0.0",
    "lodash@4.17.23",
    "lodash-es@4.17.21",
    "lodash-es@4.17.23",
    "minimatch@3.1.2",
    "nanoid@3.3.11",
    "picomatch@2.3.1",
    "picomatch@4.0.3",
    "postcss@8.5.6",
    "postcss@8.5.9",
    "rollup@4.56.0",
    "tmp@0.2.5",
    "undici@7.19.0",
    "undici@7.22.0",
    "vite@8.0.8",
}


def _load_jsonc(path: Path) -> dict[str, Any]:
    """Parse Bun's JSONC lock format, including its allowed trailing commas."""
    raw = path.read_text(encoding="utf-8")
    normalized = re.sub(r",(\s*[}\]])", r"\1", raw)
    return json.loads(normalized)


def _resolved_bun_releases(lock: dict[str, Any]) -> set[str]:
    """Return registry package releases present in a parsed Bun lock."""
    return {
        str(entry[0])
        for entry in lock["packages"].values()
        if isinstance(entry, list) and entry and isinstance(entry[0], str)
    }


def _updates_by_ecosystem(dependabot: dict[str, Any], ecosystem: str) -> list[dict[str, Any]]:
    """Return Dependabot update policies for one package ecosystem."""
    return [
        update
        for update in dependabot["updates"]
        if update["package-ecosystem"] == ecosystem
    ]


def _load_next_config(path: Path) -> dict[str, Any]:
    """Import a Next config and return the resolved config object."""
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node.js is required to validate Next configuration")

    # The argv and configuration paths are fixed by this test module.
    result = subprocess.run(  # nosec B603
        [
            node,
            "--input-type=module",
            "-e",
            "import { pathToFileURL } from 'node:url'; "
            "const config = await import(pathToFileURL(process.argv[1]).href); "
            "console.log(JSON.stringify(config.default));",
            str(path),
        ],
        check=True,
        capture_output=True,
        env={
            "NEXT_PUBLIC_API_URL": "http://127.0.0.1:8000",
            "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE": "advanced",
        },
        text=True,
    )
    return json.loads(result.stdout)


def _assert_bun_release_baseline(
    web_lock: dict[str, Any], admin_lock: dict[str, Any]
) -> None:
    """Assert resolved packages and the direct Bun workspace ownership contract."""
    web_workspaces = web_lock["workspaces"]
    assert set(web_workspaces) == {
        "",
        "extension",
        "packages/ui",
        "packages/voice-assistant-sdk",
        "tldw-frontend",
    }, "web workspace importers must retain their distinct monorepo identities"
    assert web_workspaces["extension"]["name"] == "tldw-assistant"
    assert web_workspaces["extension"]["dependencies"]["@tldw/ui"] == "workspace:*"
    assert web_workspaces["packages/ui"]["name"] == "@tldw/ui"
    assert web_workspaces["packages/ui"]["dependencies"]["@noble/hashes"] == "2.0.1"
    web_frontend = web_workspaces["tldw-frontend"]
    assert web_frontend["name"] == "tldw-frontend"
    assert web_frontend["dependencies"]["next"] == "16.3.3"
    assert web_frontend["dependencies"]["@sentry/nextjs"] == "10.46.0"
    assert web_frontend["dependencies"]["@tldw/ui"] == "workspace:*"
    assert web_frontend["devDependencies"]["@next/eslint-plugin-next"] == "16.3.3"

    admin_workspaces = admin_lock["workspaces"]
    assert set(admin_workspaces) == {""}, "Admin must retain an independent root importer"
    admin_root = admin_workspaces[""]
    assert admin_root["name"] == "tldw-admin"
    assert admin_root["dependencies"]["next"] == "16.3.3"
    assert admin_root["devDependencies"]["@next/bundle-analyzer"] == "16.3.3"
    assert admin_root["devDependencies"]["eslint-config-next"] == "16.3.3"

    assert web_lock["packages"]["@playwright/test"][0] == "@playwright/test@1.58.0"
    assert web_lock["packages"]["playwright"][0] == "playwright@1.58.0"
    assert web_lock["packages"]["wxt"][0] == "wxt@0.20.27"
    assert (
        admin_lock["packages"]["eslint-plugin-react-hooks"][0]
        == "eslint-plugin-react-hooks@7.0.1"
    )
    assert web_lock["packages"]["next"][0] == "next@16.3.3"
    assert admin_lock["packages"]["next"][0] == "next@16.3.3"
    assert not web_lock["packages"]["@sentry/nextjs"][0].startswith("@sentry/nextjs@9.")


def test_next_security_baseline_is_exact() -> None:
    """Catches a release manifest that reintroduces vulnerable Next/Sentry pins."""
    web = json.loads((ROOT / "apps/tldw-frontend/package.json").read_text())
    admin = json.loads((ROOT / "admin-ui/package.json").read_text())

    assert web["dependencies"]["next"] == "16.3.3"
    assert web["dependencies"]["@sentry/nextjs"] == "10.46.0"
    assert web["devDependencies"]["@next/eslint-plugin-next"] == "16.3.3"
    assert admin["dependencies"]["next"] == "16.3.3"
    assert admin["dependencies"]["@sentry/nextjs"] == "10.46.0"
    assert admin["devDependencies"]["@next/bundle-analyzer"] == "16.3.3"
    assert admin["devDependencies"]["eslint-config-next"] == "16.3.3"


def test_bun_roots_pin_required_security_overrides() -> None:
    """Catches removal or broadening of reviewed security and compatibility pins."""
    web = json.loads((ROOT / "apps/package.json").read_text())
    admin = json.loads((ROOT / "admin-ui/package.json").read_text())

    assert web["overrides"] == WEB_REQUIRED_OVERRIDES
    assert admin["overrides"] == ADMIN_REQUIRED_OVERRIDES


def test_web_vite_release_is_the_nearest_safe_compatible_pin() -> None:
    """Catches a workspace manifest drifting beyond the validated Vite release."""
    frontend = json.loads((ROOT / "apps/tldw-frontend/package.json").read_text())
    extension = json.loads((ROOT / "apps/extension/package.json").read_text())

    assert frontend["devDependencies"]["vite"] == "8.0.16"
    assert extension["devDependencies"]["vite"] == "8.0.16"


def test_bun_locks_exclude_known_source_gate_blockers() -> None:
    """Catches a lock refresh that restores a known High/Critical release."""
    for lock_path in (ROOT / "apps/bun.lock", ROOT / "admin-ui/bun.lock"):
        releases = _resolved_bun_releases(_load_jsonc(lock_path))
        assert not releases & KNOWN_VULNERABLE_BUN_RELEASES


def test_bun_locks_resolve_supported_next_and_sentry_generation() -> None:
    """Catches a lock refresh that leaves Next 16.3.3 or Sentry 10 unresolved."""
    web_lock = _load_jsonc(ROOT / "apps/bun.lock")
    admin_lock = _load_jsonc(ROOT / "admin-ui/bun.lock")

    _assert_bun_release_baseline(web_lock, admin_lock)


def test_bun_lock_contract_rejects_admin_importer_projection() -> None:
    """Catches a web lock substituted with Admin's otherwise compatible importer map."""
    web_lock = _load_jsonc(ROOT / "apps/bun.lock")
    admin_lock = _load_jsonc(ROOT / "admin-ui/bun.lock")
    wrong_web_lock = deepcopy(web_lock)
    wrong_web_lock["workspaces"] = deepcopy(admin_lock["workspaces"])

    with pytest.raises(AssertionError, match="web workspace importers"):
        _assert_bun_release_baseline(wrong_web_lock, admin_lock)


def test_next_configs_disable_agent_rule_generation() -> None:
    """Catches Next dev configs that can generate repository instruction files."""
    web_config = _load_next_config(ROOT / "apps/tldw-frontend/next.config.mjs")
    admin_config = _load_next_config(ROOT / "admin-ui/next.config.mjs")

    assert web_config.get("agentRules") is False
    assert admin_config.get("agentRules") is False


def test_python_release_tools_and_build_backend_are_exactly_pinned() -> None:
    """Catches mutable packaging tools that would escape the universal lock."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["build-system"]["requires"] == [
        "setuptools==84.0.0",
        "wheel==0.48.0",
    ]
    assert pyproject["dependency-groups"]["release"] == [
        "build==1.6.0",
        "twine==7.0.0",
        "setuptools==84.0.0",
        "wheel==0.48.0",
    ]
    assert pyproject["tool"]["uv"]["conflicts"] == [
        [{"group": "release"}, {"extra": "backend-vllm"}]
    ]
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["namespaces"] is True
    assert pyproject["project"]["optional-dependencies"]["ingestion_email"] == [
        "pypff>=0.6.3; python_version >= '3.11'"
    ]
    assert (
        "dicta-onnx>=0.2.0; python_version >= '3.11'"
        in pyproject["project"]["optional-dependencies"]["TTS_chatterbox_lang"]
    )


def test_universal_uv_lock_contains_root_and_release_tool_profiles() -> None:
    """Catches a missing, stale, or incomplete repository-owned Python lock."""
    lock_path = ROOT / "uv.lock"
    assert lock_path.is_file(), "the universal root uv.lock must be committed"

    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["version"] == 1
    packages = {package["name"]: package for package in lock["package"]}
    assert packages["tldw-server"]["source"] == {"editable": "."}
    assert {name: packages[name]["version"] for name in ("build", "twine", "setuptools", "wheel")} == {
        "build": "1.6.0",
        "twine": "7.0.0",
        "setuptools": "84.0.0",
        "wheel": "0.48.0",
    }


@pytest.mark.parametrize(
    ("profile", "dockerfile", "extra"),
    (
        (profile, dockerfile, extra)
        for profile, (dockerfile, extra) in PYTHON_PRODUCTION_PROFILES.items()
    ),
)
def test_python_production_images_use_locked_noneditable_uv_profiles(
    profile: str, dockerfile: str, extra: str | None
) -> None:
    """Catches dependency resolution or editable installs in production images."""
    text = (ROOT / dockerfile).read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert f"COPY --from={UV_IMAGE} /uv /uvx /bin/" in text
    assert "UV_PROJECT_ENVIRONMENT=/opt/tldw-venv" in text
    assert "UV_LINK_MODE=copy" in text
    assert "UV_COMPILE_BYTECODE=1" in text
    assert "COPY pyproject.toml uv.lock README.md LICENSE /app/" in text
    assert "COPY apps/mcp-unified/src /app/apps/mcp-unified/src" in text
    assert "COPY packages/tldw_profile_core /app/packages/tldw_profile_core" in text
    assert text.index("COPY packages/tldw_profile_core") < text.index("uv sync")
    assert text.index("COPY tldw_Server_API /app/tldw_Server_API") < text.index("uv sync")
    assert "uv sync --locked --no-dev --no-editable" in normalized
    if extra is None:
        assert "uv sync --locked --no-dev --no-editable --extra" not in normalized
    else:
        assert f"uv sync --locked --no-dev --no-editable --extra {extra}" in normalized
    assert "COPY --from=builder /opt/tldw-venv /opt/tldw-venv" in text
    assert "/opt/tldw-venv/bin" in text
    assert "pip install -e" not in normalized
    assert "pip install --upgrade pip" not in normalized
    assert "COPY Config_Files " not in text, f"{profile} copies a nonexistent root path"
    if profile in {"worker", "audio-worker"}:
        assert "TLDW_CONFIG_DIR=/app/Config_Files" in text
        assert "USER_DB_BASE_DIR=/app/Databases/user_databases" in text
        assert "DATABASE_URL=sqlite:////app/Databases/users.db" in text
        assert "/app/Databases/user_databases" in text


def test_container_build_matrix_includes_all_python_production_profiles() -> None:
    """Catches a locked worker image that is omitted from pull-request builds."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/container-build-check.yml").read_text(encoding="utf-8")
    )
    build = workflow["jobs"]["build-and-scan"]
    matrix = {entry["name"]: entry for entry in build["strategy"]["matrix"]["include"]}

    assert matrix["app"]["dockerfile"] == "Dockerfiles/Dockerfile.prod"
    assert matrix["worker"]["dockerfile"] == "Dockerfiles/Dockerfile.worker"
    assert matrix["audio-worker"]["dockerfile"] == "Dockerfiles/Dockerfile.audio_gpu_worker"
    image_build = next(step for step in build["steps"] if step.get("name") == "Build local OCI candidate")
    assert image_build["with"]["platforms"] == "linux/amd64"


def test_embedding_worker_entrypoint_is_in_the_built_package() -> None:
    """Catches namespace-only worker modules omitted by setuptools discovery."""
    assert (
        ROOT / "tldw_Server_API/app/core/Embeddings/services/__init__.py"
    ).is_file()
    worker = (ROOT / "Dockerfiles/Dockerfile.worker").read_text(encoding="utf-8")
    assert (
        'CMD ["python", "-m", '
        '"tldw_Server_API.app.core.Embeddings.services.redis_worker", '
        '"--stage", "all"]'
    ) in worker


def test_dependabot_owns_bun_uv_docker_and_actions_update_roots() -> None:
    """Catches release roots that Dependabot cannot keep current after this baseline."""
    dependabot = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text())

    bun_updates = _updates_by_ecosystem(dependabot, "bun")
    assert [update["directory"] for update in bun_updates].count("/apps") == 1
    assert [update["directory"] for update in bun_updates].count("/admin-ui") == 1
    assert len(_updates_by_ecosystem(dependabot, "uv")) == 1
    assert _updates_by_ecosystem(dependabot, "uv")[0]["directory"] == "/"
    assert any(
        update["directory"] == "/Dockerfiles"
        for update in _updates_by_ecosystem(dependabot, "docker")
    )
    assert len(_updates_by_ecosystem(dependabot, "github-actions")) == 1
