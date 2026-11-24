"""Run one opt-in outbound fixture inside the prepared backend container."""

from __future__ import annotations

import argparse
import atexit
import io
import json
import os
import shutil
import socket
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

from tldw_Server_API.tests.helpers.offline_network import block_live_network

from .outbound_inventory import AssetIdentity, Inventory, InventoryError, origin_only

_INVENTORY = Path(__file__).with_name("outbound_inventory.json")
_KOKORO = "setup.kokoro"
_KOKORO_ASSET = "kokoro.onnx"
_HUB = "https://huggingface.co"
_CDN = "https://fixture-cdn.invalid"
_CDN_FILE = f"{_CDN}/model"
_REVISION = "a" * 40
_OPENAI = "https://api.openai.com"
_MODEL = "gpt-4o-mini"
_BODY = b"onnx"


class FixtureError(ValueError):
    """An already-recorded fixture check failed."""


@dataclass
class Evidence:
    status: str = "fail"
    operation: str = "failed"
    events: list[str] = field(default_factory=list)
    raw_attempts: int = 0
    violations: list[str] = field(default_factory=list)
    sdk_version: str | None = None

    def check(self, validation: Callable[..., Any], *args: Any) -> None:
        try:
            validation(*args)
        except InventoryError as error:
            self.violations.append(error.code)
            raise

    def require(self, condition: bool, code: str) -> None:
        if not condition:
            self.violations.append(code)
            raise FixtureError(code)


def _prepare_environment(root: Path) -> None:
    config = root / "config.txt"
    shutil.copyfile(_INVENTORY.parents[2] / "tldw_Server_API/Config_Files/config.txt", config)
    dotenv = root / ".env"
    dotenv.touch()
    enabled_flag = "1"
    os.environ.update({
        "DATABASE_URL": f"sqlite:///{root / 'main.db'}",
        "USER_DB_BASE_DIR": str(root / "users"),
        "CIRCUIT_BREAKER_REGISTRY_MODE": "memory",
        "TLDW_CONFIG_FILE": str(config),
        "TLDW_ENV_FILE": str(dotenv),
        "TLDW_ENV_FILE_EXCLUSIVE": "1",
        "TLDW_INSTALL_STATE_DIR": str(root),
        "AUTH_MODE": "single_user",
        "SINGLE_USER_API_KEY": "synthetic-outbound-fixture",
        "HF_HOME": str(root / "hf"),
        "HF_HUB_DISABLE_XET": enabled_flag,
        "HF_HUB_DISABLE_TELEMETRY": enabled_flag,
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": enabled_flag,
        "TLDW_SETUP_SKIP_PIP": "1",
        "LLM_ADAPTERS_NATIVE_HTTP_OPENAI": "1",
    })


def _download(case: str, root: Path, evidence: Evidence) -> None:
    if case == "download-origin":
        os.environ["HF_ENDPOINT"] = "https://unlisted-hub.invalid"
    if case != "download-default":
        os.environ["HF_DEFAULT_REVISION"] = _REVISION

    import httpx
    import huggingface_hub

    from tldw_Server_API.app.core.Setup import install_manager

    evidence.sdk_version = huggingface_hub.__version__
    product_inventory = Inventory.load(_INVENTORY)
    fixture_manifest = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    kokoro = next(item for item in fixture_manifest["capabilities"] if item["id"] == _KOKORO)
    kokoro["origins"].append(_CDN)
    asset = next(item for item in kokoro["assets"] if item["id"] == _KOKORO_ASSET)
    if case != "download-default":
        asset["identity"]["revision"] = _REVISION
    fixture_path = root / "fixture-inventory.json"
    fixture_path.write_text(json.dumps(fixture_manifest), encoding="utf-8")
    fixture_inventory = Inventory.load(fixture_path)
    asset_inventory = product_inventory if case == "download-default" else fixture_inventory
    expected = asset_inventory.capabilities[_KOKORO].assets[_KOKORO_ASSET].identity
    metadata_path = f"/{expected.repository}/resolve/{expected.revision or 'main'}/{expected.resource}"

    def respond(method: str, url: str) -> tuple[int, dict[str, str], bytes]:
        evidence.check(fixture_inventory.check_origin, _KOKORO, url)
        if method == "HEAD" and origin_only(url) == _HUB:
            parsed = urlsplit(url)
            evidence.require(parsed.path == metadata_path and not parsed.query, "fixture_request_mismatch")
            evidence.events.append("hub_metadata")
            return 302, {
                "Location": _CDN_FILE,
                "X-Repo-Commit": _REVISION,
                "ETag": '"outbound-fixture"',
                "X-Linked-Size": str(len(_BODY)),
            }, b""
        evidence.require(method == "GET" and url == _CDN_FILE, "fixture_request_mismatch")
        if case == "download-redirect":
            evidence.events.append("fixture_redirect")
            return 302, {"Location": "https://unlisted-cdn.invalid/model"}, b""
        evidence.events.append("fixture_download")
        return 200, {"Content-Length": str(len(_BODY))}, _BODY

    real_download = huggingface_hub.hf_hub_download
    real_dispatch = install_manager._download_hf_file

    def check_asset(*, repo_id: str, filename: str, revision: str | None, **options: Any) -> str:
        actual = AssetIdentity("huggingface", repo_id, filename, revision)
        evidence.check(asset_inventory.require_asset, _KOKORO, _KOKORO_ASSET, actual)
        return real_download(repo_id=repo_id, filename=filename, revision=revision, **options)

    def dispatch(repo_id: str, filename: str, destination: Path) -> None:
        if case == "download-file":
            filename = "onnx/other.onnx"
        elif case == "download-repository":
            repo_id = "fixture-owner/other-repository"
        real_dispatch(repo_id, filename, destination)

    target = root / "onnx/model.onnx"
    config = {"TTS-Settings": {"kokoro_model_path": str(target), "kokoro_voices_json": str(root / "voices")}}
    with ExitStack() as cleanup:
        if hasattr(huggingface_hub, "set_client_factory"):
            def mock_response(request: httpx.Request) -> httpx.Response:
                status, headers, data = respond(request.method, str(request.url))
                return httpx.Response(status, headers=headers, content=data)

            huggingface_hub.set_client_factory(lambda: httpx.Client(
                transport=httpx.MockTransport(mock_response), follow_redirects=True, trust_env=False,
            ))
            cleanup.callback(huggingface_hub.close_session)
        elif hasattr(huggingface_hub, "configure_http_backend"):
            import requests

            class FixtureAdapter(requests.adapters.HTTPAdapter):
                def send(self, request, **kwargs):
                    status, headers, data = respond(request.method, request.url)
                    response = requests.Response()
                    response.request = request
                    response.url = request.url
                    response.status_code = status
                    response.headers.update(headers)
                    response.raw = io.BytesIO(data)
                    return response

            def session_factory() -> requests.Session:
                session = requests.Session()
                session.trust_env = False
                for scheme in ("http://", "https://"):
                    session.mount(scheme, FixtureAdapter())
                return cleanup.enter_context(session)

            huggingface_hub.configure_http_backend(backend_factory=session_factory)
        else:
            evidence.require(False, "unsupported_huggingface_sdk")

        cleanup.enter_context(patch.object(huggingface_hub, "hf_hub_download", check_asset))
        cleanup.enter_context(patch.object(install_manager, "_download_hf_file", dispatch))
        cleanup.enter_context(patch.object(install_manager, "_load_config", return_value=config))
        result = install_manager.execute_install_plan({"tts": [{"engine": "kokoro", "variants": ["onnx"]}]})

    status = (result or {}).get("status")
    if evidence.violations:
        evidence.require(status == "failed" and not target.exists(), "fixture_download_refusal_unverified")
    else:
        evidence.require(
            status == "completed" and target.is_file() and target.read_bytes() == _BODY,
            "fixture_download_incomplete",
        )
        evidence.operation = "completed"


def _provider(case: str, evidence: Evidence) -> None:
    import httpx

    from tldw_Server_API.app.core.exceptions import ChatAPIError
    from tldw_Server_API.app.core.LLM_Calls import gateway
    from tldw_Server_API.app.core.LLM_Calls.providers import openai_adapter
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        set_llm_provider_overrides_cache_for_tests,
    )

    inventory = Inventory.load(_INVENTORY)
    base = "https://unlisted-provider.invalid" if case == "provider-origin" else _OPENAI
    default = case == "provider-default"
    config = {} if default else {"openai_api": {"api_base_url": f"{base}/v1"}}

    def respond(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        evidence.check(inventory.check_origin, "provider.openai", url)
        evidence.require(
            request.method == "POST" and url == f"{_OPENAI}/v1/chat/completions",
            "fixture_request_mismatch",
        )
        evidence.require(json.loads(request.content).get("model") == _MODEL, "fixture_model_mismatch")
        evidence.events.append("openai_default_completion" if default else "openai_completion")
        return httpx.Response(200, json={"choices": [{"message": {"content": "fixture completion"}}]})

    def client_factory(**kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(respond), trust_env=False, **kwargs)

    set_llm_provider_overrides_cache_for_tests({})
    with (
        patch.object(gateway, "discover_provider_models", return_value=SimpleNamespace(status="ready", models=(_MODEL,))),
        patch.object(openai_adapter, "http_client_factory", client_factory),
    ):
        try:
            result = gateway.perform_chat_api_call(
                api_endpoint="openai",
                model=_MODEL,
                api_key="synthetic-outbound-provider-fixture",
                app_config=config,
                messages=[{"role": "user", "content": "fixture input"}],
                stream=False,
            )
        except ChatAPIError:
            if not evidence.violations:
                raise
            return
    evidence.require(
        result["choices"][0]["message"]["content"] == "fixture completion",
        "fixture_provider_incomplete",
    )
    evidence.operation = "completed"


def main() -> int:
    """Write a private terminal record with fixed event and reason labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    evidence = Evidence()
    cleanup = ExitStack()
    attempts = cleanup.enter_context(block_live_network(fixture_endpoints=frozenset()))

    # Register before importing the app so its own exit callbacks stay guarded.
    @atexit.register
    def write_evidence() -> None:
        cleanup.close()
        evidence.raw_attempts = len(attempts)
        if evidence.raw_attempts:
            evidence.violations.insert(0, "raw_network_attempt")
        evidence.status = "fail" if evidence.violations else "pass"
        (args.root / "evidence.json").write_text(json.dumps(asdict(evidence)), encoding="utf-8")

    try:
        _prepare_environment(args.root)
        if args.case.endswith("-caught-dns"):
            try:
                socket.getaddrinfo("recorder-tripwire.invalid", 443)
            except socket.gaierror:
                pass
        if args.case.startswith("download"):
            _download(args.case, args.root, evidence)
        else:
            _provider(args.case, evidence)
    except FixtureError:
        pass
    except ImportError:
        evidence.violations.append("fixture_dependency_missing")
    # Details from SDK errors can contain a complete request URL.
    except Exception:  # noqa: BLE001
        evidence.violations.append("fixture_error")

    return int(bool(attempts or evidence.violations))


if __name__ == "__main__":
    raise SystemExit(main())
