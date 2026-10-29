"""Discovery makes generation's egress decision for the same base (PR #3191 re-review I2).

These run the real ``fetch`` and egress policy against a real localhost server, with the
production default policy (private addresses and non-standard ports blocked).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_helpers import validate_base_url_override
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.LLM_Calls import chat_calls
from tldw_Server_API.app.core.LLM_Calls import gateway as llm_gateway
from tldw_Server_API.app.core.Security import egress

pytestmark = [pytest.mark.unit, pytest.mark.provider_inventory]


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """A LiteLLM-style OpenAI gateway on 127.0.0.1 at a non-standard port."""
    for name in ("OPENAI_API_BASE_URL", "OPENAI_API_BASE", "OPENAI_BASE_URL", "MOCK_OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    # Production default policy: the pytest-only localhost port exemption is off.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv(egress.ALLOWED_PORTS_ENV, raising=False)
    monkeypatch.setenv(egress.BLOCK_PRIVATE_ENV, "true")
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            hits.append(self.path)
            body = json.dumps({"data": [{"id": "gateway-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args: Any) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/v1"
    # The operator's server config points OpenAI at the gateway.
    server_config = {"openai_api": {"api_base_url": base}}
    monkeypatch.setattr(chat_calls, "load_and_log_configs", lambda: server_config)
    try:
        yield SimpleNamespace(base=base, hits=hits, server_config=server_config)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_operator_configured_localhost_gateway_is_discovered_through_real_fetch(gateway: SimpleNamespace) -> None:
    """A server-configured base gets the exact-origin scope server-owned local endpoints get."""
    request: dict[str, Any] = {
        "model": "gateway-model",
        "api_key": "synthetic-key",
        "app_config": gateway.server_config,
    }
    llm_gateway._validate_provider_model_selection("openai", request)
    assert gateway.hits == ["/v1/models"]
    assert request["base_url"] == gateway.base


@pytest.mark.parametrize("block_private", [True, False])
def test_byok_private_gateway_is_refused_exactly_when_generation_refuses(
    gateway: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, block_private: bool,
) -> None:
    """A BYOK base keeps the default policy its stored base_url was admitted with, never a wider scope.

    The operator has configured the same origin, so only the BYOK provenance stops discovery widening it.
    """
    monkeypatch.setenv(egress.BLOCK_PRIVATE_ENV, "true" if block_private else "false")
    if not block_private:
        monkeypatch.setenv(egress.ALLOWED_PORTS_ENV, "*")
    try:
        validate_base_url_override(gateway.base)
        generation_refuses = False
    except ValueError:
        generation_refuses = True

    request: dict[str, Any] = {
        "model": "gateway-model",
        "api_key": "synthetic-key",
        "app_config": {"openai_api": {"api_base_url": gateway.base}},
        "credentials_resolved": True,
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: SimpleNamespace(endpoint_provenance="byok", auth_source=None),
    }
    try:
        llm_gateway._validate_provider_model_selection("openai", request)
        discovery_refuses = False
    except ChatConfigurationError:
        discovery_refuses = True

    assert generation_refuses is block_private
    assert discovery_refuses is generation_refuses
    assert gateway.hits == ([] if block_private else ["/v1/models"])


def test_an_unsafe_base_never_gets_a_scope_even_when_the_server_base_is_unsafe_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two unusable bases must not compare equal and hand the raw input an exact-origin scope."""
    from tldw_Server_API.app.core.LLM_Calls import provider_model_inventory

    monkeypatch.setattr(provider_model_inventory, "resolve_provider_models_base_url", lambda _provider: None)

    assert provider_model_inventory.inventory_endpoint_scope("openai", "http://127.0.0.1:4000/v1?next=1") is None
