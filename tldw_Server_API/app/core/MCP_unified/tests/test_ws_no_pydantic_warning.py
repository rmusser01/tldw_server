"""Ensure no Pydantic deprecation warnings are emitted on WS send."""

import warnings


def test_ws_no_pydantic_deprecation_warning(ws_client):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with ws_client.websocket_connect("/api/v1/mcp/ws?client_id=warnchk") as ws:
            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {"clientInfo": {"name": "WarnChk"}},
                    "id": 1,
                }
            )
            _ = ws.receive_json()

    texts = [str(x.message) for x in w]
    assert not any("The `dict` method is deprecated" in t for t in texts)  # nosec B101
