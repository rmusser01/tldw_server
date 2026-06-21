import pytest

from mcp_unified.lsp import LspPosition, LspToolError


def test_lsp_position_is_zero_based_utf16_contract():
    position = LspPosition(line=0, character=4)

    assert position.line == 0
    assert position.character == 4


def test_lsp_position_rejects_negative_offsets():
    with pytest.raises(ValueError, match="line"):
        LspPosition(line=-1, character=0)


def test_lsp_error_payload_redacts_absolute_paths(tmp_path):
    error = LspToolError("backend_unhealthy", detail=f"failed in {tmp_path}")

    payload = error.to_payload(workspace_root=tmp_path)

    assert str(tmp_path) not in str(payload)
    assert payload["reason_code"] == "backend_unhealthy"
