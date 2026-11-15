import pytest

pytestmark = pytest.mark.unit


def test_mcts_reads_the_published_connection_manager() -> None:
    from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_websocket
    from tldw_Server_API.app.core.Prompt_Management.prompt_studio import mcts_optimizer, ws_connections

    assert ws_connections.current() is prompt_studio_websocket.connection_manager
    assert not hasattr(mcts_optimizer, "ws_connection_manager")  # no module-level copy that could go stale
