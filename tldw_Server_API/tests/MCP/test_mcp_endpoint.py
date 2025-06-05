import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any, Optional, Union

# Adjust the path to import the app from your project's structure
from tldw_Server_API.app.main import app  # Assuming your FastAPI app instance is here
from tldw_Server_API.app.api.v1.schemas.mcp_schemas import Implementation

client = TestClient(app)

def rpc_call(method: str, params: Optional[Dict[str, Any]] = None, id: Union[str, int, None] = "test-id") -> Dict[str, Any]:
    """Helper function to make JSON-RPC calls to the test client."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "id": id,
    }
    if params is not None:
        payload["params"] = params

    response = client.post("/api/v1/mcp/", json=payload)
    return response.json()

# --- Test Cases ---

def test_initialize_request_successful():
    """Test successful initialize request."""
    params = {
        "capabilities": {
            "experimental": {"customCap": True},
            "roots": {"workspaceFolders": True},
            "sampling": {"maxTokens": 1000}
        },
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    }
    response_data = rpc_call("initialize", params)

    assert response_data["jsonrpc"] == "2.0"
    assert response_data["id"] == "test-id"
    assert "result" in response_data
    assert "error" not in response_data

    result = response_data["result"]
    assert "capabilities" in result
    assert "serverInfo" in result
    assert result["serverInfo"]["name"] == "tldw-mcp-server"
    assert result["protocolVersion"] == "2025-03-26"
    assert "instructions" in result # Optional, but our server provides it

def test_initialize_invalid_params():
    """Test initialize request with missing required parameters."""
    params = { # Missing clientInfo and protocolVersion
        "capabilities": {}
    }
    response_data = rpc_call("initialize", params)

    assert response_data["jsonrpc"] == "2.0"
    assert "error" in response_data
    assert response_data["error"]["code"] == -32602  # Invalid params
    assert "Invalid initialize params" in response_data["error"]["message"]

def test_ping_request_successful():
    """Test successful ping request."""
    response_data = rpc_call("ping")

    assert response_data["jsonrpc"] == "2.0"
    assert response_data["id"] == "test-id"
    assert "result" in response_data
    assert response_data["result"] == {} # Ping result is an empty object
    assert "error" not in response_data

def test_ping_request_with_params_successful():
    """Test successful ping request even if unexpected params are sent (server should ignore)."""
    response_data = rpc_call("ping", {"extra_param": "should_be_ignored"})
    assert response_data["jsonrpc"] == "2.0"
    assert "result" in response_data
    assert response_data["result"] == {}

def test_method_not_found():
    """Test calling a non-existent method."""
    response_data = rpc_call("nonExistentMethod", {})

    assert response_data["jsonrpc"] == "2.0"
    assert "error" in response_data
    assert response_data["error"]["code"] == -32601  # Method not found

def test_invalid_json_request():
    """Test sending a malformed JSON request."""
    response = client.post("/api/v1/mcp/", data="{malformed_json")
    response_data = response.json()

    assert response_data["jsonrpc"] == "2.0"
    assert response_data["id"] is None # id might not be parsable
    assert "error" in response_data
    assert response_data["error"]["code"] == -32700  # Parse error

def test_invalid_json_rpc_structure_missing_method():
    """Test sending a valid JSON but invalid JSON-RPC (e.g., missing 'method')."""
    payload = {
        "jsonrpc": "2.0",
        # "method": "initialize", # Missing method
        "id": "test-invalid-structure"
    }
    response = client.post("/api/v1/mcp/", json=payload)
    response_data = response.json()

    assert response_data["jsonrpc"] == "2.0"
    assert response_data["id"] == "test-invalid-structure"
    assert "error" in response_data
    assert response_data["error"]["code"] == -32600  # Invalid Request

# --- Resources Method Tests ---

def test_list_resources_successful():
    """Test successful resources/list request."""
    # First, ensure server is initialized
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })

    response_data = rpc_call("resources/list")

    assert "result" in response_data
    assert "error" not in response_data
    result = response_data["result"]
    assert "resources" in result
    assert isinstance(result["resources"], list)
    # Check if default resources are present (based on mcp.py)
    assert len(result["resources"]) >= 2
    assert result["resources"][0]["uri"] == "mcp://example.com/resource/1"
    assert result["resources"][0]["name"] == "Example Resource 1"

def test_read_resource_successful_text():
    """Test successful resources/read for a text resource."""
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })

    uri_to_read = "mcp://example.com/resource/1"
    response_data = rpc_call("resources/read", {"uri": uri_to_read})

    assert "result" in response_data
    assert "error" not in response_data
    result = response_data["result"]
    assert "contents" in result
    assert isinstance(result["contents"], list)
    assert len(result["contents"]) == 1
    content = result["contents"][0]
    assert content["uri"] == uri_to_read
    assert content["mimeType"] == "text/plain"
    assert "text" in content
    assert content["text"] == "Content of Example Resource 1"

def test_read_resource_not_found():
    """Test resources/read for a non-existent URI."""
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })

    response_data = rpc_call("resources/read", {"uri": "mcp://example.com/resource/nonexistent"})

    assert "error" in response_data
    assert response_data["error"]["code"] == -32601 # Or a more specific app error code mapped to this
    assert "not found" in response_data["error"]["message"].lower()

def test_read_resource_invalid_params_missing_uri():
    """Test resources/read with missing uri parameter."""
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })
    response_data = rpc_call("resources/read", {}) # Missing "uri"

    assert "error" in response_data
    assert response_data["error"]["code"] == -32602 # Invalid params
    assert "Invalid read_resource params" in response_data["error"]["message"]


# --- Tools Method Tests ---

def test_list_tools_successful():
    """Test successful tools/list request."""
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })

    response_data = rpc_call("tools/list")

    assert "result" in response_data
    assert "error" not in response_data
    result = response_data["result"]
    assert "tools" in result
    assert isinstance(result["tools"], list)
    # Check if default tool is present
    assert len(result["tools"]) >= 1
    assert result["tools"][0]["name"] == "example_tool"
    assert "inputSchema" in result["tools"][0]

def test_call_tool_successful():
    """Test successful tools/call for the example_tool."""
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })

    params = {
        "name": "example_tool",
        "arguments": {"message": "Hello from test"}
    }
    response_data = rpc_call("tools/call", params)

    assert "result" in response_data
    assert "error" not in response_data
    result = response_data["result"]
    assert "content" in result
    assert isinstance(result["content"], list)
    assert len(result["content"]) == 1
    content_item = result["content"][0]
    assert content_item["type"] == "text"
    assert "Example tool received: Hello from test" in content_item["text"]

def test_call_tool_not_found():
    """Test tools/call for a non-existent tool."""
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })

    params = {"name": "non_existent_tool", "arguments": {}}
    response_data = rpc_call("tools/call", params)

    assert "error" in response_data
    assert response_data["error"]["code"] == -32601 # Or app specific mapped to this
    assert "Tool with name non_existent_tool not found" in response_data["error"]["message"]

def test_call_tool_invalid_params_missing_name():
    """Test tools/call with missing 'name' parameter."""
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })

    response_data = rpc_call("tools/call", {"arguments": {}}) # Missing "name"

    assert "error" in response_data
    assert response_data["error"]["code"] == -32602 # Invalid params
    assert "Invalid call_tool params" in response_data["error"]["message"]

def test_call_tool_missing_required_argument():
    """Test tools/call for example_tool with missing required 'message' argument."""
    # Note: The current example_tool implementation doesn't strictly validate
    # its own arguments but defaults. If it did raise an error for missing
    # "message", this test would need adjustment or the tool logic would.
    # For now, we test the successful call path even with "missing" (defaulted) args.
    rpc_call("initialize", {
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0.1.0"},
        "protocolVersion": "2025-03-26"
    })

    params = {
        "name": "example_tool",
        "arguments": {} # "message" is required by schema but handler has a default
    }
    response_data = rpc_call("tools/call", params)

    assert "result" in response_data
    assert "error" not in response_data
    result = response_data["result"]
    assert "content" in result
    content_item = result["content"][0]
    assert "Example tool received: No message provided" in content_item["text"]

# Consider adding more tests:
# - Different ID types (int, null) for JSON-RPC requests.
# - Behavior when server is not initialized for methods that require it (though current handlers don't strictly check this globally yet).
# - Pagination for list methods if/when implemented (cursor).
# - Different resource types for resources/read (e.g., blob) once implemented.
# - Tools with more complex input schemas and validation.
# - Test for `isError` field in `CallToolResultData`.
# - Test `protocolVersion` mismatch if server were to enforce it strictly.
# - Test `experimental`, `roots`, `sampling` effects on `initialize` if server logic used them.
# - Test `ListResourcesParams` and `ListToolsParams` (e.g. cursor) when implemented.
# - Test `BlobResourceContents` for `resources/read` when implemented.
# - Test `ToolAnnotations` effects if handlers used them.
# - Test `instructions` field in `InitializeResultData` has expected content.
# - Test `clientInfo` in `InitializeRequestParams` is stored or used if intended.
# - Test `id: null` for notifications if the server should not respond. (FastAPI TestClient might auto-handle responses)

# To run these tests:
# 1. Ensure FastAPI and Uvicorn are installed.
# 2. Ensure Pytest is installed: `pip install pytest`
# 3. Navigate to the root of your `tldw_Server_API` project in the terminal.
# 4. Run pytest: `python -m pytest` or simply `pytest`
#    You might need to set PYTHONPATH: `export PYTHONPATH=.` (Linux/macOS) or `set PYTHONPATH=.` (Windows)
#    if Python has trouble finding your modules.
#    Or, if your tests directory is at the same level as `app` directory: `python -m pytest tests/MCP/test_mcp_endpoint.py`
#    If using VSCode, it usually handles test discovery and running well.
#
# Note on `mcp_server_state` and Test Isolation:
# These tests currently interact with a shared `mcp_server_state` dictionary in `mcp.py`.
# This means the order of tests *could* matter if one test changes state that another relies on
# (e.g., the `initialize` call in one test affects subsequent tests).
# Pytest typically runs tests in a discovered order, not necessarily the order in the file.
# For true test isolation with in-memory state:
#   - Reset `mcp_server_state` before each test (e.g., in a pytest fixture).
#   - Or, design handlers to be less reliant on mutable global state for basic tests,
#     or provide a way to inject/reset state for testing.
# For now, the tests are written assuming `initialize` is a prerequisite for most other operations,
# and many tests call it. This is acceptable for initial testing but consider refactoring for better isolation
# if the state becomes more complex or tests start interfering with each other.
# A simple fixture could be:
# @pytest.fixture(autouse=True)
# def reset_mcp_state():
#     from tldw_Server_API.app.api.v1.endpoints.mcp import mcp_server_state, ServerCapabilities, Resource, Tool # Import necessary components
#     mcp_server_state["initialized"] = False
#     mcp_server_state["client_capabilities"] = None
#     # Optionally reset known_resources and known_tools to their defaults if tests modify them
#     # This would require re-importing or deep copying the initial state.
#     # For this example, we'll assume tests don't modify the default resources/tools lists themselves,
#     # only the 'initialized' and 'client_capabilities' flags.
#     # If they do, a more robust reset is needed.
#     original_server_capabilities = ServerCapabilities(...) # Re-create or copy
#     original_resources = [...] # Re-create or copy
#     original_tools = [...] # Re-create or copy
#     # Then assign them back
#     # This is a common pattern for managing state in tests.
#     yield
#     # Any cleanup after test if needed

# For the current structure, explicitly calling initialize in tests that need it is a clear way
# to manage dependencies, though less "pure" in terms of unit test isolation.
# Consider using pytest-fastapi-deps for managing dependencies if they become complex.
# Example of a fixture for resetting state (simplified, adjust imports and initial state as needed):
"""
import copy
from tldw_Server_API.app.api.v1.endpoints.mcp import mcp_server_state as global_mcp_state
from tldw_Server_API.app.api.v1.schemas.mcp_schemas import ServerCapabilities, Resource, Tool, ToolInputSchema, ToolAnnotations

# Store the initial state once
initial_mcp_server_state = {
    "initialized": False,
    "server_capabilities": ServerCapabilities(
        resources={"listChanged": False, "subscribe": False},
        tools={"listChanged": False},
        prompts={"listChanged": False},
        logging=True,
        completions=False,
    ),
    "client_capabilities": None,
    "known_resources": [
        Resource(uri="mcp://example.com/resource/1", name="Example Resource 1", description="This is the first example resource.", mimeType="text/plain", size=100),
        Resource(uri="mcp://example.com/resource/2", name="Example Resource 2", description="This is the second example resource, also plain text.", mimeType="text/plain")
    ],
    "known_tools": [
        Tool(
            name="example_tool",
            description="An example tool that echoes input.",
            inputSchema=ToolInputSchema(type="object", properties={"message": {"type": "string", "description": "The message to echo."}}, required=["message"]),
            annotations=ToolAnnotations(title="Example Echo Tool", readOnlyHint=True)
        )
    ]
}

@pytest.fixture(autouse=True)
def reset_mcp_state_automatically():
    # Deep copy the initial state to ensure tests don't affect each other through mutable objects
    global_mcp_state.clear()
    global_mcp_state.update(copy.deepcopy(initial_mcp_server_state))
    yield
"""
# The above fixture, if enabled (by removing the triple quotes and ensuring correct imports/initial state),
# would run before each test, resetting the state. This is generally good practice.
# For now, the tests are written to be somewhat independent by re-initializing where necessary.
# If tests start failing randomly, state leakage between tests is a prime suspect.
