# tldw_Server_API/app/api/v1/endpoints/mcp.py
from fastapi import APIRouter, Request as FastAPIRequest, HTTPException
from typing import Any, Dict

from tldw_Server_API.app.api.v1.schemas.mcp_schemas import (
    InitializeRequestParams,
    InitializeResultData,
    # PingRequestParams, # Not strictly needed if params are not validated explicitly
    ListResourcesParams,
    ListResourcesResultData,
    ReadResourceParams,
    ReadResourceResultData,
    ListToolsParams,
    ListToolsResultData,
    CallToolParams,
    CallToolResultData,
    ErrorResponse,
    JSONRPCRequest,
    JSONRPCResponse,
    Resource,  # For mcp_server_state typing
    Tool,      # For mcp_server_state typing
    ServerCapabilities, # For mcp_server_state typing
    ToolInputSchema,    # For mcp_server_state typing
    ToolAnnotations,     # For mcp_server_state typing
    Implementation,
    TextResourceContents, # Added missing import
    TextContent # Added missing import
)

router = APIRouter()

# In-memory store for MCP server state (replace with a proper database if needed)
mcp_server_state: Dict[str, Any] = {
    "initialized": False,
    "server_capabilities": ServerCapabilities(
        resources={
            "listChanged": False,
            "subscribe": False,
        },
        tools={
            "listChanged": False,
        },
        prompts={
            "listChanged": False,
        },
        logging=True,
        completions=False,
    ),
    "client_capabilities": None,
    "known_resources": [
        Resource(
            uri="mcp://example.com/resource/1",
            name="Example Resource 1",
            description="This is the first example resource.",
            mimeType="text/plain",
            size=100,
        ),
        Resource(
            uri="mcp://example.com/resource/2",
            name="Example Resource 2",
            description="This is the second example resource, also plain text.",
            mimeType="text/plain",
        )
    ],
    "known_tools": [
        Tool(
            name="example_tool",
            description="An example tool that echoes input.",
            inputSchema=ToolInputSchema(
                type="object",
                properties={
                    "message": {"type": "string", "description": "The message to echo."}
                },
                required=["message"],
            ),
            annotations=ToolAnnotations(
                title="Example Echo Tool",
                readOnlyHint=True,
            )
        )
    ]
}

async def handle_initialize(params_data: Dict[str, Any]) -> InitializeResultData:
    try:
        validated_params = InitializeRequestParams(**params_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid initialize params: {e}")

    mcp_server_state["client_capabilities"] = validated_params.capabilities
    mcp_server_state["initialized"] = True

    return InitializeResultData(
        capabilities=mcp_server_state["server_capabilities"],
        serverInfo=Implementation(name="tldw-mcp-server", version="0.1.0"),
        protocolVersion="2025-03-26"
    )

async def handle_ping(params_data: Dict[str, Any]) -> Dict[str, Any]:
    return {}


async def handle_list_resources(params_data: Dict[str, Any]) -> ListResourcesResultData:
    # validated_params = ListResourcesParams(**params_data)
    return ListResourcesResultData(
        resources=mcp_server_state["known_resources"],
    )

async def handle_read_resource(params_data: Dict[str, Any]) -> ReadResourceResultData:
    try:
        validated_params = ReadResourceParams(**params_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid read_resource params: {e}")

    uri_to_read = validated_params.uri
    for res_model in mcp_server_state["known_resources"]:
        if res_model.uri == uri_to_read:
            if res_model.mimeType == "text/plain":
                return ReadResourceResultData(
                    contents=[TextResourceContents(
                        uri=uri_to_read,
                        text=f"Content of {res_model.name}",
                        mimeType="text/plain"
                    )]
                )
            else:
                raise HTTPException(status_code=501, detail=f"Reading resource type {res_model.mimeType} not implemented.")
    raise HTTPException(status_code=404, detail=f"Resource with URI {uri_to_read} not found.")


async def handle_list_tools(params_data: Dict[str, Any]) -> ListToolsResultData:
    # validated_params = ListToolsParams(**params_data)
    return ListToolsResultData(
        tools=mcp_server_state["known_tools"]
    )

async def handle_call_tool(params_data: Dict[str, Any]) -> CallToolResultData:
    try:
        validated_params = CallToolParams(**params_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid call_tool params: {e}")

    tool_name = validated_params.name
    tool_args = validated_params.arguments if validated_params.arguments else {}

    if tool_name == "example_tool":
        message = tool_args.get("message", "No message provided")
        return CallToolResultData(
            content=[TextContent(
                type="text",
                text=f"Example tool received: {message}"
            )]
        )
    raise HTTPException(status_code=404, detail=f"Tool with name {tool_name} not found.")


METHOD_HANDLERS = {
    "initialize": handle_initialize,
    "ping": handle_ping,
    "resources/list": handle_list_resources,
    "resources/read": handle_read_resource,
    "tools/list": handle_list_tools,
    "tools/call": handle_call_tool,
}

@router.post("/", response_model=JSONRPCResponse)
async def mcp_endpoint(fastapi_request: FastAPIRequest):
    try:
        json_rpc_body = await fastapi_request.json()
    except ValueError:
        return JSONRPCResponse(
            id=None,
            error=ErrorResponse(code=-32700, message="Parse error")
        )

    try:
        jrpc_req = JSONRPCRequest(**json_rpc_body)
    except Exception:
        req_id = json_rpc_body.get("id") if isinstance(json_rpc_body, dict) else None
        return JSONRPCResponse(
            id=req_id,
            error=ErrorResponse(code=-32600, message="Invalid Request")
        )

    request_id = jrpc_req.id

    if jrpc_req.method not in METHOD_HANDLERS:
        return JSONRPCResponse(
            id=request_id,
            error=ErrorResponse(code=-32601, message="Method not found")
        )

    handler = METHOD_HANDLERS[jrpc_req.method]

    try:
        params_data = jrpc_req.params if jrpc_req.params is not None else {}
        handler_result_data = await handler(params_data)

        return JSONRPCResponse(id=request_id, result=handler_result_data)

    except HTTPException as http_exc:
        error_code = -32000
        if 400 <= http_exc.status_code < 500:
            error_code = -32602
        elif http_exc.status_code == 404:
            error_code = -32601
        elif http_exc.status_code == 501:
             error_code = -32601

        return JSONRPCResponse(
            id=request_id,
            error=ErrorResponse(code=error_code, message=http_exc.detail)
        )
    except Exception as e:
        return JSONRPCResponse(
            id=request_id,
            error=ErrorResponse(code=-32603, message=f"Internal error: {str(e)}")
        )
