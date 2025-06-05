# tldw_Server_API/app/api/v1/schemas/mcp_schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal, Dict, Any

class JSONRPCBase(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"

class ErrorResponse(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None

class JSONRPCResponse(JSONRPCBase):
    id: Optional[Union[str, int]]
    result: Optional[Any] = None
    error: Optional[ErrorResponse] = None

class ClientCapabilities(BaseModel):
    experimental: Optional[Dict[str, Any]] = None
    roots: Optional[Dict[str, Any]] = None
    sampling: Optional[Dict[str, Any]] = None
    # Add other client capabilities as needed

class Implementation(BaseModel):
    name: str
    version: str

class InitializeRequestParams(BaseModel):
    capabilities: ClientCapabilities
    clientInfo: Implementation
    protocolVersion: str

class InitializeRequest(JSONRPCBase): # Used for validating incoming structure if needed
    method: Literal["initialize"]
    params: InitializeRequestParams
    id: Optional[Union[str, int]] = None


class ServerCapabilities(BaseModel):
    resources: Optional[Dict[str, Any]] = Field(default_factory=dict)
    tools: Optional[Dict[str, Any]] = Field(default_factory=dict)
    prompts: Optional[Dict[str, Any]] = Field(default_factory=dict)
    logging: Optional[bool] = False
    completions: Optional[bool] = False
    # Add other server capabilities

class InitializeResultData(BaseModel):
    capabilities: ServerCapabilities
    serverInfo: Implementation
    protocolVersion: str
    instructions: Optional[str] = None

class PingRequestParams(BaseModel):
    pass

class PingRequest(JSONRPCBase): # Used for validating incoming structure if needed
    method: Literal["ping"]
    params: Optional[PingRequestParams] = None
    id: Optional[Union[str, int]] = None

# --- Resources ---
class Resource(BaseModel):
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None
    size: Optional[int] = None
    annotations: Optional[Dict[str, Any]] = None

class ListResourcesParams(BaseModel):
    cursor: Optional[str] = None

class ListResourcesRequest(JSONRPCBase): # Used for validating incoming structure if needed
    method: Literal["resources/list"]
    params: Optional[ListResourcesParams] = None
    id: Optional[Union[str, int]]

class ListResourcesResultData(BaseModel):
    resources: List[Resource]
    nextCursor: Optional[str] = None


class ReadResourceParams(BaseModel):
    uri: str

class ReadResourceRequest(JSONRPCBase): # Used for validating incoming structure if needed
    method: Literal["resources/read"]
    params: ReadResourceParams
    id: Optional[Union[str, int]]

class TextResourceContents(BaseModel):
    uri: str
    text: str
    mimeType: Optional[str] = "text/plain"

class BlobResourceContents(BaseModel):
    uri: str
    blob: str # base64 encoded
    mimeType: str

class ReadResourceResultData(BaseModel):
    contents: List[Union[TextResourceContents, BlobResourceContents]]


# --- Tools ---
class ToolInputSchema(BaseModel):
    type: Literal["object"]
    properties: Dict[str, Dict[str, Any]]
    required: Optional[List[str]] = None

class ToolAnnotations(BaseModel):
    title: Optional[str] = None
    readOnlyHint: Optional[bool] = None
    destructiveHint: Optional[bool] = None
    idempotentHint: Optional[bool] = None
    openWorldHint: Optional[bool] = None


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    inputSchema: ToolInputSchema
    annotations: Optional[ToolAnnotations] = None

class ListToolsParams(BaseModel):
    cursor: Optional[str] = None

class ListToolsRequest(JSONRPCBase): # Used for validating incoming structure if needed
    method: Literal["tools/list"]
    params: Optional[ListToolsParams] = None
    id: Optional[Union[str, int]]

class ListToolsResultData(BaseModel):
    tools: List[Tool]
    nextCursor: Optional[str] = None


class CallToolParams(BaseModel):
    name: str
    arguments: Optional[Dict[str, Any]] = None

class CallToolRequest(JSONRPCBase): # Used for validating incoming structure if needed
    method: Literal["tools/call"]
    params: CallToolParams
    id: Optional[Union[str, int]]

class TextContent(BaseModel):
    type: Literal["text"]
    text: str

class CallToolResultData(BaseModel):
    content: List[TextContent]
    isError: Optional[bool] = False

# Generic request model for parsing before dispatching
class JSONRPCRequest(JSONRPCBase):
    method: str
    params: Optional[Dict[str, Any]] = None # Or Union of all specific Param types
    id: Optional[Union[str, int]] = None
