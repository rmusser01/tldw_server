# rag_schemas.py
# Description: This file contains the Pydantic models for the Retrieval Augmented Generation API.
#
from typing import List, Optional, Dict, Any, Union, Literal as TypingLiteral
from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from uuid import UUID, uuid4
from enum import Enum

# --- Enums ---
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"

class ToolCallType(str, Enum):
    FUNCTION = "function"

class SearchModeEnum(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    CUSTOM = "custom"

class IndexMeasureEnum(str, Enum):
    COSINE_DISTANCE = "cosine_distance"
    L2_DISTANCE = "l2_distance"
    MAX_INNER_PRODUCT = "max_inner_product"
    # Add other values if specified in your /abstractions/search.py

class ToolTypeEnum(str, Enum):
    FUNCTION = "function"

class RAGToolEnum(str, Enum):
    WEB_SEARCH = "web_search"
    WEB_SCRAPE = "web_scrape"
    SEARCH_FILE_DESCRIPTIONS = "search_file_descriptions"
    SEARCH_FILE_KNOWLEDGE = "search_file_knowledge"
    GET_FILE_CONTENT = "get_file_content"

class ResearchToolEnum(str, Enum):
    RAG = "rag"
    REASONING = "reasoning"
    CRITIQUE = "critique"
    PYTHON_EXECUTOR = "python_executor"

class AgentModeEnum(str, Enum):
    RAG = "rag"
    RESEARCH = "research"

class ReasoningEffortEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# --- Message Sub-Models ---
class FunctionCall(BaseModel):
    arguments: str
    name: str

class ToolFunction(BaseModel):
    arguments: str
    name: str

class ToolCall(BaseModel):
    id: str
    function: ToolFunction
    type: ToolCallType = ToolCallType.FUNCTION

class ImageData(BaseModel):
    media_type: str = Field(..., examples=["image/png", "image/jpeg"])
    data: str = Field(..., description="Base64 encoded image data.")

class Message(BaseModel):
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    structured_content: Optional[List[Dict[str, Any]]] = None
    image_url: Optional[HttpUrl] = None
    image_data: Optional[ImageData] = None

    model_config = {
        "extra": "allow" # Allow extra fields if any come from external systems
    }

# --- SearchSettings Sub-Models ---
class HybridSearchSettings(BaseModel):
    full_text_weight: float = 1.0
    semantic_weight: float = 5.0
    full_text_limit: int = 200
    rrf_k: int = 50

class ChunkSearchSettings(BaseModel):
    index_measure: IndexMeasureEnum = IndexMeasureEnum.COSINE_DISTANCE
    probes: int = 10
    ef_search: int = 40
    enabled: bool = True

class GraphSearchSettings(BaseModel):
    limits: Dict[str, int] = Field(default_factory=dict, examples=[{"entity": 5, "relationship": 3}])
    enabled: bool = True

class SearchSettings(BaseModel):
    use_hybrid_search: bool = False
    use_semantic_search: bool = True
    use_fulltext_search: bool = False
    filters: Dict[str, Any] = Field(default_factory=dict, examples=[{"metadata.year": {"$gt": 2020}}])
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    include_metadatas: bool = True
    include_scores: bool = True
    search_strategy: str = Field(default="vanilla", examples=["vanilla", "query_fusion", "hyde"])
    hybrid_settings: Optional[HybridSearchSettings] = None
    chunk_settings: Optional[ChunkSearchSettings] = None
    graph_settings: Optional[GraphSearchSettings] = None
    num_sub_queries: int = Field(default=5, description="For 'hyde' or 'rag_fusion' strategies.")

# --- GenerationConfig Sub-Models ---
class ToolFunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None # JSON Schema for parameters

class ToolDefinition(BaseModel):
    type: ToolTypeEnum = ToolTypeEnum.FUNCTION
    function: ToolFunctionDefinition

class GenerationConfig(BaseModel):
    model: Optional[str] = Field(None, examples=["openai/gpt-4o-mini"])
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    max_tokens_to_sample: int = Field(default=1024, gt=0)
    stream: bool = False
    functions: Optional[List[Dict[str, Any]]] = Field(None, description="Deprecated. Use 'tools' instead.")
    tools: Optional[List[ToolDefinition]] = None
    add_generation_kwargs: Optional[Dict[str, Any]] = None
    api_base: Optional[HttpUrl] = None
    response_format: Optional[Dict[str, Any]] = Field(None, description="Specifies response format (e.g., JSON object). Structure depends on provider.")
    extended_thinking: bool = False
    thinking_budget: Optional[int] = Field(None, gt=0)
    reasoning_effort: Optional[ReasoningEffortEnum] = None

# --- Main Request Model ---
class RetrievalAgentRequest(BaseModel):
    message: Optional[Message] = None
    messages: Optional[List[Message]] = Field(None, description="(Deprecated) List of messages representing the conversation history. Use 'message' and 'conversation_id'.")

    search_mode: SearchModeEnum = SearchModeEnum.CUSTOM
    search_settings: Optional[SearchSettings] = None

    rag_generation_config: Optional[GenerationConfig] = None
    research_generation_config: Optional[GenerationConfig] = None

    rag_tools: Optional[List[RAGToolEnum]] = None
    research_tools: Optional[List[ResearchToolEnum]] = None

    task_prompt: Optional[str] = Field(None, examples=["You are an expert financial analyst."])
    include_title_if_available: bool = True
    conversation_id: Optional[UUID] = None
    max_tool_context_length: int = Field(default=32768, gt=0)
    use_system_context: bool = True
    mode: AgentModeEnum = AgentModeEnum.RAG
    needs_initial_conversation_name: Optional[bool] = None


    @model_validator(mode='after')
    def check_message_conditional_requirement(self) -> 'RetrievalAgentRequest':
        if self.message is None and self.messages is None:
            raise ValueError("Either 'message' or the deprecated 'messages' field must be provided.")
        if self.message is None and not self.messages: # if messages is empty list
             raise ValueError("If 'message' is not provided, 'messages' must not be empty.")
        return self

# --- Response Models ---
class Citation(BaseModel):
    document_id: Optional[str] = None
    source_name: str
    content: Optional[str] = Field(None, description="Relevant snippet from the source.")
    link: Optional[HttpUrl] = None

class RetrievalAgentResponse(BaseModel):
    conversation_id: UUID
    response_message: Message # The agent's reply message object
    search_results_summary: Optional[str] = Field(None, description="Optional summary of retrieved context or search process.")
    citations: Optional[List[Citation]] = Field(None, description="List of sources/citations used by the agent to formulate the response.")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Optional field for debugging information related to the request processing.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "conversation_id": "a1b2c3d4-e5f6-7777-8888-999a0b1c2d3e",
                "response_message": {
                    "role": "assistant",
                    "content": "AI-powered search is rapidly evolving with advancements in large language models..."
                },
                "citations": [
                    {
                        "source_name": "AI Search Trends Report 2024.pdf",
                        "content": "LLMs are now capable of understanding complex queries...",
                        "document_id": "doc-xyz-123"
                    }
                ]
            }
        }
    }


# --- Models for Search API based on OpenAPI ---

class FiltersModel(BaseModel):
    # Using Dict[str, Any] for flexibility as filters can be complex (e.g., {"$and": [...]})
    # and field-specific operator structures.
    # Example: {"document_id": {"$eq": "some-uuid"}, "tags": {"$in": ["AI", "research"]}}
    # Example: {"$and": [{"metadata.year": {"$gt": 2020}}, {"metadata.type": {"$eq": "report"}}]}
    root: Dict[str, Any] = Field(default_factory=dict)

    def __getitem__(self, item):
        return self.root[item]

    def __setitem__(self, key, value):
        self.root[key] = value

    def items(self):
        return self.root.items()

    def get(self, key, default=None):
        return self.root.get(key, default)

class HybridSettingsModel(BaseModel):
    # additionalProperties: true implies flexibility
    # If specific known properties exist, they should be defined.
    # For now, allowing any extra fields.
    model_config = {
        "extra": "allow"
    }
    # Example properties, if known, could be added here:
    # semantic_weight: Optional[float] = None
    # full_text_weight: Optional[float] = None


class ChunkSettingsModel(BaseModel):
    index_measure: Optional[IndexMeasureEnum] = None # Use existing enum, make optional as per OpenAPI not listing it as required
    probes: int = 10
    ef_search: int = 40
    enabled: bool = True

class GraphSettingsModel(BaseModel):
    limits: Dict[str, int] = Field(default_factory=dict)
    enabled: bool = True


class RagGenerationConfigStandard(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens_to_sample: Optional[int] = None
    stream: Optional[bool] = None
    functions: Optional[List[Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None # Assuming general object for tools for now
    add_generation_kwargs: Optional[Dict[str, Any]] = None
    api_base: Optional[HttpUrl] = None
    response_format: Dict[str, Any] # Required in this part of oneOf
    extended_thinking: bool = False
    thinking_budget: Optional[int] = None
    reasoning_effort: Optional[ReasoningEffortEnum] = None # Reuse existing enum
    task_prompt: Optional[str] = None # This was under RagGenerationConfig in OpenAPI, not SearchParameterFields.
                                      # But it's listed under properties here.
    include_title_if_available: bool = False
    include_web_search: bool = False

    @field_validator('thinking_budget')
    def check_thinking_budget(cls, v, values):
        if v is not None and 'max_tokens_to_sample' in values.data and values.data['max_tokens_to_sample'] is not None:
            if v >= values.data['max_tokens_to_sample']:
                raise ValueError('thinking_budget must be less than max_tokens_to_sample')
        return v

# For RagGenerationConfigBaseModel which is "additionalProperties: true"
# We can represent this as Dict[str, Any] or a BaseModel with extra='allow'
class RagGenerationConfigBase(BaseModel):
    model_config = {
        "extra": "allow"
    }

RagGenerationConfigUnion = Union[RagGenerationConfigStandard, RagGenerationConfigBase]


class SearchParameterFields(BaseModel):
    use_hybrid_search: bool = False
    use_semantic_search: bool = True
    use_fulltext_search: bool = False
    filters: Optional[FiltersModel] = Field(default_factory=FiltersModel) # Use FiltersModel
    limit: int = Field(default=10, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    include_metadatas: bool = True
    include_scores: bool = True
    search_strategy: str = Field(default="vanilla", examples=["vanilla", "query_fusion", "hyde"])
    hybrid_settings: Optional[HybridSettingsModel] = None # Use HybridSettingsModel
    chunk_settings: Optional[ChunkSettingsModel] = None   # Use ChunkSettingsModel
    graph_settings: Optional[GraphSettingsModel] = None   # Use GraphSettingsModel
    num_sub_queries: int = Field(default=5)
    rag_generation_config: Optional[RagGenerationConfigUnion] = None # Use the Union

    model_config = {
        "extra": "allow" # To accommodate any fields if SearchRequest inherits and has more
    }


class SearchApiRequest(SearchParameterFields): # Inherits top-level fields
    querystring: str
    search_mode: SearchModeEnum = SearchModeEnum.CUSTOM # Reuse existing enum
    # search_settings is Optional and of type SearchParameterFields itself
    search_settings: Optional[SearchParameterFields] = None # This allows nested SearchParameterFields


# --- Response model for the new Search API Endpoint ---
class SearchApiResultItem(BaseModel):
    id: str = Field(..., description="Unique ID of the search result item.")
    score: Optional[float] = Field(None, description="Relevance score of the item.")
    title: Optional[str] = Field(None, description="Title of the item.")
    snippet: Optional[str] = Field(None, description="A short snippet from the item's content.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Associated metadata.")
    # Add other fields like document_url, type, etc. as needed

class SearchApiResponse(BaseModel):
    query_id: UUID = Field(default_factory=uuid4, description="Unique ID for this search query.")
    querystring_echo: str = Field(..., description="The original query string that was processed.")
    results: List[SearchApiResultItem] = Field(default_factory=list, description="List of search results.")
    total_results: Optional[int] = Field(None, description="Total number of results found (might be an estimate or exact).")
    # You can add more fields like applied_filters, warnings, suggestions, etc.
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Optional debugging information.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
                "querystring_echo": "latest developments in AI",
                "results": [
                    {
                        "id": "doc-123",
                        "score": 0.89,
                        "title": "AI Breakthroughs in 2024",
                        "snippet": "Recent advancements in large language models have led to significant...",
                        "metadata": {"type": "research_paper", "year": 2024}
                    }
                ],
                "total_results": 1
            }
        }
    }

#
# End of rag_schemas.py
#######################################################################################################################