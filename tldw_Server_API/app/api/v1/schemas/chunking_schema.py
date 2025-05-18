# chunking_schema.py
#
# Imports
from typing import Optional, Any, Dict, List
#
# Third-party Libraries
from pydantic import Field, BaseModel, field_validator, model_validator
#
# Local Imports
from tldw_Server_API.app.core.Utils.Chunk_Lib import (
    DEFAULT_CHUNK_OPTIONS as default_chunk_options_from_lib
)
#
###########################################################################################################################
#
# Functions:

# --- Pydantic Schemas for Request and Response ---

class LLMOptionsForChunkerInternalSteps(BaseModel):
    """
    LLM configurations if a chunking method (e.g., 'rolling_summarize')
    internally uses an LLM for its processing steps.
    The Provider and Model for these steps are determined by the server.
    Client can suggest other parameters.
    """
    # Provider and Model are now server-determined for these internal steps.

    temperature: Optional[float] = Field(None, ge=0.0, le=2.0,
                                         description="Suggested temperature for LLM (0.0-2.0) for internal steps. Server default if None.")
    system_prompt_for_step: Optional[str] = Field(None,
                                                 description="Suggest a system prompt for each internal LLM step. Server might have a default or append to this.")
    max_tokens_per_step: Optional[int] = Field(None, gt=0,
                                            description="Suggest max tokens for the LLM to generate in each internal step. Server might cap this.")

    @field_validator('temperature', mode='before')
    @classmethod
    def ensure_float_type(cls, v: Any, info) -> Optional[float]:
        if v is None: return None
        try: return float(v)
        except (ValueError, TypeError): raise ValueError(f"Field '{info.field_name}' must be a float. Got: '{v}'")

    @field_validator('max_tokens_per_step', mode='before')
    @classmethod
    def ensure_int_type(cls, v: Any, info) -> Optional[int]:
        if v is None: return None
        try: return int(v)
        except (ValueError, TypeError): raise ValueError(f"Field '{info.field_name}' must be an int. Got: '{v}'")


class ChunkingOptionsRequest(BaseModel):
    # Core Chunking Method & Basic Parameters
    method: Optional[str] = Field(default_chunk_options_from_lib.get('method'),
                                  description="Chunking method (e.g., 'words', 'sentences', 'json', 'semantic', 'xml', 'ebook_chapters', 'rolling_summarize').")
    max_size: Optional[int] = Field(default_chunk_options_from_lib.get('max_size'), gt=0,
                                   description="Max size of chunks. Must be > 0 if set.")
    overlap: Optional[int] = Field(default_chunk_options_from_lib.get('overlap'), ge=0,
                                  description="Overlap between chunks. Must be >= 0 if set.")
    language: Optional[str] = Field(default_chunk_options_from_lib.get('language'),
                                    description="Language of the text. Auto-detected if None.")
    tokenizer_name_or_path: Optional[str] = Field(default_chunk_options_from_lib.get('tokenizer_name_or_path', "gpt2"),
                                                  description="Tokenizer model name or path.")

    # Behavior Modifiers
    adaptive: Optional[bool] = Field(default_chunk_options_from_lib.get('adaptive'),
                                     description="Enable adaptive chunk sizing.")
    multi_level: Optional[bool] = Field(default_chunk_options_from_lib.get('multi_level'),
                                        description="Enable multi-level chunking.")

    # Method-Specific Options
    custom_chapter_pattern: Optional[str] = Field(default_chunk_options_from_lib.get('custom_chapter_pattern'),
                                                  description="Custom regex for 'ebook_chapters' method.")
    semantic_similarity_threshold: Optional[float] = Field(default_chunk_options_from_lib.get('semantic_similarity_threshold'), ge=0.0, le=1.0,
                                                       description="Similarity threshold for 'semantic' chunking.")
    semantic_overlap_sentences: Optional[int] = Field(default_chunk_options_from_lib.get('semantic_overlap_sentences'), ge=0,
                                                    description="Sentence overlap for 'semantic' chunking.")
    json_chunkable_data_key: Optional[str] = Field(default_chunk_options_from_lib.get('json_chunkable_data_key', 'data'),
                                                 description="Key in JSON object whose dict value should be chunked.")

    # Options for 'rolling_summarize' (high-level controls)
    summarization_detail: Optional[float] = Field(default_chunk_options_from_lib.get('summarization_detail'), ge=0.0, le=1.0,
                                                description="Detail level for 'rolling_summarize' (0.0-1.0).")

    # Nested model for LLM parameters (client suggestions for non-provider/model aspects)
    llm_options_for_internal_steps: Optional[LLMOptionsForChunkerInternalSteps] = Field(None,
                                                                         description="Advanced: Client suggestions for LLM configurations (e.g., temperature, step-specific prompts/tokens) if the selected chunking method uses an LLM internally. Provider and Model are server-determined for these steps.")

    # ... (validators: ensure_int_type, ensure_float_type, check_overlap_less_than_max_size as before) ...
    @field_validator('max_size', 'overlap', 'semantic_overlap_sentences', mode='before')
    @classmethod
    def ensure_int_type(cls, v: Any, info) -> Optional[int]:
        if v is None: return None
        try: return int(v)
        except (ValueError, TypeError): raise ValueError(f"Field '{info.field_name}' must be an integer. Got: '{v}'")

    @field_validator('semantic_similarity_threshold', 'summarization_detail', mode='before')
    @classmethod
    def ensure_float_type(cls, v: Any, info) -> Optional[float]:
        if v is None: return None
        try: return float(v)
        except (ValueError, TypeError): raise ValueError(f"Field '{info.field_name}' must be a float. Got: '{v}'")

    @model_validator(mode='after')
    def check_overlap_less_than_max_size(cls, values: 'ChunkingOptionsRequest') -> 'ChunkingOptionsRequest':
        max_size, overlap = values.max_size, values.overlap
        current_method = values.method or default_chunk_options_from_lib.get('method')
        if max_size is not None and overlap is not None:
            if current_method in ['words', 'sentences', 'tokens', 'paragraphs', 'json_list']:
                if overlap >= max_size:
                    raise ValueError(f"Overlap ({overlap}) must be less than max_size ({max_size}) for method '{current_method}'.")
        return values

class ChunkingTextRequest(BaseModel):
    text_content: str = Field(..., description="Text content to be chunked.")
    file_name: Optional[str] = Field("input_text.txt", description="Optional name for the input.")
    options: Optional[ChunkingOptionsRequest] = Field(None, description="Chunking parameters.")

class ChunkedContentResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]

class ChunkingResponse(BaseModel):
    chunks: List[ChunkedContentResponse]
    original_file_name: Optional[str]
    applied_options: ChunkingOptionsRequest

class ChunkMetadataResponse(BaseModel):
    # This should mirror the keys present in the 'metadata' dict returned by improved_chunking_process
    # Using Dict[str, Any] for flexibility as metadata content can be dynamic
    chunk_index: int
    total_chunks: int
    chunk_method: str
    max_size: int # This will be the *actual* max_size used after defaults/overrides
    overlap: int  # This will be the *actual* overlap used
    language: Optional[str]
    relative_position: float
    header_text: Optional[str] = None
    # Add any other common fields you expect, or leave it more open if very dynamic
    # For example, if json_content was extracted:
    # initial_json_metadata: Optional[Dict[str, Any]] = None # Example

#
# End of chunking_schema.py
#######################################################################################################################
