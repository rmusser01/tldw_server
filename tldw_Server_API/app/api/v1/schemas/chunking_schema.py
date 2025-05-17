# chunking_schema.py
#
# Imports
from typing import Optional, Any, Dict, List
#
# Third-party Libraries
from pydantic import Field, BaseModel, field_validator, model_validator

from tldw_Server_API.app.api.v1.endpoints.chunking import LLMOptionsForChunking
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

class LLMOptionsForChunkerInternalSteps(BaseModel): # Renamed for clarity
    """
    LLM configurations if a chunking method (e.g., 'rolling_summarize')
    internally uses an LLM for its processing steps.
    These are suggestions; server-side configurations (especially API keys) will take precedence.
    """
    provider: Optional[str] = Field(None,
                                    description="Suggested LLM provider for internal steps (e.g., 'openai', 'anthropic'). Server default if None.")
    model: Optional[str] = Field(None,
                                 description="Suggested LLM model for internal steps. Server default if None.")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0,
                                         description="Suggested temperature for LLM (0.0-2.0). Server default if None.")
    # You could add more specific LLM params here if desired, like:
    # system_prompt_override: Optional[str] = Field(None, description="Override the default system prompt for internal LLM summarization steps.")
    # max_tokens_for_step: Optional[int] = Field(None, gt=0, description="Suggest max tokens for each internal LLM summarization step.")

class ChunkingOptionsRequest(BaseModel):
    # Core Chunking Method & Basic Parameters
    method: Optional[str] = Field(default_chunk_options_from_lib.get('method'),
                                  description="Chunking method (e.g., 'words', 'sentences', 'json', 'semantic', 'xml', 'ebook_chapters', 'rolling_summarize').")
    max_size: Optional[int] = Field(default_chunk_options_from_lib.get('max_size'), gt=0,
                                   description="Max size of chunks (e.g., word count, sentence count, token count, item count). Must be > 0 if set.")
    overlap: Optional[int] = Field(default_chunk_options_from_lib.get('overlap'), ge=0,
                                  description="Overlap between chunks. Must be >= 0 if set.")
    language: Optional[str] = Field(default_chunk_options_from_lib.get('language'),
                                    description="Language of the text (e.g., 'en', 'zh', 'ja'). Auto-detected if None.")
    tokenizer_name_or_path: Optional[str] = Field(default_chunk_options_from_lib.get('tokenizer_name_or_path', "gpt2"), # Assuming you add this to DEFAULT_CHUNK_OPTIONS
                                                  description="Tokenizer model name or path from Hugging Face (e.g., 'gpt2', 'bert-base-uncased').")

    # Behavior Modifiers for Chunking
    adaptive: Optional[bool] = Field(default_chunk_options_from_lib.get('adaptive'),
                                     description="Enable adaptive chunk sizing for applicable methods.")
    multi_level: Optional[bool] = Field(default_chunk_options_from_lib.get('multi_level'),
                                        description="Enable multi-level chunking for applicable methods (e.g., paragraphs then words).")

    # Method-Specific Options
    custom_chapter_pattern: Optional[str] = Field(default_chunk_options_from_lib.get('custom_chapter_pattern'),
                                                  description="Custom regex pattern for 'ebook_chapters' method if default patterns are insufficient.")
    semantic_similarity_threshold: Optional[float] = Field(default_chunk_options_from_lib.get('semantic_similarity_threshold'), ge=0.0, le=1.0,
                                                       description="Similarity threshold (0.0-1.0) for 'semantic' chunking breaks.")
    semantic_overlap_sentences: Optional[int] = Field(default_chunk_options_from_lib.get('semantic_overlap_sentences'), ge=0,
                                                    description="Number of sentences to overlap for 'semantic' chunking.")
    json_chunkable_data_key: Optional[str] = Field(default_chunk_options_from_lib.get('json_chunkable_data_key', 'data'),
                                                 description="The key in a JSON object whose dictionary value should be chunked (for 'json' method with a dict input).")

    # Options for 'rolling_summarize' and other LLM-dependent chunking methods
    summarization_detail: Optional[float] = Field(default_chunk_options_from_lib.get('summarization_detail'), ge=0.0, le=1.0,
                                                description="Detail level (0.0-1.0) for 'rolling_summarize' method. Higher means more, smaller summarization steps.")
    # summarize_system_prompt: Optional[str] = Field(default_chunk_options_from_lib.get('summarize_system_prompt'),
    #                                               description="System prompt to use for the LLM during 'rolling_summarize' steps.")
    # summarize_additional_instructions: Optional[str] = Field(default_chunk_options_from_lib.get('summarize_additional_instructions'),
    #                                                         description="Additional instructions appended to the user prompt for 'rolling_summarize' steps.")
    # summarize_min_chunk_tokens: Optional[int] = Field(default_chunk_options_from_lib.get('summarize_min_chunk_tokens'), gt=0,
    #                                                    description="Minimum token size for text parts fed to the LLM in 'rolling_summarize'.")
    # summarize_chunk_delimiter: Optional[str] = Field(default_chunk_options_from_lib.get('summarize_chunk_delimiter'),
    #                                                 description="Delimiter used to initially split text for 'rolling_summarize'.")
    # summarize_recursively: Optional[bool] = Field(default_chunk_options_from_lib.get('summarize_recursively'),
    #                                              description="Enable recursive summarization for 'rolling_summarize'.")


    # Nested model for LLM parameters used by certain chunking methods
    llm_options_for_internal_steps: Optional[LLMOptionsForChunkerInternalSteps] = Field(None,
                                                                         description="Advanced: LLM configurations if the selected chunking method itself uses an LLM internally (e.g. 'rolling_summarize').")

    @field_validator('max_size', 'overlap', 'semantic_overlap_sentences', mode='before')
    @classmethod
    def ensure_int_type(cls, v: Any, info) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (ValueError, TypeError):
            raise ValueError(f"Field '{info.field_name}' must be an integer or convertible to an integer. Got: '{v}'")

    @field_validator('semantic_similarity_threshold', 'summarization_detail', 'temperature', mode='before', check_fields=False) # Added temperature to this logic
    # 'check_fields=False' might be needed if 'temperature' is in the nested model, apply validator to nested model then.
    # If 'temperature' is directly in this model:
    # @field_validator('semantic_similarity_threshold', 'summarization_detail', 'llm_options_for_internal_steps.temperature', mode='before') # This syntax for nested might not work directly.
    # It's better to put validators on the nested model LLMOptionsForChunkerInternalSteps for its fields.
    @classmethod
    def ensure_float_type(cls, v: Any, info) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Field '{info.field_name}' must be a float or convertible to a float. Got: '{v}'")

    @model_validator(mode='after') # Changed from 'root_validator' for Pydantic v2
    def check_overlap_less_than_max_size(cls, values: 'ChunkingOptionsRequest') -> 'ChunkingOptionsRequest':
        # Pydantic v2 passes the model instance to 'after' validators.
        # For 'before' model_validators, it passes a dict of values.
        # Access fields via values.max_size, values.overlap
        max_size, overlap = values.max_size, values.overlap # Get values from the model instance

        if max_size is not None and overlap is not None:
            # This validation makes most sense for methods where max_size and overlap are counts of the same unit
            # (e.g., words, sentences, tokens, list items).
            # For methods like 'xml' or 'ebook_chapters', their meaning of max_size/overlap might differ.
            # Consider making this check conditional based on 'method' if needed.
            current_method = values.method or default_chunk_options_from_lib.get('method')
            if current_method in ['words', 'sentences', 'tokens', 'paragraphs', 'json_list']: # Methods where this direct comparison is valid
                if overlap >= max_size:
                    raise ValueError(f"Overlap ({overlap}) must be less than max_size ({max_size}) for method '{current_method}'.")
        return values

class ChunkingTextRequest(BaseModel):
    text_content: str = Field(..., description="Text content to be chunked.")
    file_name: Optional[str] = Field("input_text.txt",
                                     description="Optional name for the input, used in some metadata/logging.")
    options: Optional[ChunkingOptionsRequest] = Field(None,
                                                     description="Chunking parameters. Library defaults will be used if not provided or partially provided.")

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

class ChunkedContentResponse(BaseModel):
    text: str
    metadata: Dict[str, Any] # Kept as Dict for flexibility from improved_chunking_process

class ChunkingResponse(BaseModel):
    chunks: List[ChunkedContentResponse]
    original_file_name: Optional[str]
    applied_options: ChunkingOptionsRequest # Shows the actual options used for the process

#
# End of chunking_schema.py
#######################################################################################################################
