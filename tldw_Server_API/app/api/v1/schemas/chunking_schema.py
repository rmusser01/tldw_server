# chunking_schema.py
#
# Imports
from typing import Optional, Any, Dict, List
#
# Third-party Libraries
from pydantic import Field, BaseModel, field_validator
#
# Local Imports
from tldw_Server_API.app.core.Utils.Chunk_Lib import (
    improved_chunking_process,
    chunk_options as default_chunk_options_from_lib
)
#
###########################################################################################################################
#
# Functions:

# --- Pydantic Schemas for Request and Response ---

class ChunkingOptionsRequest(BaseModel):
    method: Optional[str] = Field(default_chunk_options_from_lib.get('method'),
                                  description="Chunking method (e.g., 'words', 'sentences', 'json', 'semantic', 'xml', 'ebook_chapters').")
    max_size: Optional[int] = Field(default_chunk_options_from_lib.get('max_size'),
                                   description="Max size of chunks (meaning depends on method: words, sentences, tokens, items, etc.).")
    overlap: Optional[int] = Field(default_chunk_options_from_lib.get('overlap'),
                                  description="Overlap between chunks (meaning depends on method).")
    language: Optional[str] = Field(default_chunk_options_from_lib.get('language'),
                                    description="Language of the text (e.g., 'en', 'zh'). Auto-detected if None.")
    # Add other options from your library's default_chunk_options or that improved_chunking_process might use
    # e.g., adaptive, multi_level, custom_chapter_pattern
    adaptive: Optional[bool] = Field(default_chunk_options_from_lib.get('adaptive'),
                                     description="Enable adaptive chunking (if applicable to method).")
    multi_level: Optional[bool] = Field(default_chunk_options_from_lib.get('multi_level'),
                                        description="Enable multi-level chunking (if applicable to method).")
    custom_chapter_pattern: Optional[str] = Field(None,
                                                  description="Custom regex pattern for 'ebook_chapters' method.")

    @field_validator('max_size', 'overlap', mode='before')
    @classmethod
    def ensure_int_type(cls, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ValueError("max_size and overlap must be integers")

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
