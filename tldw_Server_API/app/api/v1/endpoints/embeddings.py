import base64
from typing import List, Union, Optional

import numpy as np
import tiktoken # type: ignore
from fastapi import APIRouter, HTTPException, Body

from tldw_Server_API.app.api.v1.schemas.embeddings_models import CreateEmbeddingResponse, CreateEmbeddingRequest, \
    EmbeddingData, EmbeddingUsage

# Assuming Embeddings_Create.py and Utils.py are in a path reachable by Python.
# Adjust the import path based on your actual project structure.
try:
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        create_embeddings_batch,
        default_embedding_provider, # Updated name
        default_embedding_model,    # Updated name
        default_embedding_api_url,  # Updated name
    )
    from tldw_Server_API.app.core.Utils.Utils import logging
except ImportError:
    import logging
    logging.warning("Could not import from tldw_Server_API. Using placeholder for embedding functions and config.")
    # Placeholder functions and configs for standalone testing:
    def create_embeddings_batch(
        texts: List[str],
        model_override: Optional[str] = None,
        provider_override: Optional[str] = None,
        api_url_override: Optional[str] = None,
        timeout_seconds: int = 300
    ) -> List[List[float]]:
        model_to_use = model_override or "placeholder_model"
        provider_to_use = provider_override or "placeholder_provider"
        logging.info(f"Placeholder: Creating embeddings for {len(texts)} texts with model {model_to_use} via {provider_to_use}")
        if model_to_use == "text-embedding-ada-002":
            return [[(0.01 * len(text) + i * 0.0001 + idx * 0.1) for i in range(1536)] for idx, text in enumerate(texts)]
        return [[(0.01 * len(text) + i * 0.001 + idx * 0.1) for i in range(768)] for idx, text in enumerate(texts)]

    default_embedding_provider = "placeholder_provider"
    default_embedding_model = "placeholder_model"
    default_embedding_api_url = "http://localhost:8000/placeholder_embed"


# --- Token Counting ---
def count_tokens(text: str, model_name: str) -> int:
    """Counts tokens in a string for a given model."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        logging.warning(f"tiktoken encoding for model '{model_name}' not found. Using 'cl100k_base'.")
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e_tiktoken: # pylint: disable=broad-except
            logging.warning(f"'cl100k_base' encoding not found ({e_tiktoken}). Falling back to string split for token count.")
            return len(text.split())
    return len(encoding.encode(text))

def count_tokens_for_list(texts: List[str], model_name: str) -> int:
    """Counts total tokens for a list of strings."""
    total = 0
    for text_item in texts:
        total += count_tokens(text_item, model_name)
    return total

# --- FastAPI Router ---
router = APIRouter()

@router.post(
    "/embeddings",
    response_model=CreateEmbeddingResponse,
    summary="Creates an embedding vector representing the input text.",
    tags=["Embeddings"]
)
async def create_embedding_endpoint(
    request: CreateEmbeddingRequest = Body(...)
):
    input_data = request.input
    model_id_from_request = request.model # User explicitly provides this as per spec
    encoding_format = request.encoding_format
    # dimensions_param = request.dimensions # Parsed, usage depends on backend library
    # user_param = request.user             # Parsed, usage depends on backend library

    texts_to_embed: List[str] = []
    num_prompt_tokens: int = 0

    if isinstance(input_data, str):
        if not input_data.strip():
            raise HTTPException(status_code=400, detail="Input string cannot be empty.")
        texts_to_embed = [input_data]
        num_prompt_tokens = count_tokens(input_data, model_id_from_request)
    elif isinstance(input_data, list):
        if not input_data:
            raise HTTPException(status_code=400, detail="Input list cannot be empty.")
        if len(input_data) > 2048:
            raise HTTPException(status_code=400, detail="Input array must not exceed 2048 elements.")

        if all(isinstance(item, str) for item in input_data):
            if any(not item.strip() for item in input_data):
                 raise HTTPException(status_code=400, detail="Input strings in list cannot be empty.")
            texts_to_embed = input_data
            num_prompt_tokens = count_tokens_for_list(input_data, model_id_from_request)
        elif all(isinstance(item, int) for item in input_data):
            logging.warning(f"Received tokenized input (List[int]) for model '{model_id_from_request}'. Current backend expects text. Processing of raw tokens is not supported by this endpoint's backend.")
            num_prompt_tokens = len(input_data)
            raise HTTPException(status_code=400, detail="List[int] (tokenized) input is not directly supported by this endpoint's current text-based embedding backend. Please provide strings.")
        elif all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in input_data):
            logging.warning(f"Received batch tokenized input (List[List[int]]) for model '{model_id_from_request}'. Current backend expects text. Processing of raw tokens is not supported by this endpoint's backend.")
            num_prompt_tokens = sum(len(sublist) for sublist in input_data if isinstance(sublist, list))
            raise HTTPException(status_code=400, detail="List[List[int]] (batch tokenized) input is not directly supported by this endpoint's current text-based embedding backend. Please provide strings.")
        else:
            raise HTTPException(status_code=400, detail="Invalid input list format. List must contain all strings, all integers, or all lists of integers. Mixed types are not supported.")
    else:
        raise HTTPException(status_code=400, detail="Invalid input type. Must be str or a list.")

    if not texts_to_embed:
        raise HTTPException(status_code=400, detail="No processable text input derived from the request.")

    try:
        # The create_embeddings_batch function now uses its internal defaults for provider and API URL,
        # and takes model_override.
        # The 'model' from the request *is* the override for the library's default model.
        raw_embeddings_list = create_embeddings_batch(
            texts=texts_to_embed,
            model_override=model_id_from_request, # Pass the user-specified model here
            # provider_override and api_url_override will be None, so library defaults are used.
            # If you wanted to map model_id_from_request to a specific provider/URL,
            # you'd add that logic here before calling create_embeddings_batch.
        )
    except FileNotFoundError as e:
        logging.error(f"Model file not found for {model_id_from_request}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding model file not found or configured correctly for model '{model_id_from_request}'.")
    except ValueError as e: # Catch specific value errors from the library
        logging.warning(f"ValueError during embedding creation for model '{model_id_from_request}': {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error creating embeddings for model '{model_id_from_request}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

    output_data: List[EmbeddingData] = []
    for i, raw_embedding in enumerate(raw_embeddings_list):
        embedding_floats: List[float]
        if hasattr(raw_embedding, 'tolist'):
            embedding_floats = raw_embedding.tolist()
        elif isinstance(raw_embedding, list) and all(isinstance(x, (float, int)) for x in raw_embedding):
            embedding_floats = [float(x) for x in raw_embedding]
        else:
            logging.error(f"Unexpected embedding format for item {i} with model '{model_id_from_request}': {type(raw_embedding)}")
            raise HTTPException(status_code=500, detail="Internal error processing embedding format.")

        processed_embedding_value: Union[List[float], str]
        if encoding_format == "base64":
            byte_array = np.array(embedding_floats, dtype=np.float32).tobytes()
            processed_embedding_value = base64.b64encode(byte_array).decode('utf-8')
        else:
            processed_embedding_value = embedding_floats

        output_data.append(
            EmbeddingData(
                embedding=processed_embedding_value,
                index=i
            )
        )

    usage = EmbeddingUsage(
        prompt_tokens=num_prompt_tokens,
        total_tokens=num_prompt_tokens
    )

    return CreateEmbeddingResponse(
        data=output_data,
        model=model_id_from_request, # Return the model ID that was actually used.
        usage=usage
    )

#
# End of embeddings.py
#######################################################################################################################
