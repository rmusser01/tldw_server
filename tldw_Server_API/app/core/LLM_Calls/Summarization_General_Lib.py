# Summarization_General_Lib.py
#########################################
# General Summarization Library
# This library is used to perform summarization.
#
####
####################
# Function List
#
# 1. extract_text_from_segments(segments: List[Dict]) -> str
# 2. recursive_summarize_chunks(chunks: List[str], summarize_func: Callable[[str], str]) -> str
# 3. analyze(...)
#
#
####################
# Import necessary libraries
import inspect
import json
import os
from collections.abc import Generator
from typing import Any, Callable, Optional, Union

from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.Chat.streaming_utils import _extract_text_from_upstream_sse

#
# Import Local
from tldw_Server_API.app.core.Chunking import improved_chunking_process
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_name,
    custom_openai_provider_number,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    resolve_provider_section,
)
from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt
from tldw_Server_API.app.core.Utils.Utils import logging

#
#######################################################################################################################
# Function Definitions
#

loaded_config_data = load_and_log_configs()

_ADAPTER_PROVIDER_ALIASES = {
    "custom_openai_api": "custom-openai-api",
    "custom_openai_api_2": "custom-openai-api-2",
    "custom-openai-api": "custom-openai-api",
    "custom-openai-api-2": "custom-openai-api-2",
}

_SUMMARIZATION_PROMPT_KEY = "Summarization System Prompt"
_DEFAULT_SYSTEM_PROMPT = (
    "You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, "
    "you should follow these guidelines: Use multiple headings based on the referenced topics, "
    "not categories like quotes or terms. Headings should be surrounded by bold formatting and not be "
    "listed as bullet points themselves. Leave no space between headings and their corresponding list items "
    "underneath. Important terms within the content should be emphasized by setting them in bold font. "
    "Any text that ends with a colon should also be bolded. Before submitting your response, review the "
    "instructions, and make any corrections necessary to adhered to the specified format. Do not reference "
    "these instructions within the notes.``` \nBased on the content between backticks create comprehensive "
    "bulleted notes.\n"
    "**Bulleted Note Creation Guidelines**\n\n"
    "**Headings**:\n"
    "- Based on referenced topics, not categories like quotes or terms\n"
    "- Surrounded by **bold** formatting\n"
    "- Not listed as bullet points\n"
    "- No space between headings and list items underneath\n\n"
    "**Emphasis**:\n"
    "- **Important terms** set in bold font\n"
    "- **Text ending in a colon**: also bolded\n\n"
    "**Review**:\n"
    "- Ensure adherence to specified format\n"
    "- Do not reference these instructions in your response."
)
_SUMMARIZATION_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AssertionError,
    AttributeError,
    ConnectionError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeError,
    ValueError,
    json.JSONDecodeError,
)


def _resolve_default_system_prompt() -> str:
    prompt = load_prompt("summarization", _SUMMARIZATION_PROMPT_KEY)
    if prompt:
        return prompt
    return _DEFAULT_SYSTEM_PROMPT


def _adapter_provider_name(api_name: str) -> str:
    normalized = normalize_provider(api_name)
    custom_number = custom_openai_provider_number(normalized)
    if custom_number is not None:
        return custom_openai_provider_name(custom_number)
    return _ADAPTER_PROVIDER_ALIASES.get(normalized, normalized)


def _build_summary_prompt(text: str, custom_prompt_arg: Optional[str]) -> str:
    suffix = custom_prompt_arg or ""
    if suffix:
        return f"{text}\n\n\n\n{suffix}"
    return text


def _resolve_adapter_timeout(provider: str, app_config: dict[str, Any]) -> Optional[float]:
    section = resolve_provider_section(provider)
    if section:
        raw = (app_config.get(section) or {}).get("api_timeout")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
    return None


def _summarize_via_adapter(
    *,
    api_name: str,
    text_to_summarize: str,
    custom_prompt_arg: Optional[str],
    api_key: Optional[str],
    temp: Optional[float],
    system_message: Optional[str],
    streaming: bool,
    model_override: Optional[str],
) -> Union[str, Generator[str, None, None], None]:
    provider = _adapter_provider_name(api_name)
    if not provider:
        return f"Error: Invalid API Name '{api_name}'"
    app_config = ensure_app_config(loaded_config_data)
    adapter = get_registry().get_adapter(provider)
    if adapter is None:
        return f"Error: LLM adapter unavailable for provider '{provider}'"
    model = model_override or resolve_provider_model(provider, app_config)
    if not model:
        return f"Error: Model is required for provider '{provider}'"
    prompt = _build_summary_prompt(text_to_summarize, custom_prompt_arg)
    request: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "system_message": system_message,
        "model": model,
        "api_key": api_key or resolve_provider_api_key_from_config(provider, app_config),
        "temperature": temp,
        "stream": streaming,
        "app_config": app_config,
    }
    timeout = _resolve_adapter_timeout(provider, app_config)
    if streaming:
        def stream_generator() -> Generator[str, None, None]:
            gen = None
            try:
                gen = adapter.stream(request, timeout=timeout)
                for raw in gen:
                    line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                    text_chunk, error_payload, _done = _extract_text_from_upstream_sse(line)
                    if error_payload is not None:
                        yield f"Error: {error_payload}"
                        return
                    if text_chunk:
                        yield text_chunk
            except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS as exc:
                logging.error(f"Error during adapter streaming for {provider}: {exc}", exc_info=True)
                yield f"Error during streaming: {exc}"
            finally:
                try:
                    if gen is not None and hasattr(gen, "close"):
                        gen.close()
                except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS:
                    pass
        return stream_generator()
    try:
        response = adapter.chat(request, timeout=timeout)
        return extract_response_content(response) or str(response)
    except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS as exc:
        logging.error(f"Error during adapter summarization for {provider}: {exc}", exc_info=True)
        return f"Error calling API {api_name}: {exc}"

#######################################################################################################################
# Helper Function Definitions
#

# --- Keep existing helper functions ---
def extract_text_from_segments(segments: list[dict]) -> str:
    # (Keep existing implementation)
    logging.debug(f"Segments received: {segments}")
    logging.debug(f"Type of segments: {type(segments)}")
    text = ""
    if isinstance(segments, list):
        for segment in segments:
            # logging.debug(f"Current segment: {segment}") # Can be verbose
            # logging.debug(f"Type of segment: {type(segment)}")
            if isinstance(segment, dict) and 'Text' in segment:
                text += segment['Text'] + " "
            elif isinstance(segment, dict) and 'text' in segment: # Adding flexibility for key case
                 text += segment['text'] + " "
            else:
                logging.warning(f"Skipping segment due to missing 'Text' key or wrong type: {segment}")
    elif isinstance(segments, str): # Allow passing a pre-joined string
        logging.debug("Segments received as a single string.")
        text = segments
    else:
        logging.warning(f"Unexpected type of 'segments': {type(segments)}. Trying to convert to string.")
        text = str(segments) # Attempt conversion

    return text.strip()


def recursive_summarize_chunks(
    chunks: list[str],
    summarize_func: Callable[[str], str] # Function now only needs to accept the text
) -> str:
    """
    Recursively processes chunks by combining the result of the previous step
    with the next chunk and applying the summarize_func.

    This is suitable for tasks like recursive summarization where context
    from the previous summary is needed for the next chunk.

    Args:
        chunks: A list of text chunks to process.
        summarize_func: A function that takes a single string argument (the text
                        to process) and returns a single string result (the summary
                        or analysis). This function should handle its own configuration
                        (like API keys, prompts, temperature) internally or via closure.
                        It should also handle potential errors and return an error string
                        (e.g., starting with "Error:") if processing fails.

    Returns:
        A single string representing the final result after processing all chunks,
        or an error string if any step failed. Returns an empty string if
        the input chunks list is empty.
    """
    if not chunks:
        logging.warning("recursive_summarize_chunks called with empty chunk list.")
        return ""

    logging.info(f"Starting recursive processing of {len(chunks)} chunks.")
    current_summary = ""

    for i, chunk in enumerate(chunks):
        logging.debug(f"Processing chunk {i+1}/{len(chunks)} recursively.")
        text_to_process: str

        if i == 0:
            # Process the first chunk directly
            text_to_process = chunk
            logging.debug(f"Processing first chunk (length {len(text_to_process)}).")
        else:
            # Combine the previous summary with the current chunk
            # Add a separator for clarity for the LLM
            combined_text = f"{current_summary}\n\n---\n\n{chunk}"
            text_to_process = combined_text
            logging.debug(f"Processing combination of previous summary and chunk {i+1} (total length {len(text_to_process)}).")

        # Apply the processing function
        try:
            step_result = summarize_func(text_to_process)

            # Check if the processing function indicated an error
            if isinstance(step_result, str) and step_result.startswith("Error:"):
                logging.error(f"Error during recursive step {i+1}: {step_result}")
                return step_result # Propagate the error immediately

            if not isinstance(step_result, str):
                 # This shouldn't happen if summarize_func adheres to the contract, but good to check
                 logging.error(f"Recursive step {i+1} did not return a string. Got: {type(step_result)}")
                 return f"Error: Processing step {i+1} returned unexpected type {type(step_result)}"

            current_summary = step_result
            logging.debug(f"Chunk {i+1} processed. Current summary length: {len(current_summary)}")

        except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS as e:
            logging.exception(f"Unexpected error calling summarize_func during recursive step {i+1}: {e}", exc_info=True)
            return f"Error: Unexpected failure during recursive step {i+1}: {e}"

    logging.info("Recursive processing completed successfully.")
    return current_summary


def extract_text_from_input(input_data: Any) -> str:
    """Extracts usable text content from various input types."""
    logging.debug(f"Extracting text from input of type: {type(input_data)}")
    if isinstance(input_data, str):
        # Check if it's a file path
        if os.path.isfile(input_data):
            logging.debug(f"Input is a file path: {input_data}")
            try:
                with open(input_data, encoding='utf-8') as f:
                    content = f.read()
                # Attempt to parse as JSON, otherwise return raw content
                try:
                    data = json.loads(content)
                    logging.debug("File content parsed as JSON.")
                    return extract_text_from_input(data) # Recurse with parsed data
                except json.JSONDecodeError:
                    logging.debug("File content is not JSON, returning raw text.")
                    return content.strip()
            except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS as e:
                logging.error(f"Error reading file {input_data}: {e}")
                return ""
        # Check if it's a JSON string
        elif input_data.strip().startswith('{') or input_data.strip().startswith('['):
             logging.debug("Input is potentially a JSON string.")
             try:
                 data = json.loads(input_data)
                 logging.debug("Input string parsed as JSON.")
                 return extract_text_from_input(data) # Recurse with parsed data
             except json.JSONDecodeError:
                 logging.debug("Input string is not JSON, treating as plain text.")
                 return input_data.strip()
        # Otherwise, treat as plain text
        else:
            logging.debug("Input is a plain text string.")
            return input_data.strip()

    elif isinstance(input_data, dict):
        logging.debug("Input is a dictionary.")
        # Prioritize known structures
        if 'transcription' in input_data:
            logging.debug("Extracting text from 'transcription' field.")
            return extract_text_from_segments(input_data['transcription'])
        elif 'segments' in input_data:
            logging.debug("Extracting text from 'segments' field.")
            return extract_text_from_segments(input_data['segments'])
        elif 'text' in input_data:
             logging.debug("Extracting text from 'text' field.")
             return str(input_data['text']).strip()
        elif 'content' in input_data:
             logging.debug("Extracting text from 'content' field.")
             return str(input_data['content']).strip()
        else:
            # Fallback: try to convert the whole dict to string (might be noisy)
            logging.warning("No specific text field found in dict, converting entire dict to string.")
            try:
                return json.dumps(input_data, indent=2)
            except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS:
                 return str(input_data) # Final fallback

    elif isinstance(input_data, list):
        logging.debug("Input is a list, assuming list of segments.")
        # Assume it's a list of segments like {'Text': '...'} or {'text': '...'}
        return extract_text_from_segments(input_data)

    else:
        logging.warning(f"Unhandled input type: {type(input_data)}. Attempting string conversion.")
        return str(input_data).strip()


# --- Internal API Dispatcher ---
def _dispatch_to_api(
    text_to_summarize: str,
    custom_prompt_arg: Optional[str],
    api_name: str,
    api_key: Optional[str],
    temp: Optional[float],
    system_message: Optional[str],
    streaming: bool,
    model_override: Optional[str] = None,
) -> Union[str, Generator[str, None, None], None]:
    """
    Internal function to call the appropriate API-specific summarization function.
    Handles the mapping from api_name to the actual function call.
    """
    try:
        api_name_lower = api_name.lower()
        logging.debug(f"Dispatching to API: {api_name_lower}")

        adapter_result = _summarize_via_adapter(
            api_name=api_name_lower,
            text_to_summarize=text_to_summarize,
            custom_prompt_arg=custom_prompt_arg,
            api_key=api_key,
            temp=temp,
            system_message=system_message,
            streaming=streaming,
            model_override=model_override,
        )
        if adapter_result is None:
            error_msg = f"Error: LLM adapter unavailable for provider '{api_name}'"
            logging.error(error_msg)
            return error_msg
        return adapter_result

    except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error during dispatch to API '{api_name}': {str(e)}", exc_info=True)
        return f"Error calling API {api_name}: {str(e)}"


# --- Main Summarization Function ---
def analyze(
    api_name: str,
    input_data: Any,
    custom_prompt_arg: Optional[str],
    api_key: Optional[str] = None,
    system_message: Optional[str] = None,
    temp: Optional[float] = None,
    streaming: bool = False,
    recursive_summarization: bool = False,
    chunked_summarization: bool = False, # Summarize chunks separately & combine
    chunk_options: Optional[dict] = None,
    model_override: Optional[str] = None,
) -> Union[str, Generator[str, None, None]]:
    """
    Performs analysis(summarization by default) using a specified API, with optional chunking strategies. Provide a system prompt to avoid summarization.

    Args:
        input_data: Data to analyze(Default is summarization) (text string, file path to JSON, dict, list of dicts).
        custom_prompt_arg: Custom prompt instructions for the LLM.
        api_name: Name of the API service to use (e.g., 'openai', 'anthropic', 'ollama').
        api_key: Optional API key. If None, the specific API function will attempt to load from config.
        temp: Optional temperature setting for the LLM (default varies by API).
        system_message: Optional system message/persona for the LLM. If None, a default is used.
        streaming: If True, attempts to return a generator for streaming output.
                   NOTE: Streaming output is only supported when NO chunking strategy is used.
                   If chunking is enabled, the function will process internally and return a final string.
        recursive_summarization: If True, uses a recursive summarization strategy:
                                 Summarize chunk 1 -> Combine summary 1 + chunk 2 -> Summarize -> ...
        chunked_summarization: If True, summarizes each chunk individually and concatenates the results.
                               Mutually exclusive with recursive_summarization.
        chunk_options: Dictionary of options for the chunking process (passed to improved_chunking_process).
                       Defaults: {'method': 'words', 'max_size': 1000, 'overlap': 100}.

    Returns:
        - A string containing the final summary.
        - A generator yielding summary tokens if streaming=True AND no chunking is used.
        - An error string (starting with "Error:") if summarization fails.
    """
    # Load config here if needed for top-level decisions, otherwise let specific funcs handle it
    # loaded_config_data = load_and_log_configs() # Load once if needed globally
    logging.info(f"Starting summarization process. API: {api_name}, Recursive: {recursive_summarization}, Chunked: {chunked_summarization}, Streaming: {streaming}")

    if recursive_summarization and chunked_summarization:
        error_msg = "Error: Cannot perform both recursive and chunked summarization simultaneously."
        logging.error(error_msg)
        return error_msg

    # Set default system message if not provided
    if system_message is None:
        logging.debug("Using default system message.")
        system_message = _resolve_default_system_prompt()

    try:
        # 1. Extract text content from input_data
        text_content = extract_text_from_input(input_data)
        if not text_content:
            logging.error("Could not extract text content from input data.")
            return "Error: Could not extract text content."
        logging.info(f"Extracted text content length: {len(text_content)} characters.")
        logging.debug(f"Extracted text content (first 500 chars): {text_content[:500]}...")

        # --- Define helper to consume potential generators ---
        def consume_generator(gen):
            if inspect.isgenerator(gen):
                logging.debug("Consuming generator stream...")
                result_list = []
                try:
                    for chunk in gen:
                        if isinstance(chunk, str):
                             result_list.append(chunk)
                        else:
                             logging.warning(f"Generator yielded non-string chunk: {type(chunk)}")
                    final_string = "".join(result_list)
                    logging.debug("Generator consumed.")
                    return final_string
                except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS as e:
                     logging.error(f"Error consuming generator: {e}", exc_info=True)
                     return f"Error consuming stream: {e}"
            return gen # Return as is if not a generator

        # --- Chunking and Summarization Logic ---
        final_result: Union[str, Generator[str, None, None], None] = None
        effective_streaming_for_api_call = False # Default for chunking modes

        # Default chunk options
        default_chunk_opts = {'method': 'sentences', 'max_size': 500, 'overlap': 200}
        current_chunk_options = chunk_options if isinstance(chunk_options, dict) else default_chunk_opts

        if recursive_summarization:
            logging.info("Performing recursive summarization.")
            chunks_data = improved_chunking_process(text_content, current_chunk_options) # Renamed variable for clarity
            if not chunks_data:
                logging.warning("Recursive summarization: Chunking produced no chunks.")
                return "Error: Recursive summarization failed - no chunks generated."

            # Extract just the text from the chunk data
            text_chunks = [chunk['text'] for chunk in chunks_data]
            logging.debug(f"Generated {len(text_chunks)} text chunks for recursive summarization.")

            # Define the summarizer function for recursive_summarize_chunks
            # It must accept ONE argument (the text) and return the summary string.
            # It captures necessary variables (api_name, key, temp, prompts, etc.) from the outer scope (closure).
            # It must handle potential errors from the API call and return an error string if needed.
            def recursive_step_processor(text_to_summarize: str) -> str:
                logging.debug(f"recursive_step_processor called with text length: {len(text_to_summarize)}")
                # Force non-streaming for internal steps and consume immediately
                api_result = _dispatch_to_api(
                    text_to_summarize,
                    custom_prompt_arg,  # Custom prompt is handled by _dispatch_to_api
                    api_name,
                    api_key,
                    temp,
                    system_message,  # System message is handled by _dispatch_to_api
                    streaming=False  # IMPORTANT: Force non-streaming for internal recursive steps
                )
                # consume_generator handles both strings and generators, returning a string
                processed_result = consume_generator(api_result)

                # Ensure the result is a string (consume_generator should do this)
                if not isinstance(processed_result, str):
                    logging.error(f"API dispatch/consumption did not return a string. Got: {type(processed_result)}")
                    # Return an error string that recursive_summarize_chunks can detect
                    return f"Error: Internal summarization step failed to produce string output (got {type(processed_result)})"

                logging.debug(f"recursive_step_processor finished. Result length: {len(processed_result)}")
                # Return the result string (which could be a summary or an error message from consume_generator)
                return processed_result

            # Call the simplified recursive_summarize_chunks utility
            # It now only needs the list of text chunks and the processing function
            final_result = recursive_summarize_chunks(
                chunks=text_chunks,
                summarize_func=recursive_step_processor
            )
            # The result of recursive_summarize_chunks is now the final string summary or an error string

        elif chunked_summarization:
            logging.info("Performing chunked summarization (summarize each, then combine).")
            chunks = improved_chunking_process(text_content, current_chunk_options)
            if not chunks:
                logging.warning("Chunked summarization: Chunking produced no chunks.")
                return "Error: Chunked summarization failed - no chunks generated."
            logging.debug(f"Generated {len(chunks)} chunks for chunked summarization.")

            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                logging.debug(f"Summarizing chunk {i+1}/{len(chunks)}")
                # Summarize each chunk - force non-streaming for API call
                chunk_summary_result = _dispatch_to_api(
                    chunk['text'], custom_prompt_arg, api_name, api_key,
                    temp, system_message, streaming=False, # Force non-streaming
                    model_override=model_override,
                )
                # Consume generator immediately
                processed_chunk_summary = consume_generator(chunk_summary_result)

                if isinstance(processed_chunk_summary, str) and not processed_chunk_summary.startswith("Error:"):
                    chunk_summaries.append(processed_chunk_summary)
                else:
                    error_detail = processed_chunk_summary if isinstance(processed_chunk_summary, str) else "Unknown error"
                    logging.warning(f"Failed to summarize chunk {i+1}: {error_detail}")
                    chunk_summaries.append(f"[Error summarizing chunk {i+1}: {error_detail}]") # Add error placeholder

            # Combine the summaries
            final_result = "\n\n---\n\n".join(chunk_summaries) # Join with a separator

        else:
            # No chunking - direct summarization
            logging.info("Performing direct summarization (no chunking).")
            # Use the user's requested streaming setting for the API call
            effective_streaming_for_api_call = streaming
            final_result = _dispatch_to_api(
                 text_content, custom_prompt_arg, api_name, api_key,
                 temp, system_message, streaming=effective_streaming_for_api_call,
                 model_override=model_override,
            )

        # --- Post-processing and Return ---

        # If streaming was requested AND no chunking was done AND result is a generator, return it directly
        if streaming and not recursive_summarization and not chunked_summarization and inspect.isgenerator(final_result):
            logging.info("Returning generator for streaming output.")
            return final_result
        else:
            # Otherwise, consume any potential generator to get the final string
            logging.debug("Consuming final result (if generator) as streaming=False or chunking was used.")
            final_string_summary = consume_generator(final_result)

            # Final check and return
            if final_string_summary is None:
                logging.error("Summarization resulted in None after processing.")
                return "Error: Summarization failed unexpectedly."
            elif isinstance(final_string_summary, str) and final_string_summary.startswith("Error:"):
                logging.error(f"Summarization failed: {final_string_summary}")
                return final_string_summary
            elif isinstance(final_string_summary, str):
                logging.info(f"Summarization completed successfully. Final Length: {len(final_string_summary)}")
                logging.debug(f"Final Summary (first 500 chars): {final_string_summary[:500]}...")
                return final_string_summary
            else:
                # This case should ideally not be reached if consume_generator works correctly
                logging.error(f"Unexpected final result type after processing: {type(final_string_summary)}")
                return f"Error: Unexpected result type {type(final_string_summary)}"

    except _SUMMARIZATION_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Critical error in summarize function: {str(e)}", exc_info=True)
        return f"Error: An unexpected error occurred during summarization: {str(e)}"

#
# End of Analysis Function
###################################################################################



def extract_metadata_and_content(input_data):
    metadata = {}
    content = ""

    if isinstance(input_data, str):
        if os.path.exists(input_data):
            with open(input_data, encoding='utf-8') as file:
                data = json.load(file)
        else:
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                return {}, input_data
    elif isinstance(input_data, dict):
        data = input_data
    else:
        return {}, str(input_data)

    # Extract metadata
    metadata['title'] = data.get('title', 'No title available')
    metadata['author'] = data.get('author', 'Unknown author')

    # Extract content
    if 'transcription' in data:
        content = extract_text_from_segments(data['transcription'])
    elif 'segments' in data:
        content = extract_text_from_segments(data['segments'])
    elif 'content' in data:
        content = data['content']
    else:
        content = json.dumps(data)

    return metadata, content


def format_input_with_metadata(metadata, content):
    formatted_input = f"Title: {metadata.get('title', 'No title available')}\n"
    formatted_input += f"Author: {metadata.get('author', 'Unknown author')}\n\n"
    formatted_input += content
    return formatted_input



def extract_text_from_input(input_data):  # noqa: F811
    if isinstance(input_data, str):
        try:
            # Try to parse as JSON
            data = json.loads(input_data)
        except json.JSONDecodeError:
            # If not valid JSON, treat as plain text
            return input_data
    elif isinstance(input_data, dict):
        data = input_data
    else:
        return str(input_data)

    # Extract relevant fields from the JSON object
    text_parts = []
    if 'title' in data:
        text_parts.append(f"Title: {data['title']}")
    if 'description' in data:
        text_parts.append(f"Description: {data['description']}")
    if 'transcription' in data:
        if isinstance(data['transcription'], list):
            transcription_text = ' '.join([segment.get('Text', '') for segment in data['transcription']])
        elif isinstance(data['transcription'], str):
            transcription_text = data['transcription']
        else:
            transcription_text = str(data['transcription'])
        text_parts.append(f"Transcription: {transcription_text}")
    elif 'segments' in data:
        segments_text = extract_text_from_segments(data['segments'])
        text_parts.append(f"Segments: {segments_text}")

    return '\n\n'.join(text_parts)


#
#
############################################################################################################################################
