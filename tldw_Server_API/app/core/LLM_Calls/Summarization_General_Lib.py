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
# 2. summarize_with_openai(api_key, file_path, custom_prompt_arg)
# 3. summarize_with_anthropic(api_key, file_path, model, custom_prompt_arg, max_retries=3, retry_delay=5)
# 4. summarize_with_cohere(api_key, file_path, model, custom_prompt_arg)
# 5. summarize_with_groq(api_key, file_path, model, custom_prompt_arg)
#
#
####################
# Import necessary libraries
import inspect
import json
import os
import time
from typing import Optional, Union, Generator, Any, Dict, List, Callable
#
# 3rd-Party Imports
import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
#
# Import Local
from tldw_Server_API.app.core.Utils.Chunk_Lib import (
    improved_chunking_process
)
from tldw_Server_API.app.core.LLM_Calls.Local_Summarization_Lib import (
    summarize_with_llama,
    summarize_with_kobold,
    summarize_with_oobabooga,
    summarize_with_tabbyapi,
    summarize_with_vllm,
    summarize_with_local_llm,
    summarize_with_ollama,
    summarize_with_custom_openai,
    summarize_with_custom_openai_2
)
from tldw_Server_API.app.core.Utils.Utils import (
    load_and_log_configs,
    logging
)
#
#######################################################################################################################
# Function Definitions
#

loaded_config_data = load_and_log_configs()
openai_api_key = loaded_config_data.get('openai_api', {}).get('api_key', None)

#######################################################################################################################
# Helper Function Definitions
#

# --- Keep existing helper functions ---
def extract_text_from_segments(segments: List[Dict]) -> str:
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
    chunks: List[str],
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

        except Exception as e:
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
                with open(input_data, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Attempt to parse as JSON, otherwise return raw content
                try:
                    data = json.loads(content)
                    logging.debug("File content parsed as JSON.")
                    return extract_text_from_input(data) # Recurse with parsed data
                except json.JSONDecodeError:
                    logging.debug("File content is not JSON, returning raw text.")
                    return content.strip()
            except Exception as e:
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
            except Exception:
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
    streaming: bool
) -> Union[str, Generator[str, None, None], None]:
    """
    Internal function to call the appropriate API-specific summarization function.
    Handles the mapping from api_name to the actual function call.
    """
    try:
        api_name_lower = api_name.lower()
        logging.debug(f"Dispatching to API: {api_name_lower}")

        # Ensure required args for specific functions are handled if needed
        # (e.g., model might be loaded from config inside specific funcs)

        # NOTE: The specific functions (summarize_with_openai, etc.) should
        # handle their own internal logic for loading API keys from config if
        # the provided api_key is None. They also handle loading models, etc.
        # We just pass the parameters along.

        if api_name_lower == "openai":
            return summarize_with_openai(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "anthropic":
            # Anthropic might need model passed explicitly or loaded in its func
            return summarize_with_anthropic(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "cohere":
            return summarize_with_cohere(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "google":
            return summarize_with_google(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "groq":
            return summarize_with_groq(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "huggingface":
             # HuggingFace might need specific handling for system_message if not directly supported
            return summarize_with_huggingface(api_key, text_to_summarize, custom_prompt_arg, temp, streaming) # system_message not directly passed? Check func def.
        elif api_name_lower == "openrouter":
            return summarize_with_openrouter(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "deepseek":
            return summarize_with_deepseek(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "mistral":
            return summarize_with_mistral(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        # --- Local LLM Calls ---
        elif api_name_lower == "llama.cpp":
            return summarize_with_llama(text_to_summarize, custom_prompt_arg, api_key, temp, system_message, streaming)
        elif api_name_lower == "kobold":
            return summarize_with_kobold(text_to_summarize, api_key, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "ooba":
            # Ooba might need api_url from config inside its function
            return summarize_with_oobabooga(text_to_summarize, api_key, custom_prompt_arg, system_message, temp=temp, streaming=streaming)
        elif api_name_lower == "tabbyapi":
            return summarize_with_tabbyapi(text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "vllm":
            return summarize_with_vllm(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "local-llm":
            return summarize_with_local_llm(text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "custom-openai-api":
            # Custom OpenAI likely needs base_url from config inside its function
            return summarize_with_custom_openai(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming)
        elif api_name_lower == "custom-openai-api-2":
             # Custom OpenAI likely needs base_url from config inside its function
            return summarize_with_custom_openai_2(api_key, text_to_summarize, custom_prompt_arg, temp, system_message, streaming) # Assuming this exists
        elif api_name_lower == "ollama":
            # Ollama might need model param or load from config
            return summarize_with_ollama(text_to_summarize, custom_prompt_arg, None, api_key, temp, system_message, streaming) # Passing None for model, assuming func handles it
        # --- MOCKING TEST LLM Calls ---
        elif api_name_lower == "mock-llm":
            return summarize_with_mock_llm(text_to_summarize, custom_prompt_arg, api_key, temp, system_message, streaming)
        else:
            error_msg = f"Error: Invalid API Name '{api_name}'"
            logging.error(error_msg)
            return error_msg

    except Exception as e:
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
    chunk_options: Optional[dict] = None
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
        system_message = (
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
                except Exception as e:
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
                    temp, system_message, streaming=False # Force non-streaming
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
                 temp, system_message, streaming=effective_streaming_for_api_call
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

    except Exception as e:
        logging.error(f"Critical error in summarize function: {str(e)}", exc_info=True)
        return f"Error: An unexpected error occurred during summarization: {str(e)}"

#
# End of Analysis Function
###################################################################################


###################################################################################
#
# API Calls

def summarize_with_openai(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    try:
        # API key validation
        if not api_key or api_key.strip() == "":
            logging.info("OpenAI Summarize: API key not provided as parameter")
            logging.info("OpenAI Summarize: Attempting to use API key from config file")
            loaded_config_data = load_and_log_configs()
            api_key = loaded_config_data.get('openai_api', {}).get('api_key', "")
            logging.debug(f"OpenAI Summarize: Using API key from config file: {api_key[:5]}...{api_key[-5:]}")

        if not api_key or api_key.strip() == "":
            logging.error("OpenAI: #2 API key not found or is empty")
            return "OpenAI: API Key Not Provided/Found in Config file or is empty"

        # API key handling: prioritize parameter, then config
        effective_api_key = api_key
        if not effective_api_key or not effective_api_key.strip():
            logging.info("OpenAI Summarize: API key not provided or empty, using config.")
            effective_api_key = loaded_config_data.get('openai_api', {}).get('api_key', "")

        if not effective_api_key or not effective_api_key.strip():
            logging.error("OpenAI: API key not found or is empty (checked param and config).")
            return "Error: OpenAI API Key Not Provided/Found or is empty."

        # Model handling: load from config
        openai_model = loaded_config_data.get('openai_api', {}).get('model') or "gpt-4o" # Default model
        logging.debug(f"OpenAI: Using model: {openai_model}")

        # NOTE: input_data passed to this function should *already be the extracted text*
        # by the time the main `summarize` function calls `_dispatch_to_api`.
        # So, the complex input parsing logic previously here is removed.
        text = str(input_data) # Ensure it's a string

        logging.debug(f"OpenAI: Received text length: {len(text)}")
        logging.debug(f"OpenAI: Custom prompt: {custom_prompt_arg}")
        logging.debug(f"OpenAI: Temperature: {temp}, System Message: {system_message}, Streaming: {streaming}")

        headers = {
            'Authorization': f'Bearer {effective_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"OpenAI API Key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else None}")
        logging.debug("openai: Preparing data + prompt for submittal")
        openai_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        if temp is None: temp = 0.7
        if system_message is None: system_message = "You are a helpful AI assistant."
        try:
            temp = float(temp)
        except (ValueError, TypeError):
            logging.warning(f"Invalid temperature value '{temp}', using default 0.7")
            temp = 0.7


        payload = {
            "model": openai_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": openai_prompt}
            ],
            # FIXME - Set a Max tokens value in config file for each API
            "max_tokens": 4096,
            "temperature": temp,
            "stream": streaming
        }

        # --- Retry Logic --- (Copied from original, seems reasonable)
        session = requests.Session()
        retry_count = loaded_config_data.get('openai_api', {}).get('api_retries', 3)
        retry_delay = loaded_config_data.get('openai_api', {}).get('api_retry_delay', 1) # Using 1s default backoff factor
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504], # Added 500
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter) # Mount for http too if needed

        api_url = loaded_config_data.get('openai_api', {}).get('api_base_url', 'https://api.openai.com/v1') + '/chat/completions'


        logging.debug(f"OpenAI: Posting request to {api_url}")
        response = session.post(
            api_url,
            headers=headers,
            json=payload,
            stream=streaming,
            timeout=loaded_config_data.get('openai_api', {}).get('api_timeout', 120) # Add timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        if streaming:
            logging.debug("OpenAI: Processing streaming response.")
            def stream_generator():
                try:
                    for line in response.iter_lines():
                        line = line.decode("utf-8").strip()
                        if not line: continue
                        if line.startswith("data: "):
                            data_str = line[len("data: "):]
                            if data_str == "[DONE]": break
                            try:
                                data_json = json.loads(data_str)
                                chunk = data_json["choices"][0]["delta"].get("content", "")
                                yield chunk
                            except json.JSONDecodeError:
                                logging.error(f"OpenAI Stream: Error decoding JSON: {data_str}")
                                continue
                            except (KeyError, IndexError) as e:
                                logging.error(f"OpenAI Stream: Unexpected structure: {data_str} - Error: {e}")
                                continue
                except Exception as stream_error:
                     logging.error(f"OpenAI Stream: Error during streaming: {stream_error}", exc_info=True)
                     yield f"Error during streaming: {stream_error}" # Yield error in stream
                finally:
                     response.close() # Ensure connection is closed
            return stream_generator()
        else:
            logging.debug("OpenAI: Processing non-streaming response.")
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0 and 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                summary = response_data['choices'][0]['message']['content'].strip()
                logging.debug("OpenAI: Summarization successful (non-streaming).")
                return summary
            else:
                logging.warning(f"OpenAI: Summary not found in response: {response_data}")
                return "Error: OpenAI Summary not found in response."

    except requests.exceptions.RequestException as e:
        logging.error(f"OpenAI: API request failed: {str(e)}", exc_info=True)
        return f"Error: OpenAI API request failed: {str(e)}"
    except Exception as e:
        logging.error(f"OpenAI: Unexpected error: {str(e)}", exc_info=True)
        return f"Error: OpenAI unexpected error: {str(e)}"


def summarize_with_anthropic(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False, max_retries=3, retry_delay=5):
    logging.debug("Anthropic: Summarization process starting...")
    try:
        logging.debug("Anthropic: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            anthropic_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                anthropic_api_key = api_key
                logging.info("Anthropic: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                anthropic_api_key = loaded_config_data['anthropic_api'].get('api_key')
                if anthropic_api_key:
                    logging.info("Anthropic: Using API key from config file")
                else:
                    logging.warning("Anthropic: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not anthropic_api_key or not anthropic_api_key.strip():
            logging.error("Anthropic: No valid API key available")
            return "Anthropic: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Anthropic: Using API Key: {anthropic_api_key[:5]}...{anthropic_api_key[-5:]}")

        logging.debug("AnthropicAI: Using provided string data for summarization")
        data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"AnthropicAI: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"AnthropicAI: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Anthropic: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Anthropic: Invalid input data format")

        if temp is None:
            temp = 0.1
        temp = float(temp)

        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'x-api-key': anthropic_api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        }

        anthropic_prompt = custom_prompt_arg
        logging.debug(f"Anthropic: Prompt is {anthropic_prompt}")
        user_message = {
            "role": "user",
            "content": f"{text} \n\n\n\n{anthropic_prompt}"
        }

        model = loaded_config_data['anthropic_api']['model']

        data = {
            "model": model,
            "max_tokens": 4096,  # max possible tokens to return
            "messages": [user_message],
            "stop_sequences": ["\n\nHuman:"],
            "temperature": temp,
            "top_k": 0,
            "top_p": 1.0,
            "metadata": {
                "user_id": "example_user_id",
            },
            "stream": streaming,
            "system": system_message
        }

        for attempt in range(max_retries):
            try:
                # Create a session
                session = requests.Session()

                # Load config values
                retry_count = loaded_config_data['anthropic_api']['api_retries']
                retry_delay = loaded_config_data['anthropic_api']['api_retry_delay']

                # Configure the retry strategy
                retry_strategy = Retry(
                    total=retry_count,  # Total number of retries
                    backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                    status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
                )

                # Create the adapter
                adapter = HTTPAdapter(max_retries=retry_strategy)

                # Mount adapters for both HTTP and HTTPS
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                logging.debug("Anthropic: Posting request to API")
                response = requests.post(
                    'https://api.anthropic.com/v1/messages',
                    headers=headers,
                    json=data,
                    stream=streaming
                )

                # Check if the status code indicates success
                if response.status_code == 200:
                    if streaming:
                        # Handle streaming response
                        def stream_generator():
                            collected_text = ""
                            event_type = None
                            for line in response.iter_lines():
                                line = line.decode('utf-8').strip()
                                if line == '':
                                    continue
                                if line.startswith('event:'):
                                    event_type = line[len('event:'):].strip()
                                elif line.startswith('data:'):
                                    data_str = line[len('data:'):].strip()
                                    if data_str == '[DONE]':
                                        break
                                    try:
                                        data_json = json.loads(data_str)
                                        if event_type == 'content_block_delta' and data_json.get('type') == 'content_block_delta':
                                            delta = data_json.get('delta', {})
                                            text_delta = delta.get('text', '')
                                            collected_text += text_delta
                                            yield text_delta
                                    except json.JSONDecodeError:
                                        logging.error(f"Anthropic: Error decoding JSON from line: {line}")
                                        continue
                            # Optionally, return the full collected text at the end
                            # yield collected_text
                        return stream_generator()
                    else:
                        # Non-streaming response
                        logging.debug("Anthropic: Post submittal successful")
                        response_data = response.json()
                        try:
                            # Extract the assistant's reply from the 'content' field
                            content_blocks = response_data.get('content', [])
                            summary = ''
                            for block in content_blocks:
                                if block.get('type') == 'text':
                                    summary += block.get('text', '')
                            summary = summary.strip()
                            logging.debug("Anthropic: Summarization successful")
                            logging.debug(f"Anthropic: Summary (first 500 chars): {summary[:500]}...")
                            return summary
                        except Exception as e:
                            logging.debug("Anthropic: Unexpected data in response")
                            logging.error(f"Unexpected response format from Anthropic API: {response.text}")
                            return None
                elif response.status_code == 500:  # Handle internal server error specifically
                    logging.debug("Anthropic: Internal server error")
                    logging.error("Internal server error from API. Retrying may be necessary.")
                    time.sleep(retry_delay)
                else:
                    logging.debug(f"Anthropic: Failed to summarize, status code {response.status_code}: {response.text}")
                    logging.error(f"Failed to process summary, status code {response.status_code}: {response.text}")
                    return None

            except requests.RequestException as e:
                logging.error(f"Anthropic: Network error during attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return f"Anthropic: Network error: {str(e)}"
    except FileNotFoundError as e:
        logging.error(f"Anthropic: File not found: {input_data}")
        return f"Anthropic: File not found: {input_data}"
    except json.JSONDecodeError as e:
        logging.error(f"Anthropic: Invalid JSON format in file: {input_data}")
        return f"Anthropic: Invalid JSON format in file: {input_data}"
    except Exception as e:
        logging.error(f"Anthropic: Error in processing: {str(e)}")
        return f"Anthropic: Error occurred while processing summary with Anthropic: {str(e)}"


# Summarize with Cohere
def summarize_with_cohere(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    logging.debug("Cohere: Summarization process starting...")
    try:
        logging.debug("Cohere: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            cohere_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                cohere_api_key = api_key
                logging.info("Cohere: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                cohere_api_key = loaded_config_data['cohere_api'].get('api_key')
                if cohere_api_key:
                    logging.info("Cohere: Using API key from config file")
                else:
                    logging.warning("Cohere: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not cohere_api_key or not cohere_api_key.strip():
            logging.error("Cohere: No valid API key available")
            return "Cohere: API Key Not Provided/Found in Config file or is empty"

        if custom_prompt_arg is None:
            custom_prompt_arg = ""

        if system_message is None:
            system_message = ""

        logging.debug(f"Cohere: Using API Key: {cohere_api_key[:5]}...{cohere_api_key[-5:] if cohere_api_key else None}")

        logging.debug("Cohere: Using provided string data for summarization")
        data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"Cohere: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"Cohere: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("Cohere: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Cohere: Invalid input data format")

        cohere_model = loaded_config_data['cohere_api']['model']

        if temp is None:
            temp = 0.3
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'accept': 'application/json',
            'content-type': 'application/json',
            'Authorization': f'Bearer {cohere_api_key}'
        }

        cohere_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug(f"Cohere: Prompt being sent is {cohere_prompt}")

        data = {
            "preamble": system_message,
            "message": cohere_prompt,
            "model": cohere_model,
#            "connectors": [{"id": "web-search"}],
            "temperature": temp,
            "streaming": streaming
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['cohere_api']['api_retries']
            retry_delay = loaded_config_data['cohere_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Cohere: Submitting streaming request to API endpoint")
            response = session.post(
                'https://api.cohere.ai/v1/chat',
                headers=headers,
                json=data,
                stream=True  # Enable response streaming
            )
            response.raise_for_status()

            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        try:
                            data_json = json.loads(decoded_line)
                            if 'response' in data_json:
                                chunk = data_json['response']
                                yield chunk
                            elif 'token' in data_json:
                                # For token-based streaming (if applicable)
                                chunk = data_json['token']
                                yield chunk
                            elif 'text' in data_json:
                                # For text-based streaming
                                chunk = data_json['text']
                                yield chunk
                            else:
                                logging.debug(f"Cohere: Unhandled streaming data: {data_json}")
                        except json.JSONDecodeError:
                            logging.error(f"Cohere: Error decoding JSON from line: {decoded_line}")
                            continue

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['cohere_api']['api_retries']
            retry_delay = loaded_config_data['cohere_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Cohere: Submitting request to API endpoint")
            response = session.post('https://api.cohere.ai/v1/chat', headers=headers, json=data)
            response_data = response.json()
            logging.debug(f"API Response Data: {response_data}")

            if response.status_code == 200:
                if 'text' in response_data:
                    summary = response_data['text'].strip()
                    logging.debug("Cohere: Summarization successful")
                    return summary
                elif 'response' in response_data:
                    # Adjust if the API returns 'response' field instead of 'text'
                    summary = response_data['response'].strip()
                    logging.debug("Cohere: Summarization successful")
                    return summary
                else:
                    logging.error("Cohere: Expected data not found in API response.")
                    return "Cohere: Expected data not found in API response."
            else:
                logging.error(f"Cohere: API request failed with status code {response.status_code}: {response.text}")
                return f"Cohere: API request failed: {response.text}"

    except Exception as e:
        logging.error(f"Cohere: Error in processing: {str(e)}", exc_info=True)
        return f"Cohere: Error occurred while processing summary with Cohere: {str(e)}"



# https://console.groq.com/docs/quickstart
def summarize_with_groq(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    logging.debug("Groq: Summarization process starting...")
    try:
        logging.debug("Groq: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            groq_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                groq_api_key = api_key
                logging.info("Groq: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                groq_api_key = loaded_config_data['groq_api'].get('api_key')
                if groq_api_key:
                    logging.info("Groq: Using API key from config file")
                else:
                    logging.warning("Groq: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not groq_api_key or not groq_api_key.strip():
            logging.error("Groq: No valid API key available")
            return "Groq: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Groq: Using API Key: {groq_api_key[:5]}...{groq_api_key[-5:]}")

        # Input data handling
        logging.debug("Groq: Using provided string data for summarization")
        data = input_data

        # Debug logging to identify sent data
        logging.debug(f"Groq: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"Groq: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            logging.debug("Groq: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Groq: Invalid input data format")

        # Set the model to be used
        groq_model = loaded_config_data['groq_api']['model']

        if temp is None:
            temp = 0.2
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {groq_api_key}',
            'Content-Type': 'application/json'
        }

        groq_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        logging.debug(f"Groq: Prompt being sent is {groq_prompt}")

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": groq_prompt,
                }
            ],
            "model": groq_model,
            "temperature": temp,
            "stream": streaming
        }

        logging.debug("Groq: Submitting request to API endpoint")
        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['groq_api']['api_retries']
            retry_delay = loaded_config_data['groq_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=data,
                stream=True  # Enable response streaming
            )
            response.raise_for_status()

            def stream_generator():
                collected_messages = ""
                for line in response.iter_lines():
                    line = line.decode("utf-8").strip()

                    if line == "":
                        continue

                    if line.startswith("data: "):
                        data_str = line[len("data: "):]
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            chunk = data_json["choices"][0]["delta"].get("content", "")
                            collected_messages += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            logging.error(f"Groq: Error decoding JSON from line: {line}")
                            continue
                # Optionally, you can return the full collected message at the end
                # yield collected_messages

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['groq_api']['api_retries']
            retry_delay = loaded_config_data['groq_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=data
            )

            response_data = response.json()
            logging.debug("API Response Data: %s", response_data)

            if response.status_code == 200:
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Groq: Summarization successful")
                    return summary
                else:
                    logging.error("Groq: Expected data not found in API response.")
                    return "Groq: Expected data not found in API response."
            else:
                logging.error(f"Groq: API request failed with status code {response.status_code}: {response.text}")
                return f"Groq: API request failed: {response.text}"

    except Exception as e:
        logging.error("Groq: Error in processing: %s", str(e), exc_info=True)
        return f"Groq: Error occurred while processing summary with Groq: {str(e)}"


def summarize_with_openrouter(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False,):
    import requests
    import json
    global openrouter_model, openrouter_api_key
    try:
        logging.debug("OpenRouter: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            openrouter_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                openrouter_api_key = api_key
                logging.info("OpenRouter: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                openrouter_api_key = loaded_config_data['openrouter_api'].get('api_key')
                if openrouter_api_key:
                    logging.info("OpenRouter: Using API key from config file")
                else:
                    logging.warning("OpenRouter: No API key found in config file")

        # Model Selection validation
        logging.debug("OpenRouter: Validating model selection")
        loaded_config_data = load_and_log_configs()
        openrouter_model = loaded_config_data['openrouter_api']['model']
        logging.debug(f"OpenRouter: Using model from config file: {openrouter_model}")

        # Final check to ensure we have a valid API key
        if not openrouter_api_key or not openrouter_api_key.strip():
            logging.error("OpenRouter: No valid API key available")
            raise ValueError("No valid Anthropic API key available")
    except Exception as e:
        logging.error("OpenRouter: Error in processing: %s", str(e))
        return f"OpenRouter: Error occurred while processing config file with OpenRouter: {str(e)}"

    logging.debug(f"OpenRouter: Using API Key: {openrouter_api_key[:5]}...{openrouter_api_key[-5:]}")

    logging.debug(f"OpenRouter: Using Model: {openrouter_model}")

    logging.debug("OpenRouter: Using provided string data for summarization")
    data = input_data

    # DEBUG - Debug logging to identify sent data
    logging.debug(f"OpenRouter: Loaded data: {data[:500]}...(snipped to first 500 chars)")
    logging.debug(f"OpenRouter: Type of data: {type(data)}")

    if isinstance(data, dict) and 'summary' in data:
        # If the loaded data is a dictionary and already contains a summary, return it
        logging.debug("OpenRouter: Summary already exists in the loaded data")
        return data['summary']

    # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
    if isinstance(data, list):
        segments = data
        text = extract_text_from_segments(segments)
    elif isinstance(data, str):
        text = data
    else:
        raise ValueError("OpenRouter: Invalid input data format")

    openrouter_prompt = f"{input_data} \n\n\n\n{custom_prompt_arg}"

    if temp is None:
        temp = 0.1
    temp = float(temp)
    if system_message is None:
        system_message = "You are a helpful AI assistant who does whatever the user requests."

    if streaming:
        try:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['openrouter_api']['api_retries']
            retry_delay = loaded_config_data['openrouter_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("OpenRouter: Submitting streaming request to API endpoint")
            # Make streaming request
            response = session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Accept": "text/event-stream",  # Important for streaming
                },
                data=json.dumps({
                    "model": openrouter_model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": openrouter_prompt}
                    ],
                    #"max_tokens": 4096,
                    #"top_p": 1.0,
                    "temperature": temp,
                    "stream": True
                }),
                stream=True  # Enable streaming in requests
            )

            if response.status_code == 200:
                full_response = ""
                # Process the streaming response
                for line in response.iter_lines():
                    if line:
                        # Remove "data: " prefix and parse JSON
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_str = line[6:]  # Remove "data: " prefix
                            if json_str.strip() == '[DONE]':
                                break
                            try:
                                json_data = json.loads(json_str)
                                if 'choices' in json_data and len(json_data['choices']) > 0:
                                    delta = json_data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content = delta['content']
                                        print(content, end='', flush=True)  # Print streaming output
                                        full_response += content
                            except json.JSONDecodeError:
                                continue

                logging.debug("openrouter: Streaming completed successfully")
                return full_response.strip()
            else:
                error_msg = f"openrouter: Streaming API request failed with status code {response.status_code}: {response.text}"
                logging.error(error_msg)
                return error_msg

        except Exception as e:
            error_msg = f"openrouter: Error occurred while processing stream: {str(e)}"
            logging.error(error_msg)
            return error_msg
    else:
        try:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['openrouter_api']['api_retries']
            retry_delay = loaded_config_data['openrouter_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("OpenRouter: Submitting request to API endpoint")
            print("OpenRouter: Submitting request to API endpoint")
            response = session.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                },
                data=json.dumps({
                    "model": openrouter_model,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": openrouter_prompt}
                    ],
                    #"max_tokens": 4096,
                    #"top_p": 1.0,
                    "temperature": temp,
                    #"stream": streaming
                })
            )

            response_data = response.json()
            logging.debug("API Response Data: %s", response_data)

            if response.status_code == 200:
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("openrouter: Summarization successful")
                    print("openrouter: Summarization successful.")
                    return summary
                else:
                    logging.error("openrouter: Expected data not found in API response.")
                    return "openrouter: Expected data not found in API response."
            else:
                logging.error(f"openrouter:  API request failed with status code {response.status_code}: {response.text}")
                return f"openrouter: API request failed: {response.text}"
        except Exception as e:
            logging.error("openrouter: Error in processing: %s", str(e))
            return f"openrouter: Error occurred while processing summary with openrouter: {str(e)}"


def summarize_with_huggingface(api_key, input_data, custom_prompt_arg, temp=None, streaming=False,):
    # https://huggingface.co/docs/api-inference/tasks/chat-completion
    loaded_config_data = load_and_log_configs()
    logging.debug("HuggingFace: Summarization process starting...")
    try:
        logging.debug("HuggingFace: Loading and validating configurations")
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            huggingface_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                huggingface_api_key = api_key
                logging.info("HuggingFace: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                huggingface_api_key = loaded_config_data['huggingface_api'].get('api_key')
                logging.debug(f"HuggingFace: API key from config: {huggingface_api_key[:5]}...{huggingface_api_key[-5:]}")
                if huggingface_api_key:
                    logging.info("HuggingFace: Using API key from config file")
                else:
                    logging.warning("HuggingFace: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not huggingface_api_key or not huggingface_api_key.strip():
            logging.error("HuggingFace: No valid API key available")
            # You might want to raise an exception here or handle this case as appropriate for your application
            # FIXME
            # For example: raise ValueError("No valid Anthropic API key available")

        logging.debug(f"HuggingFace: Using API Key: {huggingface_api_key[:5]}...{huggingface_api_key[-5:]}")

        logging.debug("HuggingFace: Using provided string data for summarization")
        data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"HuggingFace: Loaded data: {data[:500]}...(snipped to first 500 chars)")
        logging.debug(f"HuggingFace: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            # If the loaded data is a dictionary and already contains a summary, return it
            logging.debug("HuggingFace: Summary already exists in the loaded data")
            return data['summary']

        # If the loaded data is a list of segment dictionaries or a string, proceed with summarization
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("HuggingFace: Invalid input data format")

        headers = {
            "Authorization": f"Bearer {huggingface_api_key}"
        }
        huggingface_model = loaded_config_data['huggingface_api']['model']
        API_URL = f"https://api-inference.huggingface.co/models/{huggingface_model}"
        if temp is None:
            temp = 0.1
        temp = float(temp)
        huggingface_prompt = f"{custom_prompt_arg}\n\n\n{text}"
        logging.debug(f"HuggingFace: Prompt being sent is {huggingface_prompt}")
        data_payload = {
            "inputs": huggingface_prompt,
            "max_tokens": 4096,
            "stream": streaming,
            "temperature": temp
        }

        logging.debug("HuggingFace: Submitting request...")
        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['huggingface_api']['api_retries']
            retry_delay = loaded_config_data['huggingface_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(API_URL, headers=headers, json=data_payload, stream=True)
            response.raise_for_status()

            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data:'):
                            data_str = decoded_line[len('data:'):].strip()
                            if data_str == '[DONE]':
                                break
                            try:
                                data_json = json.loads(data_str)
                                if 'token' in data_json:
                                    token_text = data_json['token'].get('text', '')
                                    yield token_text
                                elif 'generated_text' in data_json:
                                    # Some models may send the full generated text
                                    generated_text = data_json['generated_text']
                                    yield generated_text
                                else:
                                    logging.debug(f"HuggingFace: Unhandled streaming data: {data_json}")
                            except json.JSONDecodeError:
                                logging.error(f"HuggingFace: Error decoding JSON from line: {decoded_line}")
                                continue
                # Optionally, yield the final collected text
                # yield collected_text

            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['huggingface_api']['api_retries']
            retry_delay = loaded_config_data['huggingface_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            response = session.post(API_URL, headers=headers, json=data_payload)

            if response.status_code == 200:
                response_json = response.json()
                logging.debug(f"HuggingFace: Response JSON: {response_json}")
                if isinstance(response_json, dict) and 'generated_text' in response_json:
                    chat_response = response_json['generated_text'].strip()
                elif isinstance(response_json, list) and len(response_json) > 0 and 'generated_text' in response_json[0]:
                    chat_response = response_json[0]['generated_text'].strip()
                else:
                    logging.error("HuggingFace: Expected 'generated_text' in response")
                    return "HuggingFace: Expected 'generated_text' in API response."

                logging.debug("HuggingFace: Summarization successful")
                return chat_response
            else:
                logging.error(f"HuggingFace: Summarization failed with status code {response.status_code}: {response.text}")
                return f"HuggingFace: Failed to process summary. Status code: {response.status_code}"

    except Exception as e:
        logging.error("HuggingFace: Error in processing: %s", str(e), exc_info=True)
        return f"HuggingFace: Error occurred while processing summary with HuggingFace: {str(e)}"


def summarize_with_deepseek(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    # https://api-docs.deepseek.com/api/create-chat-completion
    logging.debug("DeepSeek: Summarization process starting...")
    try:
        logging.debug("DeepSeek: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            deepseek_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                deepseek_api_key = api_key
                logging.info("DeepSeek: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                deepseek_api_key = loaded_config_data['deepseek_api'].get('api_key')
                if deepseek_api_key:
                    logging.info("DeepSeek: Using API key from config file")
                else:
                    logging.warning("DeepSeek: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not deepseek_api_key or not deepseek_api_key.strip():
            logging.error("DeepSeek: No valid API key available")
            return "DeepSeek: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"DeepSeek: Using API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:]}")

        # Input data handling
        logging.debug("DeepSeek: Using provided string data for summarization")
        data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"DeepSeek: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"DeepSeek: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            logging.debug("DeepSeek: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("DeepSeek: Invalid input data format")

        deepseek_model = loaded_config_data['deepseek_api']['model'] or "deepseek-chat"

        if temp is None:
            temp = 0.1
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {deepseek_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"DeepSeek API Key: {deepseek_api_key[:5]}...{deepseek_api_key[-5:] if deepseek_api_key else None}")
        logging.debug("DeepSeek: Preparing data + prompt for submission")
        deepseek_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        data = {
            "model": deepseek_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": deepseek_prompt}
            ],
            "stream": streaming,
            "temperature": temp
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['deepseek_api']['api_retries']
            retry_delay = loaded_config_data['deepseek_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("DeepSeek: Posting streaming request")
            response = session.post(
                'https://api.deepseek.com/chat/completions',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            def stream_generator():
                collected_text = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line == '':
                            continue
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[len('data: '):]
                            if data_str == '[DONE]':
                                break
                            try:
                                data_json = json.loads(data_str)
                                delta_content = data_json['choices'][0]['delta'].get('content', '')
                                collected_text += delta_content
                                yield delta_content
                            except json.JSONDecodeError:
                                logging.error(f"DeepSeek: Error decoding JSON from line: {decoded_line}")
                                continue
                            except KeyError as e:
                                logging.error(f"DeepSeek: Key error: {str(e)} in line: {decoded_line}")
                                continue
                yield collected_text
            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['deepseek_api']['api_retries']
            retry_delay = loaded_config_data['deepseek_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("DeepSeek: Posting request")
            response = session.post('https://api.deepseek.com/chat/completions', headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("DeepSeek: Summarization successful")
                    return summary
                else:
                    logging.warning("DeepSeek: Summary not found in the response data")
                    return "DeepSeek: Summary not available"
            else:
                logging.error(f"DeepSeek: Summarization failed with status code {response.status_code}")
                logging.error(f"DeepSeek: Error response: {response.text}")
                return f"DeepSeek: Failed to process summary. Status code: {response.status_code}"
    except Exception as e:
        logging.error(f"DeepSeek: Error in processing: {str(e)}", exc_info=True)
        return f"DeepSeek: Error occurred while processing summary: {str(e)}"


def summarize_with_mistral(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False):
    logging.debug("Mistral: Summarization process starting...")
    try:
        logging.debug("Mistral: Loading and validating configurations")
        loaded_config_data = load_and_log_configs()
        if loaded_config_data is None:
            logging.error("Failed to load configuration data")
            mistral_api_key = None
        else:
            # Prioritize the API key passed as a parameter
            if api_key and api_key.strip():
                mistral_api_key = api_key
                logging.info("Mistral: Using API key provided as parameter")
            else:
                # If no parameter is provided, use the key from the config
                mistral_api_key = loaded_config_data['mistral_api'].get('api_key')
                if mistral_api_key:
                    logging.info("Mistral: Using API key from config file")
                else:
                    logging.warning("Mistral: No API key found in config file")

        # Final check to ensure we have a valid API key
        if not mistral_api_key or not mistral_api_key.strip():
            logging.error("Mistral: No valid API key available")
            return "Mistral: API Key Not Provided/Found in Config file or is empty"

        logging.debug(f"Mistral: Using API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:]}")

        # Input data handling
        logging.debug("Mistral: Using provided string data for summarization")
        data = input_data

        # DEBUG - Debug logging to identify sent data
        logging.debug(f"Mistral: Loaded data: {str(data)[:500]}...(snipped to first 500 chars)")
        logging.debug(f"Mistral: Type of data: {type(data)}")

        if isinstance(data, dict) and 'summary' in data:
            logging.debug("Mistral: Summary already exists in the loaded data")
            return data['summary']

        # Text extraction
        if isinstance(data, list):
            segments = data
            text = extract_text_from_segments(segments)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError("Mistral: Invalid input data format")

        mistral_model = loaded_config_data['mistral_api']['model'] or "mistral-large-latest"

        if temp is None:
            temp = 0.2
        temp = float(temp)
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."

        headers = {
            'Authorization': f'Bearer {mistral_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(f"Mistral API Key: {mistral_api_key[:5]}...{mistral_api_key[-5:] if mistral_api_key else None}")
        logging.debug("Mistral: Preparing data + prompt for submission")
        mistral_prompt = f"{custom_prompt_arg}\n\n\n\n{text} "
        data = {
            "model": mistral_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": mistral_prompt}
            ],
            "temperature": temp,
            "top_p": 1,
            "max_tokens": 4096,
            "stream": streaming,
            "safe_prompt": False
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['mistral_api']['api_retries']
            retry_delay = loaded_config_data['mistral_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Mistral: Posting streaming request")
            response = session.post(
                'https://api.mistral.ai/v1/chat/completions',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            def stream_generator():
                collected_text = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line == '':
                            continue
                        try:
                            # Assuming the response is in SSE format
                            if decoded_line.startswith('data:'):
                                data_str = decoded_line[len('data:'):].strip()
                                if data_str == '[DONE]':
                                    break
                                data_json = json.loads(data_str)
                                if 'choices' in data_json and len(data_json['choices']) > 0:
                                    delta_content = data_json['choices'][0]['delta'].get('content', '')
                                    collected_text += delta_content
                                    yield delta_content
                                else:
                                    logging.error(f"Mistral: Unexpected data format: {data_json}")
                                    continue
                            else:
                                # Handle other event types if necessary
                                continue
                        except json.JSONDecodeError:
                            logging.error(f"Mistral: Error decoding JSON from line: {decoded_line}")
                            continue
                        except KeyError as e:
                            logging.error(f"Mistral: Key error: {str(e)} in line: {decoded_line}")
                            continue
                # Optionally, you can return the full collected text at the end
                # yield collected_text
            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['mistral_api']['api_retries']
            retry_delay = loaded_config_data['mistral_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Mistral: Posting non-streaming request")
            response = session.post(
                'https://api.mistral.ai/v1/chat/completions',
                headers=headers,
                json=data
            )

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Mistral: Summarization successful")
                    return summary
                else:
                    logging.warning("Mistral: Summary not found in the response data")
                    return "Mistral: Summary not available"
            else:
                logging.error(f"Mistral: Summarization failed with status code {response.status_code}")
                logging.error(f"Mistral: Error response: {response.text}")
                return f"Mistral: Failed to process summary. Status code: {response.status_code}"
    except Exception as e:
        logging.error(f"Mistral: Error in processing: {str(e)}", exc_info=True)
        return f"Mistral: Error occurred while processing summary: {str(e)}"


def summarize_with_google(api_key, input_data, custom_prompt_arg, temp=None, system_message=None, streaming=False,):
    loaded_config_data = load_and_log_configs()
    try:
        # API key validation
        if not api_key or api_key.strip() == "":
            logging.info("Google: #1 API key not provided as parameter")
            logging.info("Google: Attempting to use API key from config file")
            api_key = loaded_config_data['google_api']['api_key']

        if not api_key or api_key.strip() == "":
            logging.error("Google: #2 API key not found or is empty")
            return "Google: API Key Not Provided/Found in Config file or is empty"

        google_api_key = api_key
        logging.debug(f"Google: Using API Key: {api_key[:5]}...{api_key[-5:]}")

        # Input data handling
        logging.debug(f"Google: Raw input data type: {type(input_data)}")
        logging.debug(f"Google: Raw input data (first 500 chars): {str(input_data)[:500]}...")

        if isinstance(input_data, str):
            if input_data.strip().startswith('{'):
                # It's likely a JSON string
                logging.debug("Google: Parsing provided JSON string data for summarization")
                try:
                    data = json.loads(input_data)
                except json.JSONDecodeError as e:
                    logging.error(f"Google: Error parsing JSON string: {str(e)}")
                    return f"Google: Error parsing JSON input: {str(e)}"
            else:
                logging.debug("Google: Using provided string data for summarization")
                data = input_data
        else:
            data = input_data

        logging.debug(f"Google: Processed data type: {type(data)}")
        logging.debug(f"Google: Processed data (first 500 chars): {str(data)[:500]}...")

        # Text extraction
        if isinstance(data, dict):
            if 'summary' in data:
                logging.debug("Google: Summary already exists in the loaded data")
                return data['summary']
            elif 'segments' in data:
                text = extract_text_from_segments(data['segments'])
            else:
                text = json.dumps(data)  # Convert dict to string if no specific format
        elif isinstance(data, list):
            text = extract_text_from_segments(data)
        elif isinstance(data, str):
            text = data
        else:
            raise ValueError(f"Google: Invalid input data format: {type(data)}")

        logging.debug(f"Google: Extracted text (first 500 chars): {text[:500]}...")
        logging.debug(f"Google: Custom prompt: {custom_prompt_arg}")

        google_model = loaded_config_data['google_api']['model'] or "gemini-1.5-pro"
        logging.debug(f"Google: Using model: {google_model}")

        headers = {
            'Authorization': f'Bearer {google_api_key}',
            'Content-Type': 'application/json'
        }

        logging.debug(
            f"Google API Key: {google_api_key[:5]}...{google_api_key[-5:] if google_api_key else None}")
        logging.debug("openai: Preparing data + prompt for submittal")
        google_prompt = f"{text} \n\n\n\n{custom_prompt_arg}"
        #if temp is None:
        #    temp = 0.7
        if system_message is None:
            system_message = "You are a helpful AI assistant who does whatever the user requests."
        #temp = float(temp)
        data = {
            "model": google_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": google_prompt}
            ],
            "stream": streaming,
            #"max_tokens": 4096,
            #"temperature": temp
        }

        if streaming:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['google_api']['api_retries']
            retry_delay = loaded_config_data['google_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Google: Posting streaming request")
            response = session.post(
                'https://generativelanguage.googleapis.com/v1beta/openai/',
                headers=headers,
                json=data,
                stream=True
            )
            response.raise_for_status()

            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line == '':
                            continue
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[len('data: '):]
                            if data_str == '[DONE]':
                                break
                            try:
                                data_json = json.loads(data_str)
                                chunk = data_json["choices"][0]["delta"].get("content", "")
                                yield chunk
                            except json.JSONDecodeError:
                                logging.error(f"Google: Error decoding JSON from line: {decoded_line}")
                                continue
                            except KeyError as e:
                                logging.error(f"Google: Key error: {str(e)} in line: {decoded_line}")
                                continue
            return stream_generator()
        else:
            # Create a session
            session = requests.Session()

            # Load config values
            retry_count = loaded_config_data['google_api']['api_retries']
            retry_delay = loaded_config_data['google_api']['api_retry_delay']

            # Configure the retry strategy
            retry_strategy = Retry(
                total=retry_count,  # Total number of retries
                backoff_factor=retry_delay,  # A delay factor (exponential backoff)
                status_forcelist=[429, 502, 503, 504],  # Status codes to retry on
            )

            # Create the adapter
            adapter = HTTPAdapter(max_retries=retry_strategy)

            # Mount adapters for both HTTP and HTTPS
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            logging.debug("Google: Posting request")
            response = session.post('https://generativelanguage.googleapis.com/v1beta/openai/', headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    summary = response_data['choices'][0]['message']['content'].strip()
                    logging.debug("Google: Summarization successful")
                    logging.debug(f"Google: Summary (first 500 chars): {summary[:500]}...")
                    return summary
                else:
                    logging.warning("Google: Summary not found in the response data")
                    return "Google: Summary not available"
            else:
                logging.error(f"Google: Summarization failed with status code {response.status_code}")
                logging.error(f"Google: Error response: {response.text}")
                return f"Google: Failed to process summary. Status code: {response.status_code}"
    except json.JSONDecodeError as e:
        logging.error(f"Google: Error decoding JSON: {str(e)}", exc_info=True)
        return f"Google: Error decoding JSON input: {str(e)}"
    except requests.RequestException as e:
        logging.error(f"Google: Error making API request: {str(e)}", exc_info=True)
        return f"Google: Error making API request: {str(e)}"
    except Exception as e:
        logging.error(f"Google: Unexpected error: {str(e)}", exc_info=True)
        return f"Google: Unexpected error occurred: {str(e)}"

def summarize_with_mock_llm(text_to_summarize: str, custom_prompt_arg: Optional[str], api_key: Optional[str] = None, temp: Optional[float] = None, system_message: Optional[str] = None, streaming: bool = False):
    """
    Mock implementation of OpenAI summarization function that mimics the behavior
    without making actual API calls.

    Returns either a string summary or a generator for streaming responses,
    matching the behavior of the real function.
    """
    try:
        # Log the same debug information as the real function
        logging.debug(f"MOCK-LLM (MOCK): Received text length: {len(str(text_to_summarize))}")
        logging.debug(f"MOCK-LLM (MOCK): Custom prompt: {custom_prompt_arg}")
        logging.debug(f"MOCK-LLM (MOCK): Temperature: {temp}, System Message: {system_message}, Streaming: {streaming}")

        # Extract a sample of text to include in mock response
        sample_text = str(text_to_summarize)[:50] + "..." if len(str(text_to_summarize)) > 50 else str(text_to_summarize)

        # Create mock summary
        mock_summary = (
            f"[MOCK OPENAI RESPONSE]\n"
            f"This is a mock summary generated for testing purposes.\n\n"
            f"Sample of input text: '{sample_text}'\n"
            f"Custom prompt: '{custom_prompt_arg}'\n"
            f"Temperature: {temp or 0.7}\n"
            f"System message: '{system_message or 'You are a helpful AI assistant.'}'\n"
            f"Time generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Add some simulated delay to mimic API latency
        time.sleep(0.5)

        mock_summary_text = f"Mocked summary for: {text_to_summarize[:30]}..."
        if streaming:
            def _stream():
                yield mock_summary_text

            return _stream()
        else:
            logging.debug("OpenAI (MOCK): Returning non-streaming mock response")
            return mock_summary

    except Exception as e:
        logging.error(f"OpenAI (MOCK): Unexpected error: {str(e)}", exc_info=True)
        return f"Error: OpenAI mock function unexpected error: {str(e)}"
#
#

# FIXME
def summarize_chunk(api_name, text, custom_prompt_input, api_key, temp=None, system_message=None):
    logging.debug("Entered 'summarize_chunk' function")
    if api_name in (None, "None", "none"):
        logging.warning("summarize_chunk: API name not provided for summarization")
        return "No summary available"

    try:
        result = analyze(text, custom_prompt_input, api_name, api_key, temp, system_message)

        # Handle streaming generator responses
        if inspect.isgenerator(result):
            logging.debug(f"Handling streaming response from {api_name}")
            collected_chunks = []
            for chunk in result:
                # Check for error chunks first
                if isinstance(chunk, str) and chunk.startswith("Error:"):
                    logging.warning(f"Streaming error from {api_name}: {chunk}")
                    return chunk
                collected_chunks.append(chunk)
            final_result = "".join(collected_chunks)
            logging.info(f"Summarization with {api_name} streaming successful")
            return final_result

        # Handle regular string responses
        elif isinstance(result, str):
            if result.startswith("Error:"):
                logging.warning(f"Summarization with {api_name} failed: {result}")
                return None
            logging.info(f"Summarization with {api_name} successful")
            return result

        # Handle unexpected response types
        else:
            logging.error(f"Unexpected response type from {api_name}: {type(result)}")
            return None

    except Exception as e:
        logging.error(f"Error in summarize_chunk with {api_name}: {str(e)}", exc_info=True)
        return None


def extract_metadata_and_content(input_data):
    metadata = {}
    content = ""

    if isinstance(input_data, str):
        if os.path.exists(input_data):
            with open(input_data, 'r', encoding='utf-8') as file:
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



def extract_text_from_input(input_data):
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
