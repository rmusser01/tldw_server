# Audio_Files.py
#########################################
# Audio Processing Library
# This library is used to download or load audio files from a local directory.
#
####
#
# Functions:
#
#
#########################################
# Imports
import json
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
#
# External Imports
import requests
import yt_dlp
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords, \
    check_media_and_whisper_model
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import summarize
from tldw_Server_API.app.core.Utils.Utils import downloaded_files, \
    sanitize_filename, logging
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import extract_metadata
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import speech_to_text, \
    convert_to_wav, ConversionError
from tldw_Server_API.app.core.Utils.Chunk_Lib import improved_chunking_process
#
#######################################################################################################################
# Function Definitions
#

MAX_FILE_SIZE = 500 * 1024 * 1024

def download_audio_file(url: str, use_cookies: bool = False, cookies: Optional[str] = None) -> str:
    """
    Downloads an audio file from a URL.

    Args:
        url: The URL of the audio file.
        use_cookies: Whether to use cookies for the download.
        cookies: A JSON string of cookies if use_cookies is True.

    Returns:
        The local path to the downloaded audio file.

    Raises:
        requests.RequestException: If the download fails due to network issues or bad status codes.
        ValueError: If the file size exceeds the limit or cookies are invalid JSON.
        Exception: For other unexpected errors.
    """
    try:
        # --- DB CHECK REMOVED ---
        # The decision to download is now made by the caller (API layer)

        logging.info(f"Attempting audio download from: {url}")

        # Set up the request headers
        headers = {}
        if use_cookies and cookies:
            try:
                # Accept JSON string or dictionary
                if isinstance(cookies, str):
                    cookie_dict = json.loads(cookies)
                elif isinstance(cookies, dict):
                    cookie_dict = cookies
                else:
                     raise TypeError("Cookies must be a JSON string or a dictionary.")
                headers['Cookie'] = '; '.join([f'{k}={v}' for k, v in cookie_dict.items()])
                logging.debug("Using cookies for download.")
            except (json.JSONDecodeError, TypeError) as e:
                logging.warning(f"Invalid cookie format provided. Proceeding without cookies. Error: {e}")
                # Decide if you want to raise an error or just proceed without cookies
                # raise ValueError(f"Invalid cookie format: {e}") from e

        # Make the request
        response = requests.get(url, headers=headers, stream=True, timeout=120) # Added timeout
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        # Get the file size
        file_size = int(response.headers.get('content-length', 0))
        if file_size > MAX_FILE_SIZE:
            raise ValueError(f"File size ({file_size / (1024*1024):.2f} MB) exceeds the {MAX_FILE_SIZE / (1024*1024):.0f}MB limit.")

        # Generate a unique filename
        # Try to get a filename from headers or URL
        content_disposition = response.headers.get('content-disposition')
        original_filename = None
        if content_disposition:
            parts = content_disposition.split('filename=')
            if len(parts) > 1:
                original_filename = parts[1].strip('"\' ')

        if not original_filename:
             try:
                 original_filename = Path(urlparse(url).path).name
             except Exception:
                 pass # Ignore errors parsing URL path

        if original_filename:
             base_name = sanitize_filename(Path(original_filename).stem)
             extension = Path(original_filename).suffix or ".mp3" # Default to mp3 if no ext
             # Limit length
             base_name = base_name[:50]
             # Use a more robust unique ID section in filename
             unique_id = uuid.uuid4().hex[:8]
             file_name = f"{base_name}_{unique_id}{extension}"
        else:
             unique_id = uuid.uuid4().hex[:8]
             file_name = f"audio_{unique_id}.mp3"

        # Use a dedicated, potentially configurable, download directory
        # For temporary processing, might want to use the provided temp_dir later
        save_dir = Path(tempfile.gettempdir()) / "tldw_audio_downloads"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name

        # Download the file efficiently
        downloaded_bytes = 0
        log_interval = 5 * 1024 * 1024 # Log every 5MB
        next_log_thresh = log_interval
        logging.info(f"Downloading to: {save_path}")
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                # Filter out keep-alive new chunks.
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)
                    if file_size > 0 and downloaded_bytes >= next_log_thresh:
                         logging.info(f"Download progress: {downloaded_bytes / (1024*1024):.1f} / {file_size / (1024*1024):.1f} MB")
                         next_log_thresh += log_interval


        logging.info(f"Audio file downloaded successfully: {save_path} ({downloaded_bytes / (1024*1024):.2f} MB)")
        return str(save_path)

    except requests.exceptions.Timeout:
         logging.error(f"Timeout occurred while downloading audio file: {url}")
         raise requests.RequestException(f"Download timed out for {url}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading audio file from {url}: {type(e).__name__} - {e}")
        # Optionally include response details if available
        err_msg = f"Error downloading audio: {e.response.status_code}" if e.response else str(e)
        raise requests.RequestException(f"Download failed for {url}. Reason: {err_msg}") from e
    except ValueError as e:
        logging.error(f"Value error during download from {url}: {e}")
        raise # Re-raise specific value errors (file size, cookies)
    except Exception as e:
        logging.error(f"Unexpected error downloading audio file from {url}: {type(e).__name__} - {e}", exc_info=True)
        raise Exception(f"Unexpected download error for {url}") from e # Re-raise generic


def process_audio_files(
    # Use 'inputs' to accept both URLs and local paths
    inputs: List[str],
    # Processing parameters
    transcription_model: str,
    transcription_language: Optional[str] = 'en', # Default to 'en'
    perform_chunking: bool = True,
    chunk_method: Optional[str] = None, # Will default based on type if None
    max_chunk_size: int = 500,
    chunk_overlap: int = 200,
    use_adaptive_chunking: bool = False,
    use_multi_level_chunking: bool = False,
    chunk_language: Optional[str] = None, # Language for chunking logic
    diarize: bool = False,
    vad_use: bool = False, # Add VAD parameter
    timestamp_option: bool = True, # Keep timestamps by default
    perform_analysis: bool = True, # Summarize by default if API provided
    api_name: Optional[str] = None, # LLM API for summarization
    api_key: Optional[str] = None,
    custom_prompt_input: Optional[str] = None,
    system_prompt_input: Optional[str] = None,
    summarize_recursively: bool = False,
    # Input handling parameters
    use_cookies: bool = False,
    cookies: Optional[str] = None,
    keep_original: bool = False, # Keep downloaded/intermediate files?
    # Optional metadata overrides (less common here, usually handled by API layer)
    custom_title: Optional[str] = None,
    author: Optional[str] = None,
    temp_dir: Optional[str] = None # Explicit temp dir passed from API
) -> Dict[str, Any]:
    """
    Processes a list of audio inputs (URLs or local file paths). Handles download,
    conversion, transcription, chunking, and analysis (summarization). Returns
    structured results suitable for API response, without DB interaction.

    Args:
        inputs: List of strings, where each string is either a URL or a local file path.
        transcription_model: Name of the Whisper model to use.
        transcription_language: Target language for transcription (e.g., 'en', 'es').
        perform_chunking: Whether to chunk the transcription.
        chunk_method: Method for chunking ('words', 'sentences', 'recursive', etc.).
        max_chunk_size: Max characters/tokens per chunk.
        chunk_overlap: Overlap between chunks.
        use_adaptive_chunking: Use adaptive chunking methods.
        use_multi_level_chunking: Use multi-level chunking.
        chunk_language: Language for chunking logic (defaults to transcription language).
        diarize: Perform speaker diarization.
        vad_use: Use Voice Activity Detection (VAD) filter during transcription.
        timestamp_option: Include timestamps in the final transcript.
        perform_analysis: Whether to perform summarization/analysis.
        api_name: Name of the LLM API to use for analysis (e.g., 'openai').
        api_key: API key for the LLM API.
        custom_prompt_input: Custom prompt for the summarization task.
        system_prompt_input: System prompt for the summarization task.
        summarize_recursively: Use recursive summarization strategy.
        use_cookies: Pass cookies when downloading URLs.
        cookies: Cookie string (JSON format) or dictionary.
        keep_original: If False, temporary downloaded/converted files are deleted.
        custom_title: Optional title override (primarily for context).
        author: Optional author override (primarily for context).
        temp_dir: Optional path to a directory for temporary files. If None, uses OS default.


    Returns:
        A dictionary containing:
          - 'processed_count': Number of successfully processed items.
          - 'errors_count': Number of failed items.
          - 'errors': List of error messages for failed items.
          - 'results': A list of dictionaries, one for each input item, containing:
            - 'input_ref': The original URL or filename provided.
            * 'processing_source': The actual file path used after download/upload.
            - 'status': 'Success', 'Error', or 'Warning'.
            - 'input_ref': Original URL or filename.
            - 'processing_source': Final path used for processing (e.g., WAV file).
            - 'media_type': 'audio'.
            - 'metadata': Dict with title, author etc.
            - 'content': Full transcribed text (string or None).
            - 'segments': List of transcribed segments (or None).
            - 'chunks': List of text chunks if chunking performed (or None).
            - 'analysis': Generated summary/analysis (string or None).
            - 'analysis_details': Dict with details (e.g., detected language).
            - 'error': Error message if processing failed (string or None).
            - 'warnings': List of non-fatal warnings.
            - 'db_id': Always None for this function.
            - 'db_message': Always None for this function.
    """
    batch_items_results: List[Dict[str, Any]] = []
    progress_log: List[str] = []
    temp_files_to_clean: List[str] = []
    start_time_all = time.time()

    # --- Setup Temporary Directory ---
    # Use TemporaryDirectory which cleans up automatically unless keep_original=True
    # Note: If keep_original=True, the caller needs to manage the lifecycle of temp_dir
    temp_directory_manager = None
    processing_temp_dir_path = None

    if temp_dir:
        processing_temp_dir_path = Path(temp_dir)
        processing_temp_dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using provided temporary directory: {processing_temp_dir_path}")
    else:
        # Create a managed temporary directory if none provided
        # It will be cleaned up automatically unless keep_original is True
        temp_directory_manager = tempfile.TemporaryDirectory(prefix="audio_proc_")
        processing_temp_dir_path = Path(temp_directory_manager.name)
        logging.info(f"Created managed temporary directory: {processing_temp_dir_path}")

    # Helper to track progress messages
    def update_progress(message: str):
        logging.info(message)
        progress_log.append(message)

    # Define chunk options dictionary
    chunk_options = {
        'method': chunk_method or ('sentences' if (chunk_language or transcription_language or 'en') == 'en' else 'recursive'),
        'max_size': max_chunk_size,
        'overlap': chunk_overlap,
        'adaptive': use_adaptive_chunking,
        'multi_level': use_multi_level_chunking,
        'language': chunk_language or transcription_language or 'en',
    } if perform_chunking else None

    try:
        # --- Process Each Input ---
        for i, input_item in enumerate(inputs, start=1):
            item_start_time = time.time()
            is_url = isinstance(input_item, str) and input_item.startswith(("http://", "https://"))
            input_ref = input_item if is_url else Path(input_item).name
            update_progress(f"--- Processing item {i}/{len(inputs)}: {input_ref} ---")

            item_result: Dict[str, Any] = { # Explicit typing
                "status": "Pending",
                "input_ref": input_ref,
                "processing_source": input_item, # Initial source
                "media_type": "audio",
                "metadata": {"title": custom_title, "author": author},
                "content": None,
                "segments": None,
                "chunks": None, # Added field
                "analysis": None, # Renamed from summary
                "analysis_details": {},
                "error": None,
                "warnings": [],
                "db_id": None, # Standard fields for response consistency
                "db_message": None,
            }
            current_audio_path = None
            downloaded_path = None
            wav_file_path = None
            item_temp_files = [] # Files specific to this item

            try:
                # 1. Get Local Audio Path (Download if URL, Copy if local?)
                if is_url:
                    update_progress(f"Downloading audio from URL: {input_item}")
                    try:
                        # Download to the processing temp dir
                        downloaded_path = download_audio_file(input_item, use_cookies, cookies)
                        # Move or copy to our managed temp dir if different
                        target_path = processing_temp_dir_path / Path(downloaded_path).name
                        if Path(downloaded_path).parent != processing_temp_dir_path:
                            Path(downloaded_path).rename(target_path)
                            current_audio_path = str(target_path)
                            # Clean up original download dir if empty? Maybe too complex.
                        else:
                            current_audio_path = downloaded_path

                        item_result["processing_source"] = current_audio_path
                        item_temp_files.append(current_audio_path) # Mark for potential cleanup
                        item_result["metadata"]["title"] = item_result["metadata"].get("title") or Path(current_audio_path).stem.replace("_"+uuid.uuid4().hex[:8],"") # Basic title
                    except Exception as download_err:
                        err_msg = f"Failed to download/prepare URL: {download_err}"
                        update_progress(err_msg)
                        item_result.update({"status": "Error", "error": err_msg})
                        continue

                else: # Local file input
                    local_path = Path(input_item)
                    if not local_path.exists():
                        raise FileNotFoundError(f"Local file not found: {input_item}")
                    if local_path.stat().st_size > MAX_FILE_SIZE:
                         raise ValueError(f"Local file '{input_ref}' size exceeds {MAX_FILE_SIZE / (1024*1024):.0f}MB limit.")

                    # Check if the file is already in the target temp directory (likely an upload)
                    # Resolve paths to handle potential symlinks or relative paths robustly
                    if local_path.resolve().parent == processing_temp_dir_path.resolve():
                        update_progress(f"Using already saved file in temp directory: {local_path.name}")
                        current_audio_path = str(local_path)
                        # No need to copy, it's already where it needs to be.
                        # Do NOT add to item_temp_files here, the main TempDirManager handles this dir.
                    else:
                        # If it's a local file from elsewhere, *then* copy it to the temp directory
                        update_progress(f"Copying local file '{local_path.name}' to temporary directory.")
                        try:
                            target_path = processing_temp_dir_path / local_path.name
                            import shutil
                            shutil.copy2(local_path, target_path) # copy2 preserves metadata
                            current_audio_path = str(target_path)
                            # Only mark the *copied* file for item-specific cleanup if needed
                            # and if keep_original=False (handled by the main finally block check)
                            item_temp_files.append(current_audio_path)
                        except Exception as copy_err:
                             # Log the specific error
                             logging.error(f"shutil.copy2 failed for source '{local_path}' to target '{target_path}': {copy_err}", exc_info=True)
                             raise RuntimeError(f"Failed to copy local file to temp directory: {copy_err}") from copy_err

                    item_result["processing_source"] = current_audio_path # Source is now the copied file
                    item_result["metadata"]["title"] = item_result["metadata"].get("title") or local_path.stem

                if not current_audio_path or not Path(current_audio_path).exists():
                     raise RuntimeError("Audio file path is missing or invalid after download/copy check.")

                # 2. Convert to WAV using the library function
                update_progress(f"Converting '{Path(current_audio_path).name}' to WAV...")
                try:
                    # Always overwrite in temp dir context
                    wav_file_path = convert_to_wav(current_audio_path, overwrite=True)
                    # ... (path checking logic - ensure wav_file_path is valid) ...
                    if not wav_file_path or not Path(wav_file_path).exists():
                         raise ConversionError(f"convert_to_wav did not return a valid path or file does not exist: {wav_file_path}")
                    item_temp_files.append(wav_file_path) # Mark WAV for potential cleanup
                    item_result["processing_source"] = wav_file_path # Update source
                    update_progress(f"Conversion to WAV successful: {Path(wav_file_path).name}")
                except (ConversionError, FileNotFoundError, RuntimeError) as conv_err:
                    # If conversion fails, set error and status, then *re-raise*
                    # to be caught by the outer 'except Exception as item_processing_exc'
                    err_msg = f"Audio conversion failed: {conv_err}"
                    update_progress(err_msg)
                    item_result.update({"status": "Error", "error": err_msg})
                    raise # Re-raise the caught exception

                # 3. Transcribe
                update_progress(f"Starting transcription (Model: {transcription_model}, Lang: {transcription_language or 'auto'}, VAD: {vad_use}, Diarize: {diarize})")
                try:
                    # Ensure wav_file_path is valid before calling speech_to_text
                    if not wav_file_path:
                         raise ValueError("Cannot transcribe, WAV file path is missing.")

                    transcription_output = speech_to_text(
                        audio_file_path=wav_file_path,
                        whisper_model=transcription_model,
                        selected_source_lang=transcription_language,
                        vad_filter=vad_use,
                        diarize=diarize,
                    )
                    raw_segments = transcription_output
                    # ... (process segments, set item_result["content"], item_result["segments"]) ...
                    if not raw_segments:
                        item_result.setdefault("warnings", [])
                        item_result["warnings"].append("Transcription produced no segments.")
                        update_progress("Warning: Transcription generated no segments.")
                        item_result["content"] = ""
                        item_result["segments"] = []
                    else:
                        item_result["segments"] = raw_segments
                        item_result["content"] = format_transcription_with_timestamps(
                            raw_segments, keep_timestamps=timestamp_option
                        )
                        if not item_result["content"].strip():
                            item_result.setdefault("warnings", [])
                            item_result["warnings"].append("Transcription resulted in empty text.")
                            update_progress("Warning: Transcription text is empty.")

                    update_progress("Transcription completed.")

                except (RuntimeError, ValueError) as trans_err:
                     # If transcription fails, set error and status, then *re-raise*
                     err_msg = f"Transcription failed: {trans_err}"
                     update_progress(err_msg)
                     item_result.update({"status": "Error", "error": err_msg})
                     raise # Re-raise the caught exception

                # 4. Chunking
                text_to_process = item_result["content"]
                generated_chunks = None
                text_to_process_for_analysis = []
                if chunk_options and text_to_process and text_to_process.strip():
                     # ... (existing chunking logic) ...
                     # Ensure generated_chunks and text_to_process_for_analysis are set
                    update_progress(f"Chunking text with options: {chunk_options}")
                    try:
                        generated_chunks = improved_chunking_process(text_to_process, chunk_options)
                        if not generated_chunks:
                            update_progress("Warning: Chunking resulted in no text chunks.")
                            item_result.setdefault("warnings", [])
                            item_result["warnings"].append("Chunking process yielded no chunks.")
                        else:
                            update_progress(f"Chunking produced {len(generated_chunks)} chunk(s).")
                            item_result["chunks"] = generated_chunks
                            # Assuming summarize needs list of strings for now
                            text_to_process_for_analysis = [chunk.get('text', '') if isinstance(chunk, dict) else str(chunk) for chunk in generated_chunks]
                    except Exception as chunk_err:
                         err_msg = f"Chunking failed: {chunk_err}"
                         update_progress(err_msg)
                         item_result.setdefault("warnings", [])
                         item_result["warnings"].append(f"Chunking error: {chunk_err}")
                         text_to_process_for_analysis = [text_to_process] if text_to_process else [] # Fallback
                         item_result["chunks"] = None
                elif chunk_options:
                    update_progress("Chunking skipped (empty transcript).")
                else:
                    update_progress("Chunking not requested.")
                    if text_to_process and text_to_process.strip():
                         text_to_process_for_analysis = [text_to_process]

                    # 5. Analysis (Summarization) (if requested and text exists)
                    if perform_analysis and api_name and api_name.lower() != "none" and text_to_process_for_analysis:
                        update_progress(f"Starting analysis using API: {api_name}")
                        try:
                            analysis_result = summarize(
                                api_name=api_name,
                                input_data=text_to_process_for_analysis,
                                custom_prompt_arg=custom_prompt_input,
                                api_key=api_key,
                                recursive_summarization=summarize_recursively,
                                chunked_summarization=(generated_chunks is not None and len(
                                    generated_chunks) > 1 and not summarize_recursively),
                                temp=None,
                                system_message=system_prompt_input
                            )
                            if isinstance(analysis_result, str) and analysis_result.startswith("Error:"):
                                raise RuntimeError(analysis_result)

                            item_result["analysis"] = analysis_result or "Analysis API returned no result."
                            update_progress("Analysis completed.")

                        except Exception as exc:
                            err_msg = f"Analysis failed: {exc}"
                            update_progress(err_msg)
                            item_result["analysis"] = "[Analysis Failed]"
                            item_result.setdefault("warnings", [])
                            item_result["warnings"].append(f"Analysis error: {exc}")
                    elif perform_analysis and (not api_name or api_name.lower() == "none"):
                        item_result["analysis"] = "[Analysis Skipped: No API specified]"
                        update_progress("Analysis skipped (no API name provided).")
                    elif perform_analysis and not text_to_process_for_analysis:
                        item_result["analysis"] = "[Analysis Skipped: No text content]"
                        update_progress("Analysis skipped (no text found after transcription/chunking).")
                    else:  # Analysis not requested
                        item_result["analysis"] = "[Analysis Not Requested]"


                # 6. Finalize Status for SUCCESS/WARNING case
                # If we reach here, no critical error was raised during conversion/transcription
                item_result["status"] = "Warning" if item_result.get("warnings") else "Success"
                item_processing_time = time.time() - item_start_time
                update_progress(f"Item {i} ({input_ref}) finished processing. Status: {item_result['status']}. Time: {item_processing_time:.2f}s")

            except Exception as item_processing_exc:
                # Catch ANY exception raised during the item's processing steps
                # (including re-raised conversion/transcription errors or others)
                error_message = f"Failed to process item {i} ({input_ref}): {type(item_processing_exc).__name__} - {item_processing_exc}"
                update_progress(error_message)
                logging.error(error_message, exc_info=True) # Log full traceback
                item_result["status"] = "Error" # Ensure status is Error
                # Store simplified error message if not already set by inner handlers
                if not item_result.get("error"):
                    item_result["error"] = str(item_processing_exc)

            finally:
                # THIS BLOCK *ALWAYS* EXECUTES FOR THE ITEM, REGARDLESS OF EXCEPTIONS ABOVE
                # Add item-specific temp files to the main list for cleanup tracking
                temp_files_to_clean.extend(item_temp_files)
                # Append the final state of item_result (Success, Warning, or Error)
                logging.debug(f"Appending result for item {i}: Status='{item_result.get('status')}', Error='{item_result.get('error')}'")
                batch_items_results.append(item_result) # Use the renamed list

        # --- End of Loop ---

    except Exception as outer_exc: # Catch setup errors before the loop
         logging.error(f"Fatal error during audio processing setup: {outer_exc}", exc_info=True)
         return {
            "processed_count": 0, "errors_count": len(inputs),
            "errors": [f"Fatal setup error: {outer_exc}"],
            # Use the renamed list name here too
            "results": [{"input_ref": item, "status": "Error", "error": f"Fatal setup error: {outer_exc}", "media_type": "audio"} for item in inputs]
         }
    finally:
        # --- Cleanup Temporary Files ---
        if not keep_original:
            update_progress("Cleaning up temporary files...")
            # Use set to avoid trying to delete the same file multiple times
            unique_files_to_clean = set(temp_files_to_clean)
            cleaned_count = 0
            for file_path_str in unique_files_to_clean:
                if file_path_str: # Ensure not None or empty
                    file_path = Path(file_path_str)
                    if file_path.exists() and file_path.is_file(): # Check if it's a file
                         # Security check: Ensure it's within the temp directory
                         try:
                              is_safe = file_path.resolve().is_relative_to(processing_temp_dir_path.resolve())
                         except ValueError: # Python < 3.9
                              is_safe = str(file_path.resolve()).startswith(str(processing_temp_dir_path.resolve()))

                         if is_safe:
                              try:
                                   file_path.unlink()
                                   cleaned_count += 1
                                   logging.debug(f"Removed temp file: {file_path}")
                              except OSError as e:
                                   update_progress(f"Warning: Failed to remove temporary file {file_path}: {e}")
                         else:
                              logging.warning(f"Skipping deletion of file outside temp dir: {file_path}")
            update_progress(f"Attempted removal of {cleaned_count} temporary files.")
        else:
            update_progress("Skipping temporary file cleanup (keep_original=True).")

        # --- Cleanup Temporary Directory (if managed) ---
        if temp_directory_manager:
            try:
                 temp_directory_manager.cleanup()
                 update_progress(f"Removed managed temporary directory: {processing_temp_dir_path}")
            except Exception as e:
                 logging.warning(f"Could not remove managed temporary directory {processing_temp_dir_path}: {e}")


    # --- Calculate Final Counts and Return ---
    # Use the renamed list
    logging.debug(f"Final batch_items_results before calculating counts: {batch_items_results}")
    processed_count = sum(1 for r in batch_items_results if r.get("status") in ["Success", "Warning"])
    failed_count = len(batch_items_results) - processed_count
    total_time = time.time() - start_time_all
    update_progress(f"Processing batch complete. Success/Warning: {processed_count}, Failed: {failed_count}. Total Time: {total_time:.2f}s")

    # Structure the final output
    final_output = {
        "processed_count": processed_count,
        "errors_count": failed_count,
        "errors": [r.get("error") for r in batch_items_results if r.get("status") == "Error" and r.get("error")],
        "results": batch_items_results, # Return the list
    }
    logging.debug(f"Returning final output: {final_output}")
    return final_output


def format_transcription_with_timestamps(segments, keep_timestamps=True):
    """
    Formats the transcription segments with or without timestamps.

    Parameters:
        segments (list): List of transcription segments.
        keep_timestamps (bool): Whether to include timestamps.

    Returns:
        str: Formatted transcription.
    """
    if keep_timestamps:
        formatted_segments = []
        for segment in segments:
            start = segment.get('Time_Start', 0)
            end = segment.get('Time_End', 0)
            text = segment.get('Text', '').strip()

            # Check if start and end are already formatted strings
            if isinstance(start, str) and ':' in start:
                # Already in HH:MM:SS format, use directly
                formatted_segments.append(f"[{start}-{end}] {text}")
            else:
                # Numeric seconds, convert to time format
                try:
                    start_time = time.strftime('%H:%M:%S', time.gmtime(float(start)))
                    end_time = time.strftime('%H:%M:%S', time.gmtime(float(end)))
                    formatted_segments.append(f"[{start_time}-{end_time}] {text}")
                except (ValueError, TypeError):
                    # Fallback if conversion fails
                    formatted_segments.append(f"[{start}-{end}] {text}")
            # Join the segments with a newline to ensure proper formatting
            formatted_segments.append(f"[{start:.2f}-{end:.2f}] {text}")
        return "\n".join(formatted_segments)
    else:
        # Join the text without timestamps
        return "\n".join([segment.get('Text', '').strip() for segment in segments])


def download_youtube_audio(url):
    try:
        # Determine ffmpeg path based on the operating system.
        ffmpeg_path = './Bin/ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract information about the video
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info_dict = ydl.extract_info(url, download=False)
                sanitized_title = sanitize_filename(info_dict['title'])

            # Setup the temporary filenames
            temp_video_path = Path(temp_dir) / f"{sanitized_title}_temp.mp4"
            temp_audio_path = Path(temp_dir) / f"{sanitized_title}.mp3"

            # Initialize yt-dlp with options for downloading
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/best[height<=480]',  # Prefer best audio, or video up to 480p
                'ffmpeg_location': ffmpeg_path,
                'outtmpl': str(temp_video_path),
                'noplaylist': True,
                'quiet': True
            }

            # Execute yt-dlp to download the video/audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Check if the file exists
            if not temp_video_path.exists():
                raise FileNotFoundError(f"Expected file was not found: {temp_video_path}")

            # Use ffmpeg to extract audio
            ffmpeg_command = [
                ffmpeg_path,
                '-i', str(temp_video_path),
                '-vn',  # No video
                '-acodec', 'libmp3lame',
                '-b:a', '192k',
                str(temp_audio_path)
            ]
            subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Check if the audio file was created
            if not temp_audio_path.exists():
                raise FileNotFoundError(f"Expected audio file was not found: {temp_audio_path}")

            # Create a persistent directory for the download if it doesn't exist
            persistent_dir = Path("downloads")
            persistent_dir.mkdir(exist_ok=True)

            # Move the file from the temporary directory to the persistent directory
            persistent_file_path = persistent_dir / f"{sanitized_title}.mp3"
            os.replace(str(temp_audio_path), str(persistent_file_path))

            # Add the file to the list of downloaded files
            downloaded_files.append(str(persistent_file_path))

            return str(persistent_file_path), f"Audio downloaded successfully: {sanitized_title}.mp3"
    except Exception as e:
        return None, f"Error downloading audio: {str(e)}"


def process_podcast(
    url: str,
    # Metadata passed from caller (API) or extracted
    title: Optional[str] = None,
    author: Optional[str] = None,
    keywords: Optional[str] = "", # Comma-separated string or list
    # Processing options
    whisper_model: str = "distil-large-v3",
    enable_diarization: bool = False,
    keep_timestamps: bool = True,
    # Analysis options
    custom_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None, # Added system prompt
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    summarize_recursively: bool = False, # Added recursive flag
    # Chunking options
    perform_chunking: bool = True, # Added perform flag
    chunk_method: Optional[str] = None,
    max_chunk_size: int = 300,
    chunk_overlap: int = 0,
    use_adaptive_chunking: bool = False,
    use_multi_level_chunking: bool = False,
    chunk_language: str = 'english',
    # Download options
    use_cookies: bool = False,
    cookies: Optional[str] = None, # JSON string or dict
    # File handling
    keep_original: bool = False, # Keep intermediate files?
    temp_dir: Optional[str] = None # Explicit temp dir
) -> dict[str, None | dict[str, str | None] | list[Any] | dict[Any, Any] | str | float | Any] | None:
    """
    Processes a podcast URL: downloads, extracts metadata, transcribes, chunks, summarizes.
    Returns a dictionary with processed data and status, without DB interaction.

    Returns:
        Dict containing keys like: status, input_ref, processing_source, transcript,
        segments, summary, metadata, error, warnings, analysis_details.
        (Similar structure to process_audio_files results)
    """
    start_time = time.time()
    progress = []
    temp_files = []
    result = {
        "status": "Pending", "input_ref": url, "processing_source": url,
        "transcript": None, "segments": None, "summary": None,
        "metadata": {"title": title, "author": author, "keywords": keywords}, # Initial metadata
        "error": None, "warnings": [], "analysis_details": {}
    }

    # --- Setup Temporary Directory ---
    temp_directory_manager = tempfile.TemporaryDirectory(prefix="podcast_proc_", dir=temp_dir)

    def update_progress(message):
        logging.info(f"Podcast ({url[:50]}...): {message}")
        progress.append(message)

    def _cleanup_temp_files():
        if not keep_original:
            cleaned = 0
            for f_path in temp_files:
                if f_path and Path(f_path).exists():
                    try: Path(f_path).unlink() ; cleaned += 1
                    except OSError as e: update_progress(f"Warning: Failed to remove temp file {f_path}: {e}")
            update_progress(f"Cleaned {cleaned} temporary podcast files.")

    try:
        processing_temp_dir = Path(temp_directory_manager.name)
        update_progress(f"Using temp directory: {processing_temp_dir}")

        # 1. Download Audio
        update_progress("Downloading podcast audio...")
        audio_file_path = download_audio_file(url, use_cookies, cookies) # Uses refactored download
        temp_files.append(audio_file_path)
        result["processing_source"] = audio_file_path # Update source to local path
        update_progress(f"Podcast downloaded: {audio_file_path}")

        # 2. Extract Metadata (Optional but useful for podcasts)
        try:
             update_progress("Attempting to extract metadata...")
             # Pass cookies if needed by extract_metadata
             metadata = extract_metadata(url, use_cookies=use_cookies, cookies=cookies)
             if metadata:
                  # Update result's metadata, prioritizing existing values if provided
                  result["metadata"]["title"] = result["metadata"].get("title") or metadata.get('title', Path(audio_file_path).stem)
                  result["metadata"]["author"] = result["metadata"].get("author") or metadata.get('uploader', 'Unknown Author')
                  result["metadata"]["series"] = metadata.get('series')
                  result["metadata"]["episode"] = metadata.get('episode')
                  result["metadata"]["season"] = metadata.get('season')
                  result["metadata"]["upload_date"] = metadata.get('upload_date')
                  result["metadata"]["duration"] = metadata.get('duration')
                  result["metadata"]["description"] = metadata.get('description')

                  # Augment keywords - handle existing string or list
                  current_keywords = result["metadata"].get("keywords") or ""
                  kw_list = []
                  if isinstance(current_keywords, str):
                       kw_list = [k.strip() for k in current_keywords.split(',') if k.strip()]
                  elif isinstance(current_keywords, list):
                       kw_list = current_keywords

                  if metadata.get('series'): kw_list.append(f"series:{metadata['series']}")
                  if metadata.get('episode'): kw_list.append(f"episode:{metadata['episode']}")
                  if metadata.get('season'): kw_list.append(f"season:{metadata['season']}")
                  # Add tags as keywords if available
                  tags = metadata.get('tags')
                  if isinstance(tags, list): kw_list.extend(tags)

                  result["metadata"]["keywords"] = list(set(kw_list)) # Store as unique list

                  update_progress(f"Metadata extracted: Title='{result['metadata']['title']}', Author='{result['metadata']['author']}'")
             else:
                  update_progress("No additional metadata extracted.")
                  # Ensure basic metadata from filename/input is present
                  result["metadata"]["title"] = result["metadata"].get("title") or Path(audio_file_path).stem
                  result["metadata"]["author"] = result["metadata"].get("author") or 'Unknown Author'
                  if isinstance(result["metadata"]["keywords"], str): # Ensure keywords is a list
                       result["metadata"]["keywords"] = [k.strip() for k in result["metadata"]["keywords"].split(',') if k.strip()]

        except Exception as meta_err:
             update_progress(f"Warning: Metadata extraction failed: {meta_err}")
             result["warnings"].append(f"Metadata extraction failed: {meta_err}")
             # Ensure basic metadata exists
             result["metadata"]["title"] = result["metadata"].get("title") or Path(audio_file_path).stem
             result["metadata"]["author"] = result["metadata"].get("author") or 'Unknown Author'
             if isinstance(result["metadata"]["keywords"], str): # Ensure keywords is a list
                  result["metadata"]["keywords"] = [k.strip() for k in result["metadata"]["keywords"].split(',') if k.strip()]


        # 3. Process Audio (Convert, Transcribe, Chunk, Summarize)
        #    Leverage the main process_audio_files function for consistency
        update_progress("Processing audio (transcription, analysis)...")
        # Pass the downloaded file path as input
        processing_result = process_audio_files(
            inputs=[audio_file_path], # Pass the downloaded path
            transcription_model=whisper_model,
            # transcription_language=... # Add if needed, defaults in process_audio_files
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            max_chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            chunk_language=chunk_language,
            diarize=enable_diarization,
            timestamp_option=keep_timestamps,
            perform_analysis=(api_name is not None and api_name.lower() != 'none'),
            api_name=api_name,
            api_key=api_key,
            custom_prompt_input=custom_prompt,
            system_prompt_input=system_prompt,
            summarize_recursively=summarize_recursively,
            keep_original=keep_original, # Let sub-function handle its temps if needed
            temp_dir=str(processing_temp_dir), # Pass down temp dir
             # Don't pass cookies down, download is done
        )

        # Merge results from process_audio_files back into our podcast result
        if processing_result and processing_result.get("results"):
            item_proc_result = processing_result["results"][0] # Get the result for the single item
            result["status"] = item_proc_result.get("status", "Error")
            result["transcript"] = item_proc_result.get("transcript")
            result["segments"] = item_proc_result.get("segments")
            result["summary"] = item_proc_result.get("summary")
            result["error"] = result.get("error") or item_proc_result.get("error") # Combine errors
            result["warnings"].extend(item_proc_result.get("warnings", []))
            result["analysis_details"].update(item_proc_result.get("analysis_details", {}))
            # Keep the richer metadata extracted earlier
            # result["metadata"] is already populated
            # Update processing_source if sub-process changed it (e.g., to WAV)
            result["processing_source"] = item_proc_result.get("processing_source", result["processing_source"])

            if result["status"] == "Error":
                 # If sub-processing failed, ensure top-level reflects it
                 update_progress(f"Audio processing failed: {result['error']}")
            else:
                 update_progress("Audio processing completed.")
        else:
            raise RuntimeError("process_audio_files returned unexpected or empty result.")

        # --- DB CALL REMOVED ---
        # No call to add_media_with_keywords here

        result["status"] = "Warning" if result.get("warnings") else result.get("status", "Success") # Final status update


    except Exception as e:
        error_message = f"Error processing podcast {url}: {type(e).__name__} - {str(e)}"
        update_progress(f"Processing failed: {error_message}")
        logging.error(error_message, exc_info=True)
        result["status"] = "Error"
        result["error"] = str(e)

    finally:
        _cleanup_temp_files()
        try:
            temp_directory_manager.cleanup()
            update_progress(f"Removed podcast temp directory: {processing_temp_dir}")
        except Exception as e:
             logging.warning(f"Could not remove podcast temp directory {processing_temp_dir}: {e}")


    processing_time = time.time() - start_time
    update_progress(f"Podcast processing finished. Status: {result['status']}. Time: {processing_time:.2f}s")
    # Add timing and progress log to result if desired
    result["processing_time_seconds"] = processing_time
    # result["progress_log"] = progress

    # Ensure metadata keywords is a list before returning
    if isinstance(result["metadata"].get("keywords"), str):
        result["metadata"]["keywords"] = [k.strip() for k in result["metadata"]["keywords"].split(',') if k.strip()]


    # Log metrics (optional, can be done in API layer too)
    metric_labels = {
        "whisper_model": whisper_model,
        "api_name": api_name or "None",
        "status": result["status"]
    }
    if result["status"] == "Error":
        log_counter("podcasts_failed_total", labels=metric_labels)
    else:
        log_counter("podcasts_processed_total", labels=metric_labels)
    log_histogram("podcast_processing_time_seconds", processing_time, labels=metric_labels)

    return result


#
#
#######################################################################################################################