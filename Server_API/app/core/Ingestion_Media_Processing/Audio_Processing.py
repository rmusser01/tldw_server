# file: Server_API/app/core/audio_processing.py

import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests

from App_Function_Libraries.DB.DB_Manager import (
    add_media_with_keywords,
    add_media_to_database,        # or whichever you'd like
    check_media_and_whisper_model,
    check_existing_media,
)
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization
from App_Function_Libraries.Chunk_Lib import improved_chunking_process
from App_Function_Libraries.Utils.Utils import (
    logging,
    create_download_directory,
    extract_text_from_segments,
    format_transcription,
)
from App_Function_Libraries.Audio.Audio_Transcription_Lib import speech_to_text, format_transcription_with_timestamps
from App_Function_Libraries.Video_DL_Ingestion_Lib import extract_metadata, download_video  # Reusable for audio?

# For ephemeral logic, we'll just skip DB calls if store_in_db=False. If you want an ephemeral storage approach,
# you can store results in an in-memory dict keyed by a UUID. The code below demonstrates skipping DB entirely.


def process_audio(
    # Core ingestion params
    urls: Optional[List[str]] = None,   # e.g. normal or “podcast” URLs
    local_files: Optional[List[str]] = None,  # paths to already-uploaded .mp3, .wav, etc.
    is_podcast: bool = False,
    # Transcription params
    whisper_model: str = "distil-large-v3",
    diarize: bool = False,
    keep_timestamps: bool = True,
    # Summarization params
    api_name: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None,
    chunk_method: Optional[str] = None,
    max_chunk_size: int = 300,
    chunk_overlap: int = 0,
    use_adaptive_chunking: bool = False,
    use_multi_level_chunking: bool = False,
    chunk_language: str = "english",
    # Additional params
    keywords: str = "",
    keep_original_audio: bool = False,
    use_cookies: bool = False,
    cookies: Optional[str] = None,
    store_in_db: bool = True,   # ephemeral if False
    custom_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified audio ingestion & processing function that can handle:
      - Normal audio files (URLs or local paths).
      - Podcasts (just set is_podcast=True to gather extra metadata if desired).
    Transcribes, optionally summarizes (if api_name != None),
    and optionally stores in DB (if store_in_db=True).

    Returns a dict with "results", "errors", etc., and the full transcripts/summaries for ephemeral usage.

    :param urls: List of remote URLs (could be normal .mp3 or “podcast”).
    :param local_files: List of local file paths to .mp3, .wav, etc.
    :param is_podcast: If True, we treat the URL as a “podcast,” attempt to gather extra metadata (title/author).
    :param whisper_model: The Whisper model name (e.g., "distil-large-v3").
    :param diarize: If True, enable speaker diarization.
    :param keep_timestamps: If True, keep timestamps in transcript.
    :param api_name: Summarization LLM name (e.g. "openai"), or None if no summarization.
    :param api_key: Summarization LLM API key, if any.
    :param custom_prompt: The user’s custom prompt for summarization, or None.
    :param chunk_method: "words", "sentences", etc.
    :param max_chunk_size: The max chunk size for summarization chunking.
    :param chunk_overlap: Overlap between chunks.
    :param use_adaptive_chunking: Whether to adapt chunk sizes based on complexity.
    :param use_multi_level_chunking: Whether to chunk in multiple passes.
    :param chunk_language: The language for chunking logic.
    :param keywords: A string of comma-separated keywords for DB or indexing.
    :param keep_original_audio: If False, we delete the downloaded file after processing.
    :param use_cookies: If True, parse the cookies param for authenticated downloads.
    :param cookies: A JSON string or similar representing cookies for the download.
    :param store_in_db: If True, store final transcripts in DB; else ephemeral.
    :param custom_title: If set, prefix or override the title in the DB.
    :return: dict with fields:
        {
          "processed_count": int,
          "errors_count": int,
          "errors": [...],
          "results": [
            {
              "input_item": str,
              "status": "Success" or "Error",
              "transcript": str or None,
              "summary": str or None,
              "db_id": optional db id
            }, ...
          ]
        }
    """

    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    all_inputs = []
    if urls:
        all_inputs.extend(urls)
    if local_files:
        all_inputs.extend(local_files)

    if not all_inputs:
        return {
            "processed_count": 0,
            "errors_count": 1,
            "errors": ["No inputs provided (no urls, no files)"],
            "results": []
        }

    # Prepare chunking
    chunk_options = {
        'method': chunk_method,
        'max_size': max_chunk_size,
        'overlap': chunk_overlap,
        'adaptive': use_adaptive_chunking,
        'multi_level': use_multi_level_chunking,
        'language': chunk_language
    }

    for audio_input in all_inputs:
        start_time = time.time()
        try:
            # 1) Distinguish remote vs. local
            is_remote = audio_input.startswith(("http://", "https://"))
            download_path = None

            # 2) Possibly skip DB if we see the same item with the same whisper model
            #    (But only if store_in_db==True.  If ephemeral, we always re-process.)
            if store_in_db and is_remote:
                # check if we already have it
                media_exists, reason = check_media_and_whisper_model(
                    url=audio_input,
                    current_whisper_model=whisper_model
                )
                if not media_exists:
                    logging.info(f"Proceeding with new audio: {reason}")
                else:
                    # If it’s the same whisper model, skip.
                    # If you want to allow overwrite, handle that here:
                    return_item = {
                        "input_item": audio_input,
                        "status": "Error",
                        "error": f"Already processed with same model: {reason}",
                    }
                    results.append(return_item)
                    errors.append(return_item["error"])
                    continue

            # 3) If remote, download
            if is_remote:
                # Re-use your existing “download_video()” or “download_audio_file()” approach
                # or unify them. For demonstration, I'll do a small snippet:
                download_path = download_video(
                    audio_input,  # ironically named, but it’s just a link
                    create_download_directory("Audio_Downloads"),
                    full_info=None,
                    download_video_flag=False,  # just audio if you want
                    current_whisper_model=whisper_model
                )
                if not download_path:
                    raise RuntimeError("Download returned None")

            else:
                # local file
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Local file not found: {audio_input}")
                download_path = audio_input

            # 4) If “is_podcast,” extract metadata for e.g. title/author
            title = custom_title or os.path.basename(download_path)
            author = "Unknown"
            if is_podcast and is_remote:
                meta = extract_metadata(audio_input, use_cookies, cookies)
                if meta:
                    title = meta.get("title", title)  # override if found
                    author = meta.get("uploader", author)

            # 5) Transcribe
            segments = speech_to_text(
                audio_file_path=download_path,
                whisper_model=whisper_model,
                diarize=diarize
            )
            if isinstance(segments, dict) and "segments" in segments:
                segments = segments["segments"]
            if not isinstance(segments, list):
                raise RuntimeError(f"Unexpected transcription result: {segments}")
            # Possibly keep or remove timestamps
            transcript = format_transcription_with_timestamps(segments, keep_timestamps=keep_timestamps)

            # 6) Summarize if api_name is set
            summary_text = None
            if api_name and api_name.lower() != "none":
                # chunk + summarize
                chunked_texts = improved_chunking_process(transcript, chunk_options)
                summary_text = perform_summarization(api_name, chunked_texts, custom_prompt, api_key)
            else:
                summary_text = "[No summary requested]"

            # 7) Possibly store in DB
            media_id = None
            if store_in_db:
                # call e.g. add_media_with_keywords or add_media_to_database
                # up to you how to store.
                ingestion_date = datetime.now().strftime('%Y-%m-%d')
                # If you prefer the add_media_with_keywords approach:
                add_res = add_media_with_keywords(
                    url=(audio_input if is_remote else "local-file"),
                    title=title,
                    media_type="podcast" if is_podcast else "audio",
                    content=transcript,
                    keywords=keywords,
                    prompt=custom_prompt or "",
                    summary=summary_text,
                    transcription_model=whisper_model,
                    author=author,
                    ingestion_date=ingestion_date
                )
                # If add_media_with_keywords returns an ID, capture it:
                if isinstance(add_res, dict) and "id" in add_res:
                    media_id = add_res["id"]

            # 8) If keep_original_audio=False and we downloaded the file, remove it
            if is_remote and (not keep_original_audio) and download_path and os.path.exists(download_path):
                try:
                    os.remove(download_path)
                except Exception as e:
                    logging.warning(f"Could not remove downloaded audio: {e}")

            # 9) Log success metrics
            processing_time = time.time() - start_time
            log_counter("audio_ingestion_success", labels={"whisper_model": whisper_model})
            log_histogram("audio_ingestion_time_seconds", processing_time, labels={"whisper_model": whisper_model})

            results.append({
                "input_item": audio_input,
                "status": "Success",
                "transcript": transcript,
                "summary": summary_text,
                "db_id": media_id
            })

        except Exception as e:
            error_msg = f"Error processing '{audio_input}': {e}"
            logging.error(error_msg)
            errors.append(error_msg)
            results.append({
                "input_item": audio_input,
                "status": "Error",
                "error": str(e)
            })
            log_counter("audio_ingestion_failure", labels={"whisper_model": whisper_model})

    return {
        "processed_count": len(results),
        "errors_count": len(errors),
        "errors": errors,
        "results": results
    }
