# Video_DL_Ingestion_Lib.py
#########################################
# Video Downloader and Ingestion Library
# This library is used to handle downloading videos from YouTube and other platforms.
# It also handles the ingestion of the videos into the database.
# It uses yt-dlp to extract video information and download the videos.
####
import json
####################
# Function List
#
# 1. get_video_info(url)
# 2. create_download_directory(title)
# 3. sanitize_filename(title)
# 4. normalize_title(title)
# 5. get_youtube(video_url)
# 6. get_playlist_videos(playlist_url)
# 7. download_video(video_url, download_path, info_dict, download_video_flag)
# 8. save_to_file(video_urls, filename)
# 9. save_summary_to_file(summary, file_path)
# 10. process_url(url, num_speakers, whisper_model, custom_prompt, offset, api_name, api_key, vad_filter, download_video, download_audio, rolling_summarization, detail_level, question_box, keywords, chunk_summarization, chunk_duration_input, words_per_second_input)
#
#
####################
# Import necessary libraries to run solo for testing
import os
import re
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs
#
# 3rd-Party Imports
import unicodedata
import yt_dlp
from loguru import logger
# Import Local
from tldw_Server_API.app.core.Evaluations.ms_g_eval import run_geval
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import perform_transcription
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import summarize
from tldw_Server_API.app.core.Utils.Utils import (
    convert_to_seconds,
    extract_text_from_segments,
    logging
)
from tldw_Server_API.app.core.Utils.Chunk_Lib import improved_chunking_process
from tldw_Server_API.app.core.Metrics.metrics_logger import (
    log_counter, log_histogram
)
#
#######################################################################################################################
# Function Definitions
#

# ffmpeg check
try:
    # Adjust .parent calls based on your actual structure to reach the project root
    # Example: If this file is in app/core/Ingestion/Video/
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
except NameError: # Fallback if __file__ is not defined
    PROJECT_ROOT = Path(os.getcwd())
    logging.warning(f"Could not determine project root from __file__, falling back to CWD: {PROJECT_ROOT}")

def normalize_title(title):
    # Normalize the string to 'NFKD' form and encode to 'ascii' ignoring non-ascii characters
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('ascii')
    title = title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('"', '').replace('*', '').replace('?',
                                                                                                                   '').replace(
        '<', '').replace('>', '').replace('|', '')
    return title

def get_video_info(url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url, download=False)
            return info_dict
        except Exception as e:
            logging.error(f"Error extracting video info: {e}")
            return None


def get_youtube(video_url):
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'noplaylist': False,
        'quiet': True,
        'extract_flat': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        logging.debug("About to extract youtube info")
        info_dict = ydl.extract_info(video_url, download=False)
        logging.debug("Youtube info successfully extracted")
    return info_dict


def get_playlist_videos(playlist_url):
    ydl_opts = {
        'extract_flat': True,
        'skip_download': True,
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

        if 'entries' in info:
            video_urls = [entry['url'] for entry in info['entries']]
            playlist_title = info['title']
            return video_urls, playlist_title
        else:
            print("No videos found in the playlist.")
            return [], None


def download_video(video_url, download_path, info_dict, download_video_flag, current_whisper_model=None):
    """
    Downloads video or audio using yt-dlp with refined option handling.
    """
    if not yt_dlp:
        logging.error("yt-dlp module not available, cannot download.")
        return None

    # --- 1. Determine Filename and Extension ---
    title = "unknown_video"
    ext = "tmp" # Default temporary extension

    if info_dict and isinstance(info_dict, dict):
        title = info_dict.get('title', title)
        ext = info_dict.get('ext') # Can be None
    else:
        logging.warning("info_dict missing or invalid, using fallbacks for title/ext.")
        # Attempt fallback from URL (less reliable)
        try:
             path_part = Path(urlparse(video_url).path)
             if path_part.stem:
                 title = path_part.stem
             if path_part.suffix:
                 ext = path_part.suffix[1:]
        except Exception:
             logging.warning("Could not parse URL for fallback title/ext.")

    normalized_video_title = normalize_title(title)
    unique_suffix = uuid.uuid4().hex[:8]
    download_path_obj = Path(download_path)

    # Define target extension and preferred codec based on download type
    valid_audio_codecs = {'m4a', 'mp3', 'opus', 'wav', 'aac', 'ogg', 'flac'}
    preferred_codec = 'm4a' # Default audio codec
    target_ext = 'mp4'      # Default overall target

    if download_video_flag:
        target_ext = 'mp4' # Video+Audio target is MP4
        logging.debug("Download type: Video+Audio")
    else: # Audio only requested
        logging.debug(f"Download type: Audio only. Extracted ext hint: '{ext}'")
        local_ext = ext or 'm4a' # Use hint or default to m4a
        if local_ext.lower() in valid_audio_codecs:
             target_ext = local_ext.lower() # Use the valid audio extension
             preferred_codec = target_ext    # Use it as the preferred codec
             logging.debug(f"Using valid extracted audio extension: {target_ext}")
        else:
             target_ext = 'm4a' # Fallback if hint wasn't a valid audio type
             preferred_codec = 'm4a'
             logging.debug(f"Extracted ext '{ext}' not a recognized audio codec, using fallback: {target_ext}")

    # Final output path construction
    final_output_path = download_path_obj / f"{normalized_video_title}_{unique_suffix}.{target_ext}"
    final_output_path_str = str(final_output_path)
    logging.debug(f"Generated unique target path: {final_output_path_str}")

    if final_output_path.exists():
        logging.warning(f"Target file already exists: {final_output_path_str}. Skipping download.")
        return final_output_path_str

    # --- 2. Setup ffmpeg Path ---
    ffmpeg_path = None
    # (Keep the previous ffmpeg path detection logic here)
    if sys.platform.startswith('win'):
        local_ffmpeg = Path(os.getcwd()) / 'Bin' / 'ffmpeg.exe'
        if local_ffmpeg.exists():
            ffmpeg_path = str(local_ffmpeg)
        else:
            logging.warning(f"Local ffmpeg not found at '{local_ffmpeg}', trying 'ffmpeg' from PATH.")
            ffmpeg_path = 'ffmpeg'
    elif sys.platform.startswith(('linux', 'darwin')):
        ffmpeg_path = 'ffmpeg'
    else:
        logging.warning(f"Unsupported platform {sys.platform} for ffmpeg path detection.")
        ffmpeg_path = 'ffmpeg'

    # Optional: Confirm ffmpeg works (can be removed if causing issues)
    try:
        subprocess.run([ffmpeg_path, '-version'], check=True, capture_output=True, timeout=5)
        logging.debug(f"Confirmed ffmpeg command: {ffmpeg_path}")
    except (FileNotFoundError, subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired) as ffmpeg_err:
        logging.warning(f"ffmpeg command '{ffmpeg_path}' check failed: {ffmpeg_err}. yt-dlp might still work or fail later.")
        # Don't necessarily fail here, let yt-dlp try

    # --- 3. Construct yt-dlp Options ---
    ydl_opts = None
    # Using a temporary path template for yt-dlp, as postprocessors rename the file
    # Note: yt-dlp handles the final renaming to the 'outtmpl' name *if* no postprocessor runs or if it merges.
    # When FFmpegExtractAudio runs, it often uses its own naming based on the original + codec.
    # We will check for the final_output_path later.
    base_template = str(download_path_obj / f"{normalized_video_title}_{unique_suffix}")

    if download_video_flag:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': {'default': final_output_path_str}, # Target the final mp4 path directly
            'ffmpeg_location': ffmpeg_path,
            'quiet': True, 'no_warnings': True,
            'merge_output_format': 'mp4', # Ensure merged output is mp4
        }
        log_msg = "yt_dlp: Downloading video and audio..."
    else: # Audio only
         ydl_opts = {
             'format': 'bestaudio/best',
             # Let yt-dlp determine intermediate name, postprocessor handles final codec/ext
             'outtmpl': {'default': base_template + '.%(ext)s'}, # Template for download before postprocessing
             'ffmpeg_location': ffmpeg_path,
             'quiet': True, 'no_warnings': True,
             'postprocessors': [{
                 'key': 'FFmpegExtractAudio',
                 'preferredcodec': preferred_codec, # Use the valid audio codec
                 # No need to specify output path here, FFmpegExtractAudio uses preferredcodec for ext
             }],
             'keepvideo': False, # Don't keep original if only audio is needed
         }
         log_msg = f"yt_dlp: Downloading and extracting audio only (codec: {preferred_codec})..."

    # --- 4. Execute Download ---
    if ydl_opts is None:
        logging.error("Logic error: ydl_opts was not set.")
        return None

    logging.debug(f"Attempting download with ydl_opts: {ydl_opts}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logging.debug(log_msg)
            ydl.download([video_url])
            logging.debug(f"yt_dlp: Download/extraction process finished.")

            # --- 5. Verify Output File ---
            # Check if the *expected* final file exists after processing
            if final_output_path.exists():
                logging.info(f"Successfully obtained media file: {final_output_path_str}")
                return final_output_path_str
            else:
                # If audio extraction happened, the file might have a slightly different name
                # Let's search for the file based on the unique part
                found_files = list(download_path_obj.glob(f"{normalized_video_title}_{unique_suffix}.*"))
                # Filter out potentially incomplete '.part' files
                valid_files = [f for f in found_files if not f.name.endswith('.part')]

                if len(valid_files) == 1:
                     actual_path = str(valid_files[0])
                     logging.warning(f"Expected path '{final_output_path_str}' not found, but found unique match '{actual_path}'. Using it.")
                     # Optionally rename to expected name if desired, but returning actual path is safer
                     # try:
                     #     valid_files[0].rename(final_output_path)
                     #     logging.info(f"Renamed '{actual_path}' to '{final_output_path_str}'")
                     #     return final_output_path_str
                     # except OSError as rename_err:
                     #     logging.error(f"Failed to rename downloaded file: {rename_err}")
                     #     return actual_path # Return path found even if rename failed
                     return actual_path
                elif len(valid_files) > 1:
                     logging.error(f"Multiple potential output files found after download for pattern '{normalized_video_title}_{unique_suffix}.*': {valid_files}. Cannot determine correct file.")
                     return None
                else:
                     logging.error(f"yt_dlp: Target file '{final_output_path_str}' (or variations) not found after download/postprocessing.")
                     return None

    except yt_dlp.utils.DownloadError as e:
        # More specific download errors (network, unavailable video etc.)
        logging.error(f"yt_dlp: DownloadError for {video_url}: {e}")
        return None
    except Exception as e:
        # Catches other errors (like potential init errors, unexpected issues)
        logging.error(f"yt_dlp: Unexpected error during download/processing for {video_url}: {e}", exc_info=True)
        # No UnboundLocalError should happen now if ydl_opts logic is correct
        return None


def extract_video_info(url):
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)

            # Log only a subset of the info to avoid overwhelming the logs
            log_info = {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'upload_date': info.get('upload_date')
            }
            logging.debug(f"Extracted info for {url}: {log_info}")

            return info
    except Exception as e:
        logging.error(f"Error extracting video info for {url}: {str(e)}", exc_info=True)
        return None


def get_youtube_playlist_urls(playlist_id):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(f'https://www.youtube.com/playlist?list={playlist_id}', download=False)
        return [entry['url'] for entry in result['entries'] if entry.get('url')]


def parse_and_expand_urls(urls):
    logging.info(f"Starting parse_and_expand_urls with input: {urls}")
    expanded_urls = []

    for url in urls:
        try:
            logging.info(f"Processing URL: {url}")
            parsed_url = urlparse(url)
            logging.debug(f"Parsed URL components: {parsed_url}")

            # YouTube playlist handling
            if 'youtube.com' in parsed_url.netloc and 'list' in parsed_url.query:
                playlist_id = parse_qs(parsed_url.query)['list'][0]
                logging.info(f"Detected YouTube playlist with ID: {playlist_id}")
                playlist_urls = get_youtube_playlist_urls(playlist_id)
                logging.info(f"Expanded playlist URLs: {playlist_urls}")
                expanded_urls.extend(playlist_urls)

            # YouTube short URL handling
            elif 'youtu.be' in parsed_url.netloc:
                video_id = parsed_url.path.lstrip('/')
                full_url = f'https://www.youtube.com/watch?v={video_id}'
                logging.info(f"Expanded YouTube short URL to: {full_url}")
                expanded_urls.append(full_url)

            # Vimeo handling
            elif 'vimeo.com' in parsed_url.netloc:
                video_id = parsed_url.path.lstrip('/')
                full_url = f'https://vimeo.com/{video_id}'
                logging.info(f"Processed Vimeo URL: {full_url}")
                expanded_urls.append(full_url)

            # Add more platform-specific handling here

            else:
                logging.info(f"URL not recognized as special case, adding as-is: {url}")
                expanded_urls.append(url)

        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}", exc_info=True)
            # Optionally, you might want to add the problematic URL to expanded_urls
            # expanded_urls.append(url)

    logging.info(f"Final expanded URLs: {expanded_urls}")
    return expanded_urls


def extract_metadata(url, use_cookies=False, cookies=None):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'skip_download': True,
    }

    if use_cookies and cookies:
        try:
            cookie_dict = json.loads(cookies)
            ydl_opts['cookiefile'] = cookie_dict
        except json.JSONDecodeError:
            logging.warning("Invalid cookie format. Proceeding without cookies.")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            metadata = {
                'title': info.get('title'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'duration': info.get('duration'),
                'tags': info.get('tags'),
                'description': info.get('description')
            }

            # Create a safe subset of metadata to log
            safe_metadata = {
                'title': metadata.get('title', 'No title'),
                'duration': metadata.get('duration', 'Unknown duration'),
                'upload_date': metadata.get('upload_date', 'Unknown upload date'),
                'uploader': metadata.get('uploader', 'Unknown uploader')
            }

            logging.info(f"Successfully extracted metadata for {url}: {safe_metadata}")
            return metadata
        except Exception as e:
            logging.error(f"Error extracting metadata for {url}: {str(e)}", exc_info=True)
            return None


def generate_timestamped_url(url, hours, minutes, seconds):
    # Extract video ID from the URL
    video_id_match = re.search(r'(?:v=|)([0-9A-Za-z_-]{11}).*', url)
    if not video_id_match:
        return "Invalid YouTube URL"

    video_id = video_id_match.group(1)

    # Calculate total seconds
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds)

    # Generate the new URL
    new_url = f"https://www.youtube.com/watch?v={video_id}&t={total_seconds}s"

    return new_url


# New FastAPI ingestion functions
def process_videos(
    inputs: List[str],
    start_time: Optional[str],
    end_time: Optional[str],
    diarize: bool,
    vad_use: bool,
    transcription_model: str,
    transcription_language: Optional[str],
    perform_analysis: bool,
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    perform_chunking: bool,
    chunk_method: Optional[str],
    max_chunk_size: int,
    chunk_overlap: int,
    use_adaptive_chunking: bool,
    use_multi_level_chunking: bool,
    chunk_language: Optional[str],
    summarize_recursively: bool,
    api_name: Optional[str],
    api_key: Optional[str],
    use_cookies: bool,
    cookies: Optional[str],
    timestamp_option: bool,
    perform_confabulation_check: bool, # Renamed from confab_checkbox
    temp_dir: Optional[str] = None, # Added temp_dir argument
    keep_original: bool = False, # Add if needed for intermediate files
    perform_diarization:bool = False,
) -> Dict[str, Any]:
    """
    Processes multiple videos or local file paths, transcribes, summarizes,
    and optionally stores in the DB (if store_in_db=True).

    This function was adapted from your old `process_videos_with_error_handling()`
    but with Gradio references removed.

    :param inputs: A list of either URLs or local file paths.
    :param start_time: Start time for partial transcription (e.g. "1:30" or "90").
    :param end_time: End time for partial transcription.
    :param diarize: Enable speaker diarization.
    :param vad_use: Enable Voice Activity Detection.
    :param transcription_model: Name of the transcription model to use.
    :param transcription_language: Language for transcription.
    :param perform_analysis: If True, perform analysis on the transcript.
    :param custom_prompt: The user’s custom text prompt for summarization.
    :param system_prompt: The system prompt for the LLM.
    :param perform_chunking: If True, break transcripts into chunks before summarizing.
    :param chunk_method: "words", "sentences", etc.
    :param max_chunk_size: Maximum chunk size for chunking.
    :param chunk_overlap: Overlap size for chunking.
    :param use_adaptive_chunking: Whether to adapt chunk sizes by text complexity.
    :param use_multi_level_chunking: If True, chunk in multiple passes.
    :param chunk_language: The language for chunking logic.
    :param summarize_recursively: If True, do multi-pass summarization of chunk summaries.
    :param api_name: The LLM API name (e.g., "openai").
    :param api_key: The user’s (or system) API key for the LLM.
    :param use_cookies: If True, use cookies for authenticated video downloads.
    :param cookies: The user-supplied cookies in JSON or Netscape format.
    :param timestamp_option: If True, keep timestamps in final transcript.
    :param perform_confabulation_check: If True, run confabulation check on the summary.
    :param keep_original: If True, keep the downloaded file
    :param perform_diarization: If True, perform diarization on inputs
    :return: A dict with the overall results, e.g.:
             {
               "processed_count": int,
               "errors_count": int,
               "errors": [...],
               "results": [...],
               "confabulation_results": "..."
             }
    """
    logging.info(f"Starting process_videos (DB-agnostic) for {len(inputs)} inputs.")
    errors = []
    results = []
    all_transcripts_for_confab = {} # Renamed for clarity
    all_summaries_for_confab = {} # Renamed for clarity

    # Save all transcriptions and summaries to these dict/strings:
    all_transcriptions = {}
    all_summaries = ""

    # Convert user times to seconds
    start_seconds = convert_to_seconds(start_time) if start_time else 0
    end_seconds = convert_to_seconds(end_time) if end_time else None

    # If user typed no inputs, bail out
    if not inputs:
        logging.warning("No input provided to process_videos()")
        return {
            "processed_count": 0,
            "errors_count": 1,
            "errors": ["No inputs provided."],
            "results": []
        }

    # Enforce temp_dir usage
    if not temp_dir:
        # If None is passed despite Fix #1, something is wrong upstream.
        logging.error("CRITICAL: process_videos called without a valid temp_dir path.")
        # Return an error immediately or raise, depending on desired behavior
        return {
            "processed_count": 0,
            "errors_count": len(inputs),
            "errors": ["Internal Error: Processing temporary directory was not provided."],
            "results": [{"status": "Error", "input_ref": inp, "error": "Internal processing setup error"} for inp in inputs]
        }
    processing_temp_dir = Path(temp_dir)
    # Ensure the directory exists (it should, as TempDirManager creates it)
    if not processing_temp_dir.is_dir():
         logging.error(f"CRITICAL: Provided temp_dir '{processing_temp_dir}' does not exist or is not a directory.")
         # Handle error appropriately
         return {
             "processed_count": 0, "errors_count": len(inputs),
             "errors": [f"Internal Error: Invalid temporary directory '{processing_temp_dir}'."],
             "results": [{"status": "Error", "input_ref": inp, "error": "Internal processing setup error"} for inp in inputs]
         }
    logging.info(f"process_videos using provided temporary directory: {processing_temp_dir}")

    for video_input in inputs:
        video_start_time = datetime.now()
        try:
            # Pass necessary parameters down, including temp_dir
            single_result = process_single_video(
                video_input=video_input,
                start_seconds=start_seconds,
                end_seconds=end_seconds, # Pass end_seconds down
                diarize=diarize,
                vad_use=vad_use,
                transcription_model=transcription_model,
                transcription_language=transcription_language,
                perform_analysis=perform_analysis,
                custom_prompt=custom_prompt,
                system_prompt=system_prompt,
                perform_chunking=perform_chunking,
                chunk_method=chunk_method,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                use_adaptive_chunking=use_adaptive_chunking,
                use_multi_level_chunking=use_multi_level_chunking,
                chunk_language=chunk_language,
                summarize_recursively=summarize_recursively,
                api_name=api_name,
                api_key=api_key,
                use_cookies=use_cookies,
                cookies=cookies,
                timestamp_option=timestamp_option,
                temp_dir=str(processing_temp_dir), # Pass temp dir path
                keep_intermediate_audio=False, # Pass if needed
                perform_diarization=perform_diarization,
            )

            results.append(single_result) # Append regardless of status

            if single_result.get("status") == "Success":
                log_counter(...) # Metrics are fine if DB-free

                # Prepare for potential confabulation check
                transcript_text = single_result.get("transcript", "") # Use 'transcript' key returned by single
                summary_text = single_result.get("summary", "") # Use 'summary' key returned by single
                if transcript_text and summary_text:
                     all_transcripts_for_confab[video_input] = transcript_text
                     all_summaries_for_confab[video_input] = summary_text

                # Logging the timing
                video_end_time = datetime.now()
                processing_time = (video_end_time - video_start_time).total_seconds()
                log_histogram(
                    metric_name="video_processing_time_seconds",
                    value=processing_time,
                    labels={"whisper_model": transcription_model, "api_name": (api_name or "none")}
                )
            elif single_result.get("status") == "Error":
                # If status is "Error"
                if single_result.get("status") == "Error":
                    errors.append(single_result.get("error", "Unknown processing error"))

                # Log failure metric
                log_counter(
                    metric_name="videos_failed_total",
                    labels={"whisper_model": transcription_model, "api_name": (api_name or "none")},
                    value=1
                )
            elif single_result.get("status") == "Warning":
                # If status is "Warning"
                warnings = single_result.get("warnings", [])
                if warnings:
                    errors.extend(warnings)

        except Exception as exc:
            msg = f"Exception processing '{video_input}': {exc}"
            logging.error(msg, exc_info=True)
            errors.append(msg)
            # Append an error result structure
            results.append({
                "status": "Error",
                "input_ref": video_input,
                "processing_source": video_input,
                "media_type": "video",
                "error": msg,
                # Fill other fields with None/defaults
                "metadata": {}, "transcript": None, "segments": None, "chunks": None, "summary": None,
                "analysis_details": None, "warnings": None
            })
            log_counter("videos_failed_total", ...)

            # Log failure metric
            log_counter(
                metric_name="videos_failed_total",
                labels={"whisper_model": transcription_model, "api_name": (api_name or 'none')},
                value=1
            )

    # --- Recalculate counts based on the correctly populated 'results' list ---
    processed_count_calc = sum(1 for r in results if r.get("status") == "Success")
    errors_count_calc = sum(1 for r in results if r.get("status") == "Error")
    warnings_count_calc = sum(1 for r in results if r.get("status") == "Warning")

    # Optionally, run a confabulation check on the entire set of summaries
    confabulation_results = None
    if confabulation_results and all_transcriptions:
        confab_results = []
        # Process each transcript-summary pair individually for g_eval check
        for url, transcript in all_transcriptions.items():
            # Extract the corresponding summary for this URL
            url_pattern = f"Video Input: {re.escape(url)}\nTranscription:.*?\nSummary:\n(.*?)\n\n---\n\n"
            summary_match = re.search(url_pattern, all_summaries, re.DOTALL)

            if summary_match:
                # FIXME - validate this call
                individual_summary = summary_match.group(1)
                # Create single-item collections for this transcript-summary pair
                single_transcript_dict = f"URL: + {url} : {transcript}"
                single_summary = f"Video Input: {url}\nTranscription:\n{transcript}\n\nSummary:\n{individual_summary}\n\n"

                # Run g_eval on this single pair
                pair_result = run_geval(single_transcript_dict, single_summary, api_key, api_name)
                confab_results.append(f"URL: {url} - {pair_result}")
            else:
                logging.warning(f"Could not find matching summary for URL: {url}")

        confabulation_results = f"Confabulation checks completed:\n" + "\n".join(confab_results)

    logger.debug(
        f"process_videos DEBUG: Final results list before return: {json.dumps(results, indent=2, default=str)}")
    logger.debug(f"process_videos DEBUG: Calculated processed_count: {processed_count_calc}")
    logger.debug(f"process_videos DEBUG: Calculated errors_count: {errors_count_calc}")

    return {
        "processed_count": processed_count_calc,
        "errors_count": errors_count_calc,
        "warnings_count": warnings_count_calc,
        "errors": errors, # List of error messages
        "results": results,
        "confabulation_results": confabulation_results
    }


def process_single_video(
    video_input: str,
    start_seconds: int,
    end_seconds: Optional[int], # Ensure this is used if needed by transcription/summarization
    diarize: bool,
    vad_use: bool,
    transcription_model: str,
    transcription_language: Optional[str],
    perform_analysis: bool,
    custom_prompt: Optional[str],
    system_prompt: Optional[str],
    perform_chunking: bool,
    chunk_method: Optional[str],
    max_chunk_size: int,
    chunk_overlap: int,
    use_adaptive_chunking: bool,
    use_multi_level_chunking: bool,
    chunk_language: Optional[str],
    summarize_recursively: bool,
    api_name: Optional[str],
    api_key: Optional[str],
    use_cookies: bool,
    cookies: Optional[str],
    timestamp_option: bool,
    temp_dir: str, # Expect temp_dir path from caller (e.g., TempDirManager context)
    keep_intermediate_audio: bool = False, # Flag to keep the WAV file from transcription
    perform_diarization: bool = False, # Flag to perform diarization
) -> Dict[str, Any]:
    """
    Processes a single video/file: Extracts metadata, downloads if URL,
    transcribes, optionally summarizes.
    Returns a dict matching MediaItemProcessResponse structure.
    'input_ref' should hold the original URL/path passed in video_input.
    'processing_source' should hold the path of the file actually processed.
    """
    # --- Initialize result with the ORIGINAL input reference ---
    processing_result = {
        "status": "Pending",
        "input_ref": video_input,  # Store the original URL or path here
        "processing_source": video_input, # Start with original, update if downloaded/copied
        "media_type": "video",
        "metadata": {},
        "content": "", # Corresponds to 'transcript'
        "segments": None,
        "chunks": None,
        "analysis": None, # Corresponds to 'summary'
        "analysis_details": {},
        "error": None,
        "warnings": [],
    }
    local_file_path_for_transcription = None
    # Temp dir for download is provided by the caller (`temp_dir`)

    try:
        logger.info(f"Processing single video input: {video_input}") # Log original
        is_remote = urlparse(video_input).scheme in ('http', 'https')
        processing_temp_dir = Path(temp_dir)

        # 1. Get Metadata & Determine LOCAL Processing Path
        if is_remote:
            logger.info("Input is URL. Extracting metadata and downloading...")
            info_dict = extract_metadata(video_input, use_cookies, cookies)
            if not info_dict:
                raise ValueError(f"Failed to extract metadata for URL: {video_input}")
            processing_result["metadata"] = info_dict
            logger.debug(f"Metadata extracted for {video_input}")

            download_target_dir_str = str(processing_temp_dir)
            logger.info(f"Downloading URL to directory: {download_target_dir_str}")
            download_audio_only_flag = True
            downloaded_path = download_video(
                video_url=video_input,
                download_path=download_target_dir_str,
                info_dict=info_dict,
                download_video_flag=not download_audio_only_flag,
            )

            if not downloaded_path or not os.path.exists(downloaded_path):
                raise FileNotFoundError(f"Download failed or file not found (target in {download_target_dir_str}) for URL: {video_input}")

            local_file_path_for_transcription = downloaded_path
            # *** Update only the processing_source, keep original input_ref ***
            processing_result["processing_source"] = local_file_path_for_transcription
            logger.info(f"Download successful. Using local path: {local_file_path_for_transcription}")

        else:
            # Input is already a local file path
            if not os.path.exists(video_input):
                raise FileNotFoundError(f"Local file not found: {video_input}")
            local_file_path_for_transcription = video_input
            # *** Update only the processing_source, keep original input_ref ***
            processing_result["processing_source"] = local_file_path_for_transcription
            # Extract/create minimal metadata for local files if not already present
            if not processing_result.get("metadata"):
                 # Basic info; could potentially use ffprobe or similar for more details if needed
                 info_dict = {
                     "title": Path(video_input).stem,
                     "description": "Local file",
                     "webpage_url": f"local://{Path(video_input).resolve()}",
                     # Add other fields as None or extract if possible
                 }
                 processing_result["metadata"] = info_dict
            logger.info(f"Input is local file: {local_file_path_for_transcription}")

        # 2. Perform Transcription using the LOCAL file path
        logging.info(f"Calling perform_transcription with LOCAL path: {local_file_path_for_transcription}")
        # Ensure perform_transcription is correctly imported
        # Note: Pass the PROCESSING TEMP DIR to perform_transcription if it needs
        # a place to put its *own* intermediate files (like the WAV).
        # Check the signature of perform_transcription. Assuming it takes `temp_dir` now.
        intermediate_wav_path, segments = perform_transcription(
            video_path=local_file_path_for_transcription, # THE LOCAL PATH
            offset=start_seconds,
            # end_seconds=end_seconds, # Pass if perform_transcription supports it
            transcription_model=transcription_model,
            vad_use=vad_use,
            diarize=diarize,
            overwrite=False, # Usually False for safety unless specifically needed
            transcription_language=transcription_language,
            temp_dir=str(processing_temp_dir) # Pass temp dir for its use
        )

        # Check transcription results carefully
        if segments is None:
            error_msg = "Transcription failed (returned None segments)."
            # Check if intermediate_wav_path holds error info (depends on perform_transcription impl.)
            if isinstance(intermediate_wav_path, dict) and 'error' in intermediate_wav_path:
                error_msg = f"Transcription failed: {intermediate_wav_path['error']}"
            elif isinstance(intermediate_wav_path, str) and "error" in intermediate_wav_path.lower():
                # Less ideal check if it returns error string in path var
                error_msg = f"Transcription failed: {intermediate_wav_path}"

            processing_result.update({"status": "Error", "error": error_msg})
            logger.error(error_msg + f" Input: {video_input}")
            return processing_result # Return early on transcription failure

        logger.info(f"Transcription successful for {local_file_path_for_transcription}")
        processing_result["segments"] = segments
        processing_result["content"] = extract_text_from_segments(segments, include_timestamps=timestamp_option)
        processing_result["analysis_details"]["whisper_model"] = transcription_model
        processing_result["analysis_details"]["transcription_language"] = transcription_language
        # Add other relevant details like diarize, vad_use if needed

        # Cleanup intermediate audio file created by transcription (if applicable)
        if not keep_intermediate_audio and intermediate_wav_path and os.path.exists(intermediate_wav_path):
             try:
                 os.remove(intermediate_wav_path)
                 logger.debug(f"Removed intermediate transcription audio file: {intermediate_wav_path}")
             except Exception as e:
                 warn_msg = f"Failed to remove intermediate audio file: {intermediate_wav_path} ({e})"
                 logging.warning(warn_msg)
                 processing_result["warnings"].append(warn_msg)

        # 3. Format Transcript (Content)
        # Possibly strip timestamps based on flag
        if not timestamp_option and isinstance(segments, list):
            logger.debug("Removing timestamps from segments.")
            for seg in segments:
                # Using .pop with default None avoids errors if keys are missing
                seg.pop("Time_Start", None)
                seg.pop("Time_End", None)
                seg.pop("start", None) # Check for alternative keys used by whisper
                seg.pop("end", None)

        # Prepare main 'content' string
        transcription_text = extract_text_from_segments(segments, include_timestamps=timestamp_option)
        processing_result["content"] = transcription_text
        if not transcription_text:
             warn_msg = "Transcription resulted in empty text content."
             logging.warning(warn_msg)
             processing_result["warnings"].append(warn_msg)

        # 4. Analysis (Chunking & Summarization) if requested and content exists
        analysis_text = None
        if perform_analysis and api_name and api_name.lower() != "none" and transcription_text:
            processing_result["analysis_details"]["llm_api"] = api_name
            processing_result["analysis_details"]["custom_prompt"] = custom_prompt
            processing_result["analysis_details"]["system_prompt"] = system_prompt

            # Maybe add metadata context to the text before chunking/summarizing?
            # text_context = f"Title: {processing_result['metadata'].get('title', 'N/A')}\n" \
            #                f"Author: {processing_result['metadata'].get('uploader', 'N/A')}\n\n" \
            #                f"{transcription_text}"
            text_to_analyze = transcription_text # Start with transcript

            if perform_chunking:
                logger.info(f"Performing chunking for {local_file_path_for_transcription}")
                chunk_opts = {
                    'method': chunk_method or 'recursive', # Default if None
                    'max_size': max_chunk_size,
                    'overlap': chunk_overlap,
                    'adaptive': use_adaptive_chunking,
                    'multi_level': use_multi_level_chunking,
                    'language': chunk_language or transcription_language or 'en' # Sensible language default
                }
                processing_result["analysis_details"]["chunking_options"] = chunk_opts
                try:
                    chunked_texts_list = improved_chunking_process(text_to_analyze, chunk_opts)
                    processing_result["chunks"] = chunked_texts_list
                    if not chunked_texts_list:
                         warn_msg = "Chunking yielded no chunks. Analysis will use full text."
                         logging.warning(warn_msg)
                         processing_result["warnings"].append(warn_msg)
                         # Fallback: Summarize original text if chunking fails/is empty
                         analysis_text = summarize(api_name, text_to_analyze, custom_prompt, api_key, system_message=system_prompt)

                    else:
                         logger.info(f"Chunking successful, created {len(chunked_texts_list)} chunks.")
                         chunk_summaries = []
                         # Summarize each chunk
                         for i, chunk_block in enumerate(chunked_texts_list):
                             chunk_text = chunk_block.get("text")
                             if chunk_text:
                                 try:
                                     csum = summarize(api_name, chunk_text, custom_prompt, api_key, system_message=system_prompt)
                                     if csum:
                                         chunk_summaries.append(csum)
                                         # Optionally store chunk summary in chunk metadata if needed later
                                         chunk_block.setdefault("metadata", {})["summary"] = csum
                                 except Exception as chunk_summ_err:
                                      warn_msg = f"Summarization failed for chunk {i}: {chunk_summ_err}"
                                      logging.warning(warn_msg)
                                      processing_result["warnings"].append(warn_msg)
                                      chunk_block.setdefault("metadata", {})["summary_error"] = str(chunk_summ_err)


                         if chunk_summaries:
                             # Combine chunk summaries
                             if summarize_recursively and len(chunk_summaries) > 1:
                                 logger.info("Performing recursive summarization on chunk summaries.")
                                 combined_chunk_summaries = "\n\n---\n\n".join(chunk_summaries) # Use separator
                                 try:
                                     analysis_text = summarize(api_name, combined_chunk_summaries, custom_prompt or "Summarize the key points from the preceding text sections.", api_key, system_message=system_prompt)
                                 except Exception as rec_summ_err:
                                     warn_msg = f"Recursive summarization failed: {rec_summ_err}"
                                     logging.warning(warn_msg)
                                     processing_result["warnings"].append(warn_msg)
                                     analysis_text = combined_chunk_summaries # Fallback
                             else:
                                 analysis_text = "\n\n---\n\n".join(chunk_summaries) # Simple join
                         else:
                              warn_msg = "Analysis: Chunk summarization yielded no results."
                              logging.warning(warn_msg)
                              processing_result["warnings"].append(warn_msg)

                except Exception as chunk_err:
                    warn_msg = f"Chunking process failed: {chunk_err}. Analysis will use full text."
                    logging.warning(warn_msg, exc_info=True)
                    processing_result["warnings"].append(warn_msg)
                    # Fallback: Summarize original text if chunking fails
                    try:
                        analysis_text = summarize(api_name, text_to_analyze, custom_prompt, api_key, system_message=system_prompt)
                    except Exception as summ_err:
                         warn_msg = f"Summarization failed after chunking error: {summ_err}"
                         logging.error(warn_msg, exc_info=True)
                         processing_result["warnings"].append(warn_msg)

            else: # No chunking requested
                 logger.info(f"Performing single-pass analysis for {local_file_path_for_transcription}")
                 try:
                     analysis_text = summarize(api_name, text_to_analyze, custom_prompt, api_key, system_message=system_prompt)
                 except Exception as summ_err:
                     warn_msg = f"Summarization failed: {summ_err}"
                     logging.error(warn_msg, exc_info=True)
                     processing_result["warnings"].append(warn_msg)

            processing_result["analysis"] = analysis_text # Store final analysis/summary
            if not analysis_text:
                 warn_msg = "Analysis was performed but resulted in empty content."
                 logging.warning(warn_msg)
                 processing_result["warnings"].append(warn_msg)
            else:
                 logger.info("Analysis completed.")

        elif not transcription_text:
            logging.warning("Analysis skipped because transcription text is empty.")
        elif not api_name or api_name.lower() == "none":
             logging.info("Analysis skipped because no API name was provided.")
        else:
             logging.info("Analysis skipped (not requested).")


        # 5. Final Status
        # If we reached here without erroring out earlier, it's at least a partial success.
        # Downgrade to Warning if any warnings were recorded.
        if processing_result["error"]: # Should have returned earlier if error was fatal
             processing_result["status"] = "Error"
        elif processing_result["warnings"]:
            processing_result["status"] = "Warning"
        else:
            processing_result["status"] = "Success"

        logger.info(f"Finished processing {video_input}. Final status: {processing_result['status']}")
        processing_result["input_ref"] = video_input
        return processing_result

    except FileNotFoundError as e:
        logger.error(f"File not found error processing {video_input}: {e}", exc_info=True)
        processing_result["status"] = "Error"
        processing_result["error"] = str(e)
        # *** Ensure input_ref is original on error ***
        processing_result["input_ref"] = video_input
        return processing_result
    except ValueError as e: # Catch metadata or other value errors
        logger.error(f"Value error processing {video_input}: {e}", exc_info=True)
        processing_result["status"] = "Error"
        processing_result["error"] = str(e)
        # *** Ensure input_ref is original on error ***
        processing_result["input_ref"] = video_input
        return processing_result
    except Exception as e:
        # Catch-all for unexpected errors during the process
        logger.error(f"Unexpected exception processing {video_input}: {e}", exc_info=True)
        processing_result["status"] = "Error"
        processing_result["error"] = f"Unexpected error: {type(e).__name__}: {str(e)}"
        # *** Ensure input_ref is original on error ***
        processing_result["input_ref"] = video_input
        return processing_result

#
# End of Video_DL_Ingestion_Lib.py
#######################################################################################################################
