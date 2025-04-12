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

#
# External Imports
import requests
import yt_dlp
#
# Local Imports
from tldw_Server_API.app.core.DB_Management.DB_Manager import add_media_with_keywords, \
    check_media_and_whisper_model
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import perform_summarization
from tldw_Server_API.app.core.Utils.Utils import downloaded_files, \
    sanitize_filename, logging
from tldw_Server_API.app.core.Ingestion_Media_Processing.Video.Video_DL_Ingestion_Lib import extract_metadata
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import speech_to_text
from tldw_Server_API.app.core.Utils.Chunk_Lib import improved_chunking_process
#
#######################################################################################################################
# Function Definitions
#

MAX_FILE_SIZE = 500 * 1024 * 1024

def download_audio_file(url, current_whisper_model="", use_cookies=False, cookies=None):
    try:
        # Check if media already exists in the database and compare whisper models
        should_download, reason = check_media_and_whisper_model(
            url=url,
            current_whisper_model=current_whisper_model
        )

        if not should_download:
            logging.info(f"Skipping audio download: {reason}")
            return None

        logging.info(f"Proceeding with audio download: {reason}")

        # Set up the request headers
        headers = {}
        if use_cookies and cookies:
            try:
                cookie_dict = json.loads(cookies)
                headers['Cookie'] = '; '.join([f'{k}={v}' for k, v in cookie_dict.items()])
            except json.JSONDecodeError:
                logging.warning("Invalid cookie format. Proceeding without cookies.")

        # Make the request
        response = requests.get(url, headers=headers, stream=True)
        # Raise an exception for bad status codes
        response.raise_for_status()

        # Get the file size
        file_size = int(response.headers.get('content-length', 0))
        if file_size > 500 * 1024 * 1024:  # 500 MB limit
            raise ValueError("File size exceeds the 500MB limit.")

        # Generate a unique filename
        file_name = f"audio_{uuid.uuid4().hex[:8]}.mp3"
        save_path = os.path.join('downloads', file_name)

        # Ensure the downloads directory exists
        os.makedirs('downloads', exist_ok=True)


        # Download the file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logging.info(f"Audio file downloaded successfully: {save_path}")
        return save_path

    except requests.RequestException as e:
        logging.error(f"Error downloading audio file: {str(e)}")
        raise
    except ValueError as e:
        logging.error(str(e))
        raise
    except Exception as e:
        logging.error(f"Unexpected error downloading audio file: {str(e)}")
        raise

def process_audio_files(
    audio_urls: Optional[List[str]],
    audio_files: Optional[List[str]],
    whisper_model: str,
    transcription_language: str,
    api_name: Optional[str],
    api_key: Optional[str],
    use_cookies: bool,
    cookies: Optional[str],
    keep_original: bool,
    custom_keywords: Optional[List[str]],
    custom_prompt_input: Optional[str],
    system_prompt_input: Optional[str],
    chunk_method: Optional[str],
    max_chunk_size: int,
    chunk_overlap: int,
    use_adaptive_chunking: bool,
    use_multi_level_chunking: bool,
    chunk_language: Optional[str],
    diarize: bool,
    keep_timestamps: bool,
    custom_title: Optional[str] = None,
    recursive_summarization=None
) -> Dict[str, Any]:
    """
    Process one or multiple audio files or URLs, returning structured results suitable for
    an API response. The function:
      - Optionally downloads remote files
      - Re-encodes to MP3, converts to WAV
      - Transcribes, optionally performs summarization
      - Returns a JSON-like dict with success/failure info
    """

    # A small helper to keep track of progress messages
    progress_log = []
    def update_progress(message: str):
        progress_log.append(message)

    # Keep track of processed items, to return them at the end
    results = []
    start_time = time.time()
    processed_count = 0
    failed_count = 0

    # Keep track of temporary files (for cleanup, if keep_original=False)
    temp_files = []

    # Choose your ffmpeg command logic
    if os.name == "nt":
        # On Windows, maybe supply your local ffmpeg path
        ffmpeg_cmd = os.path.join(os.getcwd(), "Bin", "ffmpeg.exe")
    else:
        # On non-Windows, assume 'ffmpeg' is in PATH
        ffmpeg_cmd = "ffmpeg"

    if os.name == "nt" and not os.path.exists(ffmpeg_cmd):
        raise FileNotFoundError(f"ffmpeg executable not found at path: {ffmpeg_cmd}")

    # Define your chunk options once
    chunk_options = {
        'method': chunk_method,
        'max_size': max_chunk_size,
        'overlap': chunk_overlap,
        'adaptive': use_adaptive_chunking,
        'multi_level': use_multi_level_chunking,
        'language': chunk_language,
    }

    def reencode_mp3(mp3_file_path: str) -> str:
        """
        Re-encode the original MP3 (in case it has unusual bitrates/codecs).
        """
        try:
            reencoded_mp3_path = mp3_file_path.replace(".mp3", "_reencoded.mp3")
            subprocess.run(
                [ffmpeg_cmd, '-i', mp3_file_path, '-codec:a', 'libmp3lame', reencoded_mp3_path],
                check=True
            )
            update_progress(f"Re-encoded {mp3_file_path} to {reencoded_mp3_path}.")
            return reencoded_mp3_path
        except subprocess.CalledProcessError as exc:
            msg = f"Error re-encoding {mp3_file_path}: {str(exc)}"
            update_progress(msg)
            raise RuntimeError(msg) from exc

    def convert_mp3_to_wav(mp3_file_path: str) -> str:
        """
        Convert MP3 to WAV to feed into the speech-to-text engine.
        """
        try:
            wav_file_path = mp3_file_path.replace(".mp3", ".wav")
            subprocess.run(
                [ffmpeg_cmd, '-i', mp3_file_path, wav_file_path],
                check=True
            )
            update_progress(f"Converted {mp3_file_path} to {wav_file_path}.")
            return wav_file_path
        except subprocess.CalledProcessError as exc:
            msg = f"Error converting {mp3_file_path} to WAV: {str(exc)}"
            update_progress(msg)
            raise RuntimeError(msg) from exc

    def cleanup_files():
        """
        Remove downloaded or intermediate files if the user chose not to keep them.
        """
        for file_path in temp_files:
            if not file_path:
                continue
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    update_progress(f"Temporary file {file_path} removed.")
            except Exception as exc:
                update_progress(f"Failed to remove temporary file {file_path}: {str(exc)}")

    # ---------------------
    #  Process URLs first
    # ---------------------
    if audio_urls:
        for i, url in enumerate(audio_urls, start=1):
            item_result = {
                "input": url,
                "success": False,
                "transcription": None,
                "summary": None,
                "error": None
            }
            try:
                update_progress(f"Downloading audio from URL {i}/{len(audio_urls)}: {url}")
                audio_file_path = download_audio_file(url, whisper_model, use_cookies, cookies)
                if not audio_file_path:
                    raise FileNotFoundError(f"Failed to download: {url}")

                temp_files.append(audio_file_path)

                # Re-encode and convert
                reencoded_mp3_path = reencode_mp3(audio_file_path)
                temp_files.append(reencoded_mp3_path)

                wav_file_path = convert_mp3_to_wav(reencoded_mp3_path)
                temp_files.append(wav_file_path)

                # Transcribe
                segments = speech_to_text(
                    wav_file_path,
                    whisper_model=whisper_model,
                    selected_source_lang=transcription_language,
                    vad_filter=True,
                    diarize=diarize,
                )
                # Some speech_to_text() functions return dict with "segments"; adapt as needed
                if isinstance(segments, dict) and 'segments' in segments:
                    segments = segments['segments']

                if not isinstance(segments, list):
                    raise ValueError("Unexpected segments format from speech_to_text()")

                transcription = format_transcription_with_timestamps(segments)
                if not transcription.strip():
                    raise ValueError("Empty transcription generated.")

                item_result["transcription"] = transcription

                # Summarize
                if api_name and api_name.lower() != "none":
                    try:
                        chunked_text = improved_chunking_process(transcription, chunk_options)
                        summary_result = perform_summarization(
                            api_name=api_name,
                            input_data=chunked_text,
                            custom_prompt_input=custom_prompt_input,
                            api_key=api_key,
                            recursive_summarization=recursive_summarization,
                            temp=None,
                            system_message=system_prompt_input
                        )
                        item_result["summary"] = summary_result or "No summary available"
                    except Exception as exc:
                        msg = f"Summarization failed: {str(exc)}"
                        update_progress(msg)
                        item_result["summary"] = "Summary generation failed"

                item_result["success"] = True
                processed_count += 1
                update_progress(f"Processed URL {i} successfully.")

            except Exception as exc:
                failed_count += 1
                item_result["error"] = str(exc)
                update_progress(f"Failed to process URL {i}: {str(exc)}")

            results.append(item_result)

    # ----------------------
    #  Process local files
    # ----------------------
    if audio_files:
        for i, file_path in enumerate(audio_files, start=1):
            # The “file_path” might be a real path or
            # something like an upload file with .name property, adapt as needed
            filename = os.path.basename(file_path) if file_path else f"audio_{i}"

            item_result = {
                "input": filename,
                "success": False,
                "transcription": None,
                "summary": None,
                "error": None
            }

            try:
                # Possibly check file size
                # if os.path.getsize(file_path) > MAX_FILE_SIZE:
                #     raise ValueError(f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024):.2f}MB")

                reencoded_mp3_path = reencode_mp3(file_path)
                temp_files.append(reencoded_mp3_path)

                wav_file_path = convert_mp3_to_wav(reencoded_mp3_path)
                temp_files.append(wav_file_path)

                # Transcribe
                segments = speech_to_text(
                    audio_file_path=wav_file_path,
                    whisper_model=whisper_model,
                    selected_source_lang=transcription_language,
                    vad_filter=True,
                    diarize=diarize,
                )
                if isinstance(segments, dict) and 'segments' in segments:
                    segments = segments['segments']

                if not isinstance(segments, list):
                    raise ValueError("Unexpected segments format from speech_to_text()")

                transcription = format_transcription_with_timestamps(segments)
                if not transcription.strip():
                    raise ValueError("Empty transcription generated")

                item_result["transcription"] = transcription

                # Summarize if API is provided
                if api_name and api_name.lower() != "none":
                    try:
                        chunked_text = improved_chunking_process(transcription, chunk_options)
                        summary_result = perform_summarization(
                            api_name=api_name,
                            input_data=chunked_text,
                            custom_prompt_input=custom_prompt_input,
                            api_key=api_key,
                            recursive_summarization=recursive_summarization,
                            temp=None,
                            system_message=system_prompt_input
                        )
                        item_result["summary"] = summary_result or "No summary available"
                        update_progress(f"File {filename} summarized successfully.")
                    except Exception as exc:
                        msg = f"Summarization failed: {str(exc)}"
                        update_progress(msg)
                        item_result["summary"] = "Summary generation failed"

                item_result["success"] = True
                processed_count += 1
                update_progress(f"Processed file {i}/{len(audio_files)}: {filename}")

            except Exception as exc:
                failed_count += 1
                item_result["error"] = str(exc)
                update_progress(f"Failed to process file {i} ({filename}): {str(exc)}")

            results.append(item_result)

    # ---------------------
    # Cleanup if needed
    # ---------------------
    if not keep_original:
        cleanup_files()

    # End timing
    total_time = time.time() - start_time
    update_progress(f"Processing complete. Success: {processed_count}, Failed: {failed_count}, Time: {total_time:.2f}s")

    # Return a structured response:
    return {
        "status": "success" if failed_count == 0 else "partial",
        "message": f"Processed: {processed_count}, Failed: {failed_count}",
        "progress": progress_log,  # If you want to see step-by-step progress
        "results": results
    }


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


def process_podcast(url, title, author, keywords, custom_prompt, api_name, api_key, whisper_model,
                    keep_original=False, enable_diarization=False, use_cookies=False, cookies=None,
                    chunk_method=None, max_chunk_size=300, chunk_overlap=0, use_adaptive_chunking=False,
                    use_multi_level_chunking=False, chunk_language='english', keep_timestamps=True):
    """
    Processes a podcast by downloading the audio, transcribing it, summarizing the transcription,
    and adding the results to the database. Metrics are logged throughout the process.

    Parameters:
        url (str): URL of the podcast.
        title (str): Title of the podcast.
        author (str): Author of the podcast.
        keywords (str): Comma-separated keywords.
        custom_prompt (str): Custom prompt for summarization.
        api_name (str): API name for summarization.
        api_key (str): API key for summarization.
        whisper_model (str): Whisper model to use for transcription.
        keep_original (bool): Whether to keep the original audio file.
        enable_diarization (bool): Whether to enable speaker diarization.
        use_cookies (bool): Whether to use cookies for authenticated downloads.
        cookies (str): JSON-formatted cookies string.
        chunk_method (str): Method for chunking text.
        max_chunk_size (int): Maximum size for each text chunk.
        chunk_overlap (int): Overlap size between chunks.
        use_adaptive_chunking (bool): Whether to use adaptive chunking.
        use_multi_level_chunking (bool): Whether to use multi-level chunking.
        chunk_language (str): Language for chunking.
        keep_timestamps (bool): Whether to keep timestamps in transcription.

    Returns:
        tuple: (progress_message, transcription, summary, title, author, keywords, error_message)
    """
    start_time = time.time()  # Start time for processing
    error_message = ""
    temp_files = []

    # Define labels for metrics
    labels = {
        "whisper_model": whisper_model,
        "api_name": api_name if api_name else "None"
    }

    def update_progress(message):
        """
        Updates the progress messages.

        Parameters:
            message (str): Progress message to append.

        Returns:
            str: Combined progress messages.
        """
        progress.append(message)
        return "\n".join(progress)

    def cleanup_files():
        if not keep_original:
            for file in temp_files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                        update_progress(f"Temporary file {file} removed.")
                except Exception as e:
                    update_progress(f"Failed to remove temporary file {file}: {str(e)}")

    progress = []  # Initialize progress messages

    try:
        # Handle cookies if required
        if use_cookies:
            cookies = json.loads(cookies)

        # Download the podcast audio file
        audio_file = download_audio_file(url, whisper_model, use_cookies, cookies)
        if not audio_file:
            raise RuntimeError("Failed to download podcast audio.")
        temp_files.append(audio_file)
        update_progress("Podcast downloaded successfully.")

        # Extract metadata from the podcast
        metadata = extract_metadata(url)
        title = title or metadata.get('title', 'Unknown Podcast')
        author = author or metadata.get('uploader', 'Unknown Author')

        # Format metadata for storage
        metadata_text = f"""
Metadata:
Title: {title}
Author: {author}
Series: {metadata.get('series', 'N/A')}
Episode: {metadata.get('episode', 'N/A')}
Season: {metadata.get('season', 'N/A')}
Upload Date: {metadata.get('upload_date', 'N/A')}
Duration: {metadata.get('duration', 'N/A')} seconds
Description: {metadata.get('description', 'N/A')}
"""

        # Update keywords with metadata information
        new_keywords = []
        if metadata.get('series'):
            new_keywords.append(f"series:{metadata['series']}")
        if metadata.get('episode'):
            new_keywords.append(f"episode:{metadata['episode']}")
        if metadata.get('season'):
            new_keywords.append(f"season:{metadata['season']}")

        keywords = f"{keywords},{','.join(new_keywords)}" if keywords else ','.join(new_keywords)
        update_progress(f"Metadata extracted - Title: {title}, Author: {author}, Keywords: {keywords}")

        # Transcribe the podcast audio
        try:
            if enable_diarization:
                segments = speech_to_text(audio_file, whisper_model=whisper_model, diarize=True)
            else:
                segments = speech_to_text(audio_file, whisper_model=whisper_model)
            # SEems like this could be optimized... FIXME
            def format_segment(segment):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('Text', '')

            if isinstance(segments, dict) and 'segments' in segments:
                segments = segments['segments']

            if isinstance(segments, list):
                transcription = format_transcription_with_timestamps(segments, keep_timestamps)
                update_progress("Podcast transcribed successfully.")
            else:
                raise ValueError("Unexpected segments format received from speech_to_text.")

            if not transcription.strip():
                raise ValueError("Transcription is empty.")
        except Exception as e:
            error_message = f"Transcription failed: {str(e)}"
            raise RuntimeError(error_message)

        # Apply chunking to the transcription
        chunk_options = {
            'method': chunk_method,
            'max_size': max_chunk_size,
            'overlap': chunk_overlap,
            'adaptive': use_adaptive_chunking,
            'multi_level': use_multi_level_chunking,
            'language': chunk_language
        }
        chunked_text = improved_chunking_process(transcription, chunk_options)

        # Combine metadata and transcription
        full_content = metadata_text + "\n\nTranscription:\n" + transcription

        # Summarize the transcription if API is provided
        summary = None
        if api_name:
            try:
                summary = perform_summarization(api_name, chunked_text, custom_prompt, api_key)
                update_progress("Podcast summarized successfully.")
            except Exception as e:
                error_message = f"Summarization failed: {str(e)}"
                raise RuntimeError(error_message)
        else:
            summary = "No summary available (API not provided)"

        # Add the processed podcast to the database
        try:
            add_media_with_keywords(
                url=url,
                title=title,
                media_type='podcast',
                content=full_content,
                keywords=keywords,
                prompt=custom_prompt,
                summary=summary or "No summary available",
                transcription_model=whisper_model,
                author=author,
                ingestion_date=datetime.now().strftime('%Y-%m-%d')
            )
            update_progress("Podcast added to database successfully.")
        except Exception as e:
            error_message = f"Error adding podcast to database: {str(e)}"
            raise RuntimeError(error_message)

        # Cleanup temporary files if required
        cleanup_files()

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time

        # Log successful processing
        log_counter(
            metric_name="podcasts_processed_total",
            labels=labels,
            value=1
        )

        # Log processing time
        log_histogram(
            metric_name="podcast_processing_time_seconds",
            value=processing_time,
            labels=labels
        )

        # Return the final outputs
        final_progress = update_progress("Processing complete.")
        return (final_progress, full_content, summary or "No summary generated.",
                title, author, keywords, error_message)

    except Exception as e:
        # Calculate processing time up to the point of failure
        end_time = time.time()
        processing_time = end_time - start_time

        # Log failed processing
        log_counter(
            metric_name="podcasts_failed_total",
            labels=labels,
            value=1
        )

        # Log processing time even on failure
        log_histogram(
            metric_name="podcast_processing_time_seconds",
            value=processing_time,
            labels=labels
        )

        logging.error(f"Error processing podcast: {str(e)}")
        cleanup_files()
        final_progress = update_progress(f"Processing failed: {str(e)}")
        return (final_progress, "", "", "", "", "", str(e))


#
#
#######################################################################################################################