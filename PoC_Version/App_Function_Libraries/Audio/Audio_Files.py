# Audio_Files.py
#########################################
# Audio Processing Library
# This library is used to download or load audio files from a local directory.
#
####
#
# Functions:
#
# download_audio_file(url, save_path)
# process_audio(
# process_audio_file(audio_url, audio_file, whisper_model="small.en", api_name=None, api_key=None)
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
#
# External Imports
import requests
import yt_dlp
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_with_keywords, \
    check_media_and_whisper_model
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from PoC_Version.App_Function_Libraries.Summarization.Summarization_General_Lib import perform_summarization
from PoC_Version.App_Function_Libraries.Utils.Utils import downloaded_files, \
    sanitize_filename, logging
from PoC_Version.App_Function_Libraries.Video_DL_Ingestion_Lib import extract_metadata
from PoC_Version.App_Function_Libraries.Audio.Audio_Transcription_Lib import speech_to_text
from PoC_Version.App_Function_Libraries.Chunk_Lib import improved_chunking_process
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


def process_audio_files(audio_urls, audio_files, whisper_model, api_name, api_key, use_cookies, cookies, keep_original,
                        custom_keywords, custom_prompt_input, chunk_method, max_chunk_size, chunk_overlap,
                        use_adaptive_chunking, use_multi_level_chunking, chunk_language, diarize,
                        keep_timestamps, custom_title, record_system_audio, recording_duration,
                        system_audio_device, consent):

    # Add validation at the start of the function
    if record_system_audio:
        if not consent:
            raise ValueError("You must confirm you have consent to record audio")
        if not system_audio_device:
            raise ValueError("Please select an audio output device to record from")

    # Add recording logic before processing files
    recorded_files = []
    start_time = time.time()  # Start time for processing
    processed_count = 0
    failed_count = 0
    progress = []
    all_transcriptions = []
    all_summaries = []
    temp_files = []  # Keep track of temporary files

    if record_system_audio:
        try:
            # Extract device ID from the selected device string
            device_id = int(system_audio_device.split(":")[0])
            recorded_file = record_system_audio(
                duration=recording_duration,
                device_id=device_id
            )
            recorded_files.append(recorded_file)
            temp_files.append(recorded_file)
        except Exception as e:
            return print(f"Recording failed: {str(e)}"), "", ""

    # Process recorded files along with others
    if recorded_files:
        if not isinstance(audio_files, list):
            audio_files = []
        audio_files.extend(recorded_files)

    def format_transcription_with_timestamps(segments):
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
                        start_time1 = time.strftime('%H:%M:%S', time.gmtime(float(start)))
                        end_time = time.strftime('%H:%M:%S', time.gmtime(float(end)))
                        formatted_segments.append(f"[{start_time1}-{end_time}] {text}")
                    except (ValueError, TypeError):
                        # Fallback if conversion fails
                        formatted_segments.append(f"[{start}-{end}] {text}")

            # Join the segments with a newline to ensure proper formatting
            return "\n".join(formatted_segments)
        else:
            # Join the text without timestamps
            return "\n".join([segment.get('Text', '').strip() for segment in segments])

    def update_progress(message):
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
            # Also clean recorded files
            for file in recorded_files:
                try:
                    if os.path.exists(file):
                        os.remove(file)
                except:
                    pass

    def reencode_mp3(mp3_file_path):
        try:
            reencoded_mp3_path = mp3_file_path.replace(".mp3", "_reencoded.mp3")
            subprocess.run([ffmpeg_cmd, '-i', mp3_file_path, '-codec:a', 'libmp3lame', reencoded_mp3_path], check=True)
            update_progress(f"Re-encoded {mp3_file_path} to {reencoded_mp3_path}.")
            return reencoded_mp3_path
        except subprocess.CalledProcessError as e:
            update_progress(f"Error re-encoding {mp3_file_path}: {str(e)}")
            raise

    def convert_mp3_to_wav(mp3_file_path):
        try:
            wav_file_path = mp3_file_path.replace(".mp3", ".wav")
            subprocess.run([ffmpeg_cmd, '-i', mp3_file_path, wav_file_path], check=True)
            update_progress(f"Converted {mp3_file_path} to {wav_file_path}.")
            return wav_file_path
        except subprocess.CalledProcessError as e:
            update_progress(f"Error converting {mp3_file_path} to WAV: {str(e)}")
            raise

    try:
        # Check and set the ffmpeg command
        global ffmpeg_cmd
        if os.name == "nt":
            logging.debug("Running on Windows")
            ffmpeg_cmd = os.path.join(os.getcwd(), "Bin", "ffmpeg.exe")
        else:
            ffmpeg_cmd = 'ffmpeg'  # Assume 'ffmpeg' is in PATH for non-Windows systems

        # Ensure ffmpeg is accessible
        if not os.path.exists(ffmpeg_cmd) and os.name == "nt":
            raise FileNotFoundError(f"ffmpeg executable not found at path: {ffmpeg_cmd}")

        # Define chunk options early to avoid undefined errors
        chunk_options = {
            'method': chunk_method,
            'max_size': max_chunk_size,
            'overlap': chunk_overlap,
            'adaptive': use_adaptive_chunking,
            'multi_level': use_multi_level_chunking,
            'language': chunk_language
        }

        # Process URLs if provided
        if audio_urls:
            urls = [url.strip() for url in audio_urls.split('\n') if url.strip()]
            for i, url in enumerate(urls, 1):
                try:
                    update_progress(f"Processing URL {i}/{len(urls)}: {url}")

                    # Download and process audio file
                    audio_file_path = download_audio_file(url, use_cookies, cookies)
                    if not audio_file_path:
                        raise FileNotFoundError(f"Failed to download audio from URL: {url}")

                    temp_files.append(audio_file_path)

                    # Process the audio file
                    reencoded_mp3_path = reencode_mp3(audio_file_path)
                    temp_files.append(reencoded_mp3_path)

                    wav_file_path = convert_mp3_to_wav(reencoded_mp3_path)
                    temp_files.append(wav_file_path)

                    # Transcribe audio
                    segments = speech_to_text(wav_file_path, whisper_model=whisper_model, diarize=diarize)

                    # Handle segments format
                    if isinstance(segments, dict) and 'segments' in segments:
                        segments = segments['segments']

                    if not isinstance(segments, list):
                        raise ValueError("Unexpected segments format received from speech_to_text")

                    transcription = format_transcription_with_timestamps(segments)
                    if not transcription.strip():
                        raise ValueError("Empty transcription generated")
                    logging.debug(f"Transcription: {transcription}")

                    # Initialize summary with default value
                    summary = "No summary available"

                    # Attempt summarization if API is provided
                    if api_name not in (None, "None", "none"):
                        logging.debug(f"Summarizing audio with API: {api_name}")
                        try:
                            chunked_text = improved_chunking_process(transcription, chunk_options)
                            summary_result = perform_summarization(api_name, chunked_text, custom_prompt_input, api_key)
                            if summary_result:
                                summary = summary_result
                            update_progress("Audio summarized successfully.")
                        except Exception as e:
                            logging.error(f"Summarization failed: {str(e)}")
                            summary = "Summary generation failed"

                    # Add to results
                    all_transcriptions.append(transcription)
                    all_summaries.append(summary)

                    # Add to database
                    title = custom_title if custom_title else os.path.basename(wav_file_path)
                    add_media_with_keywords(
                        url=url,
                        title=title,
                        media_type='audio',
                        content=transcription,
                        keywords=custom_keywords,
                        prompt=custom_prompt_input,
                        summary=summary,
                        transcription_model=whisper_model,
                        author="Unknown",
                        ingestion_date=datetime.now().strftime('%Y-%m-%d')
                    )

                    processed_count += 1
                    update_progress(f"Successfully processed URL {i}")
                    log_counter("audio_files_processed_total", 1, {"whisper_model": whisper_model, "api_name": api_name})

                except Exception as e:
                    failed_count += 1
                    update_progress(f"Failed to process URL {i}: {str(e)}")
                    log_counter("audio_files_failed_total", 1, {"whisper_model": whisper_model, "api_name": api_name})
                    continue

        # Process uploaded files if provided
        if audio_files:
            # Convert to list if single file
            if not isinstance(audio_files, list):
                audio_files = [audio_files]

            for i, audio_file in enumerate(audio_files, 1):
                try:
                    file_title = f"{custom_title}_{i}" if custom_title else os.path.basename(audio_file.name)
                    update_progress(f"Processing file {i}/{len(audio_files)}: {file_title}")

                    if os.path.getsize(audio_file.name) > MAX_FILE_SIZE:
                        raise ValueError(f"File {file_title} size exceeds maximum limit of {MAX_FILE_SIZE / (1024 * 1024):.2f}MB")

                    # Process the audio file
                    reencoded_mp3_path = reencode_mp3(audio_file.name)
                    temp_files.append(reencoded_mp3_path)

                    wav_file_path = convert_mp3_to_wav(reencoded_mp3_path)
                    temp_files.append(wav_file_path)

                    # Transcribe audio
                    segments = speech_to_text(wav_file_path, whisper_model=whisper_model, diarize=diarize)

                    if isinstance(segments, dict) and 'segments' in segments:
                        segments = segments['segments']

                    if not isinstance(segments, list):
                        raise ValueError("Unexpected segments format received from speech_to_text")

                    transcription = format_transcription_with_timestamps(segments)
                    if not transcription.strip():
                        raise ValueError("Empty transcription generated")

                    # Initialize summary with default value
                    summary = "No summary available"

                    # Attempt summarization if API is provided
                    if api_name and api_name.lower() != "none":
                        try:
                            chunked_text = improved_chunking_process(transcription, chunk_options)
                            summary_result = perform_summarization(api_name, chunked_text, custom_prompt_input, api_key)
                            if summary_result:
                                summary = summary_result
                            update_progress(f"Audio file {i} summarized successfully.")
                        except Exception as e:
                            logging.error(f"Summarization failed for file {i}: {str(e)}")
                            summary = "Summary generation failed"

                    # Add to results with file identifier
                    all_transcriptions.append(f"=== {file_title} ===\n{transcription}")
                    all_summaries.append(f"=== {file_title} ===\n{summary}")

                    # Add to database
                    add_media_with_keywords(
                        url="Uploaded File",
                        title=file_title,
                        media_type='audio',
                        content=transcription,
                        keywords=custom_keywords,
                        prompt=custom_prompt_input,
                        summary=summary,
                        transcription_model=whisper_model,
                        author="Unknown",
                        ingestion_date=datetime.now().strftime('%Y-%m-%d')
                    )

                    processed_count += 1
                    update_progress(f"Successfully processed file {i}")
                    log_counter("audio_files_processed_total", 1, {"whisper_model": whisper_model, "api_name": api_name})

                except Exception as e:
                    failed_count += 1
                    update_progress(f"Failed to process file {i}: {str(e)}")
                    log_counter("audio_files_failed_total", 1, {"whisper_model": whisper_model, "api_name": api_name})
                    continue

        # Cleanup temporary files
        if not keep_original:
            cleanup_files()

        # Log processing metrics
        processing_time = time.time() - start_time
        log_histogram("audio_processing_time_seconds", processing_time,
                     {"whisper_model": whisper_model, "api_name": api_name})
        log_counter("total_audio_files_processed", processed_count,
                   {"whisper_model": whisper_model, "api_name": api_name})
        log_counter("total_audio_files_failed", failed_count,
                   {"whisper_model": whisper_model, "api_name": api_name})

        # Prepare final output
        final_progress = update_progress(f"Processing complete. Processed: {processed_count}, Failed: {failed_count}")
        final_transcriptions = "\n\n".join(all_transcriptions) if all_transcriptions else "No transcriptions available"
        final_summaries = "\n\n".join(all_summaries) if all_summaries else "No summaries available"

        return final_progress, final_transcriptions, final_summaries

    except Exception as e:
        logging.error(f"Error in process_audio_files: {str(e)}")
        log_counter("audio_files_failed_total", 1, {"whisper_model": whisper_model, "api_name": api_name})
        if not keep_original:
            cleanup_files()
        return update_progress(f"Processing failed: {str(e)}"), "No transcriptions available", "No summaries available"


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