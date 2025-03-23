from Server_API.app.core.logging import logger
from Server_API.app.core.Ingestion_Media_Processing.Video_DL_Ingestion_Lib import extract_metadata, download_video
from Server_API.app.core.LLM_Calls.Summarization_General_Lib import perform_transcription, perform_summarization, save_transcription_and_summary
from Server_API.app.core.Utils.Utils import convert_to_seconds, create_download_directory, extract_text_from_segments
from Server_API.app.core.DB_Management.DB_Manager import add_media_to_database

async def process_video_task(url, whisper_model, custom_prompt, api_name, api_key, keywords, diarize,
                             start_time, end_time, include_timestamps, keep_original_video):
    try:
        # Create download path
        download_path = create_download_directory("Video_Downloads")
        logger.info(f"Download path created at: {download_path}")

        # Extract video information
        video_metadata = extract_metadata(url, use_cookies=False, cookies=None)
        if not video_metadata:
            raise ValueError(f"Failed to extract metadata for {url}")

        # Download video
        video_file_path = download_video(url, download_path, video_metadata, False, whisper_model)
        if not video_file_path:
            raise ValueError(f"Failed to download video/audio from {url}")

        # Perform transcription
        start_seconds = convert_to_seconds(start_time) if start_time else 0
        end_seconds = convert_to_seconds(end_time) if end_time else None
        audio_file_path, segments = perform_transcription(video_file_path, start_seconds, whisper_model, False, diarize)

        if audio_file_path is None or segments is None:
            raise ValueError("Transcription failed or segments not available.")

        # Process segments and extract text
        if not include_timestamps:
            segments = [{'Text': segment['Text']} for segment in segments]
        transcription_text = extract_text_from_segments(segments)

        # Perform summarization
        full_text_with_metadata = f"{video_metadata}\n\n{transcription_text}"
        if api_name in (None, "None", "none"):
            summary_text = "No summary available"
        else:
            summary_text = perform_summarization(api_name, full_text_with_metadata, custom_prompt, api_key)

        # Save transcription and summary
        json_file_path, summary_file_path = save_transcription_and_summary(full_text_with_metadata, summary_text, download_path, video_metadata)

        # Add to database
        add_media_to_database(video_metadata['webpage_url'], video_metadata, full_text_with_metadata, summary_text, keywords, custom_prompt, whisper_model)

        # Clean up files if not keeping original video
        if not keep_original_video:
            # Add cleanup logic here
            pass

        logger.info(f"Video processing completed for {url}")
        return True
    except Exception as e:
        logger.error(f"Error processing video for {url}: {str(e)}")
        return False