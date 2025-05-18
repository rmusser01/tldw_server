# test_video_transcription.py
# Integration test for video transcription functionality
# Usage: pytest tests/integration/test_video_transcription.py
import pytest
import os
from App_Function_Libraries.Video_DL_Ingestion_Lib import download_video, extract_video_info
from App_Function_Libraries.Audio.Audio_Transcription_Lib import convert_to_wav, speech_to_text
from App_Function_Libraries.DB.DB_Manager import add_media_to_database
from App_Function_Libraries.Utils.Utils import create_download_directory


@pytest.fixture
def test_video_url():
    return "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Use a known, stable video URL for testing


@pytest.fixture
def test_download_path(tmp_path):
    return create_download_directory(str(tmp_path))

@pytest.mark.slow
def test_video_transcription_integration(test_video_url, test_download_path):
    try:
        # Step 1: Extract video info
        info_dict, title = extract_video_info(test_video_url)
        assert info_dict is not None
        assert title is not None

        # Step 2: Download video
        video_path = download_video(test_video_url, test_download_path, info_dict, True, "medium")
        assert os.path.exists(video_path)

        # Step 3: Convert video to WAV
        audio_file_path = convert_to_wav(video_path, offset=0)
        assert os.path.exists(audio_file_path)
        assert audio_file_path.endswith('.wav')

        # Step 4: Perform speech-to-text
        segments = speech_to_text(audio_file_path, whisper_model="tiny", vad_filter=True)
        assert segments is not None
        assert len(segments) > 0

        # Step 5: Add media to database
        summary = "Test summary"
        keywords = "test,integration,video"
        custom_prompt = "This is a test prompt"
        whisper_model = "tiny"

        result = add_media_to_database(
            test_video_url,
            info_dict,
            segments,
            summary,
            keywords,
            custom_prompt,
            whisper_model
        )
        assert "added/updated successfully" in result

        # Additional assertions
        assert any('Text' in segment for segment in segments), "No text found in transcription segments"

        # You might want to check the content of the transcription
        transcription_text = ' '.join([segment.get('Text', '') for segment in segments])
        assert len(transcription_text) > 0, "Transcription is empty"

        # Check if certain expected words are in the transcription (adjust based on the known content of your test video)
        expected_words = ["never", "gonna", "give", "you", "up"]  # Adjust these based on your test video content
        for word in expected_words:
            assert word.lower() in transcription_text.lower(), f"Expected word '{word}' not found in transcription"

    except Exception as e:
        pytest.fail(f"Integration test failed with error: {str(e)}")

    finally:
        # Clean up: remove downloaded files
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
        if 'audio_file_path' in locals() and os.path.exists(audio_file_path):
            os.remove(audio_file_path)