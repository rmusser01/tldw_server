# test_audio_transcription.py
# Integration test for audio transcription functionality
# Usage: pytest tests/integration/test_audio_transcription.py
import pytest
import os
import requests
from pydub import AudioSegment
from App_Function_Libraries.Audio.Audio_Transcription_Lib import speech_to_text
from App_Function_Libraries.DB.DB_Manager import add_media_to_database
from App_Function_Libraries.Utils.Utils import create_download_directory


@pytest.fixture
def test_audio_url():
    # URL to a short, public domain audio file
    return "https://upload.wikimedia.org/wikipedia/commons/1/1f/Dial_up_modem_noises.ogg"


@pytest.fixture
def test_download_path(tmp_path):
    return create_download_directory(str(tmp_path))


def download_audio(url, download_path):
    response = requests.get(url)
    if response.status_code == 200:
        file_name = url.split("/")[-1]
        file_path = os.path.join(download_path, file_name)
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        raise Exception(f"Failed to download audio file. Status code: {response.status_code}")

@pytest.mark.slow
def test_audio_transcription_integration(test_audio_url, test_download_path):
    try:
        # Step 1: Download test audio file
        original_audio_path = download_audio(test_audio_url, test_download_path)
        assert os.path.exists(original_audio_path)

        # Step 2: Convert audio to WAV if it's not already in WAV format
        if not original_audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_file(original_audio_path)
            wav_path = os.path.splitext(original_audio_path)[0] + '.wav'
            audio.export(wav_path, format="wav")
        else:
            wav_path = original_audio_path

        assert os.path.exists(wav_path)
        assert wav_path.lower().endswith('.wav')

        # Step 3: Perform speech-to-text
        whisper_model = "tiny"  # Using 'tiny' for faster testing, adjust as needed
        segments = speech_to_text(wav_path, whisper_model=whisper_model, vad_filter=True)

        assert segments is not None
        assert len(segments) > 0

        # Step 4: Add media to database
        info_dict = {
            "title": "Test Audio Transcription",
            "uploader": "Integration Test",
            "upload_date": "20220101"
        }
        summary = "Test audio transcription summary"
        keywords = "test,integration,audio,transcription"
        custom_prompt = "This is a test prompt for audio transcription"

        result = add_media_to_database(
            test_audio_url,
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

        # Combine all transcribed text
        transcription_text = ' '.join([segment.get('Text', '') for segment in segments])
        assert len(transcription_text) > 0, "Transcription is empty"

        # Check for expected content in the transcription
        # Note: This audio file contains modem noises, so actual speech content might be limited
        # Adjust these expected phrases based on what you expect in your test audio
        expected_phrases = ["noise", "sound", "modem"]
        for phrase in expected_phrases:
            assert phrase.lower() in transcription_text.lower(), f"Expected phrase '{phrase}' not found in transcription"

        # You could also check for the presence of timestamps or other expected metadata in the segments

    except Exception as e:
        pytest.fail(f"Integration test failed with error: {str(e)}")

    finally:
        # Clean up: remove downloaded files
        if 'original_audio_path' in locals() and os.path.exists(original_audio_path):
            os.remove(original_audio_path)
        if 'wav_path' in locals() and os.path.exists(wav_path) and wav_path != original_audio_path:
            os.remove(wav_path)