# TTS_Providers_Local.py
# Description: This file contains the functions to allow for usage of offline/local TTS providers.
#
# Imports
import os
import re
from typing import Optional, Generator
#
# External Imports
import nltk
import numpy as np
import pyaudio
from pydub.playback import play
import torch
from transformers import AutoTokenizer
#
# Local Imports
from PoC_Version.App_Function_Libraries.Utils.Utils import download_file, logging
# Kokoro-Specific Imports
from scipy.io import wavfile
from pydub import AudioSegment, effects  # For optional post-processing
#
#######################################################################################################################
#
# Functions:

########################################################
#
# Local Audio Generation Functions

def play_mp3(file_path):
    """Play an MP3 file using the pydub library."""
    try:
        from pydub.utils import which
        logging.debug(f"Debug: ffmpeg path: {which('ffmpeg')}")
        logging.debug(f"Debug: ffplay path: {which('ffplay')}")
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return
        absolute_path = os.path.abspath(file_path)
        audio = AudioSegment.from_mp3(absolute_path)
        logging.debug("Debug: File loaded successfully")
        play(audio)
    except Exception as e:
        logging.debug(f"Debug: Exception type: {type(e)}")
        logging.debug(f"Debug: Exception args: {e.args}")
        logging.error(f"Error playing the audio file: {e}")


def play_audio_file(file_path):
    """Play an audio file using the pydub library."""
    try:
        absolute_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return
        audio = AudioSegment.from_file(absolute_path)
        play(audio)
    except Exception as e:
        logging.error(f"Error playing the audio file: {e}")


# Text Normalization Function
def normalize_text(text):
    """Normalize input text for TTS processing."""
    # Replace special quotes and punctuation
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # Remove non-printable characters
    text = re.sub(r"[^\S \n]", " ", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # You can add more normalization rules here if needed
    return text

#
# End of Local Audio Generation Helper Functions
########################################################


########################################################
#
# Kokoro TTS Provider Functions
# Thank you to:
# https://github.com/thewh1teagle/kokoro-onnx
# https://huggingface.co/hexgrad/Kokoro-82M
# https://github.com/PierrunoYT/Kokoro-TTS-Local
# https://huggingface.co/hexgrad/Kokoro-82M/discussions/64

# Global caches for the model and voicepacks
_kokoro_model_cache = {}
_kokoro_voicepack_cache = {}
_onnx_kokoro_cache = {}

def check_espeak_installed() -> bool:
    """Check if eSpeak NG is installed and accessible."""
    espeak_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
    espeak_path = os.getenv("PHONEMIZER_ESPEAK_PATH")
    if espeak_lib and os.path.exists(espeak_lib) and espeak_path and os.path.exists(espeak_path):
        return True
    from shutil import which
    return which("espeak-ng") is not None


# FIXME - Replace
def split_text_into_sentence_chunks(text: str, max_tokens: int, tokenizer) -> list[str]:
    """Split text into chunks based on token limits."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_tokens = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_tokens)
        if current_length + sentence_length > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def get_kokoro_model(device: str) -> torch.nn.Module:
    """Retrieve the Kokoro model from the proper directory, cached if already loaded with dynamic downloading."""
    global _kokoro_model_cache
    logging.debug("Getting Kokoro model...")
    try:
        from PoC_Version.App_Function_Libraries.TTS.Kokoro.kokoro import generate
        from PoC_Version.App_Function_Libraries.TTS.Kokoro.models import build_model
        from PoC_Version.App_Function_Libraries.TTS.Kokoro import istftnet
    except ImportError:
        raise ImportError(
            "The required packages needed for Kokoro are not currently installed, . Please follow the instructions in the README to install them.")

    # FIXME - Add check/loading of model from kokoro_model_path in config.txt
    # Construct the correct model path
    base_dir = os.getcwd()  # Parent script runs from tldw/
    model_dir = os.path.join(base_dir, "App_Function_Libraries", "models", "kokoro_models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "kokoro-v0_19.pth")

    if (model_path, device) not in _kokoro_model_cache:
        if not os.path.exists(model_path):
            # version 0.19
            url = "https://huggingface.co/hexgrad/kLegacy/resolve/main/v0.19/kokoro-v0_19.pth?download=true"
            logging.info(f"Downloading model from {url}")
            download_file(url, model_path)
        MODEL = build_model(model_path, device=device)
        _kokoro_model_cache[(model_path, device)] = MODEL
        logging.info(f"Loaded model from {model_path}")
    return _kokoro_model_cache[(model_path, device)]


def get_kokoro_voicepack(voice: str, device: str) -> torch.Tensor:
    """Retrieve the Kokoro voicepack with dynamic downloading."""
    global _kokoro_voicepack_cache

    # Always store voices in tldw\App_Function_Libraries\TTS\Kokoro\voices
    voice_dir = os.path.join("App_Function_Libraries", "TTS", "Kokoro", "voices")
    os.makedirs(voice_dir, exist_ok=True)

    voice_path = os.path.join(voice_dir, f"{voice}.pt")

    if (voice_path, device) not in _kokoro_voicepack_cache:
        if not os.path.exists(voice_path):
            url = f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{voice}.pt?download=true"
            logging.info(f"Downloading voicepack from {url} to {voice_path}")
            download_file(url, voice_path)
        VOICEPACK = torch.load(voice_path, weights_only=True).to(device)
        _kokoro_voicepack_cache[(voice_path, device)] = VOICEPACK
    return _kokoro_voicepack_cache[(voice_path, device)]


def generate_audio_kokoro(
        input_text: str,
        voice: str = "af",
        device: Optional[str] = "cpu",
        output_format: str = "wav",
        output_file: str = "speech.wav",
        speed: float = 1.0,
        post_process: bool = True,
        use_onnx: bool = False,
        # FIXME - Remove the file path declaration here and stuff it into the function
        # FIXME - Also make sure it checks against config.txt for the model path
        onnx_model_path: str = "kokoro-v0_19.onnx",
        onnx_voices_json: str = "voices.json",
        stream: bool = False,
) -> str:
    """Generate audio with chunking, dynamic downloads, ONNX support, and streaming."""
    logging.info("Kokoro TTS: Generating audio...")

    logging.debug("Checking eSpeak NG installation...")
    if not check_espeak_installed():
        logging.error("eSpeak NG not found. Install and set environment variables.")
        return "Error: eSpeak NG required"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device type (CPU or CUDA): {device}")

    if use_onnx:
        logging.debug("Using ONNX model for Kokoro TTS...")
        return _handle_onnx_generation(input_text, voice, onnx_model_path, onnx_voices_json,
                                       output_format, output_file, speed, post_process, stream)
    else:
        logging.debug("Using PyTorch model for Kokoro TTS...")
        return _handle_pytorch_generation(input_text, voice, device,
                                          output_format, output_file, speed, post_process)


def _handle_onnx_generation(
        text: str,
        voice: str,
        model_path: str,
        voices_json: str,
        output_format: str,
        output_file: str,
        speed: float,
        post_process: bool,
        stream: bool,
) -> str:
    """Handle ONNX model audio generation."""
    try:
        import onnxruntime, kokoro_onnx
    except ImportError:
        raise ImportError(
            "The 'onnxruntime' and `kokoro_onnx` packages are required for this function. Please install it using 'pip install onnxruntime kokoro_onnx'.")
    from kokoro_onnx import Kokoro, EspeakConfig
    logging.debug("Using ONNX model for Kokoro TTS...")

    if not os.path.exists(model_path):
        logging.debug("Downloading Kokoro ONNX model...")
        download_file(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
            model_path
        )
    if not os.path.exists(voices_json):
        logging.debug("Downloading Kokoro ONNX voices JSON...")
        download_file(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json",
            voices_json
        )

    logging.debug("Initializing Kokoro TTS...")
    espeak_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
    kokoro = Kokoro(
        model_path,
        voices_json,
        espeak_config=EspeakConfig(lib_path=espeak_lib) if espeak_lib else None
    )

    lang = 'en-us' if voice.startswith('a') else 'en-gb'

    if stream:
        logging.debug("Kokoro ONNX: Streaming audio generation...")
        return _handle_onnx_streaming(kokoro, text, voice, speed, lang, output_file, output_format)
    else:
        logging.debug("Kokoro ONNX: Single audio generation...")
        samples, sr = kokoro.create(text, voice=voice, speed=speed, lang=lang)
        return _save_audio(samples, sr, output_file, output_format, post_process)


async def _handle_onnx_streaming(
        kokoro,
        text: str,
        voice: str,
        speed: float,
        lang: str,
        output_file: str,
        output_format: str,
) -> str:
    """Handle async streaming for ONNX."""
    logging.debug("Kokoro ONNX: Streaming audio generation...")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    stream = kokoro.create_stream(text, voice=voice, speed=speed, lang=lang)
    full_audio = []

    async for samples, sr in stream:
        # Open a PyAudio stream
        audio_stream = p.open(format=pyaudio.paFloat32,
                              channels=1,
                              rate=sr,
                              output=True)

        # Play the audio samples
        audio_stream.write(samples.tobytes())

        # Close the stream
        audio_stream.stop_stream()
        audio_stream.close()

        full_audio.append(samples)

    # Terminate PyAudio
    p.terminate()

    combined = np.concatenate(full_audio)
    return _save_audio(combined, sr, output_file, output_format, post_process=True)


def _handle_pytorch_generation(
        text: str,
        voice: str,
        device: str,
        output_format: str,
        output_file: str,
        speed: float,
        post_process: bool,
) -> str:
    """Handle PyTorch model audio generation with chunking."""
    logging.debug("Using PyTorch model for Kokoro TTS...")
    try:
        from PoC_Version.App_Function_Libraries.TTS.Kokoro.kokoro import generate
        from PoC_Version.App_Function_Libraries.TTS.Kokoro.models import build_model
        from PoC_Version.App_Function_Libraries.TTS.Kokoro import istftnet
    except ImportError:
        raise ImportError(
            "The required packages needed for Kokoro are not currently installed. Please follow the instructions in the README to install them.")

    logging.debug("Checking for nltk install...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    logging.debug("Splitting text into chunks...")
    text_chunks = split_text_into_sentence_chunks(text, 150, tokenizer)

    logging.debug("Getting Kokoro model and voicepack...")
    # Get default voicepack if not found
    if voice not in _kokoro_voicepack_cache and not isinstance(voice, str):
        logging.debug("Voicepack not found, using default voice...")
        voice = "af_bella"
    MODEL = get_kokoro_model(device)
    logging.debug("Model loaded successfully.")
    VOICEPACK = get_kokoro_voicepack(voice, device)
    logging.debug("Voicepack loaded successfully.")
    lang = 'a' if voice.startswith('a') else 'b'
    logging.debug(f"Language set to {'en-us' if lang == 'a' else 'en-gb'}")

    audio_segments = []
    logging.debug("Generating audio for text chunks...")
    for chunk in text_chunks:
        logging.debug(f"Generating audio for chunk: {chunk}")
        audio, _ = generate(MODEL, chunk, VOICEPACK, lang=lang, speed=speed)

        # Convert tensor to numpy array if needed
        if isinstance(audio, torch.Tensor):
            logging.debug("Converting audio tensor to numpy array...")
            audio = audio.cpu().numpy()
        audio_segments.append(audio)
        logging.debug(f"Generated audio for chunk: {chunk}")

    logging.debug("Combining audio segments...")
    audio_data = np.concatenate(audio_segments)
    return _save_audio(audio_data, 24000, output_file, output_format, post_process)


def _save_audio(
        audio_data: np.ndarray,
        sample_rate: int,
        output_file: str,
        output_format: str,
        post_process: bool,
) -> str:
    """Save audio data to file with optional post-processing."""
    logging.debug("Saving audio data to file...")
    max_amp = np.max(np.abs(audio_data))
    if max_amp > 1.0:
        audio_data /= max_amp
    audio_int16 = np.int16(audio_data * 32767)

    temp_wav = "temp_audio.wav"
    wavfile.write(temp_wav, sample_rate, audio_int16)

    if post_process:
        try:
            audio = AudioSegment.from_wav(temp_wav)
            audio = effects.strip_silence(audio, silence_thresh=-40.0, padding=100)
            audio = effects.normalize(audio)
            audio.export(output_file, format=output_format)
        except Exception as e:
            logging.error(f"Post-processing failed: {e}")
            return f"Error: {e}"
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    else:
        os.rename(temp_wav, output_file)

    return output_file if os.path.exists(output_file) else "Error: File creation failed"


async def generate_audio_stream_kokoro(
        input_text: str,
        voice: str = "af",
        onnx_model_path: str = "kokoro-v0_19.onnx",
        onnx_voices_json: str = "voices.json",
) -> Generator[np.ndarray, None, None]:
    """Async generator for streaming audio chunks (ONNX only)."""
    try:
        import onnxruntime, kokoro_onnx
    except ImportError:
        raise ImportError(
            "The 'onnxruntime' and `kokoro_onnx` packages are required for this function. Please install it using 'pip install onnxruntime kokoro_onnx'.")
    logging.info("Kokoro Streaming: Generating audio stream...")

    logging.debug("Kokoro Streaming: Checking eSpeak NG installation...")
    if not check_espeak_installed():
        raise RuntimeError("eSpeak NG not installed")

    if not os.path.exists(onnx_model_path):
        logging.debug("Kokoro Streaming: Downloading Kokoro ONNX model...")
        download_file(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx",
            onnx_model_path
        )
    if not os.path.exists(onnx_voices_json):
        logging.debug("Downloading Kokoro ONNX voices JSON...")
        download_file(
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json",
            onnx_voices_json
        )

    espeak_lib = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
    logging.debug("Kokoro Streaming: Initializing Kokoro TTS...")
    from kokoro_onnx import Kokoro
    from kokoro_onnx import EspeakConfig
    kokoro = Kokoro(
        onnx_model_path,
        onnx_voices_json,
        espeak_config=EspeakConfig(lib_path=espeak_lib) if espeak_lib else None
    )

    lang = 'en-us' if voice.startswith('a') else 'en-gb'
    logging.debug("Kokoro Streaming: Starting audio stream...")
    stream = kokoro.create_stream(input_text, voice=voice, speed=1.0, lang=lang)

    logging.debug("Kokoro Streaming: Yielding audio chunks...")
    async for samples, sr in stream:
        logging.debug(f"Yielding audio chunk for stream...")
        yield samples, sr


def test_generate_audio_kokoro():
    """Test function updated for new features."""
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

    try:
        result = generate_audio_kokoro(
            input_text="Hello, this is a test of the Kokoro TTS system.",
            voice="af_bella",
            use_onnx=False
        )
        if "Error" in result:
            logging.error(result)
        else:
            logging.info(f"Audio saved: {result}")
    except Exception as e:
        logging.error(f"Test failed: {e}")


# v1
# def generate_audio_kokoro(
#         input_text: str,
#         voice: str = "af",  # Default voice: American English
#         model_path: str = "kokoro-v0_19.pth",
#         device: str = None,  # Auto-detect CUDA or fallback to CPU
#         output_format: str = "wav",
#         output_file: str = "speech.wav",
#         speed: float = 1.0,
#         post_process: bool = True  # Enable trimming, normalization, etc.
# ) -> str:
#     """
#     Generate audio using Kokoro TTS.
#
#     :param input_text: The text to be synthesized.
#     :param voice: Voicepack name, e.g., "af", "af_sarah", etc.
#     :param model_path: Path to the Kokoro model checkpoint file.
#     :param device: Device to run the model on ("cuda" or "cpu"). Default auto-detects.
#     :param output_format: Output audio format ("wav", "mp3", etc.).
#     :param output_file: Path to save the generated audio.
#     :param speed: TTS speed factor. Default is 1.0.
#     :param post_process: If True, apply audio trimming and normalization.
#     :return: Path to the saved audio file, or an error message on failure.
#     """
#     logging.debug("Kokoro TTS: Generating audio...")
#     if not input_text.strip():
#         logging.error("Kokoro TTS: No text provided.")
#         return "Kokoro TTS: Error - No text provided."
#
#     # Auto-detect device
#     device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#     logging.info(f"Kokoro TTS: Using device '{device}'")
#
#     try:
#         # Load or retrieve model and voicepack
#         MODEL = get_kokoro_model(model_path, device)
#         VOICEPACK = get_kokoro_voicepack(voice, device)
#     except Exception as e:
#         logging.error(f"Kokoro TTS: Failed to initialize model or voicepack: {e}")
#         return f"Kokoro TTS: Initialization error: {e}"
#
#     # Determine language from voice
#     lang = voice[0].lower() if voice and voice[0].lower() in {'a', 'b'} else 'a'
#     logging.info(f"Kokoro TTS: Language set to {'en-us' if lang == 'a' else 'en-gb'}")
#
#     try:
#         # Generate audio
#         audio_data, out_phonemes = generate_full(
#             MODEL, input_text, VOICEPACK, lang=lang, speed=speed
#         )
#         if audio_data is None or len(audio_data) == 0:
#             logging.error("No audio data generated.")
#             return "Kokoro TTS: Failed to generate audio data."
#         else:
#             logging.debug(f"Generated audio data of shape: {audio_data.shape}")
#
#         logging.debug(f"Kokoro TTS: Phonemes generated: {out_phonemes}")
#
#         # Normalize to int16 and prepare for saving
#         max_amp = np.max(np.abs(audio_data))
#         if max_amp > 1.0:
#             audio_data /= max_amp
#         audio_int16 = np.int16(audio_data * 32767)
#
#     except Exception as e:
#         logging.error(f"Kokoro TTS: Error during synthesis: {e}")
#         return f"Kokoro TTS: Error during synthesis: {e}"
#
#     try:
#         # Save audio to file
#         temp_wav = "temp_audio.wav"
#         wavfile.write(temp_wav, 24000, audio_int16)
#
#         if not os.path.exists(temp_wav):
#             logging.error("Kokoro TTS: Temporary WAV file not created.")
#             return "Kokoro TTS: Error - Temporary file not created."
#
#         # Post-process and save final file
#         if post_process:
#             try:
#                 audio_segment = AudioSegment.from_wav(temp_wav)
#                 trimmed_segment = effects.strip_silence(audio_segment, silence_thresh=-40.0, padding=100)
#                 normalized_segment = effects.normalize(trimmed_segment)
#                 normalized_segment.export(output_file, format=output_format.lower())
#                 logging.info(f"Kokoro TTS: Processed audio saved to '{output_file}'")
#             except Exception as e:
#                 logging.error(f"Kokoro TTS: Post-processing error: {e}")
#                 return f"Kokoro TTS: Error during post-processing: {e}"
#         else:
#             os.rename(temp_wav, output_file)
#             logging.info(f"Kokoro TTS: Raw audio saved to '{output_file}'")
#
#     except Exception as e:
#         logging.error(f"Kokoro TTS: Error during synthesis: {e}")
#         return f"Kokoro TTS: Error during synthesis: {e}"
#
#     if os.path.exists(output_file):
#         logging.info(f"Kokoro TTS: File successfully created at '{output_file}'")
#     else:
#         logging.error(f"Kokoro TTS: File was not created: {output_file}")
#
#     return output_file


#v1
def test_generate_audio_kokoro2():
    # https://huggingface.co/hexgrad/Kokoro-82M/discussions/12
    # https://github.com/bootphon/phonemizer/issues/44#issuecomment-1540885186
    import os
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
    os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
    try:
        logging.debug("Testing Kokoro TTS...")
        result_file = generate_audio_kokoro(
            input_text="Hello, this is a test of the Kokoro TTS system.",
            voice="af_bella",
            # Dynamically construct the model path
            device=None,  # Auto-detect device
            output_format="wav",
            output_file="kokoro_output.wav",
            speed=1.0,
            post_process=True
        )

        logging.debug("Kokoro TTS test complete.")
        play_audio_file(result_file)
        if "Error" in result_file:
            logging.debug("Kokoro TTS test failed.")
            logging.error(result_file)
        else:
            logging.info(f"Audio saved at: {result_file}")
    except Exception as e:
        logging.error(f"Error during Kokoro TTS test: {e}")


# Tests
def test_empty_input():
    assert "Error" in generate_audio_kokoro(input_text="", output_file="test_empty.wav")

def test_basic_synthesis():
    assert os.path.exists(generate_audio_kokoro(
        input_text="This is a basic test.",
        voice="af",
        output_file="test_basic.wav"
    ))

def test_invalid_voicepack():
    assert "Error" in generate_audio_kokoro(
        input_text="Testing invalid voice.",
        voice="nonexistent_voice",
        output_file="test_invalid_voice.wav"
    )

#
# End of Local TTS Provider Functions
#######################################################