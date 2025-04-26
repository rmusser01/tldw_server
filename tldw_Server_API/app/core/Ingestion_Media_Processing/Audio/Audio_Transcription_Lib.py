# Audio_Transcription_Lib.py
#########################################
# Transcription Library
# This library is used to perform transcription of audio files.
# Currently, uses faster_whisper for transcription.
#
####################
# Function List
#
# 1. convert_to_wav(video_file_path, offset=0, overwrite=False)
# 2. speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='small.en', vad_filter=False)
#
####################
#
# Import necessary libraries to run solo for testing
import gc
import json
import multiprocessing
import os
import shutil
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional, Union, List, Dict, Any
#
# DEBUG Imports
#from memory_profiler import profile
# Third-Party Imports
import pyaudio
from faster_whisper import WhisperModel as OriginalWhisperModel
import numpy as np
import torch
from scipy.io import wavfile
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
import sounddevice as sd
import wave

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Diarization_Lib_v2 import audio_diarization, \
    combine_transcription_and_diarization, DiarizationError
#
# Import Local
from tldw_Server_API.app.core.Utils.Utils import sanitize_filename, load_and_log_configs, logging
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram, timeit
#
#######################################################################################################################
# Function Definitions
#

# Convert video .m4a into .wav using ffmpeg
#   ffmpeg -i "example.mp4" -ar 16000 -ac 1 -c:a pcm_s16le "output.wav"
#       https://www.gyan.dev/ffmpeg/builds/
#

# FIXME
# 1. Implement chunking for large audio files
# def speech_to_text(audio_file_path, selected_source_lang='en', whisper_model='medium.en', vad_filter=False, chunk_size=30):
#     # ... existing code ...
#     segments = []
#     for segment_chunk in whisper_model_instance.transcribe(audio_file_path, beam_size=10, best_of=10, vad_filter=vad_filter, chunk_size=chunk_size):
#         # Process each chunk
#         # ... existing code ...
#
# 2. Use generators
#     def generate_segments(segments_raw):
#         for segment_chunk in segments_raw:
#             yield {
#                 "Time_Start": segment_chunk.start,
#                 "Time_End": segment_chunk.end,
#                 "Text": segment_chunk.text
#             }
#     # Usage
#     segments = list(generate_segments(segments_raw))
#
# 3. Use subprocess instead of os.system for ffmpeg
# 4. Adjust CPU threads properly
# 5. Use quantized models - compute_type="int8"


def perform_transcription(
    video_path: str,
    offset: int, # Note: Offset is passed to convert_to_wav but not used there currently
    transcription_model: str,
    vad_use: bool,
    diarize: bool = False,
    overwrite: bool = False,
    transcription_language: str = 'en',
    temp_dir: Optional[str] = None,
    ):
    """
    Converts video/audio to WAV, performs transcription (optionally with diarization),
    handles existing files, and manages errors.

    Returns:
        Tuple[Optional[str], Optional[list]]: (audio_file_path, segments_list) on success,
                                             (None, None) on critical failure (like conversion).
                                             (audio_file_path, None) if transcription fails but conversion succeeded.
    """
    local_media_path_to_convert = None
    temp_dir_manager = None
    downloaded_file_path = None  # Track the specific downloaded file
    try:
        logging.info(f"Initiating transcription process for: {video_path}")
        # 1. Convert to WAV - Catch ConversionError specifically
        try:
            audio_file_path = convert_to_wav(video_path, offset=offset, overwrite=overwrite) # Pass overwrite flag?
            if not audio_file_path or not os.path.exists(audio_file_path):
                 # This case might occur if convert_to_wav returns None/empty path without raising error
                 logging.error(f"Conversion to WAV failed or produced no file for {video_path}")
                 return None, None # Critical failure
            logging.debug(f"Converted audio file path: {audio_file_path}")
        except ConversionError as e:
            logging.error(f"Audio conversion failed for {video_path}: {e}")
            return None, None # Critical failure, stop processing

        # 2. Define paths
        base_path = os.path.splitext(audio_file_path)[0]
        # Sanitize model name for filename
        transcription_model_sanitized = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in transcription_model)
        segments_json_path = f"{base_path}-transcription_model-{transcription_model_sanitized}.segments.json"
        diarized_json_path = f"{base_path}-transcription_model-{transcription_model_sanitized}.diarized.json"

        # --- Perform Diarization and Combination (if requested) ---
        if diarize:
            final_segments = None
            try:
                # Define path to your diarization config
                # Resolve path relative to current file or use absolute path
                base_dir = Path(__file__).parent.resolve()  # Or wherever your config lives
                diarization_config_path = base_dir / 'models' / 'pyannote_diarization_config.yaml'  # ADJUST PATH AS NEEDED

                logging.info(f"Performing diarization using config: {diarization_config_path}...")
                diarization_segments = audio_diarization(
                    audio_file_path=audio_file_path,
                    config_path=diarization_config_path
                    # Optionally pass num_speakers, min_speakers, max_speakers here if known
                )

                logging.info("Combining transcription and diarization results...")
                final_segments = combine_transcription_and_diarization(
                    transcription_segments=adapted_transcription_segments,  # Use adapted segments
                    diarization_segments=diarization_segments
                )
                logging.info("Diarization and combination successful.")

            except (DiarizationError, FileNotFoundError) as e:
                logging.error(f"Diarization or combination failed for {audio_file_path}: {e}")
                # Decide how to handle diarization failure: return transcription only? or error?
                # Option 1: Return transcription with warning
                logging.warning("Proceeding with transcription only due to diarization error.")
                final_segments = adapted_transcription_segments  # Fallback to transcription segments
                # Add a warning marker maybe?
                if final_segments:
                    final_segments[0]['text'] = f"[Warning: Diarization failed ({e})] " + final_segments[0]['text']

                # Option 2: Treat as overall failure for this step
                # return audio_file_path, None # Or raise the error

            # Return the combined (or fallback) segments
            return audio_file_path, final_segments

        # 4. Handle Non-Diarized Path
        else:
            logging.info(f"Processing without diarization for {audio_file_path}")
            # Check if non-diarized JSON exists
            if os.path.exists(segments_json_path) and not overwrite:
                logging.info(f"Segments file already exists (overwrite=False): {segments_json_path}")
                try:
                    with open(segments_json_path, 'r', encoding='utf-8') as file:
                        loaded_data = json.load(file)
                    # Handle potential structures: {'segments': [...]} or just [...]
                    if isinstance(loaded_data, dict) and "segments" in loaded_data:
                        segments = loaded_data["segments"]
                    elif isinstance(loaded_data, list):
                        segments = loaded_data
                    else:
                        raise ValueError("JSON structure is not a list or {'segments': list}")

                    # Basic validation
                    if isinstance(segments, list) and all(isinstance(s, dict) and 'Text' in s for s in segments):
                        logging.debug(f"Loaded valid segments from existing file.")
                        return audio_file_path, segments
                    else:
                        logging.warning(f"Existing segments file {segments_json_path} has invalid format, but overwrite=False.")
                        return audio_file_path, None # Treat as transcription failure
                except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e:
                    logging.warning(f"Failed to read/parse existing segments file {segments_json_path}: {e}. Overwrite=False.")
                    return audio_file_path, None # Treat as transcription failure

            # Generate new transcription (or overwrite existing)
            logging.info(f"Generating/Overwriting transcription for {audio_file_path}")
            # Ensure re_generate_transcription handles errors from speech_to_text
            _ , segments = re_generate_transcription(
                audio_file_path,
                transcription_model,
                vad_use,
                selected_source_lang=transcription_language # Pass language
            )

            if segments is None: # Check if generation failed
                 logging.error(f"Transcription generation failed for {audio_file_path}")
                 return audio_file_path, None # Return path, None segments

            # Saving is handled within speech_to_text called by re_generate_transcription
            # but we already checked for overwrite flag above. If overwrite=True,
            # speech_to_text should ideally handle the overwrite when saving.
            # If speech_to_text doesn't handle overwrite, you might need to explicitly delete
            # the old file here before calling re_generate_transcription if overwrite is True.

            logging.info(f"Successfully generated/loaded transcription for {audio_file_path}")
            return audio_file_path, segments

    except Exception as e:
        # Catch-all for unexpected errors during the process
        logging.error(f"Unexpected error in perform_transcription for {video_path}: {e}", exc_info=True)
        # If conversion succeeded, return path, else None. Always return None for segments on error.
        return (audio_file_path, None) if audio_file_path else (None, None)


def re_generate_transcription(audio_file_path, whisper_model, vad_filter, selected_source_lang='en'):
    """Placeholder: Assumed to call speech_to_text and handle errors."""
    logging.info(f"Regenerating transcription for {audio_file_path} using model {whisper_model}")
    try:
        # IMPORTANT: Pass all necessary parameters to speech_to_text
        segments = speech_to_text(
            audio_file_path=audio_file_path,
            whisper_model=whisper_model,
            selected_source_lang=selected_source_lang,  # Ensure language is passed
            vad_filter=vad_filter,
            diarize=False  # Explicitly false for non-diarized regeneration
        )
        # speech_to_text now returns the segments list directly on success (or raises error)
        # It might return {'segments': [...]} if loading from an existing file structure. Adapt if necessary.
        if isinstance(segments, dict) and 'segments' in segments:
            actual_segments = segments['segments']
        else:
            actual_segments = segments  # Assuming it returns the list directly now or handles errors by raising

        if not actual_segments:
            logging.warning(f"Re-generation yielded no segments for {audio_file_path}")
            return audio_file_path, None  # Return path but None segments on empty result

        logging.info(f"Successfully re-generated transcription for {audio_file_path}")
        return audio_file_path, actual_segments
    except RuntimeError as e:
        logging.error(f"RuntimeError during re_generate_transcription for {audio_file_path}: {e}")
        return audio_file_path, None  # Return path but None segments on error
    except Exception as e:
        logging.error(f"Unexpected error during re_generate_transcription for {audio_file_path}: {e}", exc_info=True)
        return audio_file_path, None  # Return path but None segments on error


#####################################
# Memory-Saving Indefinite Recording
#####################################

class PartialTranscriptionThread(threading.Thread):
    def __init__(
        self,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        partial_text_state: dict,
        lock: threading.Lock,
        live_model: str,          # model for partial
        sample_rate=44100,
        channels=2,
        partial_update_interval=2.0,   # how often we attempt a partial transcription
        partial_chunk_seconds=5,
    ):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.partial_text_state = partial_text_state
        self.lock = lock
        self.live_model = live_model

        self.sample_rate = sample_rate
        self.channels = channels
        self.partial_update_interval = partial_update_interval
        self.partial_chunk_seconds = partial_chunk_seconds

        # Rolling buffer for partial
        self.audio_buffer = b""
        # We only keep last X seconds in memory for partial
        self.max_partial_bytes = int(self.partial_chunk_seconds * self.sample_rate * self.channels * 2)

        self.last_ts = time.time()

        # Keep track of any exceptions
        self.exception_encountered = None

    def run(self):
        while not self.stop_event.is_set():
            now = time.time()
            if now - self.last_ts < self.partial_update_interval:
                time.sleep(0.1)
                continue

            # Gather new chunks from the queue
            new_data = []
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get_nowait()
                new_data.append(chunk)

            if new_data:
                combined_new_data = b"".join(new_data)
                # Append to rolling buffer
                self.audio_buffer += combined_new_data

                # Enforce maximum partial buffer size
                if len(self.audio_buffer) > self.max_partial_bytes:
                    self.audio_buffer = self.audio_buffer[-self.max_partial_bytes:]

            # If rolling buffer is large enough, do partial transcription
            if len(self.audio_buffer) > (self.sample_rate * self.channels * 2):  # ~1s
                try:
                    # Convert from 16-bit PCM to float32
                    audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                    # If channels=2, you may want to downmix to mono:
                    # If your STT supports stereo, skip this step.
                    if self.channels == 2:
                        audio_np = audio_np.reshape((-1, 2))
                        audio_np = np.mean(audio_np, axis=1)  # simple stereo -> mono

                    # FIXME - Add support for multiple languages/whisper models
                    partial_text = transcribe_audio(
                        audio_np,
                        sample_rate=self.sample_rate,
                        whisper_model=self.live_model,
                        speaker_lang="en",
                        transcription_provider="faster-whisper"
                    )

                    with self.lock:
                        self.partial_text_state["text"] = partial_text
                except Exception as e:
                    self.exception_encountered = e
                    logging.error(f"Partial transcription error: {e}")

            self.last_ts = time.time()


def record_audio_to_disk(device_id, output_file_path, stop_event, audio_queue):
    """
    Thread function that:
    - Opens PyAudio with (44.1kHz, 2ch).
    - Reads data in a loop, writes directly to disk, and also puts chunk into `audio_queue`.
    - We store minimal data in RAM since each chunk is appended to file.
    """
    p = pyaudio.PyAudio()
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    try:
        # Validate device ID
        device_count = p.get_device_count()
        if device_id is None or device_id < 0 or device_id >= device_count:
            err_msg = f"Invalid device ID: {device_id}. Valid range is 0-{device_count - 1}"
            logging.error(err_msg)
            raise ValueError(err_msg)

        # Check device capabilities
        device_info = p.get_device_info_by_index(device_id)
        logging.info(f"Using device: {device_info['name']}")

        if device_info['maxInputChannels'] < 1:
            err_msg = f"Device {device_id} ({device_info['name']}) doesn't support audio input"
            logging.error(err_msg)
            raise ValueError(err_msg)

        # Adjust channels to device capability
        actual_channels = min(CHANNELS, int(device_info['maxInputChannels']))
        if actual_channels != CHANNELS:
            logging.info(f"Adjusted channels from {CHANNELS} to {actual_channels} for device limitations")

        # Open audio stream
        stream = p.open(
            format=FORMAT,
            channels=actual_channels,
            rate=RATE,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=CHUNK
        )

        # Open the WAV for writing
        wf = wave.open(output_file_path, 'wb')
        wf.setnchannels(actual_channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(RATE)

        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                # write to disk
                wf.writeframes(data)
                # also push to queue for partial
                audio_queue.put(data)
            except Exception as e:
                logging.error(f"Recording error: {e}")
                break

    except Exception as e:
        # Enhanced error messages for common issues
        if "9999" in str(e):
            logging.error(f"Device {device_id} is likely in use by another application")
        elif "Invalid sample rate" in str(e):
            logging.error(f"Device {device_id} doesn't support {RATE}Hz sample rate")
        else:
            logging.error(f"Error with device {device_id}: {e}")
        raise

    finally:
        # Ensure proper cleanup even if errors occur
        if 'stream' in locals():
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
        if 'wf' in locals():
            try:
                wf.close()
            except:
                pass
        p.terminate()


def stop_recording_short(record_state):
    """
    - Signals the threads to stop
    - Joins them with a timeout
    - If partial thread had an exception, returns that
    """
    if not record_state:
        return None, "[No active recording to stop]", None

    stop_event = record_state["stop_event"]
    rec_thread = record_state["record_thread"]
    partial_thread = record_state["partial_thread"]
    output_file_path = record_state["wav_path"]

    stop_event.set()
    rec_thread.join(timeout=5)
    if rec_thread.is_alive():
        logging.warning("record_thread didn't stop in time.")

    partial_thread.join(timeout=5)
    if partial_thread.is_alive():
        logging.warning("partial_thread didn't stop in time.")

    if partial_thread.exception_encountered:
        return None, f"[Partial transcription error: {partial_thread.exception_encountered}]", output_file_path

    return partial_thread.partial_text_state["text"], "", output_file_path


def parse_device_id(selected_device_text: str):
    if not selected_device_text:
        return None
    try:
        parts = selected_device_text.split(":", 1)
        return int(parts[0].strip())
    except Exception as e:
        logging.error(f"Could not parse device from '{selected_device_text}': {e}")
        return None



##########################################################
# Transcription Sink Function
def transcribe_audio(audio_data: np.ndarray, transcription_provider, sample_rate: int = 16000, speaker_lang=None, whisper_model="distil-large-v3") -> str:
    """
    Unified transcribe entry point.
    Chooses faster-whisper or Qwen2Audio based on config.
    """
    loaded_config_data = load_and_log_configs()
    if not transcription_provider:
        # Load default transcription provider via config file
        transcription_provider = loaded_config_data['STT-Settings']['default_transcriber']

    if transcription_provider.lower() == 'qwen2audio':
        logging.info("Transcribing using Qwen2Audio")
        return transcribe_with_qwen2audio(audio_data, sample_rate)

    elif transcription_provider.lower() == "parakeet":
        logging.info("Transcribing using Parakeet")
        # FIXME - implement Parakeet
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            return "Nemo package not found. Please install 'nemo_toolkit[asr]' to use Parakeet."

        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-rnnt-1.1b")

        # Enable local attention
        asr_model.change_attention_model("rel_pos_local_attn", [128, 128])  # local attn

        # Enable chunking for subsampling module
        asr_model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select

        # Transcribe a huge audio file
        transcript = asr_model.transcribe(["<path to a huge audio file>.wav"])

        return transcript

    else:
        logging.info(f"Transcribing using faster-whisper with model: {whisper_model}")
        # The function from your Audio_Transcription_Lib speech_to_text() expects a file path,
        #   so we save the audio_data to a temporary WAV
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            tmp_wav_path = tmp_file.name

        # Now pass to faster-whisper
        try:
            segments = speech_to_text(
                tmp_wav_path,
                whisper_model=whisper_model,
                selected_source_lang=speaker_lang
            )
            if isinstance(segments, dict) and 'error' in segments:
                # handle error
                return f"Error in transcription: {segments['error']}"

            # Merge all segment texts
            final_text = " ".join(seg["Text"] for seg in segments['segments']) if isinstance(segments, dict) else " ".join(
                seg["Text"] for seg in segments)
            return final_text

        finally:
            # Clean up temporary file
            try:
                os.remove(tmp_wav_path)
            except:
                pass

#
# End of Sink Function
##########################################################


##########################################################
#
# Live Audio Transcription Functions

# FIXME - Sample code for live audio transcription
class LiveAudioStreamer:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=1.6):
        """
        :param silence_threshold: amplitude threshold below which we consider "silence"
        :param silence_duration: how many seconds of silence needed to finalize
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stop_event = threading.Event()

        self.last_audio_chunk_time = time.time()
        self.silence_start_time = None

        self.pa = pyaudio.PyAudio()

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback"""
        if status:
            print(f"Stream status: {status}")
        if not self.is_recording:
            return (in_data, pyaudio.paContinue)

        # Convert the raw audio data to a numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        self.audio_queue.put(audio_data.copy())
        return (in_data, pyaudio.paContinue)

    def start(self):
        """Open the audio stream and start recording in a separate thread."""
        self.is_recording = True
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        self.listener_thread = threading.Thread(target=self.listen_loop)
        self.listener_thread.start()

    def stop(self):
        """Stop recording and close the stream."""
        self.is_recording = False
        self.stop_event.set()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.listener_thread.join()
        self.pa.terminate()

    def listen_loop(self):
        """Continuously pull chunks from the queue and detect silence."""
        audio_buffer = []

        while not self.stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            audio_buffer.append(chunk)

            # Check amplitude in this chunk
            amplitude = np.abs(chunk).mean()
            # If amplitude < threshold, we might be in silence
            if amplitude < self.silence_threshold:
                # Mark time
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()
                else:
                    elapsed = time.time() - self.silence_start_time
                    if elapsed >= self.silence_duration:
                        # We have enough silence: finalize
                        print("Silence detected. Finalizing the chunk.")
                        final_audio = np.concatenate(audio_buffer, axis=0).flatten()
                        audio_buffer.clear()
                        # Transcribe the finalized audio
                        # FIXME - Add support for multiple languages/whisper models
                        user_text = transcribe_audio(final_audio, sample_rate=self.sample_rate, whisper_model="distil-large-v3", speaker_lang="en", transcription_provider="faster-whisper")

                        # Then do something with user_text (e.g. add to chatbot)
                        self.handle_transcribed_text(user_text)
                        self.silence_start_time = None
            else:
                # reset silence timer
                self.silence_start_time = None

    def handle_transcribed_text(self, text: str):
        """Hook/callback: override or connect a signal to do something with the transcribed text."""
        print(f"USER SAID: {text}")

# # Usage example
# if __name__ == "__main__":
#     streamer = LiveAudioStreamer(silence_threshold=0.01, silence_duration=1.5)
#     streamer.start()
#     print("Recording... talk, then remain silent for 1.5s to finalize.")
#     time.sleep(15)  # Let it run for 15 seconds
#     streamer.stop()
#     print("Stopped.")

#
# End of Live Audio Transcription Functions
##########################################################


##########################################################
#
# Qwen2-Audio-related Functions

# Load Qwen2Audio (lazy load or load once at startup)
qwen_processor = None
qwen_model = None

def load_qwen2audio():
    global qwen_processor, qwen_model
    if qwen_processor is None or qwen_model is None:
        logging.info("Loading Qwen2Audio model...")
        qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return qwen_processor, qwen_model

def transcribe_with_qwen2audio(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Given a raw audio array, transcribe using Qwen2Audio's built-in ASR capabilities."""
    processor, model = load_qwen2audio()

    # We build a prompt that includes <|audio_bos|><|AUDIO|><|audio_eos|> token(s)
    # The simplest approach is "User: <|AUDIO|>"
    # But Qwen2Audio also uses special tokens <|audio_bos|> and <|audio_eos|>.
    prompt_text = "System: You are a transcription model.\nUser: <|audio_bos|><|AUDIO|><|audio_eos|>\nAssistant:"

    inputs = processor(
        text=prompt_text,
        audios=audio,
        return_tensors="pt",
        sampling_rate=sample_rate
    )
    device = model.device
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    # The raw output has prompt + transcription + possibly more text
    transcription = processor.decode(generated_ids[0], skip_special_tokens=True)

    # Post-process transcription
    # Qwen2Audio might produce additional text.
    # Typically you look for the part after "Assistant:"
    # or remove your system prompt if it appears in the output.
    # A quick approach:
    if "Assistant:" in transcription:
        # e.g. "System: ... User: <|AUDIO|>\nAssistant: Hello here's your text"
        transcription = transcription.split("Assistant:")[-1].strip()

    return transcription

#
# End of Qwen2-Audio-related Functions
##########################################################


##########################################################
#
# Faster Whisper related functions
whisper_model_instance = None
config = load_and_log_configs()
processing_choice = config['processing_choice'] or 'cpu'
total_thread_count = multiprocessing.cpu_count()

class WhisperModel(OriginalWhisperModel):
    tldw_dir = os.path.dirname(os.path.dirname(__file__))
    default_download_root = os.path.join(tldw_dir, 'models', 'Whisper')

    valid_model_sizes = [
        "tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium",
        "large-v1", "large-v2", "large-v3", "large", "distil-large-v2", "distil-medium.en",
        "distil-small.en", "distil-large-v3", "deepdml/faster-distil-whisper-large-v3.5", "deepdml/faster-whisper-large-v3-turbo-ct2",
        "nyrahealth/faster_CrisperWhisper"
    ]

    def __init__(
        self,
        model_size_or_path: str,
        device: str = processing_choice,
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,#total_thread_count, FIXME - I think this should be 0
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        files: Optional[Dict[str, Any]] = None,
        **model_kwargs: Any
    ):
        if download_root is None:
            download_root = self.default_download_root # Use your default path

        os.makedirs(download_root, exist_ok=True) # Ensure your target directory exists
        resolved_identifier = model_size_or_path # Start with the original input

        # Check 1: Does it contain '/' and is NOT an existing local directory/file?
        # This is likely a Hugging Face Hub ID.
        is_potential_hub_id = '/' in model_size_or_path and not os.path.exists(model_size_or_path)

        # Check 2: Is it an existing local directory or file?
        is_existing_local_path = os.path.exists(model_size_or_path)

        if is_potential_hub_id:
            # Assume it's a Hub ID - pass it directly to faster-whisper.
            # faster-whisper will handle downloading it (potentially respecting download_root if configured).
            logging.info(f"Treating '{model_size_or_path}' as a Hugging Face Hub ID.")
            resolved_identifier = model_size_or_path # Pass the Hub ID string as is
        elif is_existing_local_path:
            # It's a local path that exists - pass the absolute path.
             logging.info(f"Treating '{model_size_or_path}' as an existing local path.")
             resolved_identifier = os.path.abspath(model_size_or_path)
        else:
            # Assume it's a standard model size name (e.g., "large-v3").
            # Let faster-whisper handle finding/downloading this standard model.
            # It will likely use the provided `download_root` or its internal default cache.
            logging.info(f"Treating '{model_size_or_path}' as a standard model size name.")
            resolved_identifier = model_size_or_path # Pass the name

            custom_path_check = os.path.join(download_root, model_size_or_path)
            if os.path.isdir(custom_path_check):
                logging.info(f"Found standard model '{model_size_or_path}' in custom download root: {custom_path_check}")
                resolved_identifier = custom_path_check # Use the local path
            else:
                logging.info(f"Standard model '{model_size_or_path}' not in custom root. Passing name to faster-whisper.")
                # resolved_identifier remains the model size name

        # --- Pass the determined identifier and other args to the parent ---
        logging.info(
             f"Initializing faster-whisper with: model='{resolved_identifier}', "
             f"device='{device}', compute_type='{compute_type}', "
             f"download_root='{download_root}', local_files_only={local_files_only}"
        )

        try:
            super().__init__(
                model_size_or_path=resolved_identifier, # Use the corrected identifier
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                download_root=download_root, # Pass your custom root
                local_files_only=local_files_only,
                **model_kwargs
            )
            self.model_identifier = resolved_identifier # Store for reference if needed
            logging.info(f"Successfully initialized WhisperModel: {resolved_identifier}")

        except ValueError as e:
            # Error during faster-whisper init (e.g., invalid model, download failed)
            logging.error(f"Failed to initialize faster_whisper.WhisperModel with '{resolved_identifier}': {e}", exc_info=True)
            # Provide a more specific error message based on the likely cause
            if "Invalid model size" in str(e) or "could not be found" in str(e):
                 raise ValueError(f"The model identifier '{resolved_identifier}' is invalid or could not be loaded/downloaded. Check the name/path and ensure it's accessible.") from e
            else:
                 raise ValueError(f"Error initializing model '{resolved_identifier}': {e}") from e
        except Exception as e:
             # Catch other unexpected errors
             logging.error(f"An unexpected error occurred during faster_whisper.WhisperModel initialization with '{resolved_identifier}': {e}", exc_info=True)
             raise RuntimeError(f"Unexpected error loading model: {resolved_identifier} - {e}") from e

# Implement FIXME
def unload_whisper_model():
    global whisper_model_instance
    if whisper_model_instance is not None:
        del whisper_model_instance
        whisper_model_instance = None
        gc.collect()

whisper_model_cache = {}

def get_whisper_model(model_name, device):
    compute_type = "float16" if "cuda" in device else "int8" # Example compute type logic
    cache_key = (model_name, device, compute_type)

    if cache_key not in whisper_model_cache:
        logging.info(f"Cache miss. Initializing WhisperModel for key: {cache_key}")
        try:
            # This now calls the *corrected* WhisperModel.__init__
            instance = WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type
                # Pass download_root explicitly here if it's NOT handled by the class default
                # download_root=WhisperModel.default_download_root
            )
            whisper_model_cache[cache_key] = instance
        except (ValueError, RuntimeError) as e:
            logging.error(f"Fatal error creating whisper model instance for key {cache_key}: {e}")
            raise # Re-raise the exception
    else:
        logging.debug(f"Cache hit. Reusing existing WhisperModel instance for key: {cache_key}")

    return whisper_model_cache[cache_key]


# Transcribe .wav into .segments.json
#DEBUG
#@profile
# FIXME - I feel like the `vad_filter` should be enabled by default....
@timeit
def format_time(total_seconds: float) -> str:
    """
    Convert a float number of seconds into HH:MM:SS format.
    E.g., 123.45 -> '00:02:03'
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def speech_to_text(
    audio_file_path: str,
    whisper_model: str = 'distil-large-v3',
    selected_source_lang: str = 'en',  # Changed order of parameters
    vad_filter: bool = False,
    diarize: bool = False
):
    """
    Transcribe audio to text using a Whisper model.
    Returns a list of segment dictionaries or raises RuntimeError on failure.

    Args:
        audio_file_path: Path to the WAV audio file.
        whisper_model: Name or path of the faster-whisper model.
        selected_source_lang: Language code (e.g., 'en', 'es') or None for auto-detect.
        vad_filter: Apply Voice Activity Detection filter.
        diarize: Placeholder for diarization flag (not implemented in this function).

    Returns:
        List[Dict]: A list of segments, where each segment is a dict containing
                   'start_seconds', 'end_seconds', and 'Text'.

    Raises:
        ValueError: If no audio file path is provided.
        RuntimeError: If transcription fails or produces no segments.
        FileNotFoundError: If the audio file does not exist.
    """

    log_counter("speech_to_text_attempt", labels={"file_path": audio_file_path, "model": whisper_model})
    time_start = time.time()

    if not audio_file_path:
        log_counter("speech_to_text_error", labels={"error": "No audio file provided"})
        raise ValueError("speech-to-text: No audio file provided")

    # Convert the string to a Path object and ensure it's resolved (absolute path)
    file_path = Path(audio_file_path).resolve()
    if not file_path.exists():
        log_counter("speech_to_text_error", labels={"error": "Audio file not found", "file_path": str(file_path)})
        raise FileNotFoundError(f"speech-to-text: Audio file not found at {file_path}")

    logging.info(f"speech-to-text: Starting transcription for: {file_path}")
    logging.info(f"speech-to-text: Model={whisper_model}, Lang={selected_source_lang or 'auto'}, VAD={vad_filter}")

    try:
        # Construct output filenames in the same directory as the input file
        sanitized_whisper_model_name = sanitize_filename(whisper_model)
        out_file = file_path.with_name(f"{file_path.stem}-whisper_model-{sanitized_whisper_model_name}.segments.json")
        prettified_out_file = file_path.with_name(f"{file_path.stem}-whisper_model-{sanitized_whisper_model_name}.segments_pretty.json")

        options = dict(beam_size=5, best_of=5, vad_filter=vad_filter) # Simplified beam options
        # FIXME - was 10? Evaluate...
        if selected_source_lang:
            options["language"] = selected_source_lang
        # Add word_timestamps=True if needed later for more granular data
        # options["word_timestamps"] = True

        transcribe_options = dict(task="transcribe", **options)

        # Get model instance (cached)
        whisper_model_instance = get_whisper_model(whisper_model, processing_choice)

        # Perform transcription
        segments_raw, info = whisper_model_instance.transcribe(str(file_path), **transcribe_options)

        detected_lang = info.language
        lang_prob = info.language_probability
        logging.info(f"speech-to-text: Detected language: {detected_lang} (Confidence: {lang_prob:.2f})")
        # You might want to store detected_lang somewhere if using auto-detect

        segments = []
        for segment_chunk in segments_raw:
            # Store raw float seconds
            chunk = {
                "start_seconds": segment_chunk.start,
                "end_seconds": segment_chunk.end,
                "Text": segment_chunk.text.strip() # Strip whitespace from text
            }
            logging.debug(f"Segment: {chunk}")
            segments.append(chunk)
            # Log with limited precision for readability
            logging.debug(f"Segment: [{chunk['start_seconds']:.2f}-{chunk['end_seconds']:.2f}] {chunk['Text'][:100]}...")

        if segments:
            # Insert metadata at the start of the first segment if desired
            segments[0]["Text"] = (
                f"This text was transcribed using whisper model: {whisper_model}\n"
                f"Detected language: {detected_lang}\n\n"
                f"{segments[0]['Text']}"
            )

        if not segments:
            log_counter("speech_to_text_error", labels={"error": "No transcription produced"})
            raise RuntimeError("No transcription produced. The audio file may be invalid or empty.")

        transcription_time = time.time() - time_start
        logging.info(f"speech-to-text: Transcription completed in {transcription_time:.2f} seconds. Segments: {len(segments)}")
        log_histogram(
            "speech_to_text_duration",
            transcription_time,
            labels={"file_path": str(file_path), "model": whisper_model}
        )
        log_counter("speech_to_text_success", labels={"file_path": str(file_path), "model": whisper_model, "segments": len(segments)})

        gc.collect() # Suggest garbage collection
        return segments # Return the list of segment dictionaries

    except Exception as e:
        logging.error(f"speech-to-text: Error transcribing audio file {file_path}: {e}", exc_info=True)
        log_counter(
            "speech_to_text_error",
            labels={"file_path": str(file_path), "model": whisper_model, "error": type(e).__name__}
        )
        # Re-raise as a runtime error for the caller to handle
        raise RuntimeError(f"speech-to-text: Error during transcription of {file_path.name}") from e

#
# End of Faster Whisper related functions
##########################################################

##########################################################
#
# Audio Conversion

class ConversionError(Exception):
    """Custom exception for errors during audio/video conversion."""
    pass

def _find_ffmpeg() -> str:
    """Finds the ffmpeg executable."""
    # 1. Check specific relative path (if applicable to your structure)
    if os.name == 'nt':
        # Adjust this path based on your project structure relative to this file
        # Example: Assuming 'Bin' is two levels up from this script's dir
        script_dir = Path(__file__).parent
        bin_dir = script_dir.parent.parent / "Bin" # Adjust depth as needed
        ffmpeg_exe = bin_dir / "ffmpeg.exe"
        if ffmpeg_exe.exists():
            logging.debug(f"Found ffmpeg at specific Windows path: {ffmpeg_exe}")
            return str(ffmpeg_exe)

    # 2. Check environment variable (useful for Docker/server setups)
    ffmpeg_env = os.environ.get("FFMPEG_PATH")
    if ffmpeg_env and Path(ffmpeg_env).exists():
        logging.debug(f"Found ffmpeg via FFMPEG_PATH env var: {ffmpeg_env}")
        return ffmpeg_env

    # 3. Check PATH using shutil.which (cross-platform)
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logging.debug(f"Found ffmpeg in system PATH: {ffmpeg_path}")
        return ffmpeg_path

    # 4. If not found, raise error
    raise FileNotFoundError("ffmpeg executable not found in Bin directory, FFMPEG_PATH, or system PATH.")

# os.system(r'.\Bin\ffmpeg.exe -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
#DEBUG
#@profile
@timeit
def convert_to_wav(video_file_path: str, offset: int = 0, overwrite: bool = False) -> str:
    """
    Converts a video or audio file to a standardized WAV format (16kHz, mono, PCM s16le)
    suitable for Whisper transcription.

    Args:
        video_file_path: Path to the input media file.
        offset: Start offset (currently unused by ffmpeg command).
        overwrite: If True, overwrite the output WAV file if it exists.

    Returns:
        The path to the generated WAV file.

    Raises:
        FileNotFoundError: If the input file doesn't exist.
        ConversionError: If the ffmpeg conversion process fails.
        RuntimeError: If ffmpeg executable cannot be found.
    """
    log_counter("convert_to_wav_attempt", labels={"file_path": video_file_path})
    start_time = time.time()

    input_path = Path(video_file_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {video_file_path}")

    # Output path in the same directory as input
    out_path = input_path.with_suffix(".wav")

    if out_path.exists() and not overwrite:
        logging.info(f"Skipping conversion as WAV file already exists and overwrite=False: {out_path}")
        log_counter("convert_to_wav_skipped", labels={"file_path": video_file_path})
        return str(out_path)

    # Determine ffmpeg executable path
    ffmpeg_cmd = "ffmpeg" # Default for non-Windows or if specific path fails
    if sys.platform.startswith('win'):
        # Look for ffmpeg relative to this file's location structure
        # Assumes: .../app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Lib.py
        # Goal: .../app/Bin/ffmpeg.exe
        try:
            APP_DIR = Path(__file__).resolve().parents[3] # .../app
            BIN_DIR = APP_DIR / "Bin"
            FFMPEG_WIN_PATH = BIN_DIR / "ffmpeg.exe"
            if FFMPEG_WIN_PATH.exists():
                ffmpeg_cmd = str(FFMPEG_WIN_PATH)
                logging.debug(f"Using specific ffmpeg path: {ffmpeg_cmd}")
            else:
                logging.warning(f"ffmpeg.exe not found at {FFMPEG_WIN_PATH}. Falling back to 'ffmpeg' in PATH.")
        except IndexError:
             logging.warning("Could not determine app directory structure. Falling back to 'ffmpeg' in PATH.")

    # Verify ffmpeg command works
    try:
        subprocess.run([ffmpeg_cmd, "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        logging.debug(f"Confirmed ffmpeg command '{ffmpeg_cmd}' is available.")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        error_msg = f"ffmpeg command '{ffmpeg_cmd}' not found or failed execution. Please ensure ffmpeg is installed and in the system PATH or in the expected ./Bin directory. Error: {e}"
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e


    logging.info(f"Starting conversion to WAV: '{input_path.name}' -> '{out_path.name}'")

    command = [
        ffmpeg_cmd,
        "-i", str(input_path),
        "-ss", str(offset),           # Use offset if needed (e.g., "00:00:10" or 10)
        "-ar", "16000",               # Audio sample rate (good for Whisper)
        "-ac", "1",                   # Mono audio channel (good for Whisper)
        "-c:a", "pcm_s16le",          # Standard WAV audio codec
        "-y",                         # Overwrite output file without asking
        str(out_path)
    ]

    try:
        # Execute ffmpeg command
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL, # Prevent ffmpeg from waiting for stdin
            capture_output=True,      # Capture stdout and stderr
            text=True,                # Decode output as text
            check=False               # Don't raise exception on non-zero exit code automatically
        )

        # Check result
        if result.returncode != 0:
            error_details = result.stderr or result.stdout or "No output captured"
            # Clean up potentially corrupted output file
            if out_path.exists():
                try: out_path.unlink()
                except OSError: pass
            raise ConversionError(f"FFmpeg conversion failed (code {result.returncode}) for '{input_path.name}'. Error: {error_details.strip()}")
        else:
            logging.info(f"Conversion to WAV completed successfully: {out_path}")
            if result.stderr: # Log warnings even on success
                logging.warning(f"FFmpeg potential warnings for '{input_path.name}': {result.stderr.strip()}")
            log_counter("convert_to_wav_success", labels={"file_path": video_file_path})

    except ConversionError:
         # Re-raise ConversionError explicitly to ensure it's caught
         log_counter("convert_to_wav_error", labels={"file_path": video_file_path, "error": "ffmpeg_failed"})
         raise
    except Exception as e:
        # Catch other potential errors like permissions, etc.
        error_msg = f"Unexpected error during ffmpeg execution for '{input_path.name}': {e}"
        logging.error(error_msg, exc_info=True)
        log_counter("convert_to_wav_error", labels={"file_path": video_file_path, "error": str(e)})
        # Clean up potentially corrupted output file
        if out_path.exists():
             try: out_path.unlink()
             except OSError: pass
        raise ConversionError(error_msg) from e # Wrap other errors in ConversionError

    conversion_time = time.time() - start_time
    log_histogram("convert_to_wav_duration", conversion_time, labels={"file_path": video_file_path})

    gc.collect()
    return str(out_path)
#
# End of Audio Conversion Functions
##########################################################


##########################################################
#
# Audio Recording Functions

def test_device_availability(device_id):
    """Test if a device is actually available for recording."""
    if device_id is None:
        return False

    p = pyaudio.PyAudio()
    try:
        # Try to get device info
        device_info = p.get_device_info_by_index(device_id)
        if not device_info or device_info['maxInputChannels'] < 1:
            return False

        # Try to open stream briefly
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=44100,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=1024,
            start=False
        )
        stream.close()
        return True
    except Exception as e:
        logging.debug(f"Device {device_id} not available: {e}")
        return False
    finally:
        p.terminate()


@timeit
def record_audio(duration, sample_rate=16000, chunk_size=1024):
    log_counter("record_audio_attempt", labels={"duration": duration})
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    print("Recording...")
    frames = []
    stop_recording = threading.Event()
    audio_queue = queue.Queue()

    def audio_callback():
        for _ in range(0, int(sample_rate / chunk_size * duration)):
            if stop_recording.is_set():
                break
            data = stream.read(chunk_size)
            audio_queue.put(data)

    audio_thread = threading.Thread(target=audio_callback)
    audio_thread.start()

    return p, stream, audio_queue, stop_recording, audio_thread


@timeit
def stop_recording_infinite(p, stream, audio_queue, stop_recording_event, audio_thread):
    log_counter("stop_recording_attempt")
    start_time = time.time()
    stop_recording_event.set()
    audio_thread.join()

    frames = []
    while not audio_queue.empty():
        frames.append(audio_queue.get())

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    stop_time = time.time() - start_time
    log_histogram("stop_recording_duration", stop_time)
    log_counter("stop_recording_success")
    return b''.join(frames)


@timeit
def save_audio_temp(audio_data, sample_rate=16000):
    """Save audio data to temporary WAV file with proper format handling."""
    log_counter("save_audio_temp_attempt")

    try:
        # Convert tensor to numpy array if needed
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()

        # Ensure float32 format and make writable
        audio_data = np.asarray(audio_data, dtype=np.float32).copy()

        # Normalize audio
        max_amp = np.max(np.abs(audio_data))
        if max_amp > 1.0:
            audio_data /= max_amp

        # Convert to int16
        audio_data_int16 = np.int16(audio_data * 32767)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wavfile.write(temp_file.name, sample_rate, audio_data_int16)
            log_counter("save_audio_temp_success")
            return temp_file.name

    except Exception as e:
        logging.error(f"Error saving temp audio: {str(e)}")
        log_counter("save_audio_temp_error")
        return None


# Non-Filtering version
def get_system_audio_devices() -> List[Dict]:
    """
    Return available audio devices for system audio recording with better
    identification of loopback capabilities.
    """
    # Keywords commonly found in device names that can capture system output
    loopback_keywords = [
        "loopback",  # WASAPI loopback
        "stereo mix",  # Realtek driver
        "monitor",  # PulseAudio monitor on Linux
        "blackhole",  # macOS loopback driver
        "soundflower",  # older macOS loopback driver
        "what u hear",  # Sound Blaster
        "output",  # Generic term that might indicate system output
        "mix"  # Common in stereo mix devices
    ]

    devices = []
    try:
        host_apis = sd.query_hostapis()
        all_devs = sd.query_devices()

        for device_index, device in enumerate(all_devs):
            # Only include input devices
            if device["max_input_channels"] > 0:
                name_lower = device["name"].lower()
                api_name = host_apis[device["hostapi"]]["name"]

                # Check if it might be a loopback device
                is_likely_loopback = any(keyword in name_lower for keyword in loopback_keywords)

                devices.append({
                    "id": device_index,
                    "name": f"{device['name']} ({api_name})" +
                            (" [SYSTEM AUDIO]" if is_likely_loopback else ""),
                    "hostapi": device["hostapi"],
                    "max_input_channels": device["max_input_channels"],
                    "max_output_channels": device["max_output_channels"],
                    "rate": device["default_samplerate"],
                    "is_loopback": is_likely_loopback
                })

        # Sort to put potential loopback devices first
        devices.sort(key=lambda x: (not x.get("is_loopback"), x["name"]))
    except Exception as e:
        logging.error(f"Error enumerating audio devices: {e}")

    return devices
# Filtering version
# def get_system_audio_devices() -> List[Dict]:
#     """Get list of available system audio devices with their capabilities"""
#     devices = []
#     host_apis = sd.query_hostapis()
#
#     for device_index, device in enumerate(sd.query_devices()):
#         if device['max_input_channels'] > 0:
#             # Windows loopback devices show up as inputs
#             api_name = host_apis[device['hostapi']]['name']
#             devices.append({
#                 'id': device_index,
#                 'name': f"{device['name']} ({api_name})",
#                 'is_loopback': 'loopback' in device['name'].lower(),
#                 'hostapi': device['hostapi'],
#                 'max_channels': device['max_input_channels'],
#                 'rate': device['default_samplerate']
#             })
#
#     # Sort devices with loopback first
#     return sorted(devices, key=lambda x: not x['is_loopback'])


def record_system_audio(duration: float, device_id: int, sample_rate: int = 44100,
                        channels: int = 2, subtype: str = 'PCM_16') -> str:
    """
    Record system audio output to a temporary WAV file
    Returns path to recorded file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    try:
        # Configure recording settings based on device capabilities
        device_info = sd.query_devices(device_id)
        actual_sample_rate = int(device_info['default_samplerate'] if device_info['default_samplerate'] > 0
                                 else sample_rate)

        logging.info(f"Starting system audio recording (Duration: {duration}s, "
                     f"Device: {device_info['name']}, SR: {actual_sample_rate})")

        audio_data = sd.rec(
            int(duration * actual_sample_rate),
            samplerate=actual_sample_rate,
            channels=min(channels, device_info['max_input_channels']),
            device=device_id,
            dtype=np.int16,
            blocking=True
        )

        # Save to WAV file
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(min(channels, device_info['max_input_channels']))
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(actual_sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        logging.info(f"Recording saved to {temp_file.name}")
        return temp_file.name

    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        raise RuntimeError(f"Recording failed: {str(e)}")

#
# End of Audio Recording Functions
##########################################################


##########################################################
#
# Transcript Handling/Processing

def format_transcription_with_timestamps(segments, keep_timestamps=True):
    """
    Formats the transcription segments with or without timestamps.
    Handles numeric seconds or pre-formatted HH:MM:SS strings.

    Parameters:
        segments (list): List of transcription segments (dicts with 'Time_Start', 'Time_End', 'Text').
        keep_timestamps (bool): Whether to include timestamps.

    Returns:
        str: Formatted transcription.
    """
    if not segments:
        return ""

    formatted_lines = []
    if keep_timestamps:
        for segment in segments:
            start = segment.get('Time_Start', 0)
            end = segment.get('Time_End', 0)
            text = segment.get('Text', '').strip()

            start_str = start
            end_str = end

            # Convert numeric seconds to HH:MM:SS if needed
            if isinstance(start, (int, float)):
                try:
                    start_str = time.strftime('%H:%M:%S', time.gmtime(float(start)))
                except (ValueError, TypeError, OSError): # Handle potential errors like large floats
                    start_str = f"{start:.2f}s" # Fallback to seconds
            if isinstance(end, (int, float)):
                try:
                    end_str = time.strftime('%H:%M:%S', time.gmtime(float(end)))
                except (ValueError, TypeError, OSError):
                    end_str = f"{end:.2f}s" # Fallback to seconds

            formatted_lines.append(f"[{start_str}-{end_str}] {text}")
    else:
        for segment in segments:
            text = segment.get('Text', '').strip()
            if text: # Avoid adding empty lines if a segment has no text
                formatted_lines.append(text)

    return "\n".join(formatted_lines)
#
# End of Transcript Handling/Processing
##########################################################


#
#
#######################################################################################################################
