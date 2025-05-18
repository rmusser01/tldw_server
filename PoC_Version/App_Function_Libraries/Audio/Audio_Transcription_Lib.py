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
#
# Import Local
from App_Function_Libraries.Utils.Utils import sanitize_filename, load_and_log_configs, logging
from App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram, timeit
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
        "distil-small.en", "distil-large-v3", "deepdml/faster-whisper-large-v3-turbo-ct2",
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
            download_root = self.default_download_root

        os.makedirs(download_root, exist_ok=True)

        # FIXME - validate....
        # Also write an integration test...
        # Check if model_size_or_path is a valid model size
        if model_size_or_path in self.valid_model_sizes:
            # It's a model size, so we'll use the download_root
            model_path = os.path.join(download_root, model_size_or_path)
            if not os.path.isdir(model_path):
                # If it doesn't exist, we'll let the parent class download it
                model_size_or_path = model_size_or_path  # Keep the original model size
            else:
                # If it exists, use the full path
                model_size_or_path = model_path
        else:
            # It's not a valid model size, so assume it's a path
            model_size_or_path = os.path.abspath(model_size_or_path)

        super().__init__(
            model_size_or_path,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=local_files_only,
# Maybe? idk, FIXME
#            files=files,
#            **model_kwargs
        )

# Implement FIXME
def unload_whisper_model():
    global whisper_model_instance
    if whisper_model_instance is not None:
        del whisper_model_instance
        whisper_model_instance = None
        gc.collect()


def get_whisper_model(model_name, device, ):
    #FIXME - remove call to huggingface if whisper model exists on device
    global whisper_model_instance
    if whisper_model_instance is None:
        logging.info(f"Initializing new WhisperModel with size {model_name} on device {device}")
        # FIXME - add compute_type="int8"
        whisper_model_instance = WhisperModel(model_name, device=device, compute_type="default")
    return whisper_model_instance


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
    Transcribe audio to text using a Whisper model and optionally handle diarization.
    Saves JSON output to {filename}-whisper_model-{model}.segments.json in the same directory.
    """

    log_counter("speech_to_text_attempt", labels={"file_path": audio_file_path, "model": whisper_model})
    time_start = time.time()

    if not audio_file_path:
        log_counter("speech_to_text_error", labels={"error": "No audio file provided"})
        raise ValueError("speech-to-text: No audio file provided")

    # Convert the string to a Path object and ensure it's resolved (absolute path)
    file_path = Path(audio_file_path).resolve()
    logging.info("speech-to-text: Audio file path: {file_path}")

    try:
        # Get file extension and base name
        file_ending = file_path.suffix

        # Construct output filenames in the same directory as the input file
        sanitized_whisper_model_name = sanitize_filename(whisper_model)
        out_file = file_path.with_name(f"{file_path.stem}-whisper_model-{sanitized_whisper_model_name}.segments.json")
        prettified_out_file = file_path.with_name(f"{file_path.stem}-whisper_model-{sanitized_whisper_model_name}.segments_pretty.json")

        if out_file.exists():
            logging.info(f"speech-to-text: Segments file already exists: {out_file}")
            with out_file.open() as f:
                segments = json.load(f)
            return segments

        logging.info('speech-to-text: Starting transcription...')
        # FIXME - revisit this
        options = dict(language=selected_source_lang, beam_size=10, best_of=10, vad_filter=vad_filter)
        transcribe_options = dict(task="transcribe", **options)
        # use function and config at top of file
        logging.debug(f"speech-to-text: Using whisper model: {whisper_model}", )

        whisper_model_instance = get_whisper_model(whisper_model, processing_choice)
        segments_raw, info = whisper_model_instance.transcribe(str(file_path), **transcribe_options)

        segments = []
        for segment_chunk in segments_raw:
            # Format time from seconds to HH:MM:SS
            start_str = format_time(segment_chunk.start)
            end_str = format_time(segment_chunk.end)

            chunk = {
                "Time_Start": start_str,
                "Time_End": end_str,
                "Text": segment_chunk.text
            }
            logging.debug(f"Segment: {chunk}")
            segments.append(chunk)
            logging.info(f"{start_str} - {end_str} | {segment_chunk.text}")  # Use HH:MM:SS in logs

        if segments:
            # Insert metadata at the start of the first segment if desired
            segments[0]["Text"] = (
                f"This text was transcribed using whisper model: {whisper_model}\n\n"
                + segments[0]["Text"]
            )

        if not segments:
            log_counter("speech_to_text_error", labels={"error": "No transcription produced"})
            raise RuntimeError("No transcription produced. The audio file may be invalid or empty.")

        transcription_time = time.time() - time_start
        logging.info(f"speech-to-text: Transcription completed in {transcription_time} seconds")
        log_histogram(
            "speech_to_text_duration",
            transcription_time,
            labels={"file_path": str(file_path), "model": whisper_model}
        )
        log_counter("speech_to_text_success", labels={"file_path": str(file_path), "model": whisper_model})

        # Save the segments to a JSON file - prettified and non-prettified
        # FIXME refactor so this is an optional flag to save either the prettified json file or the normal one
        save_json = True
        if save_json:
            logging.info("speech-to-text: Saving segments to JSON file")
            output_data = {'segments': segments}

            logging.info(f"speech-to-text: Saving JSON to {out_file}",)
            with out_file.open('w', encoding='utf-8') as f:
                json.dump(output_data, f)

            # free up memory
            del output_data
            gc.collect()

            logging.info(f"speech-to-text: Saving prettified JSON to {prettified_out_file}")
            with prettified_out_file.open('w', encoding='utf-8') as f:
                json.dump({'segments': segments}, f, indent=2)

        logging.debug(f"speech-to-text: returning {segments[:500]}")
        gc.collect()
        return segments

    except Exception as e:
        logging.error(f"speech-to-text: Error transcribing audio: {e}")
        log_counter(
            "speech_to_text_error",
            labels={"file_path": str(file_path), "model": whisper_model, "error": str(e)}
        )
        raise RuntimeError("speech-to-text: Error transcribing audio") from e

#
# End of Faster Whisper related functions
##########################################################

##########################################################
#
# Audio Conversion

# os.system(r'.\Bin\ffmpeg.exe -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
#DEBUG
#@profile
@timeit
def convert_to_wav(video_file_path, offset=0, overwrite=False):
    log_counter("convert_to_wav_attempt", labels={"file_path": video_file_path})
    start_time = time.time()

    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    if os.path.exists(out_path) and not overwrite:
        print(f"File '{out_path}' already exists. Skipping conversion.")
        logging.info(f"Skipping conversion as file already exists: {out_path}")
        log_counter("convert_to_wav_skipped", labels={"file_path": video_file_path})
        return out_path

    print("Starting conversion process of .m4a to .WAV")
    out_path = os.path.splitext(video_file_path)[0] + ".wav"

    try:
        if os.name == "nt":
            logging.debug("ffmpeg being ran on windows")

            if sys.platform.startswith('win'):
                ffmpeg_cmd = ".\\Bin\\ffmpeg.exe"
                logging.debug(f"ffmpeg_cmd: {ffmpeg_cmd}")
            else:
                ffmpeg_cmd = 'ffmpeg'  # Assume 'ffmpeg' is in PATH for non-Windows systems

            command = [
                ffmpeg_cmd,  # Assuming the working directory is correctly set where .\Bin exists
                "-ss", "00:00:00",  # Start at the beginning of the video
                "-i", video_file_path,
                "-ar", "16000",  # Audio sample rate
                "-ac", "1",  # Number of audio channels
                "-c:a", "pcm_s16le",  # Audio codec
                out_path
            ]
            try:
                # Redirect stdin from null device to prevent ffmpeg from waiting for input
                with open(os.devnull, 'rb') as null_file:
                    result = subprocess.run(command, stdin=null_file, text=True, capture_output=True)
                if result.returncode == 0:
                    logging.info("FFmpeg executed successfully")
                    logging.debug(f"FFmpeg output: {result.stdout}")
                else:
                    logging.error("Error in running FFmpeg")
                    logging.error(f"FFmpeg stderr: {result.stderr}")
                    raise RuntimeError(f"FFmpeg error: {result.stderr}")
            except Exception as e:
                logging.error("Error occurred - ffmpeg doesn't like windows")
                raise RuntimeError("ffmpeg failed")
        elif os.name == "posix":
            os.system(f'ffmpeg -ss 00:00:00 -i "{video_file_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{out_path}"')
        else:
            raise RuntimeError("Unsupported operating system")
        logging.info(f"Conversion to WAV completed: {out_path}")
        log_counter("convert_to_wav_success", labels={"file_path": video_file_path})
    except Exception as e:
        logging.error(f"speech-to-text: Error transcribing audio: {str(e)}")
        log_counter("convert_to_wav_error", labels={"file_path": video_file_path, "error": str(e)})
        return {"error": str(e)}

    conversion_time = time.time() - start_time
    log_histogram("convert_to_wav_duration", conversion_time, labels={"file_path": video_file_path})

    gc.collect()
    return out_path

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

#
#
#######################################################################################################################
