# Live_Recording.py
# Description: Gradio UI for live audio recording and transcription.
#
# Import necessary modules and functions
import os
import queue
import threading
import time
#
# External Imports
import gradio as gr
import numpy as np
import pyaudio
#
# Local Imports
from PoC_Version.App_Function_Libraries.Audio.Audio_Transcription_Lib import (save_audio_temp,
                                                                              stop_recording_infinite, transcribe_audio)
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_to_database
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from PoC_Version.App_Function_Libraries.Utils.Utils import logging

#
#######################################################################################################################
#
# Functions:

#################### MODELS ####################

whisper_models = [
    "tiny.en", "tiny", "base.en", "base", "small.en", "small",
    "medium.en", "medium", "large-v1", "large-v2", "large-v3", "large",
    "distil-large-v2", "distil-medium.en", "distil-small.en",
    "distil-large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "nyrahealth/faster_CrisperWhisper"
]

TRANSCRIPTION_METHODS = ["faster-whisper", "qwen2audio", "parakeet"]

#################### RECORDING ####################

def record_audio_indef(sample_rate=16000, chunk_size=1024):
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size
    )
    stop_event = threading.Event()
    audio_queue = queue.Queue()

    def audio_callback():
        while not stop_event.is_set():
            data = stream.read(chunk_size, exception_on_overflow=False)
            audio_queue.put(data)

    audio_thread = threading.Thread(target=audio_callback, daemon=True)
    audio_thread.start()
    return p, stream, audio_queue, stop_event, audio_thread


class PartialTranscriptionThread(threading.Thread):
    def __init__(
        self,
        audio_queue,
        stop_event,
        transcription_method,
        partial_text_state,
        lock,
        sample_rate=16000,
        update_interval=2.0,
        whisper_model="distil-large-v3",
        speaker_lang="en"
    ):
        super().__init__(daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.transcription_method = transcription_method
        self.partial_text_state = partial_text_state
        self.lock = lock
        self.sample_rate = sample_rate
        self.update_interval = update_interval
        self.full_audio = []
        self.audio_buffer = []

        self.last_ts = time.time()
        self.whisper_model = whisper_model
        self.speaker_lang = speaker_lang

    def run(self):
        while not self.stop_event.is_set():
            now = time.time()
            if now - self.last_ts < self.update_interval:
                time.sleep(0.1)
                continue

            # Gather any new audio chunks from the queue
            new_chunks = []
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get_nowait()
                new_chunks.append(chunk)
                # ### CHANGED: keep a copy in full_audio
                self.full_audio.append(chunk)

            if new_chunks:
                self.audio_buffer.extend(new_chunks)

            # If we have no new audio, skip
            if not self.audio_buffer:
                continue

            combined_data = b"".join(self.audio_buffer)
            audio_np = np.frombuffer(combined_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Only do partial if at least 1s of audio
            if len(audio_np) > self.sample_rate:
                try:
                    partial_result = transcribe_audio(
                        audio_np,
                        self.transcription_method,
                        sample_rate=self.sample_rate,
                        whisper_model=self.whisper_model,
                        speaker_lang=self.speaker_lang
                    )
                    with self.lock:
                        self.partial_text_state["text"] = partial_result
                except Exception as e:
                    logging.error(f"Partial transcript error: {str(e)}")

                # Keep only last 5s in the rolling buffer for partial
                max_bytes = 5 * self.sample_rate * 2
                if len(combined_data) > max_bytes:
                    self.audio_buffer = [combined_data[-max_bytes:]]
                else:
                    self.audio_buffer = [combined_data]

            self.last_ts = time.time()


def toggle_recording(
    is_recording,
    recording_state,
    transcription_method,
    live_update,
    save_recording,
    partial_text_state,
    final_text_state,
    whisper_model,
    speaker_lang="en"
):
    """
    Returns:
      (new_recording_state,
       new_is_recording_bool,
       new_button_label,
       partial_text,
       final_text,
       audio_file_path)
    """

    # ---------------- START ----------------
    if not is_recording:
        # Start indefinite recording
        log_counter("live_recording_start_attempt")
        p, stream, audio_queue, stop_event, audio_thread = record_audio_indef()
        log_counter("live_recording_start_success")

        time.sleep(0.25)  # small buffer

        partial_thread = None
        lock = threading.Lock()
        if live_update:
            # Start the partial transcription thread if live updates enabled
            partial_thread = PartialTranscriptionThread(
                audio_queue=audio_queue,
                stop_event=stop_event,
                transcription_method=transcription_method,
                partial_text_state=partial_text_state,
                lock=lock,
                sample_rate=16000,
                update_interval=2.0,
                whisper_model=whisper_model,
                speaker_lang=speaker_lang
            )
            partial_thread.start()

        # store
        new_state = {
            "p": p,
            "stream": stream,
            "audio_queue": audio_queue,
            "stop_event": stop_event,
            "audio_thread": audio_thread,
            "partial_thread": partial_thread,
            "lock": lock,
            "whisper_model": whisper_model
        }

        # CLEAR partial/final from prior session
        partial_text_state["text"] = ""
        final_text_state["text"] = ""

        return (
            new_state,
            True,
            "Stop Recording",
            partial_text_state["text"],
            final_text_state["text"],
            None  # No audio yet
        )

    # ---------------- STOP ----------------
    # if no recording_state => nothing to stop
    if not recording_state:
        return (
            None,
            False,
            "Start Recording",
            partial_text_state["text"],
            final_text_state["text"],
            None
        )

    try:
        p = recording_state["p"]
        stream = recording_state["stream"]
        audio_queue = recording_state["audio_queue"]
        stop_event = recording_state["stop_event"]
        audio_thread = recording_state["audio_thread"]
        partial_thread = recording_state["partial_thread"]
        whisper_model = recording_state["whisper_model"]

        # Signal threads to stop
        stop_event.set()
        audio_thread.join()

        # if partial_thread is running, all audio is in partial_thread.full_audio
        if partial_thread is not None:
            partial_thread.join(timeout=5)  # wait for partial thread to finish
            # Final raw audio is everything partial_thread collected
            raw_audio = b"".join(partial_thread.full_audio)
        else:
            # If partial transcription was NOT used, we still do the old approach
            raw_audio = stop_recording_infinite(
                p, stream, audio_queue, stop_event, audio_thread
            )
            while not audio_queue.empty():
                raw_audio += audio_queue.get_nowait()

        p.terminate()

        if not raw_audio:
            final_text_state["text"] = "[No audio recorded]"
            return (
                None,
                False,
                "Start Recording",
                partial_text_state["text"],
                final_text_state["text"],
                None
            )

        # Transcribe
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            final_res = transcribe_audio(
                audio_np,
                transcription_method,
                sample_rate=16000,
                whisper_model=whisper_model,
                speaker_lang=speaker_lang
            )
        except Exception as e:
            final_res = f"[Error in final transcription: {str(e)}]"
            logging.error(final_res)

        final_text_state["text"] = final_res

        # Save the WAV => for audio player
        audio_file = save_audio_temp(audio_np)
        if not save_recording:
            # Create a background thread that will delete the file after 15 seconds
            removal_thread = threading.Thread(
                target=remove_file_after_delay,
                args=(audio_file, 15),
                daemon=True
            )
            removal_thread.start()

        return (
            None,
            False,
            "Start Recording",
            partial_text_state["text"],  # partial stays as-is
            final_text_state["text"],    # final set
            audio_file
        )

    except Exception as e:
        msg = f"Error stopping recording: {e}"
        logging.error(msg)
        final_text_state["text"] = msg
        return (
            None,
            False,
            "Start Recording",
            partial_text_state["text"],
            final_text_state["text"],
            None
        )



def poll_partial_text(live, partial_text_state):
    if live:
        return partial_text_state.get("text", "")
    return "(Live update disabled)"


def save_transcription_to_db(transcription, custom_title):
    log_counter("save_transcription_to_db_attempt")
    start_time = time.time()
    if not custom_title.strip():
        custom_title = "Self-recorded Audio"

    try:
        url = "self_recorded"
        info_dict = {
            "title": custom_title,
            "uploader": "self-recorded",
            "webpage_url": url
        }
        segments = [{"Text": transcription}]
        summary = ""
        keywords = ["self-recorded", "audio"]
        custom_prompt_input = ""
        whisper_model = "self-recorded"
        media_type = "audio"

        result = add_media_to_database(
            url=url,
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keywords,
            custom_prompt_input=custom_prompt_input,
            whisper_model=whisper_model,
            media_type=media_type
        )
        dur = time.time() - start_time
        log_histogram("save_transcription_to_db_duration", dur)
        log_counter("save_transcription_to_db_success")
        return f"Transcription saved to DB. {result}"
    except Exception as e:
        logging.error(str(e))
        log_counter("save_transcription_to_db_error", labels={"error": str(e)})
        return f"Error saving: {str(e)}"


def update_custom_title_visibility(save_to_db):
    return gr.update(visible=save_to_db)


def remove_file_after_delay(file_path, delay=15):
    """Remove a file after `delay` seconds."""
    time.sleep(delay)
    if os.path.exists(file_path):
        os.remove(file_path)


#################### MAIN TAB ####################

def create_live_recording_tab():
    with gr.Tab("Live Recording and Transcription"):
        gr.Markdown("## Live Recording + Partial and Final Transcripts")

        with gr.Row():
            with gr.Column():
                transcription_method = gr.Dropdown(
                    label="Transcription Method",
                    choices=TRANSCRIPTION_METHODS,
                    value="faster-whisper"
                )
                whisper_model = gr.Dropdown(
                    label="Whisper Model",
                    choices=whisper_models,
                    value="distil-large-v3"
                )
                live_update = gr.Checkbox(label="Enable Live Transcription", value=False)
                save_recording = gr.Checkbox(label="Save WAV after Stopping", value=False)
                save_to_db_chk = gr.Checkbox(label="Save to DB after Stopping?", value=False)
                custom_title = gr.Textbox(label="Custom Title for DB", visible=False)

                record_btn = gr.Button("Start Recording")
                status_markdown = gr.Markdown("Not Recording")

            with gr.Column():
                partial_txt = gr.Textbox(label="Partial Transcript", lines=5, interactive=False)
                final_txt = gr.Textbox(label="Final Transcript", lines=5, interactive=False)
                audio_player = gr.Audio(label="Recorded Audio", visible=True)
                db_save_out = gr.Textbox(label="DB Save Status", lines=2)

        # States
        recording_state = gr.State()
        is_recording = gr.State(value=False)
        partial_text_state = gr.State({"text": ""})
        final_text_state = gr.State({"text": ""})

        def wrapper_fn(*args):
            # (recording_state, is_recording, btn_label, partial_str, final_str, audio_file)
            result = toggle_recording(*args)
            is_rec = result[1]
            # fancy HTML
            status_html = (
                "<div style='background:#e53935; color:white; padding:8px; font-weight:bold;'>"
                "🔴 Recording in Progress</div>"
                if is_rec else
                "<div style='background:#424242; color:white; padding:8px; font-weight:bold;'>"
                "⚪ Not Recording</div>"
            )
            return result + (status_html,)

        record_btn.click(
            fn=wrapper_fn,
            inputs=[
                is_recording,
                recording_state,
                transcription_method,
                live_update,
                save_recording,
                partial_text_state,
                final_text_state,
                whisper_model
            ],
            outputs=[
                recording_state,  # 0
                is_recording,     # 1
                record_btn,       # 2
                partial_txt,      # 3
                final_txt,        # 4
                audio_player,     # 5
                status_markdown   # 6 (the fancy HTML)
            ]
        )

        save_to_db_chk.change(
            fn=update_custom_title_visibility,
            inputs=[save_to_db_chk],
            outputs=[custom_title]
        )

        gr.Button("Save to DB").click(
            fn=save_transcription_to_db,
            inputs=[final_txt, custom_title],
            outputs=[db_save_out]
        )

        # Timer for partial
        poller = gr.Timer(value=2.0)
        poller.tick(
            fn=poll_partial_text,
            inputs=[live_update, partial_text_state],
            outputs=partial_txt
        )

#
# End of Functions
########################################################################################################################
