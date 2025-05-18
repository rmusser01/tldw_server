# Audio_ingestion_tab.py
# Description: Gradio UI for ingesting audio files into the database
#
# Imports
#
# External Imports
import os
import platform
import queue
import threading
import wave

import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.Audio.Audio_Files import process_audio_files
from PoC_Version.App_Function_Libraries.Audio.Audio_Transcription_Lib import PartialTranscriptionThread
from PoC_Version.App_Function_Libraries.DB.DB_Manager import list_prompts
from PoC_Version.App_Function_Libraries.Gradio_UI.Chat_ui import update_user_prompt
from PoC_Version.App_Function_Libraries.Gradio_UI.Gradio_Shared import whisper_models
from PoC_Version.App_Function_Libraries.Utils.Utils import cleanup_temp_files, default_api_endpoint, global_api_endpoints, \
    format_api_name, logging
# Import metrics logging
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from PoC_Version.App_Function_Libraries.Metrics.logger_config import logger
#
#######################################################################################################################
# Functions:

def create_audio_processing_tab():
    with gr.TabItem("Audio File Transcription + Summarization", visible=True):
        gr.Markdown("## Transcribe Audio from URLs, Uploads, or System Audio (44.1 kHz, 2ch)")

        # Attempt to get a default API dropdown value
        try:
            default_value = None
            if default_api_endpoint and default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found.")
        except Exception as e:
            logging.error(f"Error setting default API endpoint: {str(e)}")
            default_value = None

        ###############################################################################
        # Indefinite Recording UI: states and controls
        ###############################################################################
        recording_state = gr.State()  # Stores PyAudio stream, file handle, and device info
        is_recording = gr.State(value=False)
        partial_text_state = gr.State({"text": ""})
        final_wav_path_state = gr.State(None)

        # Shared variable (with lock) for partial transcript updates
        lock = threading.Lock()
        shared_partial_text = {"text": ""}

        record_system_audio = gr.Checkbox(
            label="Record System Audio Output?",
            value=False,
            info="Enable indefinite system-audio recording (loopback)."
        )

        consent_checkbox = gr.Checkbox(
            label="✅ I have obtained all necessary consents to record this audio",
            value=False,
            visible=False
        )
        partial_update_interval = gr.Slider(
            label="Partial Update Interval (seconds)",
            minimum=1, maximum=10, step=1, value=2,
            visible=False
        )
        live_trans_model = gr.Dropdown(
            label="Live (Partial) Model",
            choices=whisper_models,
            value="distil-large-v3",
            visible=False
        )
        final_trans_model = gr.Dropdown(
            label="Final Model",
            choices=whisper_models,
            value="distil-large-v3",
            visible=False
        )

        record_button = gr.Button("Start Recording", visible=False)
        transcribe_now_button = gr.Button("Transcribe Full Recording", visible=False)
        partial_txt = gr.Textbox(label="Partial Transcript (Live)", lines=6, interactive=False, visible=False)
        final_txt = gr.Textbox(label="Final Transcript (Stopped)", lines=18, interactive=False, visible=False)

        # NEW: Audio component for playback/download after recording stops.
        recorded_audio = gr.Audio(label="Recorded Audio", interactive=True, visible=False)

        ###############################################################################
        # Show/Hide logic for indefinite UI
        ###############################################################################
        def on_record_system_audio_change(enabled):
            """Updated to remove device selection"""
            return (
                gr.update(visible=enabled),  # consent_checkbox
                gr.update(visible=enabled),  # partial_update_interval
                gr.update(visible=enabled),  # live_trans_model
                gr.update(visible=enabled),  # final_trans_model
                gr.update(visible=not enabled)  # Hide file uploader
            )

        # Toggles the visibility of the indefinite-recording UI elements.
        def toggle_indefinite_ui(do_show):
            return (
                gr.update(visible=do_show),  # record_button
                gr.update(visible=do_show),  # partial_txt
                gr.update(visible=do_show),  # final_txt
                gr.update(visible=do_show)   # transcribe_now_button
            )

        ###############################################################################
        # The indefinite record start/stop logic
        ###############################################################################
        def toggle_recording(currently_recording, current_state, got_consent, partial_interval, live_model, final_model):
            """Handles start/stop recording and updates the partial transcript using a shared variable.
               Also updates an audio component for playback/download."""
            if not got_consent:
                return (
                    current_state,
                    False,
                    "Start Recording",
                    "[Consent Required]",
                    "",
                    None,
                    gr.update(visible=False)  # Hide audio component if no consent
                )

            if not currently_recording:
                # --- START indefinite recording
                try:
                    os_name = platform.system()
                    if os_name == "Windows":
                        # Windows: use pyaudiowpatch for WASAPI loopback recording
                        import pyaudiowpatch as pyaudio
                        p = pyaudio.PyAudio()
                        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
                        if not default_speakers["isLoopbackDevice"]:
                            for loopback in p.get_loopback_device_info_generator():
                                if default_speakers["name"] in loopback["name"]:
                                    default_speakers = loopback
                                    break
                            else:
                                raise Exception("Loopback device not found")
                        device_index = default_speakers["index"]
                    elif os_name in ["Linux", "Darwin"]:
                        # Linux and macOS: use PyAudio and search for a device with 'monitor' in its name
                        import pyaudio
                        p = pyaudio.PyAudio()
                        device_index = None
                        for i in range(p.get_device_count()):
                            info = p.get_device_info_by_index(i)
                            if "monitor" in info["name"].lower() and info["maxInputChannels"] > 0:
                                device_index = i
                                default_speakers = info
                                break
                        if device_index is None:
                            raise Exception("No loopback/monitor device found for system audio recording. Please configure your system audio loopback.")
                    else:
                        raise Exception("Unsupported OS for system audio recording.")

                    # Create WAV file
                    wav_path = os.path.join(os.getcwd(), "recorded_system_audio.wav")
                    wave_file = wave.open(wav_path, 'wb')
                    wave_file.setnchannels(default_speakers["maxInputChannels"])
                    # Use sample size from the correct library depending on OS.
                    if os_name == "Windows":
                        sample_size = p.get_sample_size(pyaudio.paInt16)
                    else:
                        sample_size = p.get_sample_size(pyaudio.paInt16)
                    wave_file.setsampwidth(sample_size)
                    wave_file.setframerate(int(default_speakers["defaultSampleRate"]))

                    # Create audio queue for partial transcription
                    audio_queue = queue.Queue()
                    stop_event = threading.Event()

                    # Define callback to write frames and enqueue audio data.
                    def callback(in_data, frame_count, time_info, status):
                        wave_file.writeframes(in_data)
                        audio_queue.put(in_data)
                        # paContinue is available in both libraries.
                        return (in_data, pyaudio.paContinue)

                    # Open stream using the proper device index.
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=default_speakers["maxInputChannels"],
                        rate=int(default_speakers["defaultSampleRate"]),
                        frames_per_buffer=512,
                        input=True,
                        input_device_index=device_index,
                        stream_callback=callback
                    )

                    # Start partial transcription thread using the shared variable.
                    partial_thread = PartialTranscriptionThread(
                        audio_queue=audio_queue,
                        stop_event=stop_event,
                        partial_text_state=shared_partial_text,
                        lock=lock,
                        live_model=live_model,
                        sample_rate=int(default_speakers["defaultSampleRate"]),
                        channels=default_speakers["maxInputChannels"],
                        partial_update_interval=partial_interval,
                        partial_chunk_seconds=5
                    )
                    partial_thread.start()

                    new_state = {
                        "pyaudio": p,
                        "stream": stream,
                        "wave_file": wave_file,
                        "partial_thread": partial_thread,
                        "stop_event": stop_event,
                        "wav_path": wav_path
                    }
                    return (
                        new_state,
                        True,
                        "Stop Recording",
                        "",  # Clear partial text on start
                        "",
                        None,
                        gr.update(visible=False)  # Hide audio component while recording
                    )
                except Exception as e:
                    logging.error(f"Recording start failed: {str(e)}")
                    return (
                        current_state,
                        False,
                        "Start Recording",
                        f"Error: {str(e)}",
                        "",
                        None,
                        gr.update(visible=False)
                    )
            else:
                # --- STOP indefinite recording
                try:
                    current_state["stop_event"].set()
                    current_state["partial_thread"].join()
                    current_state["stream"].stop_stream()
                    current_state["stream"].close()
                    current_state["wave_file"].close()
                    current_state["pyaudio"].terminate()

                    with lock:
                        final_partial_text = shared_partial_text.get("text", "")

                    return (
                        None,
                        False,
                        "Start Recording",
                        final_partial_text,  # Update final transcript with shared text
                        "",
                        current_state["wav_path"],
                        gr.update(visible=True, value=current_state["wav_path"])  # Show audio component with recorded file
                    )
                except Exception as e:
                    logging.error(f"Recording stop failed: {str(e)}")
                    return (
                        None,
                        False,
                        "Start Recording",
                        f"Error stopping: {str(e)}",
                        "",
                        None,
                        gr.update(visible=False)
                    )

        # Update record_button callback to include the new audio component output.
        record_button.click(
            fn=toggle_recording,
            inputs=[
                is_recording,
                recording_state,
                consent_checkbox,
                partial_update_interval,
                live_trans_model,
                final_trans_model
            ],
            outputs=[
                recording_state,
                is_recording,
                record_button,
                partial_txt,
                final_txt,
                final_wav_path_state,
                recorded_audio  # New output for audio playback/download
            ]
        )

        ###############################################################################
        # Timer to refresh partial transcript in UI
        ###############################################################################
        # Note: Removed the None input. Now the poll_partial function only accepts is_recording.
        def poll_partial(is_rec):
            with lock:
                current_text = shared_partial_text.get("text", "")
            if is_rec:
                return current_text
            return current_text or "[Not Recording]"

        partial_refresher = gr.Timer(value=1.0)
        partial_refresher.tick(
            fn=poll_partial,
            inputs=[is_recording],
            outputs=partial_txt
        )

        ###############################################################################
        # A button to transcribe the final WAV "now"
        ###############################################################################
        def do_final_transcription(final_wav_path, chosen_final_model):
            """
            Reads the entire WAV from disk, transcribes with chosen_final_model.
            Returns final transcript text + the path for re-use if needed.
            """
            import os
            if not final_wav_path or not os.path.exists(final_wav_path):
                return "[No valid recorded WAV]", None

            from PoC_Version.App_Function_Libraries.Audio.Audio_Transcription_Lib import transcribe_audio
            import wave
            import numpy as np

            try:
                with wave.open(final_wav_path, 'rb') as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    channels = wf.getnchannels()
                    raw = wf.readframes(frames)

                audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
                # Downmix if stereo
                if channels == 2:
                    audio_np = audio_np.reshape((-1, 2)).mean(axis=1)

                # transcribe_audio(...) has signature
                # transcribe_audio(audio_data, transcription_provider, sample_rate=16000, speaker_lang=None, whisper_model="distil-large-v3")
                final_res = transcribe_audio(
                    audio_data=audio_np,
                    transcription_provider="faster-whisper",  # or user config
                    sample_rate=rate,
                    whisper_model=chosen_final_model
                )
                return final_res, final_wav_path
            except Exception as e:
                return f"[Error transcribing final WAV: {str(e)}]", final_wav_path

        transcribe_now_button.click(
            fn=do_final_transcription,
            inputs=[final_wav_path_state, final_trans_model],
            outputs=[final_txt, final_wav_path_state]
        )

        ###############################################################################
        # Standard UI for uploading files or URLs, chunking, diarization, etc.
        ###############################################################################
        with gr.Row():
            with gr.Column():
                audio_url_input = gr.Textbox(
                    label="Audio File URL(s)",
                    placeholder="Enter the URL(s) of the audio file(s), one per line"
                )
                audio_file_input = gr.File(
                    label="Upload Audio Files (MP3, WAV, etc.)",
                    file_types=["audio", ".mp3", ".wav", ".m4a", ".flac", ".aac", ".alac", ".ogg", ".opus"],
                    file_count="multiple"
                )
                custom_title_input = gr.Textbox(label="Custom Title Prefix", placeholder="Prefix for your audio files")

                # Hide the normal file upload if indefinite is on
                # We'll unify that with record_system_audio.change
                record_system_audio.change(
                    fn=lambda en: gr.update(visible=not en),
                    inputs=[record_system_audio],
                    outputs=[audio_file_input]
                )

                use_cookies_input = gr.Checkbox(label="Use cookies for authenticated download", value=False)
                cookies_input = gr.Textbox(
                    label="Audio Download Cookies",
                    placeholder="Paste your cookies here (JSON format)",
                    lines=3,
                    visible=False
                )

                def toggle_cookies_box(x):
                    return gr.update(visible=x)

                use_cookies_input.change(
                    fn=toggle_cookies_box,
                    inputs=[use_cookies_input],
                    outputs=[cookies_input]
                )

                diarize_input = gr.Checkbox(label="Enable Speaker Diarization", value=False)
                whisper_model_input = gr.Dropdown(
                    choices=whisper_models,
                    value="distil-large-v3",
                    label="Whisper Model"
                )
                keep_timestamps_input = gr.Checkbox(label="Keep Timestamps", value=True)

                with gr.Row():
                    custom_prompt_checkbox = gr.Checkbox(
                        label="Use a Custom Prompt",
                        value=False,
                        visible=True
                    )
                    preset_prompt_checkbox = gr.Checkbox(
                        label="Use a pre-set Prompt",
                        value=False,
                        visible=True
                    )

                # State variables for pagination of prompts
                current_page_state = gr.State(value=1)
                total_pages_state = gr.State(value=1)

                with gr.Row():
                    preset_prompt = gr.Dropdown(label="Select Preset Prompt", choices=[], visible=False)
                with gr.Row():
                    prev_page_button = gr.Button("Previous Page", visible=False)
                    page_display = gr.Markdown("Page 1 of X", visible=False)
                    next_page_button = gr.Button("Next Page", visible=False)

                with gr.Row():
                    custom_prompt_input = gr.Textbox(
                        label="Custom Prompt",
                        placeholder="Enter custom prompt here",
                        lines=6,
                        visible=False
                    )
                with gr.Row():
                    system_prompt_input = gr.Textbox(
                        label="System Prompt",
                        value="""<s>You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhere to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
                        **Bulleted Note Creation Guidelines**

                        **Headings**:
                        - Based on referenced topics, not categories like quotes or terms
                        - Surrounded by **bold** formatting 
                        - Not listed as bullet points
                        - No space between headings and list items underneath

                        **Emphasis**:
                        - **Important terms** set in bold font
                        - **Text ending in a colon**: also bolded

                        **Review**:
                        - Ensure adherence to specified format
                        - Do not reference these instructions in your response.</s> {{ .Prompt }}
                        """,
                        lines=6,
                        visible=False,
                        interactive=True
                    )

                # Show/hide custom prompt input
                def toggle_custom_prompts(use_custom):
                    return (gr.update(visible=use_custom), gr.update(visible=use_custom))

                custom_prompt_checkbox.change(
                    fn=toggle_custom_prompts,
                    inputs=[custom_prompt_checkbox],
                    outputs=[custom_prompt_input, system_prompt_input]
                )

                # Pre-set prompts logic
                def on_preset_prompt_checkbox_change(is_checked):
                    if is_checked:
                        prompts, total_pages, current_page = list_prompts(page=1, per_page=10)
                        page_display_text = f"Page {current_page} of {total_pages}"
                        return (
                            gr.update(visible=True, interactive=True, choices=prompts),
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(value=page_display_text, visible=True),
                            current_page,
                            total_pages
                        )
                    else:
                        return (
                            gr.update(visible=False, interactive=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            1,
                            1
                        )

                preset_prompt_checkbox.change(
                    fn=on_preset_prompt_checkbox_change,
                    inputs=[preset_prompt_checkbox],
                    outputs=[preset_prompt, prev_page_button, next_page_button,
                             page_display, current_page_state, total_pages_state]
                )

                record_system_audio.change(
                    fn=on_record_system_audio_change,
                    inputs=[record_system_audio],
                    outputs=[consent_checkbox, partial_update_interval, live_trans_model, final_trans_model,
                             audio_file_input]
                )

                def on_prev_page_click(cur_page, tot_pages):
                    new_page = max(cur_page - 1, 1)
                    prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
                    page_text = f"Page {current_page} of {total_pages}"
                    return (gr.update(choices=prompts), gr.update(value=page_text), current_page)

                prev_page_button.click(
                    fn=on_prev_page_click,
                    inputs=[current_page_state, total_pages_state],
                    outputs=[preset_prompt, page_display, current_page_state]
                )

                def on_next_page_click(cur_page, tot_pages):
                    new_page = min(cur_page + 1, tot_pages)
                    prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
                    page_text = f"Page {current_page} of {total_pages}"
                    return (gr.update(choices=prompts), gr.update(value=page_text), current_page)

                next_page_button.click(
                    fn=on_next_page_click,
                    inputs=[current_page_state, total_pages_state],
                    outputs=[preset_prompt, page_display, current_page_state]
                )

                # Update custom/system prompts once user picks a preset
                def update_prompts(preset_name):
                    data = update_user_prompt(preset_name)
                    return (
                        gr.update(value=data["user_prompt"], visible=True),
                        gr.update(value=data["system_prompt"], visible=True)
                    )

                preset_prompt.change(
                    fn=update_prompts,
                    inputs=[preset_prompt],
                    outputs=[custom_prompt_input, system_prompt_input]
                )

                # Choose your summarization/analysis API
                api_name_input = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Analysis/Summarization (Optional)"
                )
                api_key_input = gr.Textbox(
                    label="API Key (if required)",
                    placeholder="Enter your API key here",
                    type="password"
                )
                custom_keywords_input = gr.Textbox(
                    label="Custom Keywords",
                    placeholder="Enter custom keywords, comma-separated"
                )
                keep_original_input = gr.Checkbox(label="Keep original audio files?", value=False)

                # Chunking options
                chunking_options_checkbox = gr.Checkbox(label="Show Chunking Options", value=False)
                with gr.Row(visible=False) as chunking_options_box:
                    gr.Markdown("### Chunking Options")
                    with gr.Column():
                        chunk_method = gr.Dropdown(
                            choices=['words', 'sentences', 'paragraphs', 'tokens'],
                            label="Chunking Method"
                        )
                        max_chunk_size = gr.Slider(
                            minimum=100, maximum=1000, value=300, step=50,
                            label="Max Chunk Size"
                        )
                        chunk_overlap = gr.Slider(
                            minimum=0, maximum=100, value=0, step=10,
                            label="Chunk Overlap"
                        )
                        use_adaptive_chunking = gr.Checkbox(label="Use Adaptive Chunking")
                        use_multi_level_chunking = gr.Checkbox(label="Use Multi-level Chunking")
                        chunk_language = gr.Dropdown(
                            choices=['english', 'french', 'german', 'spanish'],
                            label="Chunking Language"
                        )

                def toggle_chunking(x):
                    return gr.update(visible=x)

                chunking_options_checkbox.change(
                    fn=toggle_chunking,
                    inputs=[chunking_options_checkbox],
                    outputs=[chunking_options_box]
                )

                # Process button
                process_audio_button = gr.Button("Process Audio File(s)", variant="primary")

            # Outputs
            with gr.Column():
                audio_progress_output = gr.Textbox(label="Progress", lines=10)
                audio_transcription_output = gr.Textbox(label="Transcriptions", lines=10)
                audio_summary_output = gr.Textbox(label="Summaries", lines=10)
                download_transcription = gr.File(label="Download All Transcriptions as JSON")
                download_summary = gr.File(label="Download All Summaries as Text")

        ###############################################################################
        # The main "Process Audio File(s)" function call
        ###############################################################################
        process_audio_button.click(
            fn=process_audio_files,
            inputs=[
                audio_url_input,
                audio_file_input,
                whisper_model_input,
                api_name_input,
                api_key_input,
                use_cookies_input,
                cookies_input,
                keep_original_input,
                custom_keywords_input,
                custom_prompt_input,
                chunk_method,
                max_chunk_size,
                chunk_overlap,
                use_adaptive_chunking,
                use_multi_level_chunking,
                chunk_language,
                diarize_input,
                keep_timestamps_input,
                custom_title_input,
                # The indefinite-recording inputs follow. If your process_audio_files
                # doesn't handle them, it's harmless:
                record_system_audio,
                consent_checkbox
            ],
            outputs=[
                audio_progress_output,
                audio_transcription_output,
                audio_summary_output
            ]
        )
        record_system_audio.change(
            fn=toggle_indefinite_ui,
            inputs=[record_system_audio],
            outputs=[record_button, partial_txt, final_txt, transcribe_now_button]
        )

        ###############################################################################
        # Cleanup if user clears the file input
        ###############################################################################
        def on_file_clear(files):
            if not files:
                cleanup_temp_files()

        audio_file_input.clear(
            fn=on_file_clear,
            inputs=[audio_file_input],
            outputs=[]
        )

#
# End of Audio_ingestion_tab.py
#######################################################################################################################
