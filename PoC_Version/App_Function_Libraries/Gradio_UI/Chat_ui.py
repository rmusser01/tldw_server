# Chat_ui.py
# Description: Chat interface functions for Gradio
#
# Imports
import os
import queue
import sqlite3
import threading
import time
import wave
from datetime import datetime
#
# External Imports
import gradio as gr
import numpy as np
import pyaudio

from PoC_Version.App_Function_Libraries.Audio.Audio_Transcription_Lib import transcribe_audio
#
# Local Imports
from PoC_Version.App_Function_Libraries.Chat.Chat_Functions import approximate_token_count, chat, save_chat_history, \
    update_chat_content, save_chat_history_to_db_wrapper, parse_user_dict_markdown_file, ChatDictionary, \
    process_user_input
from PoC_Version.App_Function_Libraries.DB.DB_Manager import db, load_chat_history, start_new_conversation, \
    save_message, search_conversations_by_keywords, \
    get_all_conversations, delete_messages_in_conversation, search_media_db, list_prompts
from PoC_Version.App_Function_Libraries.DB.RAG_QA_Chat_DB import get_db_connection
from PoC_Version.App_Function_Libraries.Gradio_UI.Gradio_Shared import update_dropdown, update_user_prompt
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
from PoC_Version.App_Function_Libraries.TTS.TTS_Providers import generate_audio
from PoC_Version.App_Function_Libraries.Utils.Utils import default_api_endpoint, format_api_name, global_api_endpoints, \
    loaded_config_data, logging


#
#
########################################################################################################################
#
# Functions:


def show_edit_message(selected):
    if selected:
        return gr.update(value=selected[0], visible=True), gr.update(value=selected[1], visible=True), gr.update(
            visible=True)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def show_delete_message(selected):
    if selected:
        return gr.update(value=selected[1], visible=True), gr.update(visible=True)
    return gr.update(visible=False), gr.update(visible=False)


def debug_output(media_content, selected_parts):
    logging.debug(f"Debug - Media Content: {media_content}")
    logging.debug(f"Debug - Selected Parts: {selected_parts}")
    return ""


def update_selected_parts(use_content, use_summary, use_prompt):
    selected_parts = []
    if use_content:
        selected_parts.append("content")
    if use_summary:
        selected_parts.append("summary")
    if use_prompt:
        selected_parts.append("prompt")
    logging.debug(f"Debug - Update Selected Parts: {selected_parts}")
    return selected_parts


# Old update_user_prompt shim for backwards compatibility
def get_system_prompt(preset_name):
    # For backwards compatibility
    prompts = update_user_prompt(preset_name)
    return prompts["system_prompt"]

def clear_chat():
    """
    Return empty list for chatbot and None for conversation_id
    @return:
    """
    return gr.update(value=[]), None


def clear_chat_single():
    """
    Clears the chatbot and chat history.

    Returns:
        list: Empty list for chatbot messages.
        list: Empty list for chat history.
    """
    return [], []

# FIXME - add additional features....
def chat_wrapper(message, history, media_content, selected_parts, api_endpoint, api_key, custom_prompt, conversation_id,
                 save_conversation, temperature, system_prompt, streaming=False, chatdict_entries=None, max_tokens=500,
                 strategy="sorted_evenly"):
    try:
        logging.debug("chat_wrapper(): Starting chat wrapper")

        # Check if the conversation should be saved
        if save_conversation:
            logging.info("chat_wrapper(): Saving conversation")
            if conversation_id is None:
                logging.info("chat_wrapper(): Creating a new conversation")
                # Create a new conversation
                media_id = media_content.get('id', None)
                conversation_name = f"Chat about {media_content.get('title', 'Unknown Media')} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                conversation_id = start_new_conversation(title=conversation_name, media_id=media_id)
            # Add user message to the database
            user_message_id = save_message(conversation_id, role="user", content=message)

        # Include the selected parts and custom_prompt only for the first message
        if not history and selected_parts:
            message_body = "\n".join(selected_parts)
            base_message = f"{custom_prompt}\n\n{message}\n\n{message_body}" if custom_prompt else message
        elif custom_prompt:
            base_message = f"{custom_prompt}\n\n{message}"
        else:
            base_message = message

        # Apply chat dictionary processing
        if chatdict_entries:
            full_message = process_user_input(
                base_message,
                chatdict_entries,
                max_tokens=max_tokens,
                strategy=strategy
            )
        else:
            full_message = base_message

        # Generate bot response
        logging.debug("chat_wrapper(): Generating bot response")
        bot_message = ""
        response = chat(full_message,
                        history,
                        media_content,
                        selected_parts,
                        api_endpoint,
                        api_key,
                        custom_prompt,
                        temperature,
                        system_message=system_prompt,
                        streaming=streaming,
                        minp=None,
                        maxp=None,
                        model=None,
                        topp=None,
                        topk=None,
                        chatdict_entries=chatdict_entries,
                        max_tokens=max_tokens,
                        strategy=strategy)

        # Handle streaming and non-streaming responses
        if streaming:
            # For streaming responses, iterate over the generator
            for chunk in response:
                bot_message += chunk  # Accumulate the streamed response
                logging.debug(f"chat_wrapper(): Bot message being returned: {bot_message}")
                # Yield the incremental response and updated history
                yield bot_message, history + [(message, bot_message)], conversation_id
        else:
            # For non-streaming responses, handle the generator object
            if hasattr(response, "__iter__") and not isinstance(response, (str, dict)):
                # Consume the entire generator into a single string
                chunks = list(response)  # Pull everything from the generator
                bot_message = "".join(chunks)
            elif isinstance(response, dict) and "message" in response:
                bot_message = response["message"]
            else:
                # Fallback to a direct string conversion
                bot_message = str(response)

            logging.debug(f"chat_wrapper(): Bot message being returned: {bot_message}")
            # Yield the full response and updated history
            yield bot_message, history + [(message, bot_message)], conversation_id

        if save_conversation:
            # Add assistant message to the database
            save_message(conversation_id, role="assistant", content=bot_message)

    except Exception as e:
        logging.error(f"chat_wrapper(): Error in chat wrapper: {str(e)}")
        yield "chat_wrapper(): An error occurred.", history, conversation_id


def search_conversations(query):
    """Convert existing chat search to use RAG chat functions"""
    try:
        # Use the RAG search function - search by title if given a query
        if query and query.strip():
            results, _, _ = search_conversations_by_keywords(
                title_query=query.strip()
            )
        else:
            # Get all conversations if no query
            results, _, _ = get_all_conversations()

        if not results:
            return gr.update(choices=[])

        # Format choices to match existing UI format
        conversation_options = [
            (f"{conv['title']} (ID: {conv['conversation_id'][:8]})", conv['conversation_id'])
            for conv in results
        ]

        return gr.update(choices=conversation_options)
    except Exception as e:
        logging.error(f"Error searching conversations: {str(e)}")
        return gr.update(choices=[])


def load_conversation(conversation_id):
    """Convert existing load to use RAG chat functions"""
    if not conversation_id:
        return [], None

    try:
        # Use RAG load function
        messages, _, _ = load_chat_history(conversation_id)

        # Convert to chatbot history format
        history = [
            (content, None) if role == 'user' else (None, content)
            for role, content in messages
        ]

        return history, conversation_id
    except Exception as e:
        logging.error(f"Error loading conversation: {str(e)}")
        return [], None


def regenerate_last_message(history, media_content, selected_parts, api_endpoint, api_key, custom_prompt, temperature,
                            system_prompt, streaming=False):
    if not history:
        return history, "No messages to regenerate."

    logging.debug("Starting regenerate_last_message")

    # Find the last user message and its corresponding bot response
    last_user_message = None
    last_bot_message = None
    for i in range(len(history) - 1, -1, -1):
        if history[i][0]:  # This is a user message
            last_user_message = history[i][0]
            if i + 1 < len(history):
                last_bot_message = history[i + 1][1]
            break

    if not last_user_message:
        return history, "No user message found to regenerate the bot response."

    # Remove the last bot message from history
    new_history = history[:-1] if last_bot_message else history

    # Generate the new bot response
    bot_message = ""
    for chunk in chat(last_user_message, new_history, media_content, selected_parts, api_endpoint, api_key,
                      custom_prompt, temperature, system_prompt, streaming):
        if isinstance(chunk, str):
            bot_message += chunk
        elif isinstance(chunk, dict) and "choices" in chunk:
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            bot_message += content

        # Update the chatbot interface with the partial response
        new_history_with_regenerated = new_history + [(last_user_message, bot_message)]
        yield new_history_with_regenerated, "Regenerating..."

    # Update the history with the final regenerated message
    new_history_with_regenerated = new_history + [(last_user_message, bot_message)]
    logging.debug("Finished regenerating message")
    yield new_history_with_regenerated, "Last message regenerated successfully."


def update_dropdown_multiple(query, search_type, keywords=""):
    """Updated function to handle multiple search results using search_media_db"""
    try:
        # Define search fields based on search type
        search_fields = []
        if search_type.lower() == "keyword":
            # When searching by keyword, we'll search across multiple fields
            search_fields = ["title", "content", "author"]
        else:
            # Otherwise use the specific field
            search_fields = [search_type.lower()]

        # Perform the search
        results = search_media_db(
            search_query=query,
            search_fields=search_fields,
            keywords=keywords,
            page=1,
            results_per_page=50  # Adjust as needed
        )

        # Process results
        item_map = {}
        formatted_results = []

        for row in results:
            id, url, title, type_, content, author, date, prompt, summary = row
            # Create a display text that shows relevant info
            display_text = f"{title} - {author or 'Unknown'} ({date})"
            formatted_results.append(display_text)
            item_map[display_text] = id

        return gr.update(choices=formatted_results), item_map
    except Exception as e:
        logging.error(f"Error in update_dropdown_multiple: {str(e)}")
        return gr.update(choices=[]), {}


##################################################################
#
# Live Chat Functions

# FIXME - Setup defaults / loading config values
class LiveChat:
    def __init__(self, sample_rate=16000, chunk_size=1024, silence_threshold=0.01, silence_duration=2.5):
        """
        :param sample_rate: Sampling rate for recording (Hz)
        :param chunk_size: Number of frames per audio chunk
        :param silence_threshold: Amplitude threshold below which audio is considered "silent"
        :param silence_duration: Duration (in seconds) of continuous silence needed to trigger a response
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.silence_start_time = None
        self.audio_buffer = []  # Stores raw audio chunks until finalized

        self.recording_thread = None
        self.processing_thread = None

        self.pa = pyaudio.PyAudio()

    def start(self):
        self.stop_event.clear()
        # Start the audio recording thread
        self.recording_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.recording_thread.start()
        # Start the processing thread (to detect silence and trigger processing)
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.recording_thread is not None:
            self.recording_thread.join()
        if self.processing_thread is not None:
            self.processing_thread.join()
        self.pa.terminate()

    def record_audio(self):
        stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        while not self.stop_event.is_set():
            try:
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put(data)
            except Exception as e:
                print(f"Error recording audio: {e}")
        stream.stop_stream()
        stream.close()

    def process_audio(self):
        while not self.stop_event.is_set():
            try:
                # Wait briefly for a new chunk
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Append chunk to our rolling buffer
            self.audio_buffer.append(chunk)

            # Compute the average amplitude of this chunk
            audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            amplitude = np.abs(audio_data).mean()

            if amplitude < self.silence_threshold:
                # If silence detected, start (or continue) the silence timer
                if self.silence_start_time is None:
                    self.silence_start_time = time.time()
                else:
                    elapsed = time.time() - self.silence_start_time
                    if elapsed >= self.silence_duration:
                        print("Silence detected. Finalizing user input...")
                        self.finalize_and_process_buffer()
                        self.silence_start_time = None
            else:
                # Reset silence timer if speech is detected
                self.silence_start_time = None

    def finalize_and_process_buffer(self):
        if not self.audio_buffer:
            return

        # Combine all recorded chunks into one raw audio byte string
        raw_audio = b"".join(self.audio_buffer)
        # Clear the buffer so recording can resume fresh
        self.audio_buffer = []

        # Convert raw audio from int16 to normalized float32 array
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        # --- Step 1: Transcribe the finalized audio ---
        try:
            user_text = transcribe_audio(
                audio_np,
                transcription_provider="faster-whisper",
                sample_rate=self.sample_rate,
                whisper_model="deepdml/faster-whisper-large-v3-turbo-ct2",
                speaker_lang="en"
            )
        except Exception as e:
            user_text = f"[Transcription error: {str(e)}]"
        print("User said:", user_text)

        if not user_text.strip():
            print("No speech detected.")
            return

        # --- Step 2: Get a chat response from the LLM ---
        try:
            # Initialize parameters required by chat_wrapper
            history = []
            media_content = {}
            selected_parts = []
            api_endpoint = "None"  # Default endpoint
            api_key = None
            custom_prompt = None
            conversation_id = None
            save_conversation = False
            temperature = 0.7
            system_prompt = "You are a helpful AI assistant"
            streaming = False  # No streaming in LiveChat

            # Call chat_wrapper and get the response
            responses = list(chat_wrapper(
                user_text, history, media_content, selected_parts,
                api_endpoint, api_key, custom_prompt, conversation_id,
                save_conversation, temperature, system_prompt, streaming
            ))

            if responses:
                chat_response, _, _ = responses[0]  # Get the first (and likely only) response
            else:
                chat_response = "[No response generated]"
        except Exception as e:
            chat_response = f"[Chat error: {str(e)}]"
        print("LLM response:", chat_response)

        # --- Step 3: Generate TTS for the chat response ---
        try:
            tts_audio_file = generate_audio(
                api_key=None,  # Use your configured key or None if not needed
                text=chat_response,
                provider="openai",  # or another provider like "elevenlabs"
                voice="alloy",      # choose a voice from your configuration
                model=None,
                voice2=None,
                output_file=None,   # Let the function generate a temporary filename
                response_format="mp3",
                streaming=False,
                speed=1.0
            )
        except Exception as e:
            print(f"TTS generation error: {str(e)}")
            return

        if tts_audio_file and os.path.exists(tts_audio_file):
            print("Playing TTS audio...")
            self.play_audio(tts_audio_file)
        else:
            print("TTS audio file was not generated.")

    def play_audio(self, file_path):
        # A simple audio playback using PyAudio and wave
        wf = wave.open(file_path, 'rb')
        stream = self.pa.open(
            format=self.pa.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        data = wf.readframes(self.chunk_size)
        while data:
            stream.write(data)
            data = wf.readframes(self.chunk_size)
        stream.stop_stream()
        stream.close()
        wf.close()


# Global variable to store the live chat instance
global_live_chat = None

def start_live_chat_fn():
    """
    Creates and starts a LiveChat instance if one isn't running,
    and returns a status string.
    """
    global global_live_chat
    if global_live_chat is None:
        # Instantiate the LiveChat class with desired parameters
        # (Make sure LiveChat is imported or defined in your code)
        global_live_chat = LiveChat(
            sample_rate=16000,
            chunk_size=1024,
            silence_threshold=0.01,
            silence_duration=2.5
        )
        global_live_chat.start()
    return "Live speech2speech Chat Started."

def stop_live_chat_fn():
    """
    Stops the LiveChat instance if it exists and resets the global variable.
    """
    global global_live_chat
    if global_live_chat is not None:
        # This is valid because it becomes the object at runtime
        global_live_chat.stop()
        global_live_chat = None
    return "Live speech2speech Chat Stopped."


# FIXME - Add support for changing transcription provider, whisper model and speaker language
def record_and_transcribe(duration=5, sample_rate=16000, chunk_size=1024,
                          transcription_provider="faster-whisper", whisper_model="deepdml/faster-whisper-large-v3-turbo-ct2", speaker_lang="en"):
    """
    Records audio from the microphone for a fixed duration,
    transcribes it, and returns the resulting text.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)
    frames = []
    print("Recording for {} seconds...".format(duration))
    # Record for the specified duration
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        try:
            data = stream.read(chunk_size, exception_on_overflow=False)
        except Exception as e:
            print(f"Recording error: {e}")
            break
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    raw_data = b"".join(frames)
    # Convert to float32 (normalize by 32768 for int16)
    audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        transcribed_text = transcribe_audio(
            audio_np,
            transcription_provider=transcription_provider,
            sample_rate=sample_rate,
            whisper_model=whisper_model,
            speaker_lang=speaker_lang
        )
    except Exception as e:
        transcribed_text = f"[Transcription error: {str(e)}]"
    print("Transcribed Text:", transcribed_text)
    return transcribed_text


class DictationManager:
    def __init__(self, sample_rate=16000, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_recording = False
        self.recording_thread = None
        self.pa = None  # Will initialize PyAudio when needed

    def start_recording(self):
        """Start recording audio for dictation"""
        if self.is_recording:
            return "Already recording. Press 'Stop Dictation' first."

        # Initialize PyAudio if needed
        if not self.pa:
            self.pa = pyaudio.PyAudio()

        # Reset state
        self.stop_event.clear()
        self.is_recording = True

        # Start recording thread
        self.recording_thread = threading.Thread(target=self._record_audio_thread, daemon=True)
        self.recording_thread.start()

        return "🔴 Recording active. Press 'Stop Dictation' when finished."

    def stop_recording_and_transcribe(self):
        """Stop recording and transcribe the collected audio"""
        if not self.is_recording:
            return "No active recording", ""

        # Signal the recording thread to stop
        self.stop_event.set()
        self.is_recording = False

        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5)

        # Process collected audio from queue
        frames = []
        try:
            while not self.audio_queue.empty():
                frames.append(self.audio_queue.get_nowait())
        except queue.Empty:
            pass

        if not frames:
            return "No audio recorded", ""

        # Process the recorded audio
        raw_data = b"".join(frames)
        audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

        try:
            transcribed_text = transcribe_audio(
                audio_np,
                transcription_provider="faster-whisper",
                sample_rate=self.sample_rate,
                whisper_model="deepdml/faster-whisper-large-v3-turbo-ct2",
                speaker_lang="en"
            )

            return "Transcription complete", transcribed_text
        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            print(error_msg)
            return error_msg, ""

    def _record_audio_thread(self):
        """Thread function that records audio chunks"""
        stream = None
        try:
            stream = self.pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            print("Dictation recording started")
            # Empty the queue first
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

            # Start recording
            while not self.stop_event.is_set():
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception as e:
                    print(f"Error reading audio: {e}")
                    break

        except Exception as e:
            print(f"Error setting up audio: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            print("Dictation recording stopped")

    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_event.set()
            self.is_recording = False

        if self.pa:
            self.pa.terminate()
            self.pa = None

dictation_manager = DictationManager()

#
# End of Live Chat Functions
##################################################################


##################################################################
#
# Chat Interfaces

def create_chat_interface():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    #tts-status {
        font-weight: bold;
        padding: 5px;
        border-radius: 4px;
        margin-top: 5px;
    }
    #tts-status[value*="Error"], #tts-status[value*="Failed"] {
        color: red;
        background-color: #ffe6e6;
    }
    #tts-status[value*="Generating"], #tts-status[value*="Playing"] {
        color: #0066cc;
        background-color: #e6f2ff;
    }
    #tts-status[value*="Finished"] {
        color: green;
        background-color: #e6ffe6;
    }
    """
    confirm_clear_chat_js = """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
      // Grab the clear chat button by its elem_id
      const btn = document.querySelector("#clear_chat_btn");
      if(btn) {
        btn.addEventListener("click", function(e) {
          // Show a confirmation dialog
          if(!confirm("Are you sure you want to clear the chat?")) {
            // If user cancels, stop the click from reaching the server
            e.stopPropagation();
            e.preventDefault();
          }
        });
      }
    });
    </script>
    """
    with gr.TabItem("Remote LLM Chat (Horizontal)", visible=True):
        gr.Markdown("# Chat with a designated LLM Endpoint, using your selected item as starting context")
        chat_history = gr.State([])
        media_content = gr.State({})
        selected_parts = gr.State([])
        conversation_id = gr.State(None)
        # --- Live Chat Controls ---
        with gr.Row():
            start_live_chat_btn = gr.Button("Start speech2speech Chat")
            stop_live_chat_btn = gr.Button("Stop speech2speech Chat")
            live_chat_status = gr.Textbox(label="Live Chat Status", value="Inactive", interactive=False)

        # Wire the live chat control buttons to their functions
        start_live_chat_btn.click(
            fn=start_live_chat_fn,
            inputs=[],
            outputs=[live_chat_status]
        )
        stop_live_chat_btn.click(
            fn=stop_live_chat_fn,
            inputs=[],
            outputs=[live_chat_status]
        )
        # --- End Live Chat Controls ---
        with gr.Row():
            with gr.Column(scale=1):
                # Refactored API selection dropdown
                api_endpoint = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Interaction (Optional)"
                )
                api_key = gr.Textbox(label="API Key (if required)", type="password")

                # Initialize state variables for pagination
                current_page_state = gr.State(value=1)
                total_pages_state = gr.State(value=1)

                custom_prompt_checkbox = gr.Checkbox(label="Use a Custom Prompt",
                                                     value=False,
                                                     visible=True)
                preset_prompt_checkbox = gr.Checkbox(label="Use a pre-set Prompt",
                                                     value=False,
                                                     visible=True)
                with gr.Row():
                    # Add pagination controls
                    preset_prompt = gr.Dropdown(label="Select Preset Prompt",
                                                choices=[],
                                                visible=False)
                with gr.Row():
                    prev_page_button = gr.Button("Previous Page", visible=False)
                    page_display = gr.Markdown("Page 1 of X", visible=False)
                    next_page_button = gr.Button("Next Page", visible=False)
                    system_prompt_input = gr.Textbox(label="System Prompt",
                                                     value="You are a helpful AI assistant",
                                                     lines=3,
                                                     visible=False)
                with gr.Row():
                    user_prompt = gr.Textbox(label="Custom Prompt",
                                             placeholder="Enter custom prompt here",
                                             lines=3,
                                             visible=False)
                search_query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query here..."
                )
                search_type_input = gr.Radio(
                    choices=["Title", "Content", "Author", "Keyword"],
                    value="Keyword",
                    label="Search By"
                )
                keyword_filter_input = gr.Textbox(
                    label="Filter by Keywords (comma-separated)",
                    placeholder="ml, ai, python, etc..."
                )
                search_button = gr.Button("Search")
                items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                item_mapping = gr.State({})
                with gr.Row():
                    use_content = gr.Checkbox(label="Use Content")
                    use_summary = gr.Checkbox(label="Use Summary")
                    use_prompt = gr.Checkbox(label="Use Prompt")
                    save_conversation = gr.Checkbox(label="Save Conversation", value=False, visible=True)
                with gr.Row():
                    temperature = gr.Slider(label="Temperature", minimum=0.00, maximum=4.0, step=0.05, value=0.7)

                with gr.Row():
                    conversation_search = gr.Textbox(label="Search Conversations")
                with gr.Row():
                    search_conversations_btn = gr.Button("Search Conversations")
                with gr.Row():
                    previous_conversations = gr.Dropdown(label="Select Conversation", choices=[], interactive=True)
                with gr.Row():
                    load_conversations_btn = gr.Button("Load Selected Conversation")
                with gr.Row():
                    chatdict_files = gr.Files(label="Upload Chat Dictionary Markdown Files")
                    chatdict_entries = gr.State([])
                    max_tokens = gr.Slider(
                        label="Max Replacement Tokens",
                        minimum=100,
                        maximum=2000,
                        value=500,
                        step=50
                    )
                    strategy = gr.Dropdown(
                        label="Replacement Strategy",
                        choices=["sorted_evenly", "character_lore_first", "global_lore_first"],
                        value="sorted_evenly"
                    )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=800, elem_classes="chatbot-container")
                streaming = gr.Checkbox(label="Streaming", value=False, visible=True)
                msg = gr.Textbox(label="Enter your message")
                with gr.Row():
                    # --- Message Input and Transcribe Button ---
                    # When the transcribe button is clicked, record and transcribe audio, then populate the message
                    # textbox with the transcribed text.
                    dictate_status = gr.Textbox(
                        label="Dictation Status",
                        value="Ready to record",
                        interactive=False
                    )
                    start_dictation_btn = gr.Button("Start Dictation")
                    stop_dictation_btn = gr.Button("Stop Dictation")

                submit = gr.Button("Submit")
                with gr.Row():
                    speak_button = gr.Button("Speak Response")
                    tts_status = gr.Textbox(label="TTS Status", interactive=False)
                with gr.Row():
                    audio_output = gr.Audio(label="Generated Audio", visible=False)
                regenerate_button = gr.Button("Regenerate Last Message")
                with gr.Row():
                    token_count_display = gr.Number(label="Approximate Token Count", value=0, interactive=False)
                    clear_chat_button = gr.Button("Clear Chat", elem_id="clear_chat_btn")
                chat_media_name = gr.Textbox(label="Custom Chat Name(optional)")
                with gr.Row():
                    save_chat_history_to_db = gr.Button("Save Chat History to DataBase")
                    save_status = gr.Textbox(label="Save Status", interactive=False)
                with gr.Row():
                    save_chat_history_as_file = gr.Button("Save Chat History as File")
                    download_file = gr.File(label="Download Chat History")

        # Connect the dictation buttons to their functions
        start_dictation_btn.click(
            fn=dictation_manager.start_recording,
            inputs=[],
            outputs=[dictate_status]
        )

        stop_dictation_btn.click(
            fn=dictation_manager.stop_recording_and_transcribe,
            inputs=[],
            outputs=[dictate_status, msg]
        )

        gr.HTML(confirm_clear_chat_js)
        # Restore original functionality
        search_button.click(
            fn=update_dropdown_multiple,
            inputs=[search_query_input, search_type_input, keyword_filter_input],
            outputs=[items_output, item_mapping]
        )

        def save_chat_wrapper(history, conversation_id, media_content):
            file_path = save_chat_history(history, conversation_id, media_content)
            if file_path:
                return file_path, f"Chat history saved successfully as {os.path.basename(file_path)}!"
            else:
                return None, "Error saving chat history. Please check the logs and try again."

        save_chat_history_as_file.click(
            save_chat_wrapper,
            inputs=[chatbot, conversation_id, media_content],
            outputs=[download_file, save_status]
        )

        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return (
                gr.update(value=prompts["user_prompt"], visible=True),
                gr.update(value=prompts["system_prompt"], visible=True)
            )

        def clear_chat():
            return [], None, 0  # Return empty list for chatbot and None for conversation_id and token count

        clear_chat_button.click(
            clear_chat,
            outputs=[chatbot, conversation_id, token_count_display]
        )

        # Function to handle preset prompt checkbox change
        def on_preset_prompt_checkbox_change(is_checked):
            if is_checked:
                prompts, total_pages, current_page = list_prompts(page=1, per_page=20)
                page_display_text = f"Page {current_page} of {total_pages}"
                return (
                    gr.update(visible=True, interactive=True, choices=prompts),  # preset_prompt
                    gr.update(visible=True),  # prev_page_button
                    gr.update(visible=True),  # next_page_button
                    gr.update(value=page_display_text, visible=True),  # page_display
                    current_page,  # current_page_state
                    total_pages   # total_pages_state
                )
            else:
                return (
                    gr.update(visible=False, interactive=False),  # preset_prompt
                    gr.update(visible=False),  # prev_page_button
                    gr.update(visible=False),  # next_page_button
                    gr.update(visible=False),  # page_display
                    1,  # current_page_state
                    1   # total_pages_state
                )

        preset_prompt_checkbox.change(
            fn=on_preset_prompt_checkbox_change,
            inputs=[preset_prompt_checkbox],
            outputs=[preset_prompt, prev_page_button, next_page_button, page_display, current_page_state, total_pages_state]
        )

        # Chat Dictionary stuff
        def load_chatdict_files(files):
            entries = []
            for file in files:
                entries_dict = parse_user_dict_markdown_file(file.name)
                logging.debug(f"Loaded entries from {file.name}: {entries_dict.keys()}")
                for key, content in entries_dict.items():
                    entries.append(ChatDictionary(key=key, content=content))
            logging.debug(f"Total entries loaded: {len(entries)}")
            return entries

        chatdict_files.upload(
            fn=load_chatdict_files,
            inputs=[chatdict_files],
            outputs=[chatdict_entries]
        )

        # TTS Generation and Playback
        def speak_last_response(chatbot):
            """Generate audio for the last response and return the audio file"""
            logging.debug("Starting speak_last_response")
            try:
                if not chatbot or len(chatbot) == 0:
                    return "No messages to speak", None

                # Get the last assistant message from the chatbot component
                last_message = chatbot[-1][1]
                logging.debug(f"Last message to speak: {last_message}")

                # Generate unique filename with timestamp
                timestamp = int(time.time())
                output_file = f"response_{timestamp}.mp3"

                # Generate audio file
                audio_file = generate_audio(
                    text=last_message,
                    provider="openai",
                    output_file=output_file,
                    api_key=None
                )

                if audio_file and os.path.exists(audio_file):
                    return "Audio ready", audio_file
                return "Audio generation failed", None

            except Exception as e:
                logging.error(f"Error in speak_last_response: {str(e)}")
                return f"Error: {str(e)}", None

        speak_button.click(
            fn=speak_last_response,
            inputs=[chatbot],
            outputs=[tts_status, audio_output],
            api_name="speak_response"
        ).then(
            lambda: gr.update(visible=True),
            outputs=audio_output
        )

        def on_prev_page_click(current_page, total_pages):
            new_page = max(current_page - 1, 1)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=20)
            page_display_text = f"Page {current_page} of {total_pages}"
            return gr.update(choices=prompts), gr.update(value=page_display_text), current_page

        prev_page_button.click(
            fn=on_prev_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[preset_prompt, page_display, current_page_state]
        )

        def on_next_page_click(current_page, total_pages):
            new_page = min(current_page + 1, total_pages)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=20)
            page_display_text = f"Page {current_page} of {total_pages}"
            return gr.update(choices=prompts), gr.update(value=page_display_text), current_page

        next_page_button.click(
            fn=on_next_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[preset_prompt, page_display, current_page_state]
        )

        preset_prompt.change(
            update_prompts,
            inputs=[preset_prompt],
            outputs=[user_prompt, system_prompt_input]
        )

        custom_prompt_checkbox.change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[custom_prompt_checkbox],
            outputs=[user_prompt, system_prompt_input]
        )

        submit.click(
            chat_wrapper,
            inputs=[msg, chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt, conversation_id,
                    save_conversation, temperature, system_prompt_input, streaming, chatdict_entries, max_tokens, strategy],
            outputs=[msg, chatbot, conversation_id]
        ).then(  # Clear the message box after submission
            lambda x: gr.update(value=""),
            inputs=[chatbot],
            outputs=[msg]
        ).then(  # Clear the user prompt after the first message
            lambda: (gr.update(value=""), gr.update(value="")),
            outputs=[user_prompt, system_prompt_input]
        ).then(
            lambda history: approximate_token_count(history),
            inputs=[chatbot],
            outputs=[token_count_display]
        )

        items_output.change(
            update_chat_content,
            inputs=[items_output, use_content, use_summary, use_prompt, item_mapping],
            outputs=[media_content, selected_parts]
        )

        use_content.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                           outputs=[selected_parts])
        use_summary.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                           outputs=[selected_parts])
        use_prompt.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                          outputs=[selected_parts])
        items_output.change(debug_output, inputs=[media_content, selected_parts], outputs=[])

        search_conversations_btn.click(
            search_conversations,
            inputs=[conversation_search],
            outputs=[previous_conversations]
        )

        load_conversations_btn.click(
            clear_chat,
            outputs=[chatbot, chat_history]
        ).then(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chatbot, conversation_id]
        )

        previous_conversations.change(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chat_history]
        )

        save_chat_history_as_file.click(
            save_chat_history,
            inputs=[chatbot, conversation_id],
            outputs=[download_file]
        )

        save_chat_history_to_db.click(
            save_chat_history_to_db_wrapper,
            inputs=[chatbot, conversation_id, media_content, chat_media_name],
            outputs=[conversation_id, gr.Textbox(label="Save Status")]
        )

        regenerate_button.click(
            regenerate_last_message,
            inputs=[chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt, temperature,
                    system_prompt_input, streaming],
            outputs=[chatbot, gr.Textbox(label="Regenerate Status")]
        ).then(
            lambda history: approximate_token_count(history),
            inputs=[chatbot],
            outputs=[token_count_display]
        )


def create_chat_interface_stacked():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None

    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    """
    with gr.TabItem("Remote LLM Chat - Stacked", visible=True):
        gr.Markdown("# Stacked Chat")
        chat_history = gr.State([])
        media_content = gr.State({})
        selected_parts = gr.State([])
        conversation_id = gr.State(None)

        with gr.Row():
            with gr.Column():
                search_query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search query here..."
                )
                search_type_input = gr.Radio(
                    choices=["Title", "Content", "Author", "Keyword"],
                    value="Keyword",
                    label="Search By"
                )
                keyword_filter_input = gr.Textbox(
                    label="Filter by Keywords (comma-separated)",
                    placeholder="ml, ai, python, etc..."
                )
                search_button = gr.Button("Search")
                items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                item_mapping = gr.State({})
                with gr.Row():
                    use_content = gr.Checkbox(label="Use Content")
                    use_summary = gr.Checkbox(label="Use Summary")
                    use_prompt = gr.Checkbox(label="Use Prompt")
                    save_conversation = gr.Checkbox(label="Save Conversation", value=False, visible=True)
                    temp = gr.Slider(label="Temperature", minimum=0.00, maximum=4.0, step=0.05, value=0.7)
                with gr.Row():
                    conversation_search = gr.Textbox(label="Search Conversations")
                with gr.Row():
                    previous_conversations = gr.Dropdown(label="Select Conversation", choices=[], interactive=True)
                with gr.Row():
                    search_conversations_btn = gr.Button("Search Conversations")
                    load_conversations_btn = gr.Button("Load Selected Conversation")
            with gr.Column():
                # Refactored API selection dropdown
                api_endpoint = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Interaction (Optional)"
                )
                api_key = gr.Textbox(label="API Key (if required)", type="password")

                # Initialize state variables for pagination
                current_page_state = gr.State(value=1)
                total_pages_state = gr.State(value=1)

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
                streaming = gr.Checkbox(label="Streaming",
                                        value=False,
                                        visible=True
                )

                with gr.Row():
                    preset_prompt = gr.Dropdown(
                        label="Select Preset Prompt",
                        choices=[],
                        visible=False
                    )
                with gr.Row():
                    prev_page_button = gr.Button("Previous Page", visible=False)
                    page_display = gr.Markdown("Page 1 of X", visible=False)
                    next_page_button = gr.Button("Next Page", visible=False)

                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful AI assistant.",
                    lines=4,
                    visible=False
                )
                user_prompt = gr.Textbox(
                    label="Custom User Prompt",
                    placeholder="Enter custom prompt here",
                    lines=4,
                    visible=False
                )
                gr.Markdown("Scroll down for the chat window...")
        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(height=800, elem_classes="chatbot-container")
                msg = gr.Textbox(label="Enter your message")
        with gr.Row():
            with gr.Column():
                submit = gr.Button("Submit")
                with gr.Row():
                    speak_button = gr.Button("Speak Response")
                    tts_status = gr.Textbox(label="TTS Status", interactive=False)
                with gr.Row():
                    audio_output = gr.Audio(label="Generated Audio", visible=False)
                regenerate_button = gr.Button("Regenerate Last Message")
                with gr.Row():
                    token_count_display = gr.Number(label="Approximate Token Count", value=0, interactive=False)
                    clear_chat_button = gr.Button("Clear Chat")

                chat_media_name = gr.Textbox(label="Custom Chat Name(optional)")
                with gr.Row():
                    save_chat_history_to_db = gr.Button("Save Chat History to DataBase")
                    save_status = gr.Textbox(label="Save Status", interactive=False)
                with gr.Row():
                    save_chat_history_as_file = gr.Button("Save Chat History as File")
                    download_file = gr.File(label="Download Chat History")

        # Restore original functionality
        search_button.click(
            fn=update_dropdown_multiple,
            inputs=[search_query_input, search_type_input, keyword_filter_input],
            outputs=[items_output, item_mapping]
        )

        def search_conversations(query):
            try:
                # Use RAG search with title search
                if query and query.strip():
                    results, _, _ = search_conversations_by_keywords(title_query=query.strip())
                else:
                    results, _, _ = get_all_conversations()

                if not results:
                    return gr.update(choices=[])

                # Format choices to match UI
                conversation_options = [
                    (f"{conv['title']} (ID: {conv['conversation_id'][:8]})", conv['conversation_id'])
                    for conv in results
                ]

                return gr.update(choices=conversation_options)
            except Exception as e:
                logging.error(f"Error searching conversations: {str(e)}")
                return gr.update(choices=[])

        def load_conversation(conversation_id):
            if not conversation_id:
                return [], None

            try:
                # Use RAG load function
                messages, _, _ = load_chat_history(conversation_id)

                # Convert to chatbot history format
                history = [
                    (content, None) if role == 'user' else (None, content)
                    for role, content in messages
                ]

                return history, conversation_id
            except Exception as e:
                logging.error(f"Error loading conversation: {str(e)}")
                return [], None

        def save_chat_history_to_db_wrapper(chatbot, conversation_id, media_content, chat_name=None):
            log_counter("save_chat_history_to_db_attempt")
            start_time = time.time()
            logging.info(f"Attempting to save chat history. Media content type: {type(media_content)}")

            try:
                # First check if we can access the database
                try:
                    with get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                except sqlite3.DatabaseError as db_error:
                    logging.error(f"Database is corrupted or inaccessible: {str(db_error)}")
                    return conversation_id, gr.update(
                        value="Database error: The database file appears to be corrupted. Please contact support.")

                # For both new and existing conversations
                try:
                    if not conversation_id:
                        title = chat_name if chat_name else "Untitled Conversation"
                        conversation_id = start_new_conversation(title=title)
                        logging.info(f"Created new conversation with ID: {conversation_id}")

                    # Update existing messages
                    delete_messages_in_conversation(conversation_id)
                    for user_msg, assistant_msg in chatbot:
                        if user_msg:
                            save_message(conversation_id, "user", user_msg)
                        if assistant_msg:
                            save_message(conversation_id, "assistant", assistant_msg)
                except sqlite3.DatabaseError as db_error:
                    logging.error(f"Database error during message save: {str(db_error)}")
                    return conversation_id, gr.update(
                        value="Database error: Unable to save messages. Please try again or contact support.")

                save_duration = time.time() - start_time
                log_histogram("save_chat_history_to_db_duration", save_duration)
                log_counter("save_chat_history_to_db_success")

                return conversation_id, gr.update(value="Chat history saved successfully!")

            except Exception as e:
                log_counter("save_chat_history_to_db_error", labels={"error": str(e)})
                error_message = f"Failed to save chat history: {str(e)}"
                logging.error(error_message, exc_info=True)
                return conversation_id, gr.update(value=error_message)

        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return (
                gr.update(value=prompts["user_prompt"], visible=True),
                gr.update(value=prompts["system_prompt"], visible=True)
            )

        def clear_chat():
            return [], None, 0  # Empty history, conversation_id, and token count

        clear_chat_button.click(
            clear_chat,
            outputs=[chatbot, conversation_id, token_count_display]
        )

        # Handle custom prompt checkbox change
        def on_custom_prompt_checkbox_change(is_checked):
            return (
                gr.update(visible=is_checked),
                gr.update(visible=is_checked)
            )

        custom_prompt_checkbox.change(
            fn=on_custom_prompt_checkbox_change,
            inputs=[custom_prompt_checkbox],
            outputs=[user_prompt, system_prompt]
        )

        # Handle preset prompt checkbox change
        def on_preset_prompt_checkbox_change(is_checked):
            if is_checked:
                prompts, total_pages, current_page = list_prompts(page=1, per_page=20)
                page_display_text = f"Page {current_page} of {total_pages}"
                return (
                    gr.update(visible=True, interactive=True, choices=prompts),  # preset_prompt
                    gr.update(visible=True),  # prev_page_button
                    gr.update(visible=True),  # next_page_button
                    gr.update(value=page_display_text, visible=True),  # page_display
                    current_page,  # current_page_state
                    total_pages   # total_pages_state
                )
            else:
                return (
                    gr.update(visible=False, interactive=False),  # preset_prompt
                    gr.update(visible=False),  # prev_page_button
                    gr.update(visible=False),  # next_page_button
                    gr.update(visible=False),  # page_display
                    1,  # current_page_state
                    1   # total_pages_state
                )

        preset_prompt_checkbox.change(
            fn=on_preset_prompt_checkbox_change,
            inputs=[preset_prompt_checkbox],
            outputs=[preset_prompt, prev_page_button, next_page_button, page_display, current_page_state, total_pages_state]
        )

        # Pagination button functions
        def on_prev_page_click(current_page, total_pages):
            new_page = max(current_page - 1, 1)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=20)
            page_display_text = f"Page {current_page} of {total_pages}"
            return gr.update(choices=prompts), gr.update(value=page_display_text), current_page

        prev_page_button.click(
            fn=on_prev_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[preset_prompt, page_display, current_page_state]
        )

        def on_next_page_click(current_page, total_pages):
            new_page = min(current_page + 1, total_pages)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=20)
            page_display_text = f"Page {current_page} of {total_pages}"
            return gr.update(choices=prompts), gr.update(value=page_display_text), current_page

        next_page_button.click(
            fn=on_next_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[preset_prompt, page_display, current_page_state]
        )
        # TTS Generation and Playback
        def speak_last_response(chat_history):
            """Generate audio for the last response and return the audio file"""
            logging.debug("Starting speak_last_response")
            try:
                if not chat_history or len(chat_history) == 0:
                    return "No messages to speak", None

                # Get the last assistant message
                last_message = chat_history[-1][1]
                logging.debug(f"Last message to speak: {last_message}")

                # Generate unique filename with timestamp
                timestamp = int(time.time())
                output_file = f"response_{timestamp}.mp3"

                # Generate audio file
                audio_file = generate_audio(
                    text=last_message,
                    provider="openai",
                    output_file=output_file,
                    api_key=None
                )

                if audio_file and os.path.exists(audio_file):
                    return "Audio ready", audio_file
                return "Audio generation failed", None

            except Exception as e:
                logging.error(f"Error in speak_last_response: {str(e)}")
                return f"Error: {str(e)}", None

        speak_button.click(
            fn=speak_last_response,
            inputs=[chatbot],
            outputs=[tts_status, audio_output],
            api_name="speak_response"
        ).then(
            lambda: gr.update(visible=True),
            outputs=audio_output
        )
        # Update prompts when a preset is selected
        preset_prompt.change(
            update_prompts,
            inputs=[preset_prompt],
            outputs=[user_prompt, system_prompt]
        )

        submit.click(
            chat_wrapper,
            inputs=[msg, chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt,
                    conversation_id, save_conversation, temp, system_prompt, streaming],
            outputs=[msg, chatbot, conversation_id]
        ).then(
            lambda x: gr.update(value=""),
            inputs=[chatbot],
            outputs=[msg]
        ).then(
            lambda history: approximate_token_count(history),
            inputs=[chatbot],
            outputs=[token_count_display]
        )

        items_output.change(
            update_chat_content,
            inputs=[items_output, use_content, use_summary, use_prompt, item_mapping],
            outputs=[media_content, selected_parts]
        )
        use_content.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                           outputs=[selected_parts])
        use_summary.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                           outputs=[selected_parts])
        use_prompt.change(update_selected_parts, inputs=[use_content, use_summary, use_prompt],
                          outputs=[selected_parts])
        items_output.change(debug_output, inputs=[media_content, selected_parts], outputs=[])

        search_conversations_btn.click(
            search_conversations,
            inputs=[conversation_search],
            outputs=[previous_conversations]
        )

        load_conversations_btn.click(
            clear_chat,
            outputs=[chatbot, chat_history]
        ).then(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chatbot, conversation_id]
        )

        previous_conversations.change(
            load_conversation,
            inputs=[previous_conversations],
            outputs=[chat_history]
        )

        save_chat_history_as_file.click(
            save_chat_history,
            inputs=[chatbot, conversation_id],
            outputs=[download_file]
        )

        save_chat_history_to_db.click(
            save_chat_history_to_db_wrapper,
            inputs=[chatbot, conversation_id, media_content, chat_media_name],
            outputs=[conversation_id, save_status]
        )

        regenerate_button.click(
            regenerate_last_message,
            inputs=[chatbot, media_content, selected_parts, api_endpoint, api_key, user_prompt, temp, system_prompt],
            outputs=[chatbot, gr.Textbox(label="Regenerate Status")]
        ).then(
            lambda history: approximate_token_count(history),
            inputs=[chatbot],
            outputs=[token_count_display]
        )


def create_chat_interface_multi_api():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    .chat-window {
        height: 400px;
        overflow-y: auto;
    }
    """
    with gr.TabItem("One Prompt - Multiple APIs", visible=True):
        gr.Markdown("# One Prompt but Multiple APIs Chat Interface")

        with gr.Row():
            with gr.Column(scale=1):
                search_query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
                search_type_input = gr.Radio(choices=["Title", "URL", "Keyword", "Content"], value="Title",
                                             label="Search By")
                search_button = gr.Button("Search")
                items_output = gr.Dropdown(label="Select Item", choices=[], interactive=True)
                item_mapping = gr.State({})
                with gr.Row():
                    use_content = gr.Checkbox(label="Use Content")
                    use_summary = gr.Checkbox(label="Use Summary")
                    use_prompt = gr.Checkbox(label="Use Prompt")
            with gr.Column():
                # Initialize state variables for pagination
                current_page_state = gr.State(value=1)
                total_pages_state = gr.State(value=1)

                custom_prompt_checkbox = gr.Checkbox(label="Use a Custom Prompt",
                                                     value=False,
                                                     visible=True)
                preset_prompt_checkbox = gr.Checkbox(label="Use a pre-set Prompt",
                                                     value=False,
                                                     visible=True)
                with gr.Row():
                    # Add pagination controls
                    preset_prompt = gr.Dropdown(label="Select Preset Prompt",
                                                choices=[],
                                                visible=False)
                with gr.Row():
                    prev_page_button = gr.Button("Previous Page", visible=False)
                    page_display = gr.Markdown("Page 1 of X", visible=False)
                    next_page_button = gr.Button("Next Page", visible=False)
                system_prompt = gr.Textbox(label="System Prompt",
                                           value="You are a helpful AI assistant.",
                                           lines=5,
                                           visible=True)
                user_prompt = gr.Textbox(label="Modify Prompt (Prefixed to your message every time)", lines=5,
                                         value="", visible=True)

        with gr.Row():
            chatbots = []
            api_endpoints = []
            api_keys = []
            temperatures = []
            regenerate_buttons = []
            token_count_displays = []
            for i in range(3):
                with gr.Column():
                    gr.Markdown(f"### Chat Window {i + 1}")
                    # Refactored API selection dropdown
                    api_endpoint = gr.Dropdown(
                        choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                        value=default_value,
                        label="API for Chat Interaction (Optional)"
                    )
                    api_key = gr.Textbox(label=f"API Key {i + 1} (if required)", type="password")
                    temperature = gr.Slider(label=f"Temperature {i + 1}", minimum=0.0, maximum=4.0, step=0.05,
                                            value=0.7)
                    chatbot = gr.Chatbot(height=800, elem_classes="chat-window")
                    token_count_display = gr.Number(label=f"Approximate Token Count {i + 1}", value=0,
                                                    interactive=False)
                    token_count_displays.append(token_count_display)
                    regenerate_button = gr.Button(f"Regenerate Last Message {i + 1}")
                    chatbots.append(chatbot)
                    api_endpoints.append(api_endpoint)
                    api_keys.append(api_key)
                    temperatures.append(temperature)
                    regenerate_buttons.append(regenerate_button)

        with gr.Row():
            msg = gr.Textbox(label="Enter your message", scale=4)
            submit = gr.Button("Submit", scale=1)
            clear_chat_button = gr.Button("Clear All Chats")

        # State variables
        chat_history = [gr.State([]) for _ in range(3)]
        media_content = gr.State({})
        selected_parts = gr.State([])
        conversation_id = gr.State(None)

        # Event handlers
        search_button.click(
            fn=update_dropdown,
            inputs=[search_query_input, search_type_input],
            outputs=[items_output, item_mapping]
        )

        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return (
                gr.update(value=prompts["user_prompt"], visible=True),
                gr.update(value=prompts["system_prompt"], visible=True)
            )

        def on_custom_prompt_checkbox_change(is_checked):
            return (
                gr.update(visible=is_checked),
                gr.update(visible=is_checked)
            )

        custom_prompt_checkbox.change(
            fn=on_custom_prompt_checkbox_change,
            inputs=[custom_prompt_checkbox],
            outputs=[user_prompt, system_prompt]
        )

        def clear_all_chats():
            return [[]] * 3 + [[]] * 3 + [0] * 3

        clear_chat_button.click(
            clear_all_chats,
            outputs=chatbots + chat_history + token_count_displays
        )

        def on_preset_prompt_checkbox_change(is_checked):
            if is_checked:
                prompts, total_pages, current_page = list_prompts(page=1, per_page=10)
                page_display_text = f"Page {current_page} of {total_pages}"
                return (
                    gr.update(visible=True, interactive=True, choices=prompts),  # preset_prompt
                    gr.update(visible=True),  # prev_page_button
                    gr.update(visible=True),  # next_page_button
                    gr.update(value=page_display_text, visible=True),  # page_display
                    current_page,  # current_page_state
                    total_pages   # total_pages_state
                )
            else:
                return (
                    gr.update(visible=False, interactive=False),  # preset_prompt
                    gr.update(visible=False),  # prev_page_button
                    gr.update(visible=False),  # next_page_button
                    gr.update(visible=False),  # page_display
                    1,  # current_page_state
                    1   # total_pages_state
                )

        preset_prompt.change(update_user_prompt, inputs=preset_prompt, outputs=user_prompt)

        preset_prompt_checkbox.change(
            fn=on_preset_prompt_checkbox_change,
            inputs=[preset_prompt_checkbox],
            outputs=[preset_prompt, prev_page_button, next_page_button, page_display, current_page_state,
                     total_pages_state]
        )

        def on_prev_page_click(current_page, total_pages):
            new_page = max(current_page - 1, 1)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return gr.update(choices=prompts), gr.update(value=page_display_text), current_page

        prev_page_button.click(
            fn=on_prev_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[preset_prompt, page_display, current_page_state]
        )

        def on_next_page_click(current_page, total_pages):
            new_page = min(current_page + 1, total_pages)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return gr.update(choices=prompts), gr.update(value=page_display_text), current_page

        next_page_button.click(
            fn=on_next_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[preset_prompt, page_display, current_page_state]
        )

        # Update prompts when a preset is selected
        preset_prompt.change(
            update_prompts,
            inputs=[preset_prompt],
            outputs=[user_prompt, system_prompt]
        )

        def clear_all_chats():
            return [[]] * 3 + [[]] * 3 + [0] * 3

        clear_chat_button.click(
            clear_all_chats,
            outputs=chatbots + chat_history + token_count_displays
        )

        def chat_wrapper_multi(message, custom_prompt, system_prompt, *args):
            chat_histories = args[:3]
            chatbots = args[3:6]
            api_endpoints = args[6:9]
            api_keys = args[9:12]
            temperatures = args[12:15]
            media_content = args[15]
            selected_parts = args[16]

            new_chat_histories = []
            new_chatbots = []

            for i in range(3):
                # Call chat_wrapper with dummy values for conversation_id and save_conversation
                bot_message, new_history, _ = chat_wrapper(
                    message, chat_histories[i], media_content, selected_parts,
                    api_endpoints[i], api_keys[i], custom_prompt, None,  # None for conversation_id
                    False,  # False for save_conversation
                    temperature=temperatures[i],
                    system_prompt=system_prompt
                )

                new_chatbot = chatbots[i] + [(message, bot_message)]

                new_chat_histories.append(new_history)
                new_chatbots.append(new_chatbot)

            return [gr.update(value="")] + new_chatbots + new_chat_histories

        def update_token_counts(*histories):
            token_counts = []
            for history in histories:
                token_counts.append(approximate_token_count(history))
            return token_counts

        def regenerate_last_message(chat_history, chatbot, media_content, selected_parts, api_endpoint, api_key, custom_prompt, temperature, system_prompt):
            if not chat_history:
                return chatbot, chat_history, "No messages to regenerate."

            last_entry = chat_history[-1]
            last_user_message, last_bot_message = last_entry

            if last_bot_message is None:
                return chatbot, chat_history, "The last message is not from the bot."

            new_history = chat_history[:-1]

            if not last_user_message:
                return chatbot[:-1], new_history, "No user message to regenerate the bot response."

            bot_message = chat(
                last_user_message,
                new_history,
                media_content,
                selected_parts,
                api_endpoint,
                api_key,
                custom_prompt,
                temperature,
                system_prompt
            )

            new_history.append((last_user_message, bot_message))
            new_chatbot = chatbot[:-1] + [(last_user_message, bot_message)]

            return new_chatbot, new_history, "Last message regenerated successfully."

        for i in range(3):
            regenerate_buttons[i].click(
                regenerate_last_message,
                inputs=[chat_history[i], chatbots[i], media_content, selected_parts, api_endpoints[i], api_keys[i],
                        user_prompt, temperatures[i], system_prompt],
                outputs=[chatbots[i], chat_history[i], gr.Textbox(label=f"Regenerate Status {i + 1}")]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[chat_history[i]],
                outputs=[token_count_displays[i]]
            )

        # In the create_chat_interface_multi_api function:
        submit.click(
            chat_wrapper_multi,
            inputs=[msg, user_prompt,
                    system_prompt] + chat_history + chatbots + api_endpoints + api_keys + temperatures +
                   [media_content, selected_parts],
            outputs=[msg] + chatbots + chat_history
        ).then(
            lambda: (gr.update(value=""), gr.update(value="")),
            outputs=[msg, user_prompt]
        ).then(
            update_token_counts,
            inputs=chat_history,
            outputs=token_count_displays
        )

        items_output.change(
            update_chat_content,
            inputs=[items_output, use_content, use_summary, use_prompt, item_mapping],
            outputs=[media_content, selected_parts]
        )

        for checkbox in [use_content, use_summary, use_prompt]:
            checkbox.change(
                update_selected_parts,
                inputs=[use_content, use_summary, use_prompt],
                outputs=[selected_parts]
            )


def create_chat_interface_four():
    try:
        default_value = None
        if default_api_endpoint:
            if default_api_endpoint in global_api_endpoints:
                default_value = format_api_name(default_api_endpoint)
            else:
                logging.warning(f"Default API endpoint '{default_api_endpoint}' not found in global_api_endpoints")
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    custom_css = """
    .chatbot-container .message-wrap .message {
        font-size: 14px !important;
    }
    .chat-window {
        height: 400px;
        overflow-y: auto;
    }
    """

    with gr.TabItem("Four Independent API Chats", visible=True):
        gr.Markdown("# Four Independent API Chat Interfaces")

        # Initialize prompts during component creation
        prompts, total_pages, current_page = list_prompts(page=1, per_page=10)
        current_page_state = gr.State(value=current_page)
        total_pages_state = gr.State(value=total_pages)
        page_display_text = f"Page {current_page} of {total_pages}"

        with gr.Row():
            with gr.Column():
                preset_prompt = gr.Dropdown(
                    label="Select Preset Prompt (This will be prefixed to your messages, recommend copy/pasting and then clearing the User Prompt box)",
                    choices=prompts,
                    visible=True
                )
                prev_page_button = gr.Button("Previous Page", visible=True)
                page_display = gr.Markdown(page_display_text, visible=True)
                next_page_button = gr.Button("Next Page", visible=True)
                user_prompt = gr.Textbox(
                    label="Modify User Prompt",
                    lines=3
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value="You are a helpful AI assistant.",
                    lines=3
                )

            with gr.Column():
                gr.Markdown("Scroll down for the chat windows...")

        chat_interfaces = []

        def create_single_chat_interface(index, user_prompt_component):
            with gr.Column():
                gr.Markdown(f"### Chat Window {index + 1}")
                # Refactored API selection dropdown
                api_endpoint = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Chat Interaction (Optional)"
                )
                api_key = gr.Textbox(
                    label=f"API Key {index + 1} (if required)",
                    type="password"
                )
                temperature = gr.Slider(
                    label=f"Temperature {index + 1}",
                    minimum=0.0,
                    maximum=4.0,
                    step=0.05,
                    value=0.7
                )
                chatbot = gr.Chatbot(height=400, elem_classes="chat-window")
                msg = gr.Textbox(label=f"Enter your message for Chat {index + 1}")
                submit = gr.Button(f"Submit to Chat {index + 1}")
                regenerate_button = gr.Button(f"Regenerate Last Message {index + 1}")
                token_count_display = gr.Number(label=f"Approximate Token Count {index + 1}", value=0,
                                                interactive=False)
                clear_chat_button = gr.Button(f"Clear Chat {index + 1}")

                # State to maintain chat history
                chat_history = gr.State([])

                # Append to chat_interfaces list
                chat_interfaces.append({
                    'api_endpoint': api_endpoint,
                    'api_key': api_key,
                    'temperature': temperature,
                    'chatbot': chatbot,
                    'msg': msg,
                    'submit': submit,
                    'regenerate_button': regenerate_button,
                    'clear_chat_button': clear_chat_button,
                    'chat_history': chat_history,
                    'token_count_display': token_count_display
                })

        # Create four chat interfaces arranged in a 2x2 grid
        with gr.Row():
            for i in range(2):
                with gr.Column():
                    for j in range(2):
                        create_single_chat_interface(i * 2 + j, user_prompt)

        # Update user_prompt based on preset_prompt selection
        def update_prompts(preset_name):
            prompts = update_user_prompt(preset_name)
            return gr.update(value=prompts["user_prompt"]), gr.update(value=prompts["system_prompt"])

        preset_prompt.change(
            fn=update_prompts,
            inputs=[preset_prompt],
            outputs=[user_prompt, system_prompt]
        )

        # Pagination button functions
        def on_prev_page_click(current_page, total_pages):
            new_page = max(current_page - 1, 1)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return (
                gr.update(choices=prompts),
                gr.update(value=page_display_text),
                current_page
            )

        prev_page_button.click(
            fn=on_prev_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[preset_prompt, page_display, current_page_state]
        )

        def on_next_page_click(current_page, total_pages):
            new_page = min(current_page + 1, total_pages)
            prompts, total_pages, current_page = list_prompts(page=new_page, per_page=10)
            page_display_text = f"Page {current_page} of {total_pages}"
            return (
                gr.update(choices=prompts),
                gr.update(value=page_display_text),
                current_page
            )

        next_page_button.click(
            fn=on_next_page_click,
            inputs=[current_page_state, total_pages_state],
            outputs=[preset_prompt, page_display, current_page_state]
        )

        # FIXME
        def chat_wrapper_single(message, chat_history, api_endpoint, api_key, temperature, user_prompt):
            logging.debug(f"Chat Wrapper Single - Message: {message}, Chat History: {chat_history}")

            new_msg, new_history, _ = chat_wrapper(
                message,
                chat_history,
                {},  # Empty media_content
                [],  # Empty selected_parts
                api_endpoint,
                api_key,
                user_prompt,  # custom_prompt
                None,  # conversation_id
                False,  # save_conversation
                temperature,  # temperature
                system_prompt="",  # system_prompt
                max_tokens=None,
                top_p=None,
                frequency_penalty=None,
                presence_penalty=None,
                stop_sequence=None
            )
            if "API request failed" not in new_msg:
                chat_history.append((message, new_msg))
            else:
                logging.error(f"API request failed: {new_msg}")

            return "", chat_history, chat_history

        def regenerate_last_message(chat_history, api_endpoint, api_key, temperature, user_prompt):
            if not chat_history:
                return chat_history, chat_history, "No messages to regenerate."

            last_user_message, _ = chat_history[-1]

            new_msg, new_history, _ = chat_wrapper(
                last_user_message,
                chat_history[:-1],
                {},  # Empty media_content
                [],  # Empty selected_parts
                api_endpoint,
                api_key,
                user_prompt,  # custom_prompt
                None,  # conversation_id
                False,  # save_conversation
                temperature,  # temperature
                system_prompt="",  # system_prompt
                max_tokens=4096,
            )

            if "API request failed" not in new_msg:
                new_history.append((last_user_message, new_msg))
                return new_history, new_history, "Last message regenerated successfully."
            else:
                logging.error(f"API request failed during regeneration: {new_msg}")
                return chat_history, chat_history, f"Failed to regenerate: {new_msg}"

        # Attach click events for each chat interface
        for interface in chat_interfaces:
            interface['submit'].click(
                chat_wrapper_single,
                inputs=[
                    interface['msg'],
                    interface['chat_history'],
                    interface['api_endpoint'],
                    interface['api_key'],
                    interface['temperature'],
                    user_prompt
                ],
                outputs=[
                    interface['msg'],
                    interface['chatbot'],
                    interface['chat_history']
                ]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[interface['chat_history']],
                outputs=[interface['token_count_display']]
            )

            interface['regenerate_button'].click(
                regenerate_last_message,
                inputs=[
                    interface['chat_history'],
                    interface['api_endpoint'],
                    interface['api_key'],
                    interface['temperature'],
                    user_prompt
                ],
                outputs=[
                    interface['chatbot'],
                    interface['chat_history'],
                    gr.Textbox(label="Regenerate Status")
                ]
            ).then(
                lambda history: approximate_token_count(history),
                inputs=[interface['chat_history']],
                outputs=[interface['token_count_display']]
            )

            def clear_chat_single():
                return [], [], 0

            interface['clear_chat_button'].click(
                clear_chat_single,
                outputs=[interface['chatbot'], interface['chat_history'], interface['token_count_display']]
            )


def chat_wrapper_single(message, chat_history, chatbot, api_endpoint, api_key, temperature, media_content,
                        selected_parts, conversation_id, save_conversation, user_prompt):
    new_msg, new_history, new_conv_id = chat_wrapper(
        message, chat_history, media_content, selected_parts,
        api_endpoint, api_key, user_prompt, conversation_id,
        save_conversation, temperature, system_prompt=""
    )

    if new_msg:
        updated_chatbot = chatbot + [(message, new_msg)]
    else:
        updated_chatbot = chatbot

    return new_msg, updated_chatbot, new_history, new_conv_id

# Mock function to simulate LLM processing
def process_with_llm(workflow, context, prompt, api_endpoint, api_key):
    api_key_snippet = api_key[:5] + "..." if api_key else "Not provided"
    return f"LLM output using {api_endpoint} (API Key: {api_key_snippet}) for {workflow} with context: {context[:30]}... and prompt: {prompt[:30]}..."

#
# End of Chat_ui.py
#######################################################################################################################