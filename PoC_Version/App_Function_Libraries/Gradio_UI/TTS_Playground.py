# Audio_Generation_Playground.py
# Description: Gradio UI for text-to-speech audio generation
#
import os
import time
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.TTS.TTS_Providers import generate_audio
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_to_database
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
#
########################################################################################################################
# Constants and Configuration

TTS_PROVIDERS = ["openai", "elevenlabs", "alltalk", "kokoro"]
DEFAULT_VOICES = {
    "openai": ["alloy", "echo", "fable", "nova", "onyx", "shimmer"],
    "elevenlabs": ["21m00", "AZnzlk1XvdvUeBnXmlld"],
    "alltalk": ["alloy", "echo", "fable", "nova", "onyx", "shimmer"],
    "kokoro": [ "af", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael", "bf_emma",
                "bf_isabella", "bm_george", "bm_lewis" ],
}


########################################################################################################################
# Core Functions

def generate_tts_audio(
        text,
        provider,
        voice,
        model,
        speed,
        output_format,
):
    """Wrapper function for TTS generation with logging and error handling"""
    log_counter("tts_generation_attempt", labels={"provider": provider})
    start_time = time.time()

    try:
        # Generate audio using selected provider
        audio_file = generate_audio(
            api_key=None,  # Will use config credentials
            text=text,
            provider=provider,
            voice=voice,
            model=model,
            voice2=None,
            output_file=None,
            response_format=output_format,
            streaming=False,
            speed=speed
        )

        # Validate output
        if not os.path.exists(audio_file):
            raise ValueError(f"Generated file not found at {audio_file}")

        log_counter("tts_generation_success", labels={"provider": provider})
        log_histogram("tts_generation_duration", time.time() - start_time)

        return audio_file, audio_file, "Audio generated successfully!"

    except Exception as e:
        log_counter("tts_generation_error", labels={"provider": provider, "error": str(e)})
        return None, None, f"Error generating audio: {str(e)}"

########################################################################
# UI Components

def create_audio_generation_tab():
    """Create the Gradio UI tab for audio generation"""
    with gr.Tab("Audio Generation Playground"):
        gr.Markdown("# Text-to-Speech Audio Generation")

        with gr.Row():
            with gr.Column(scale=2):
                # Input Controls
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=5,
                    placeholder="Enter text to convert to speech..."
                )

                provider_dropdown = gr.Dropdown(
                    label="TTS Provider",
                    choices=TTS_PROVIDERS,
                    value="openai"
                )

                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=DEFAULT_VOICES["openai"],
                    value="alloy"
                )

                model_dropdown = gr.Dropdown(
                    label="Model",
                    visible=False,  # Only show when relevant
                    interactive=True
                )

                with gr.Row():
                    speed_slider = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1
                    )

                    format_dropdown = gr.Dropdown(
                        label="Output Format",
                        choices=["mp3", "wav"],
                        value="mp3"
                    )

                generate_btn = gr.Button("Generate Audio", variant="primary")

            with gr.Column(scale=1):
                # Output Components
                audio_output = gr.Audio(label="Generated Audio")
                download_button = gr.File(label="Download File")
                status_box = gr.Textbox(label="Status", interactive=False)

        # Update voice dropdown based on provider
        provider_dropdown.change(
            fn=lambda p: gr.Dropdown(choices=DEFAULT_VOICES.get(p, [])),
            inputs=[provider_dropdown],
            outputs=[voice_dropdown]
        )

        # Generate button: return a single file path for audio_output
        generate_btn.click(
            fn=generate_tts_audio,
            inputs=[
                text_input,
                provider_dropdown,
                voice_dropdown,
                model_dropdown,
                speed_slider,
                format_dropdown
            ],
            outputs=[audio_output, download_button, status_box]
        )

    # Return an empty update or omit
    return gr.update()

#
# End of TTS_Playground.py
######################################################################################################################
