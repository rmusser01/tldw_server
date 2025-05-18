# Podcast_tab.py
# Description: Gradio UI for ingesting podcasts into the database
#
# Imports
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.Audio.Audio_Files import process_podcast
from PoC_Version.App_Function_Libraries.DB.DB_Manager import list_prompts
from PoC_Version.App_Function_Libraries.Gradio_UI.Gradio_Shared import whisper_models, update_user_prompt
from PoC_Version.App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging
#
########################################################################################################################
#
# Functions:

def create_podcast_tab():
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
    with gr.TabItem("Podcast", visible=True):
        gr.Markdown("# Podcast Transcription and Ingestion", visible=True)
        # Initialize state variables for pagination
        current_page_state = gr.State(value=1)
        total_pages_state = gr.State(value=1)

        with gr.Row():
            with gr.Column():
                podcast_url_input = gr.Textbox(label="Podcast URL", placeholder="Enter the podcast URL here")
                podcast_title_input = gr.Textbox(label="Podcast Title", placeholder="Will be auto-detected if possible")
                podcast_author_input = gr.Textbox(label="Podcast Author", placeholder="Will be auto-detected if possible")

                podcast_keywords_input = gr.Textbox(
                    label="Keywords",
                    placeholder="Enter keywords here (comma-separated, include series name if applicable)",
                    value="podcast,audio",
                    elem_id="podcast-keywords-input"
                )

                keep_timestamps_input = gr.Checkbox(label="Keep Timestamps", value=True)

                with gr.Row():
                    podcast_custom_prompt_checkbox = gr.Checkbox(
                        label="Use a Custom Prompt",
                        value=False,
                        visible=True
                    )
                    preset_prompt_checkbox = gr.Checkbox(
                        label="Use a pre-set Prompt",
                        value=False,
                        visible=True
                    )

                with gr.Row():
                    # Add pagination controls
                    preset_prompt = gr.Dropdown(
                        label="Select Preset Prompt",
                        choices=[],
                        visible=False
                    )
                with gr.Row():
                    prev_page_button = gr.Button("Previous Page", visible=False)
                    page_display = gr.Markdown("Page 1 of X", visible=False)
                    next_page_button = gr.Button("Next Page", visible=False)

                with gr.Row():
                    podcast_custom_prompt_input = gr.Textbox(
                        label="Custom Prompt",
                        placeholder="Enter custom prompt here",
                        lines=10,
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
                        lines=10,
                        visible=False
                    )

                # Handle custom prompt checkbox change
                podcast_custom_prompt_checkbox.change(
                    fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
                    inputs=[podcast_custom_prompt_checkbox],
                    outputs=[podcast_custom_prompt_input, system_prompt_input]
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
                    prompts, total_pages, current_page = list_prompts(page=new_page, per_page=20)
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

                # Update prompts when a preset is selected
                def update_prompts(preset_name):
                    prompts = update_user_prompt(preset_name)
                    return (
                        gr.update(value=prompts["user_prompt"], visible=True),
                        gr.update(value=prompts["system_prompt"], visible=True)
                    )

                preset_prompt.change(
                    fn=update_prompts,
                    inputs=[preset_prompt],
                    outputs=[podcast_custom_prompt_input, system_prompt_input]
                )

                # Refactored API selection dropdown
                podcast_api_name_input = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Analysis/Summarization (Optional)"
                )
                podcast_api_key_input = gr.Textbox(label="API Key (if required)", type="password")
                podcast_whisper_model_input = gr.Dropdown(choices=whisper_models, value="distil-large-v3", label="Whisper Model")

                keep_original_input = gr.Checkbox(label="Keep original audio file", value=False)
                enable_diarization_input = gr.Checkbox(label="Enable speaker diarization", value=False)

                use_cookies_input = gr.Checkbox(label="Use cookies for yt-dlp", value=False)
                cookies_input = gr.Textbox(
                    label="yt-dlp Cookies",
                    placeholder="Paste your cookies here (JSON format)",
                    lines=3,
                    visible=False
                )

                use_cookies_input.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[use_cookies_input],
                    outputs=[cookies_input]
                )

                chunking_options_checkbox = gr.Checkbox(label="Show Chunking Options", value=False)
                with gr.Row(visible=False) as chunking_options_box:
                    gr.Markdown("### Chunking Options")
                    with gr.Column():
                        chunk_method = gr.Dropdown(choices=['words', 'sentences', 'paragraphs', 'tokens'], label="Chunking Method")
                        max_chunk_size = gr.Slider(minimum=100, maximum=1000, value=300, step=50, label="Max Chunk Size")
                        chunk_overlap = gr.Slider(minimum=0, maximum=100, value=0, step=10, label="Chunk Overlap")
                        use_adaptive_chunking = gr.Checkbox(label="Use Adaptive Chunking")
                        use_multi_level_chunking = gr.Checkbox(label="Use Multi-level Chunking")
                        chunk_language = gr.Dropdown(choices=['english', 'french', 'german', 'spanish'], label="Chunking Language")

                chunking_options_checkbox.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[chunking_options_checkbox],
                    outputs=[chunking_options_box]
                )

                podcast_process_button = gr.Button("Process Podcast")

            with gr.Column():
                podcast_progress_output = gr.Textbox(label="Progress")
                podcast_error_output = gr.Textbox(label="Error Messages")
                podcast_transcription_output = gr.Textbox(label="Transcription")
                podcast_summary_output = gr.Textbox(label="Summary")
                download_transcription = gr.File(label="Download Transcription as JSON")
                download_summary = gr.File(label="Download Summary as Text")

        podcast_process_button.click(
            fn=process_podcast,
            inputs=[
                podcast_url_input,
                podcast_title_input,
                podcast_author_input,
                podcast_keywords_input,
                podcast_custom_prompt_input,
                podcast_api_name_input,
                podcast_api_key_input,
                podcast_whisper_model_input,
                keep_original_input,
                enable_diarization_input,
                use_cookies_input,
                cookies_input,
                chunk_method,
                max_chunk_size,
                chunk_overlap,
                use_adaptive_chunking,
                use_multi_level_chunking,
                chunk_language,
                keep_timestamps_input,
                system_prompt_input  # Include system prompt input
            ],
            outputs=[
                podcast_progress_output,
                podcast_transcription_output,
                podcast_summary_output,
                podcast_title_input,
                podcast_author_input,
                podcast_keywords_input,
                podcast_error_output,
                download_transcription,
                download_summary
            ]
        )