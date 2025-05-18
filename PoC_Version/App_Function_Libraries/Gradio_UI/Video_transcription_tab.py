# Video_transcription_tab.py
# Description: This file contains the code for the video transcription tab in the Gradio UI.
#
# Imports
import inspect
import json
import os
from datetime import datetime
#
# External Imports
import gradio as gr
import yt_dlp

from PoC_Version.App_Function_Libraries.Chunk_Lib import improved_chunking_process
#
# Local Imports
from PoC_Version.App_Function_Libraries.DB.DB_Manager import add_media_to_database, \
    check_media_and_whisper_model, check_existing_media, update_media_content_with_version, list_prompts
from PoC_Version.App_Function_Libraries.Gradio_UI.Gradio_Shared import whisper_models, update_user_prompt
from PoC_Version.App_Function_Libraries.Gradio_UI.Gradio_Shared import error_handler
from PoC_Version.App_Function_Libraries.Summarization.Summarization_General_Lib import perform_transcription, perform_summarization, \
    save_transcription_and_summary
from PoC_Version.App_Function_Libraries.Utils.Utils import convert_to_seconds, safe_read_file, format_transcription, \
    create_download_directory, generate_unique_identifier, extract_text_from_segments, default_api_endpoint, \
    global_api_endpoints, format_api_name, load_and_log_configs, logging
from PoC_Version.App_Function_Libraries.Video_DL_Ingestion_Lib import parse_and_expand_urls, extract_metadata, download_video
from PoC_Version.App_Function_Libraries.Benchmarks_Evaluations.ms_g_eval import run_geval
# Import metrics logging
from PoC_Version.App_Function_Libraries.Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:

def create_video_transcription_tab():
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
    with (gr.TabItem("Video Transcription + Summarization", visible=True)):
        gr.Markdown("# Transcribe & Summarize Videos from URLs")
        with gr.Row():
            gr.Markdown("""Follow this project at [tldw - GitHub](https://github.com/rmusser01/tldw)""")
        with gr.Row():
            gr.Markdown(
                """If you're wondering what all this is, please see the 'Introduction/Help' tab up above for more detailed information and how to obtain an API Key.""")
        with gr.Row():
            with gr.Column():
                url_input = gr.Textbox(label="URL(s) (Mandatory)",
                                       placeholder="Enter video URLs here, one per line. Supports YouTube, Vimeo, other video sites and YouTube playlists.",
                                       lines=5)
                video_files = gr.File(label="Upload Video File(s) (Optional)", file_types=[".mp4", ".avi", ".mov", ".mkv", ".webm"], file_count="multiple")
                whisper_model_input = gr.Dropdown(choices=whisper_models, value="distil-large-v3", label="Whisper Model")

                with gr.Row():
                    diarize_input = gr.Checkbox(label="Enable Speaker Diarization", value=False)
                    vad_checkbox = gr.Checkbox(label="Enable Voice-Audio-Detection (VAD)", value=True)

                with gr.Row():
                    custom_prompt_checkbox = gr.Checkbox(label="Use a Custom Prompt",
                                                         value=False,
                                                         visible=True)
                    preset_prompt_checkbox = gr.Checkbox(label="Use a pre-set Prompt",
                                                         value=False,
                                                         visible=True)

                # Initialize state variables for pagination
                current_page_state = gr.State(value=1)
                total_pages_state = gr.State(value=1)

                with gr.Row():
                    # Add pagination controls
                    preset_prompt = gr.Dropdown(label="Select Preset Prompt",
                                                choices=[],
                                                visible=False)
                with gr.Row():
                    prev_page_button = gr.Button("Previous Page", visible=False)
                    page_display = gr.Markdown("Page 1 of X", visible=False)
                    next_page_button = gr.Button("Next Page", visible=False)
                with gr.Row():
                    system_prompt_input = gr.Textbox(label="System Prompt",
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
                                                     lines=3,
                                                     visible=False,
                                                     interactive=True)
                with gr.Row():
                    custom_prompt_input = gr.Textbox(label="Custom Prompt",
                                                     placeholder="Enter custom prompt here",
                                                     lines=3,
                                                     visible=False)

                custom_prompt_checkbox.change(
                    fn=lambda x: (gr.update(visible=x, interactive=x), gr.update(visible=x, interactive=x)),
                    inputs=[custom_prompt_checkbox],
                    outputs=[custom_prompt_input, system_prompt_input]
                )

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
                            total_pages  # total_pages_state
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

                def update_prompts(preset_name):
                    prompts = update_user_prompt(preset_name)
                    return (
                        gr.update(value=prompts["user_prompt"], visible=True, interactive=True),
                        gr.update(value=prompts["system_prompt"], visible=True, interactive=True)
                    )

                preset_prompt.change(
                    update_prompts,
                    inputs=preset_prompt,
                    outputs=[custom_prompt_input, system_prompt_input]
                )

                # Refactored API selection dropdown
                api_name_input = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Analysis/Summarization (Optional)"
                )
                api_key_input = gr.Textbox(label="API Key (Optional - Set in Config.txt)",
                                           placeholder="Enter your API key here",
                                           type="password")
                keywords_input = gr.Textbox(label="Keywords",
                                            placeholder="Enter keywords here (comma-separated)",
                                            value="default,no_keyword_set")
                # FIXME - Add proper support for this feature
                batch_size_input = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                             label="Batch Size (Number of videos to process simultaneously)", visible=False)
                timestamp_option = gr.Checkbox(label="Include Timestamps", value=True)
                keep_original_video = gr.Checkbox(label="Keep Original Video", value=False)
                perform_chunking = gr.Checkbox(label="Enable Chunking", value=False)
                chunking_options_checkbox = gr.Checkbox(label="Show Chunking Options", value=False)
                summarize_recursively = gr.Checkbox(label="Enable Recursive Summarization", value=False)
                use_cookies_input = gr.Checkbox(label="Use cookies for authenticated download", value=False)
                use_time_input = gr.Checkbox(label="Use Start and End Time", value=False)
                confab_checkbox = gr.Checkbox(label="Perform Confabulation Check of Summary", value=False)
                overwrite_checkbox = gr.Checkbox(label="Overwrite Existing Media", value=False)
                with gr.Row(visible=False) as time_input_box:
                    gr.Markdown("### Start and End time")
                    with gr.Column():
                        start_time_input = gr.Textbox(label="Start Time (Optional)",
                                                      placeholder="e.g., 1:30 or 90 (in seconds)")
                        end_time_input = gr.Textbox(label="End Time (Optional)",
                                                    placeholder="e.g., 5:45 or 345 (in seconds)")

                use_time_input.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[use_time_input],
                    outputs=[time_input_box]
                )

                cookies_input = gr.Textbox(
                    label="User Session Cookies",
                    placeholder="Paste your cookies here (JSON format)",
                    lines=3,
                    visible=False
                )

                use_cookies_input.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[use_cookies_input],
                    outputs=[cookies_input]
                )
                # Then, create a Box to group the chunking options
                with gr.Row(visible=False) as chunking_options_box:
                    gr.Markdown("### Chunking Options")
                    with gr.Column():
                        chunk_method = gr.Dropdown(choices=['words', 'sentences', 'paragraphs', 'tokens'],
                                                   label="Chunking Method")
                        max_chunk_size = gr.Slider(minimum=100, maximum=8000, value=400, step=1,
                                                   label="Max Chunk Size")
                        chunk_overlap = gr.Slider(minimum=0, maximum=5000, value=100, step=1,
                                                  label="Chunk Overlap")
                        use_adaptive_chunking = gr.Checkbox(
                            label="Use Adaptive Chunking (Adjust chunking based on text complexity)")
                        use_multi_level_chunking = gr.Checkbox(label="Use Multi-level Chunking")
                        chunk_language = gr.Dropdown(choices=['english', 'french', 'german', 'spanish'],
                                                     label="Chunking Language")

                # Add JavaScript to toggle the visibility of the chunking options box
                chunking_options_checkbox.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[chunking_options_checkbox],
                    outputs=[chunking_options_box]
                )
                process_button = gr.Button("Process Videos")

            with gr.Column():
                progress_output = gr.Textbox(label="Progress")
                error_output = gr.Textbox(label="Errors", visible=False)
                results_output = gr.HTML(label="Results")
                confabulation_output = gr.Textbox(label="Confabulation Check Results", visible=False)
                download_transcription = gr.File(label="Download All Transcriptions as JSON")
                download_summary = gr.File(label="Download All Summaries as Text")

            @error_handler
            def process_videos_with_error_handling(inputs,
                        start_time,
                        end_time,
                        diarize,
                        vad_use,
                        whisper_model,
                        custom_prompt_checkbox,
                        custom_prompt,
                        chunking_options_checkbox,
                        perform_chunking,
                        chunk_method,
                        max_chunk_size,
                        chunk_overlap,
                        use_adaptive_chunking,
                        use_multi_level_chunking,
                        chunk_language,
                        summarize_recursively,
                        api_name,
                        api_key,
                        keywords,
                        use_cookies,
                        cookies,
                        batch_size,
                        timestamp_option,
                        keep_original_video,
                        confab_checkbox,
                        overwrite_existing=False,
                        progress: gr.Progress = gr.Progress()) -> tuple:
                try:
                    # Start overall processing timer
                    proc_start_time = datetime.now()
                    logging.info("Entering process_videos_with_error_handling")
                    logging.info(f"Received inputs: {inputs}")

                    if not inputs:
                        raise ValueError("No inputs provided")

                    logging.debug("Input(s) is(are) valid")

                    # Ensure batch_size is an integer
                    try:
                        batch_size = int(batch_size)
                    except (ValueError, TypeError):
                        batch_size = 1

                    # Separate URLs and local files
                    urls = [input for input in inputs if
                            isinstance(input, str) and input.startswith(('http://', 'https://'))]
                    local_files = [input for input in inputs if
                                   isinstance(input, str) and not input.startswith(('http://', 'https://'))]

                    # Parse and expand URLs if there are any
                    expanded_urls = parse_and_expand_urls(urls) if urls else []

                    valid_local_files = []
                    invalid_local_files = []

                    for file_path in local_files:
                        if os.path.exists(file_path):
                            valid_local_files.append(file_path)
                        else:
                            invalid_local_files.append(file_path)
                            logging.error(f"Local file not found: {file_path}")

                    if invalid_local_files:
                        logging.warning(f"Found {len(invalid_local_files)} invalid local file paths")
                        # FIXME - Add more complete error handling for invalid local files

                    all_inputs = expanded_urls + valid_local_files
                    logging.info(f"Total valid inputs to process: {len(all_inputs)} "
                                 f"({len(expanded_urls)} URLs, {len(valid_local_files)} local files)")

                    all_inputs = expanded_urls + local_files
                    logging.info(f"Total inputs to process: {len(all_inputs)}")
                    results = []
                    errors = []
                    results_html = ""
                    all_transcriptions = {}
                    all_summaries = ""

                    # Start timing
                    start_proc = datetime.now()

                    for i in range(0, len(all_inputs), batch_size):
                        batch = all_inputs[i:i + batch_size]
                        batch_results = []

                        for input_item in batch:
                            # Start individual video processing timer
                            video_start_time = datetime.now()
                            try:
                                start_seconds = convert_to_seconds(start_time)
                                end_seconds = convert_to_seconds(end_time) if end_time else None
                                logging.info(f"Processing {input_item}")

                                if input_item.startswith(('http://', 'https://')):
                                    logging.info(f"Attempting to extract metadata for URL: {input_item}")
                                    video_metadata = extract_metadata(input_item, use_cookies, cookies)
                                    if not video_metadata:
                                        raise ValueError(f"Failed to extract metadata for {input_item}")
                                else:
                                    logging.info(f"Processing local file: {input_item}")
                                    video_metadata = {"title": os.path.basename(input_item), "url": input_item}

                                chunk_options = {
                                    'method': chunk_method,
                                    'max_size': max_chunk_size,
                                    'overlap': chunk_overlap,
                                    'adaptive': use_adaptive_chunking,
                                    'multi_level': use_multi_level_chunking,
                                    'language': chunk_language
                                } if chunking_options_checkbox else None

                                if custom_prompt_checkbox:
                                    custom_prompt = custom_prompt
                                else:
                                    custom_prompt = ("""
                                    <s>You are a bulleted notes specialist. ```When creating comprehensive bulleted notes, you should follow these guidelines: Use multiple headings based on the referenced topics, not categories like quotes or terms. Headings should be surrounded by bold formatting and not be listed as bullet points themselves. Leave no space between headings and their corresponding list items underneath. Important terms within the content should be emphasized by setting them in bold font. Any text that ends with a colon should also be bolded. Before submitting your response, review the instructions, and make any corrections necessary to adhered to the specified format. Do not reference these instructions within the notes.``` \nBased on the content between backticks create comprehensive bulleted notes.
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
                                    """)

                                logging.debug("Gradio_Related.py: process_url_with_metadata being called")
                                # FIXME - Would assume this is where the multi-processing for recursive summarization would occur
                                result = process_url_with_metadata(
                                    input_item, 2, whisper_model,
                                    custom_prompt,
                                    start_seconds, api_name, api_key,
                                    vad_use, False, False, summarize_recursively, 0.01, None, keywords, None, diarize,
                                    end_time=end_seconds,
                                    include_timestamps=timestamp_option,
                                    metadata=video_metadata,
                                    use_chunking=perform_chunking,
                                    chunk_options=chunk_options,
                                    keep_original_video=keep_original_video,
                                    current_whisper_model=whisper_model,
                                    overwrite_existing=overwrite_existing
                                )

                                if result[0] is None:
                                    error_message = "Processing failed without specific error"
                                    batch_results.append((input_item, error_message, "Error", video_metadata, None, None))
                                    errors.append(f"Error processing {input_item}: {error_message}")

                                    # Log failure metric
                                    log_counter(
                                        metric_name="videos_failed_total",
                                        labels={"whisper_model": whisper_model, "api_name": api_name},
                                        value=1
                                    )

                                else:
                                    url, transcription, summary, json_file, summary_file, result_metadata = result
                                    if transcription is None:
                                        error_message = f"Processing failed for {input_item}: Transcription is None"
                                        batch_results.append(
                                            (input_item, error_message, "Error", result_metadata, None, None))
                                        errors.append(error_message)

                                        # Log failure metric
                                        log_counter(
                                            metric_name="videos_failed_total",
                                            labels={"whisper_model": whisper_model, "api_name": api_name},
                                            value=1
                                        )

                                    else:
                                        batch_results.append(
                                            (input_item, transcription, "Success", result_metadata, json_file,
                                             summary_file))

                                        # Log success metric
                                        log_counter(
                                            metric_name="videos_processed_total",
                                            labels={"whisper_model": whisper_model, "api_name": api_name},
                                            value=1
                                        )

                                        # Calculate processing time
                                        video_end_time = datetime.now()
                                        processing_time = (video_end_time - video_start_time).total_seconds()
                                        log_histogram(
                                            metric_name="video_processing_time_seconds",
                                            value=processing_time,
                                            labels={"whisper_model": whisper_model, "api_name": api_name}
                                        )

                                        # Log transcription and summary metrics
                                        if transcription:
                                            log_counter(
                                                metric_name="transcriptions_generated_total",
                                                labels={"whisper_model": whisper_model},
                                                value=1
                                            )
                                        if summary:
                                            log_counter(
                                                metric_name="summaries_generated_total",
                                                labels={"whisper_model": whisper_model},
                                                value=1
                                            )

                            except Exception as e:
                                # Log failure
                                log_counter(
                                    metric_name="videos_failed_total",
                                    labels={"whisper_model": whisper_model, "api_name": api_name},
                                    value=1
                                )
                                error_message = f"Error processing {input_item}: {str(e)}"
                                logging.error(error_message, exc_info=True)
                                batch_results.append((input_item, error_message, "Error", {}, None, None))
                                errors.append(error_message)
                        results.extend(batch_results)
                        logging.info(f"Processed {len(batch_results)} videos in batch")
                        if isinstance(progress, gr.Progress):
                            progress((i + len(batch)) / len(all_inputs),
                                     f"Processed {i + len(batch)}/{len(all_inputs)} videos")

                    # Generate HTML for results
                    logging.info(f"Generating HTML for {len(results)} results")
                    for url, transcription, status, metadata, json_file, summary_file in results:
                        if status == "Success":
                            title = metadata.get('title', 'Unknown Title')

                            # Check if transcription is a string (which it should be now)
                            if isinstance(transcription, str):
                                # Split the transcription into metadata and actual transcription
                                parts = transcription.split('\n\n', 1)
                                if len(parts) == 2:
                                    metadata_text, transcription_text = parts
                                else:
                                    metadata_text = "Metadata not found"
                                    transcription_text = transcription
                            else:
                                metadata_text = "Metadata format error"
                                transcription_text = "Transcription format error"

                            loaded_config = load_and_log_configs()
                            save_transcripts = loaded_config["system_preferences"]["save_video_transcripts"]
                            if save_transcripts:
                                summary = safe_read_file(summary_file) if summary_file else "No summary available"
                            else:
                                summary = summary if summary else "No summary available"
                            # FIXME - Add to other functions that generate HTML
                            # Format the transcription
                            formatted_transcription = format_transcription(transcription_text)
                            # Format the summary
                            formatted_summary = format_transcription(summary)

                            logging.info("Creating HTML for results")
                            results_html += f"""
                            <div class="result-box">
                                <gradio-accordion>
                                    <gradio-accordion-item label="{title}">
                                        <p><strong>URL:</strong> <a href="{url}" target="_blank">{url}</a></p>
                                        <h4>Metadata:</h4>
                                        <pre>{metadata_text}</pre>
                                        <h4>Transcription:</h4>
                                        <div class="transcription" style="white-space: pre-wrap; word-wrap: break-word;">
                                            {formatted_transcription}
                                        </div>
                                        <h4>Summary:</h4>
                                        <div class="summary">{formatted_summary}</div>
                                    </gradio-accordion-item>
                                </gradio-accordion>
                            </div>
                            """
                            logging.debug(f"Transcription for {url}: {transcription[:200]}...")
                            all_transcriptions[url] = transcription
                            all_summaries += f"Title: {title}\nURL: {url}\n\n{metadata_text}\n\nTranscription:\n{transcription_text}\n\nSummary:\n{summary}\n\n---\n\n"
                            logging.info(f"HTML created for {title}")
                        else:
                            results_html += f"""
                            <div class="result-box error">
                                <h3>Error processing {url}</h3>
                                <p>{transcription}</p>
                            </div>
                            """

                    # Save all transcriptions and summaries to files
                    logging.debug("Saving all transcriptions and summaries to files")
                    with open('all_transcriptions.json', 'w', encoding='utf-8') as f:
                        json.dump(all_transcriptions, f, indent=2, ensure_ascii=False)

                    with open('all_summaries.txt', 'w', encoding='utf-8') as f:
                        f.write(all_summaries)

                    error_summary = "\n".join(errors) if errors else "No errors occurred."

                    total_inputs = len(all_inputs)

                    # End overall processing timer
                    proc_end_time = datetime.now()
                    total_processing_time = (proc_end_time - proc_start_time).total_seconds()
                    log_histogram(
                        metric_name="total_processing_time_seconds",
                        value=total_processing_time,
                        labels={"whisper_model": whisper_model, "api_name": api_name}
                    )

                    return (
                        f"Processed {total_inputs} videos. {len(errors)} errors occurred.",
                        error_summary,
                        results_html,
                        'all_transcriptions.json',
                        'all_summaries.txt'
                    )
                except Exception as e:
                    logging.error(f"Unexpected error in process_videos_with_error_handling: {str(e)}", exc_info=True)

                    # Log unexpected failure metric
                    log_counter(
                        metric_name="videos_failed_total",
                        labels={"whisper_model": whisper_model, "api_name": api_name},
                        value=1
                    )

                    return (
                        f"An unexpected error occurred: {str(e)}",
                        str(e),
                        "<div class='result-box error'><h3>Unexpected Error</h3><p>" + str(e) + "</p></div>",
                        None,
                        None
                    )

            def process_videos_wrapper(url_input, video_files, start_time, end_time, diarize, vad_use, whisper_model,
                                       custom_prompt_checkbox, custom_prompt, chunking_options_checkbox,
                                       perform_chunking, chunk_method, max_chunk_size, chunk_overlap,
                                       use_adaptive_chunking, use_multi_level_chunking, chunk_language,
                                       summarize_recursively, api_name, api_key, keywords, use_cookies, cookies,
                                       batch_size, timestamp_option, keep_original_video, confab_checkbox,
                                       overwrite_existing=False):
                global result
                try:
                    logging.info("process_videos_wrapper(): process_videos_wrapper called")

                    # Define file paths
                    transcriptions_file = os.path.join('all_transcriptions.json')
                    summaries_file = os.path.join('all_summaries.txt')

                    # Delete existing files if they exist
                    for file_path in [transcriptions_file, summaries_file]:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                logging.info(f"Deleted existing file: {file_path}")
                        except Exception as e:
                            logging.warning(f"Failed to delete file {file_path}: {str(e)}")

                    # Handle both URL input and multiple file uploads
                    inputs = []

                    # Process URLs if provided
                    if url_input:
                        inputs.extend([url.strip() for url in url_input.split('\n') if url.strip()])

                    # Process multiple video files if provided
                    if video_files is not None:
                        files_list = video_files if isinstance(video_files, list) else [video_files]

                        for file_obj in files_list:
                            if isinstance(file_obj, str):
                                inputs.append(file_obj)
                            elif hasattr(file_obj, 'path'):
                                inputs.append(file_obj.path)
                            elif isinstance(file_obj, dict) and 'path' in file_obj:
                                inputs.append(file_obj['path'])
                            elif hasattr(file_obj, 'name'):
                                inputs.append(file_obj.name)
                            else:
                                logging.warning(f"Unhandled file object type: {type(file_obj)}")
                                continue

                    if not inputs:
                        raise ValueError("No input provided. Please enter URLs or upload video files.")

                    logging.info(f"Processing inputs: {inputs}")

                    result = process_videos_with_error_handling(
                        inputs,
                        start_time,
                        end_time,
                        diarize,
                        vad_use,
                        whisper_model,
                        custom_prompt_checkbox,
                        custom_prompt,
                        chunking_options_checkbox,
                        perform_chunking,
                        chunk_method,
                        max_chunk_size,
                        chunk_overlap,
                        use_adaptive_chunking,
                        use_multi_level_chunking,
                        chunk_language,
                        summarize_recursively,
                        api_name,
                        api_key,
                        keywords,
                        use_cookies,
                        cookies,
                        batch_size,
                        timestamp_option,
                        keep_original_video,
                        confab_checkbox,
                        overwrite_existing
                    )

                    confabulation_result = None
                    if confab_checkbox:
                        logging.info("Confabulation check enabled")
                        # Assuming result[1] contains the transcript and result[2] contains the summary
                        confabulation_result = run_geval(result[1], result[2], api_key, api_name)
                        logging.info(f"Simplified G-Eval result: {confabulation_result}")

                    # Ensure that result is a tuple with 5 elements
                    if not isinstance(result, tuple) or len(result) != 5:
                        raise ValueError(
                            f"process_videos_wrapper(): Expected 5 outputs, but got {len(result) if isinstance(result, tuple) else 1}")

                    # Return the confabulation result along with other outputs
                    return (*result, confabulation_result)

                except Exception as e:
                    logging.error(f"process_videos_wrapper(): Error in process_videos_wrapper: {str(e)}", exc_info=True)
                    # Return a tuple with 6 elements in case of any error (including None for simple_geval_result)
                    return (
                        f"process_videos_wrapper(): An error occurred: {str(e)}",  # progress_output
                        str(e),  # error_output
                        f"<div class='error'>Error: {str(e)}</div>",  # results_output
                        None,  # download_transcription
                        None,  # download_summary
                        None  # simple_geval_result
                    )

            # FIXME - remove dead args for process_url_with_metadata
            @error_handler
            def process_url_with_metadata(
                input_item,
                num_speakers,
                whisper_model,
                custom_prompt,
                offset,
                api_name,
                api_key,
                vad_filter,
                download_video_flag,
                download_audio,
                rolling_summarization,
                detail_level,
                question_box,
                keywords,
                local_file_path,
                diarize,
                end_time=None,
                include_timestamps=True,
                metadata=None,
                use_chunking=False,
                chunk_options=None,
                keep_original_video=False,
                current_whisper_model="Blank",
                overwrite_existing=False
            ):
                """
                Downloads (if needed) and processes a single video or local file, then performs transcription & summarization.

                :param input_item: String representing either a URL or local file path.
                :param num_speakers: (Unused) Number of speakers for diarization (if enabled).
                :param whisper_model: Name of the Whisper model to use for transcription.
                :param custom_prompt: Custom prompt to supply to the LLM for summarization.
                :param offset: Start offset in seconds for partial transcriptions.
                :param api_name: Name of the selected LLM API (e.g., "OpenAI").
                :param api_key: API key to pass to the LLM for summarization.
                :param vad_filter: Boolean, enable/disable VAD during transcription.
                :param download_video_flag: Boolean, whether to force downloading the entire video (vs. just the audio).
                :param download_audio: (Unused) Boolean to download audio only if True.
                :param rolling_summarization: Boolean, if True do multi-pass (recursive) summarization of chunk summaries.
                :param detail_level: (Unused) Additional detail level parameter.
                :param question_box: (Unused) Additional question prompt param.
                :param keywords: A string of comma-separated keywords or a list/tuple of keywords.
                :param local_file_path: (Unused) Possibly used if a downloaded file needs a custom path.
                :param diarize: Boolean, if True perform speaker diarization (requires specialized pipeline).
                :param end_time: End offset in seconds for partial transcriptions.
                :param include_timestamps: Boolean, whether to preserve timestamps in the final transcript.
                :param metadata: An optional pre-extracted metadata dict for the video.
                :param use_chunking: Boolean, if True chunk large text for summarization.
                :param chunk_options: Dictionary specifying chunking parameters (method, max size, overlap, etc.).
                :param keep_original_video: Boolean, if True do not delete the downloaded video file.
                :param current_whisper_model: String, the current Whisper model for DB checks.
                :param overwrite_existing: Boolean, if True re-process even if the file was processed previously.

                :return: (url, full_text_with_metadata, summary_text, json_file_path, summary_file_path, info_dict)
                         or (None, None, None, None, None, None) on error.
                """
                try:
                    logging.info(f"Starting process_url_with_metadata for URL: {input_item}")

                    # FIXME Add toggle to save to a different directory
                    # FIXME Add toggle to save to disk vs temp file

                    # Load config to see if transcripts should be saved
                    loaded_config = load_and_log_configs()
                    keep_transcripts = loaded_config["system_preferences"]["save_video_transcripts"]
                    if keep_transcripts:
                        download_path = create_download_directory("Video_Downloads")
                        logging.info(f"Download path created at: {download_path}")

                    # Initialize info_dict
                    info_dict = {}

                    # -------------------------------------------------
                    # 1) Distinguish local file vs. remote URL
                    # -------------------------------------------------
                    # Handle URL or local file
                    if os.path.isfile(input_item):
                        # Local file path
                        video_file_path = input_item
                        unique_id = generate_unique_identifier(input_item)
                        # Extract basic info from local file
                        info_dict = {
                            'webpage_url': unique_id,
                            'title': os.path.basename(input_item),
                            'description': "Local file",
                            'channel_url': None,
                            'duration': None,
                            'channel': None,
                            'uploader': None,
                            'upload_date': None
                        }
                    else:
                        # URL case: Extract video information via yt_dlp
                        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                            try:
                                full_info = ydl.extract_info(input_item, download=False)
                                # Create a safe subset of info to log
                                safe_info = {
                                    'title': full_info.get('title', 'No title'),
                                    'duration': full_info.get('duration', 'Unknown duration'),
                                    'upload_date': full_info.get('upload_date', 'Unknown upload date'),
                                    'uploader': full_info.get('uploader', 'Unknown uploader'),
                                    'view_count': full_info.get('view_count', 'Unknown view count')
                                }

                                logging.debug(f"Full info extracted for {input_item}: {safe_info}")
                            except Exception as e:
                                logging.error(f"Error extracting video info: {str(e)}")
                                return None, None, None, None, None, None

                        # Filter the required metadata
                        if full_info:
                            info_dict = {
                                'webpage_url': full_info.get('webpage_url', input_item),
                                'title': full_info.get('title'),
                                'description': full_info.get('description'),
                                'channel_url': full_info.get('channel_url'),
                                'duration': full_info.get('duration'),
                                'channel': full_info.get('channel'),
                                'uploader': full_info.get('uploader'),
                                'upload_date': full_info.get('upload_date')
                            }
                            logging.debug(f"Filtered info_dict: {info_dict}")
                        else:
                            logging.error("Failed to extract video information")
                            return None, None, None, None, None, None

                        # -------------------------------------------------
                        # 2) Check media in DB, possibly skip
                        # -------------------------------------------------
                        logging.info("Checking if media already exists in the database...")
                        media_exists, reason = check_media_and_whisper_model(
                            title=info_dict.get('title'),
                            url=info_dict.get('webpage_url'),
                            current_whisper_model=current_whisper_model
                        )

                        if not media_exists:
                            logging.info(
                                f"process_url_with_metadata: Media does not exist in the database. Reason: {reason}")
                        else:
                            if "same whisper model" in reason and not overwrite_existing:
                                logging.info(
                                    f"process_url_with_metadata: Skipping download and processing as media exists and uses the same Whisper model. Reason: {reason}")
                                return input_item, None, None, None, None, info_dict
                            else:
                                logging.info(
                                    f"process_url_with_metadata: Media found, but with a different Whisper model. Reason: {reason}")

                        # -------------------------------------------------
                        # 3) Download video (or skip if existing)
                        # -------------------------------------------------
                        logging.info("Downloading video/audio...")
                        video_file_path = download_video(input_item, download_path, full_info, download_video_flag,
                                                         current_whisper_model=current_whisper_model)
                        if video_file_path is None:
                            logging.info(
                                f"process_url_with_metadata: Download skipped for {input_item}. Media might already exist or be processed.")
                            return input_item, None, None, None, None, info_dict

                    # -------------------------------------------------
                    # 4) For local files, also check DB if needed
                    # -------------------------------------------------
                    media_exists, reason = check_media_and_whisper_model(
                        title=info_dict.get('title'),
                        url=info_dict.get('webpage_url'),
                        current_whisper_model=current_whisper_model
                    )
                    if media_exists and "same whisper model" in reason and not overwrite_existing:
                        logging.info("Skipping: local file already processed with same Whisper model.")
                        return input_item, None, None, None, None, info_dict
                    else:
                        logging.info(f"Proceeding with file: {video_file_path}")

                    # -------------------------------------------------
                    # 5) Perform transcription
                    # -------------------------------------------------
                    logging.info("process_url_with_metadata: Starting transcription...")
                    logging.info(f"process_url_with_metadata: overwrite existing?: {overwrite_existing}")
                    audio_file_path, segments = perform_transcription(video_file_path, offset, whisper_model,
                                                                      vad_filter, diarize, overwrite_existing)

                    if audio_file_path is None or segments is None:
                        logging.error("process_url_with_metadata: Transcription failed or segments not available.")
                        return None, None, None, None, None, None

                    logging.info(f"process_url_with_metadata: Transcription completed. Number of segments: {len(segments)}")

                    # Merge metadata + segments and save them
                    segments_with_metadata = {
                        "metadata": info_dict,
                        "segments": segments
                    }

                    # Save segments with metadata to JSON file
                    segments_json_path = os.path.splitext(audio_file_path)[0] + ".segments.json"
                    with open(segments_json_path, 'w') as f:
                        json.dump(segments_with_metadata, f, indent=2)

                    # Delete the .wav file after successful transcription
                    # FIXME - swap to pathlib
                    files_to_delete = [audio_file_path]
                    for file_path in files_to_delete:
                        if file_path and os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                logging.info(f"process_url_with_metadata: Successfully deleted file: {file_path}")
                            except Exception as e:
                                logging.warning(f"process_url_with_metadata: Failed to delete file {file_path}: {str(e)}")

                    # Delete the video file after successful transcription if not keeping original
                    # Modify the file deletion logic to respect keep_original_video
                    if not keep_original_video:
                        files_to_delete = [audio_file_path, video_file_path]
                        for file_path in files_to_delete:
                            if file_path and os.path.exists(file_path):
                                try:
                                    os.remove(file_path)
                                    logging.info(f"process_url_with_metadata: Successfully deleted file: {file_path}")
                                except Exception as e:
                                    logging.warning(f"process_url_with_metadata: Failed to delete file {file_path}: {str(e)}")
                    else:
                        logging.info(f"process_url_with_metadata: Keeping original video file: {video_file_path}")
                        logging.info(f"process_url_with_metadata: Keeping original audio file: {audio_file_path}")

                    # -------------------------------------------------
                    # 6) Possibly drop timestamps
                    # -------------------------------------------------
                    if not include_timestamps:
                        segments = [{'Text': segment['Text']} for segment in segments]

                    logging.info(f"Segments processed for timestamp inclusion: {segments}")

                    # Extract text from segments
                    transcription_text = extract_text_from_segments(segments)

                    if transcription_text.startswith("Error:"):
                        logging.error(f"process_url_with_metadata: Failed to extract transcription: {transcription_text}")
                        return None, None, None, None, None, None

                    # Combine full text with metadata at the top
                    full_text_with_metadata = f"{json.dumps(info_dict, indent=2)}\n\n{transcription_text}"
                    logging.debug(f"Full text with metadata extracted: {full_text_with_metadata[:100]}...")

                    logging.debug(f"process_url_with_metadata: Full text with metadata extracted: {full_text_with_metadata[:100]}...")

                    # -------------------------------------------------
                    # 7) Summarize (if API is specified)
                    # -------------------------------------------------
                    summary_text = None
                    # If an API is provided and is not "None" or an empty string, perform summarization.
                    if api_name and api_name.lower() != "none" and api_name.strip() != "":
                        # API key resolution handled at base of function if none provided
                        api_key = api_key if api_key else None
                        logging.info(f"process_url_with_metadata: Starting summarization with {api_name}...")

                        # Perform Chunking if enabled
                        # FIXME - Setup a proper prompt for Recursive Summarization
                        if use_chunking:
                            logging.info("process_url_with_metadata: Chunking enabled. Starting chunking...")
                            chunked_texts = improved_chunking_process(full_text_with_metadata, chunk_options)

                            if not chunked_texts:
                                logging.warning("Chunking failed; falling back to full-text summarization.")
                                summary_text = perform_summarization(api_name, full_text_with_metadata, custom_prompt, api_key)
                            else:
                                # Summarize each chunk once
                                chunk_summaries = []
                                for chunk in chunked_texts:
                                    chunk_summary = perform_summarization(api_name, chunk['text'], custom_prompt, api_key)
                                    if chunk_summary:
                                        chunk_summaries.append(chunk_summary)
                                    else:
                                        idx = chunk['metadata'].get('chunk_index', '?')
                                        logging.error(f"Summarization failed for chunk {idx}.")

                                logging.debug(f"chunk_summaries: {chunk_summaries}")
                                if chunk_summaries:
                                    if rolling_summarization and len(chunk_summaries) > 1:
                                        # Recursively summarize the combined chunk summaries
                                        combined_text = "\n\n".join(chunk_summaries)
                                        summary_text = perform_summarization(api_name, combined_text, custom_prompt, api_key)
                                        if summary_text:
                                            logging.info("Recursive summarization successful.")
                                        else:
                                            logging.error("Recursive summarization failed.")
                                    else:
                                        # Join each chunk's summary
                                        summary_text = "\n\n".join(chunk_summaries)
                                        logging.info(f"Summarized {len(chunk_summaries)} chunk(s).")
                                    if not summary_text:
                                        logging.error("All chunk summarizations failed.")
                                        summary_text = None
                        else:
                            # Summarize the entire transcription
                            summary_text = perform_summarization(api_name, full_text_with_metadata, custom_prompt, api_key) if api_name else None
                            logging.debug(f"Just got summary_text from perform_summarization: {repr(summary_text)}")
                            if summary_text:
                                logging.info("Summarization completed successfully without chunking.")
                            else:
                                logging.error("Summarization failed for full text.")

                        # If still no summary, abort
                        if not summary_text:
                            logging.error("Summarization failed overall.")
                            return None, None, None, None, None, None

                    # If summary_text is a generator, consume it.
                    if inspect.isgenerator(summary_text):
                        summary_text = "".join(summary_text)
                        logging.debug(f"process_url_with_metadata: Consumed generator for summary_text. Generated summary: {summary_text}")
                    logging.debug(f"process_url_with_metadata: Summarization complete (first 100 chars): {summary_text[:100]}...")

                    # -------------------------------------------------
                    # 8) Save transcription + summary to disk
                    # -------------------------------------------------
                    load_config = load_and_log_configs()
                    save_transcripts = load_config["system_preferences"]["save_video_transcripts"]
                    if save_transcripts:
                        logging.info("process_url_with_metadata: Saving transcription and summary to disk")
                        download_path = create_download_directory("Audio_Processing")
                        logging.debug(f"Type of summary_text: {type(summary_text)}")
                        logging.debug(f"Preview of summary_text: {repr(summary_text)[:300]}")
                        json_file_path, summary_file_path = save_transcription_and_summary(full_text_with_metadata,
                                                                                           summary_text,
                                                                                           download_path, info_dict)
                        logging.info(f"process_url_with_metadata: Transcription saved to: {json_file_path}")
                        logging.info(f"process_url_with_metadata: Summary saved to: {summary_file_path}")
                    else:
                        logging.info("process_url_with_metadata: Not saving transcripts. Using temporary files.(save_video_transcripts set to False).")
                        # FIXME - Add temporary file handling
                        # For now,
                        json_file_path, summary_file_path = None, None

                    # -------------------------------------------------
                    # 9) Store data in the DB
                    # -------------------------------------------------
                    if isinstance(keywords, str):
                        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                    elif isinstance(keywords, (list, tuple)):
                        keywords_list = keywords
                    else:
                        keywords_list = []
                    logging.info(f"process_url_with_metadata: Keywords prepared: {keywords_list}")

                    existing_media = check_existing_media(info_dict['webpage_url'])

                    if existing_media:
                         # We have an entry, so update it (creating a new "version" row in DB)
                        media_id = existing_media['id']
                        update_result = update_media_content_with_version(media_id, info_dict, full_text_with_metadata,
                                                                          custom_prompt, summary_text, whisper_model)
                        logging.info(f"process_url_with_metadata: {update_result}")
                    else:
                        # Create a new row
                        add_result = add_media_to_database(info_dict['webpage_url'], info_dict, full_text_with_metadata,
                                                           summary_text,
                                                           keywords_list, custom_prompt, whisper_model)
                        logging.info(f"process_url_with_metadata: {add_result}")

                    # -------------------------------------------------
                    # 10) Return results
                    # -------------------------------------------------
                    return (
                        info_dict['webpage_url'],
                        full_text_with_metadata,
                        summary_text,
                        json_file_path,
                        summary_file_path,
                        info_dict
                    )
                except Exception as e:
                    logging.error(f"Error in process_url_with_metadata: {str(e)}", exc_info=True)
                    return None, None, None, None, None, None

            def toggle_confabulation_output(checkbox_value):
                return gr.update(visible=checkbox_value)

            confab_checkbox.change(
                fn=toggle_confabulation_output,
                inputs=[confab_checkbox],
                outputs=[confabulation_output]
            )

            process_button.click(
                fn=process_videos_wrapper,
                inputs=[
                    url_input,
                    video_files,
                    start_time_input,
                    end_time_input,
                    diarize_input,
                    vad_checkbox,
                    whisper_model_input,
                    custom_prompt_checkbox,
                    custom_prompt_input,
                    chunking_options_checkbox,
                    perform_chunking,
                    chunk_method,
                    max_chunk_size,
                    chunk_overlap,
                    use_adaptive_chunking,
                    use_multi_level_chunking,
                    chunk_language,
                    summarize_recursively,
                    api_name_input,
                    api_key_input,
                    keywords_input,
                    use_cookies_input,
                    cookies_input,
                    batch_size_input,
                    timestamp_option,
                    keep_original_video,
                    confab_checkbox,
                    overwrite_checkbox
                ],
                outputs=[progress_output, error_output, results_output, download_transcription, download_summary, confabulation_output]
            )

#
# End of Video_transcription_tab.py
#######################################################################################################################
