# Plaintext_tab_import.py
# Contains the code for the "Import Plain Text Files" tab in the Gradio UI.
# This tab allows users to upload plain text files (Markdown, Text, RTF) or a zip file containing multiple files.
# The user can provide a title, author, keywords, system prompt, custom user prompt, and select an API for
#   auto-summarization.
#
#######################################################################################################################
#
# Import necessary libraries
import json
#
# Import Non-Local
import gradio as gr
#
# Import Local libraries
from App_Function_Libraries.Plaintext.Plaintext_Files import final_ingest_handler, preview_import_handler
from App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging


#
#######################################################################################################################
#
# Functions:

def show_contents_unescaped(json_data):
    # Parse the JSON
    previews = json.loads(json_data)
    # previews is a list of dicts, each with "filename", "content", etc.

    # Build up a string that includes real newlines
    output_lines = []
    for preview in previews:
        filename = preview["filename"]
        content = preview["content"]
        output_lines.append(f"=== {filename} ===\n{content}\n")

    return "\n".join(output_lines)


def create_plain_text_import_tab():
    try:
        default_value = None
        if default_api_endpoint and default_api_endpoint in global_api_endpoints:
            default_value = format_api_name(default_api_endpoint)
    except Exception as e:
        logging.error(f"Error setting default API endpoint: {str(e)}")
        default_value = None
    with gr.TabItem("Import Plain Text & .docx Files", visible=True):
        gr.Markdown("# Import Plaintext files")
        gr.Markdown("Upload txt/docx/md/rtf/zip collections of them for import")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: Upload / Preview")
                import_files = gr.File(
                    label="Upload files or .zip",
                    file_count="multiple",
                    file_types=["text", ".md", ".txt", ".rtf", ".docx", ".zip"]
                )
                author_input = gr.Textbox(label="Author")
                keywords_input = gr.Textbox(label="Keywords (comma-separated)")
                system_prompt_input = gr.Textbox(label="System Prompt", lines=3)
                custom_prompt_input = gr.Textbox(label="User Prompt", lines=2)
                auto_summarize_checkbox = gr.Checkbox(label="Auto Summarize?")
                api_name_input = gr.Dropdown(
                    label="API for Summaries",
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value
                )
                api_key_input = gr.Textbox(label="API Key", type="password")

                preview_button = gr.Button("Preview")

                preview_status = gr.Textbox(label="Preview Status", lines=5)

            with gr.Column():
                # We'll store the preview data in a hidden state
                preview_data_state = gr.State()
                preview_data_display = gr.Textbox(
                    label="Preview Data",
                    lines=30,
                    interactive=False,
                    elem_id="scrollable-textbox"
                )

        # Step 2: Ingest with possible metadata overrides
        gr.Markdown("### Step 2: Final Ingest with Overrides")
        metadata_overrides_input = gr.Textbox(
            label="Metadata Overrides (JSON)",
            lines=6,
            placeholder='{"myfile.md": {"title": "New Title", "author": "New Author"}}'
        )
        ingest_button = gr.Button("Confirm Ingest")
        ingest_status = gr.Textbox(label="Ingest Status", lines=5)

        # Wire up events
        preview_button.click(
            fn=preview_import_handler,
            inputs=[
                import_files, author_input, keywords_input,
                system_prompt_input, custom_prompt_input,
                auto_summarize_checkbox, api_name_input, api_key_input
            ],
            outputs=[preview_status, preview_data_state]
        )

        # Show the text in preview_data_display
        preview_button.click(
            fn=preview_import_handler,
            inputs=[
                import_files, author_input, keywords_input,
                system_prompt_input, custom_prompt_input,
                auto_summarize_checkbox, api_name_input, api_key_input
            ],
            outputs=[preview_status, preview_data_state]
        ).then(
            fn=show_contents_unescaped,
            inputs=preview_data_state,
            outputs=preview_data_display
        )

        ingest_button.click(
            fn=final_ingest_handler,
            inputs=[preview_data_state, metadata_overrides_input],
            outputs=ingest_status
        )

    return (
        import_files,
        author_input,
        keywords_input,
        system_prompt_input,
        custom_prompt_input,
        auto_summarize_checkbox,
        api_name_input,
        api_key_input,
        preview_button,
        preview_status,
        preview_data_state,
        preview_data_display,
        metadata_overrides_input,
        ingest_button,
        ingest_status
    )

#
# End of Plain_text_import.py
#######################################################################################################################
