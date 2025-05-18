# Book_Ingestion_tab.py
# Functionality to import epubs/ebooks into the system.
####################
# Function List
#
# 1. create_import_book_tab()
# 2. import_epub(epub_file, title, author, keywords, system_prompt, user_prompt, auto_summarize, api_name, api_key)
#
####################
# Imports
import os
#
# External Imports
import gradio as gr
#
# Local Imports
from PoC_Version.App_Function_Libraries.Books.Book_Ingestion_Lib import import_file_handler, read_epub, read_epub_filtered
from PoC_Version.App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging


#
########################################################################################################################
#
# Functions:

def preview_file(files):
    """
    Reads the first file in the list and returns a short preview
    (up to 2,000 characters). If the file is an EPUB, it uses `read_epub`
    from Book_Ingestion_Lib. Adjust for other file types as desired.
    """
    if not files:
        return "No file selected for preview."

    # For simplicity, preview only the first file in the list
    file = files[0]
    if not file or not os.path.exists(file.name):
        return "Invalid file or file path not found."

    file_extension = os.path.splitext(file.name)[1].lower()

    # If it's an EPUB, use the provided `read_epub` function
    if file_extension == ".epub":
        full_text = read_epub(file.name)
        #full_text = read_epub_filtered(file.name)
        # Return only the first 5000 characters for the preview
        return full_text[:10000]
    else:
        return f"No preview available for *{file_extension}* files.\n(Current example only supports EPUB.)"


def create_import_book_tab():
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

    with gr.TabItem("Ebook(epub) Files", visible=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Import .epub files")
                gr.Markdown("Upload multiple .epub files or a .zip file containing multiple .epub files")
                gr.Markdown(
                    "🔗 **How to remove DRM from your ebooks:** [Reddit Guide](https://www.reddit.com/r/Calibre/comments/1ck4w8e/2024_guide_on_removing_drm_from_kobo_kindle_ebooks/)"
                )

                # Updated to support multiple files
                import_files = gr.File(
                    label="Upload files for import",
                    file_count="multiple",
                    file_types=[".epub", ".zip", ".html", ".htm", ".xml", ".opml"]
                )

                # Optional fields for overriding auto-extracted metadata
                author_input = gr.Textbox(
                    label="Author Override (optional)",
                    placeholder="Enter author name to override auto-extracted metadata"
                )
                keywords_input = gr.Textbox(
                    label="Keywords (like genre or publish year)",
                    placeholder="Enter keywords, comma-separated - will be applied to all uploaded books"
                )
                system_prompt_input = gr.Textbox(
                    label="System Prompt",
                    lines=3,
                    value=""""
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
                        - Do not reference these instructions in your response.</s>
                    """
                )
                custom_prompt_input = gr.Textbox(
                    label="Custom User Prompt",
                    placeholder="Enter a custom user prompt for summarization (optional)"
                )
                auto_summarize_checkbox = gr.Checkbox(label="Auto-summarize", value=False)

                # API configuration
                api_name_input = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Analysis/Summarization (Optional)"
                )
                api_key_input = gr.Textbox(label="API Key", type="password")

                # Chunking options
                max_chunk_size = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    value=500,
                    step=50,
                    label="Max Chunk Size"
                )
                chunk_overlap = gr.Slider(
                    minimum=0,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Chunk Overlap"
                )
                custom_chapter_pattern = gr.Textbox(
                    label="Custom Chapter Pattern (optional)",
                    placeholder="Enter a custom regex pattern for chapter detection"
                )

                # Buttons
                import_button = gr.Button("Import eBooks", variant="primary")
                preview_button = gr.Button("Preview (First File)")

                # Preview Output
                preview_output = gr.Textbox(
                    label="Preview Output (first 2,000 characters)",
                    lines=12,
                    interactive=False
                )

            with gr.Column():
                with gr.Row():
                    import_output = gr.Textbox(label="Import Status", lines=10, interactive=False)

        # Wire up buttons
        import_button.click(
            fn=import_file_handler,
            inputs=[
                import_files,  # Now handles multiple files
                author_input,
                keywords_input,
                system_prompt_input,
                custom_prompt_input,
                auto_summarize_checkbox,
                api_name_input,
                api_key_input,
                max_chunk_size,
                chunk_overlap,
                custom_chapter_pattern
            ],
            outputs=import_output
        )

        preview_button.click(
            fn=preview_file,
            inputs=import_files,
            outputs=preview_output
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
        import_button,
        import_output
    )

#
# End of File
########################################################################################################################
