# XML_Ingestion_Tab.py
# Description: This file contains functions for reading and writing XML files.
#
# Imports
#
# External Imports
import gradio as gr
#
# Local Imports
from App_Function_Libraries.Utils.Utils import default_api_endpoint, global_api_endpoints, format_api_name, logging
from App_Function_Libraries.Plaintext.XML_Ingestion_Lib import import_xml_handler
#
#######################################################################################################################
#
# Functions:

def create_xml_import_tab():
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

    with gr.TabItem("Import XML Files", visible=True):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Import XML Files")
                gr.Markdown("Upload XML files for import")
                import_file = gr.File(label="Upload XML file", file_types=[".xml"])
                title_input = gr.Textbox(label="Title", placeholder="Enter the title of the content")
                keywords_input = gr.Textbox(label="Keywords", placeholder="Enter keywords, comma-separated")
                system_prompt_input = gr.Textbox(label="System Prompt (for Summarization)", lines=3,
                                                 value="""<s>[Your default system prompt here]</s>""")
                custom_prompt_input = gr.Textbox(label="Custom User Prompt",
                                                 placeholder="Enter a custom user prompt for summarization (optional)")
                auto_summarize_checkbox = gr.Checkbox(label="Auto-summarize/analyze", value=False)
                api_name_input = gr.Dropdown(
                    choices=["None"] + [format_api_name(api) for api in global_api_endpoints],
                    value=default_value,
                    label="API for Analysis/Summarization (Optional)"
                )
                api_key_input = gr.Textbox(label="API Key", type="password")
                import_button = gr.Button("Import XML File")
            with gr.Column():
                import_output = gr.Textbox(label="Import Status")

        import_button.click(
            fn=import_xml_handler,
            inputs=[import_file, title_input, keywords_input, system_prompt_input,
                    custom_prompt_input, auto_summarize_checkbox, api_name_input, api_key_input],
            outputs=import_output
        )

    return import_file, title_input, keywords_input, system_prompt_input, custom_prompt_input, auto_summarize_checkbox, api_name_input, api_key_input, import_button, import_output

#
# End of XML_Ingestion_Tab.py
#######################################################################################################################
